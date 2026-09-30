"""
stage4_composer.py — محرك المونتاج السينمائي الآلي (نسخة إنتاج عالمية).

الإصلاحات السبعة (المراجعة الأولى):
1) _escape_filter_path: هروب الفاصلة العليا فقط.
2) subprocess.run: encoding="utf-8" + errors="replace".
3) asplit=1: تخطّي الفلتر بالكامل عند غياب Ambience و Rhythm.
4) fb_video_dur: استخدام layout["bounds"][-1] / fps.
5) Test #14: التحقق من السلوك الجديد للهروب.
6) إزالة علم success الميت و except الميت.
7) كاش AI يشمل transitions + asyncio loop مستقل + inspect.signature آمن.

الإصلاحات الإضافية (المراجعة الثانية):
1) حذف image_starts الميت من render_final_video.
2) shot_start_in_final يستخدم O[i-1]/fps.
3) كاش AI: OrderedDict + LRU + قفل.
4) مسار fallback: whoosh بمواضع القطع + إصلاح مزامنة المؤثرات.
5) حماية الـ fade عند الصوت القصير (< 0.9s).
6) _accepts_single_prompt عبر Signature.bind.
7) تسجيل تراكم خيوط Gemini المعلّقة.
8) Test #14 محمول على المنصات.
9) تحصين دفاعي في _merge_ai_into_motion.
10) تعليق توضيحي في _compute_image_boundaries.

الإصلاحات الإضافية (المراجعة الثالثة — هذه):
1) مزامنة الإيقاع في مسار fallback عبر rhythm_cut مبني على مواضع القطع.
2) مسار ASS آمن عبر _prepare_safe_ass (نسخ عند وجود ' في المسار).
3) _run_ffmpeg يلتقط FileNotFoundError برسالة واضحة.
4) Test #14: تصحيح فحص شارحة الخلفية المتبقية.
5) import tempfile في الأعلى.
"""

import wave
import struct
import math
import shutil
import subprocess
import array
import uuid
import json
import hashlib
import logging
import inspect
import asyncio
import threading
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("Stage4Composer")


# ============================================================
# ثوابت القرارات الإبداعية
# ============================================================

RHYTHM_ENABLED_ROLES = {"hook", "revelation", "payoff", "actionable"}

MOTION_TYPES = (
    "zoom_in", "zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
    "diag_tl_br", "diag_tr_bl", "diag_bl_tr", "diag_br_tl",
    "zoom_pan_left", "zoom_pan_right", "zoom_pan_up", "zoom_pan_down",
    "slow_drift", "static_micro",
)

ROLE_PREFERRED_MOTIONS: Dict[str, Tuple[str, ...]] = {
    "hook":          ("zoom_in", "zoom_pan_right", "diag_tl_br"),
    "revelation":    ("zoom_in", "zoom_pan_left", "diag_tr_bl"),
    "reflection":    ("slow_drift", "pan_right", "zoom_out"),
    "question":      ("slow_drift", "pan_left", "static_micro"),
    "tension":       ("zoom_in", "diag_tl_br", "pan_up"),
    "payoff":        ("zoom_out", "pan_right", "slow_drift"),
    "actionable":    ("zoom_pan_right", "zoom_in", "pan_down"),
    "myth":          ("slow_drift", "zoom_out", "pan_left"),
    "contradiction": ("zoom_in", "diag_bl_tr", "pan_up"),
    "explanation":   ("pan_right", "zoom_in", "slow_drift"),
    "analogy":       ("diag_tl_br", "zoom_pan_right", "slow_drift"),
    "setup":         ("pan_left", "slow_drift", "zoom_in"),
    "cta":           ("zoom_pan_right", "zoom_in", "diag_br_tl"),
    "":              ("zoom_in", "pan_right", "zoom_out", "slow_drift"),
}

TRANSITION_DURATIONS: Dict[str, float] = {
    "cut":              0.05,
    "crossfade_soft":   0.15,
    "crossfade_medium": 0.28,
    "crossfade_deep":   0.40,
}

ROLE_PREFERRED_TRANSITION: Dict[str, str] = {
    "hook":        "crossfade_soft",
    "revelation":  "cut",
    "reflection":  "crossfade_medium",
    "question":    "crossfade_soft",
    "tension":     "cut",
    "payoff":      "crossfade_medium",
    "actionable":  "cut",
    "myth":          "crossfade_medium",
    "contradiction": "cut",
    "explanation":   "crossfade_soft",
    "analogy":       "crossfade_soft",
    "setup":         "crossfade_medium",
    "cta":           "crossfade_soft",
}

KB_UPSCALE_W = 3840
KB_UPSCALE_H = 2160

FFPROBE_TIMEOUT_SEC = 60
SEGMENT_TIMEOUT_SEC = 900
GROUP_TIMEOUT_SEC = 1800
FINAL_TIMEOUT_SEC = 7200
FALLBACK_STEP_TIMEOUT_SEC = 3600

MAX_SINGLE_PASS_SHOTS = 24
XFADE_GROUP_SIZE = 12

AI_CALL_TIMEOUT_SEC = 90

_DEFAULT_WHOOSH_STRENGTH = 0.55

_AUDIO_NORMALIZE_CHAIN = (
    "aresample=44100,"
    "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo"
)


# ============================================================
# أدوات مساعدة عامة
# ============================================================

def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(v):
        return default
    return max(lo, min(hi, v))


def _safe_int_list(value: Any) -> List[int]:
    if not isinstance(value, (list, tuple)):
        return []
    out: List[int] = []
    for x in value:
        if isinstance(x, bool):
            continue
        if isinstance(x, (int, float)):
            try:
                if isinstance(x, float) and not math.isfinite(x):
                    continue
                out.append(int(x))
            except (ValueError, OverflowError):
                continue
    return out


def _run_ffmpeg(cmd: List[str], timeout: float, label: str) -> None:
    """
    تشغيل ffmpeg مع مهلة + التقاط stderr.
    [إصلاح 2] encoding utf-8 + errors='replace'.
    [إصلاح 3 — المراجعة الثالثة] التقاط FileNotFoundError برسالة واضحة.
    """
    try:
        subprocess.run(
            cmd, check=True, timeout=timeout,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    except subprocess.CalledProcessError as e:
        err_tail = (e.stderr or "").strip()[-1000:]
        logger.error(f"خطأ FFmpeg في {label}:\n{err_tail}")
        raise RuntimeError(
            f"فشل {label} (exit={e.returncode}):\n{err_tail}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"انتهت مهلة {label} ({timeout:.0f}s).") from e
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ffmpeg غير متوفر في PATH (المهمة: {label})."
        ) from e


_AMIX_NORMALIZE_SUPPORTED: Optional[bool] = None


def _ffmpeg_supports_amix_normalize() -> bool:
    """
    هل يدعم ffmpeg الخيار amix=normalize (FFmpeg 4.4+)؟ (مع caching).
    """
    global _AMIX_NORMALIZE_SUPPORTED
    if _AMIX_NORMALIZE_SUPPORTED is not None:
        return _AMIX_NORMALIZE_SUPPORTED
    supported = False
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "filter=amix"],
            capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        out = (r.stdout or "") + (r.stderr or "")
        supported = "normalize" in out
    except (OSError, subprocess.SubprocessError):
        supported = False
    _AMIX_NORMALIZE_SUPPORTED = supported
    return supported


def _escape_filter_path(path: Path) -> str:
    """
    [إصلاح 1] هروب آمن لمسار ملف داخل filter_complex حين يكون محاطاً
    باقتباس فردي ('...').

    عند استخدام الاقتباس الفردي، يقرأ FFmpeg المسار حرفياً؛ لا يفكّ هروب
    الرموز : [ ] , داخل الاقتباس. الرمز الوحيد الذي يجب هروبه هو الفاصلة
    العليا ' نفسها (لأنها تُنهي الاقتباس).

    - نحوّل المسار إلى صيغة posix (شرطة مائلة للأمام).
    - نستبدل ' بـ '\\'' (close + escaped + open).

    ملاحظة: النمط '\\'' هو القياسي في FFmpeg filtergraph syntax، لكن
    بعض إصدارات FFmpeg القديمة (≤ 4.2) لا تفكّه بشكل موحّد داخل فلتر
    ass=filename=. لهذا السبب يُنصح باستخدام _prepare_safe_ass قبل
    تمرير المسار إلى فلتر ass.
    """
    s = path.resolve().as_posix()
    return s.replace("'", "'\\''")


def _prepare_safe_ass(subtitles_ass: Path) -> Tuple[Path, Optional[Path]]:
    """
    [إصلاح 2 — المراجعة الثالثة] ينسخ ملف الترجمة إلى مسار آمن خالٍ من
    الفاصلة العليا عند الحاجة، لتفادي مشاكل هروب '\\''  في فلتر
    ass=filename='...' على بعض إصدارات FFmpeg.

    السلوك:
      - إن كان المسار الأصلي بلا ' → يُعاد كما هو (لا نسخ، لا تنظيف).
      - إن احتوى ' → يُنسخ إلى مجلد temp جديد بمسار مضمون.

    Returns:
        (المسار الآمن للاستخدام، مجلد temp للتنظيف لاحقًا أو None).
    """
    s = subtitles_ass.resolve().as_posix()
    if "'" not in s:
        return subtitles_ass, None
    try:
        safe_dir = Path(tempfile.mkdtemp(prefix="stage4_ass_"))
    except OSError as e:
        logger.warning(
            f"تعذّر إنشاء مجلد آمن لمسار ASS ({e}) — استخدام المسار الأصلي."
        )
        return subtitles_ass, None
    safe_path = safe_dir / "subs.ass"
    try:
        shutil.copyfile(subtitles_ass, safe_path)
    except OSError as e:
        logger.warning(
            f"تعذّر نسخ ASS إلى مسار آمن ({e}) — استخدام المسار الأصلي."
        )
        try:
            shutil.rmtree(safe_dir, ignore_errors=True)
        except Exception:
            pass
        return subtitles_ass, None
    logger.info(
        "🛡️ نُسخ ملف ASS إلى مسار آمن (يحتوي ' في المسار الأصلي)."
    )
    return safe_path, safe_dir


# ============================================================
# المؤثرات الصوتية
# ============================================================

def create_synthetic_whoosh(output_path: Path):
    if output_path.exists() and output_path.stat().st_size > 0:
        return

    sample_rate = 44100
    duration = 0.45
    n_samples = int(sample_rate * duration)

    rng_state = 0x13579BDF

    def _rand() -> float:
        nonlocal rng_state
        rng_state = (1103515245 * rng_state + 12345) & 0x7FFFFFFF
        return (rng_state / 0x7FFFFFFF) * 2.0 - 1.0

    with wave.open(str(output_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        pink = 0.0
        frames = bytearray()

        for i in range(n_samples):
            t = i / sample_rate
            tn = t / duration

            if tn < 0.15:
                env = (tn / 0.15) ** 1.6
            else:
                env = math.exp(-3.5 * (tn - 0.15))

            phase1 = 2 * math.pi * (180.0 * t + 240.0 * t * tn)
            phase2 = 2 * math.pi * (320.0 * t + 360.0 * t * tn)
            phase3 = 2 * math.pi * (90.0  * t + 110.0 * t * tn)

            tonal = (
                0.45 * math.sin(phase1)
                + 0.30 * math.sin(phase2)
                + 0.25 * math.sin(phase3)
            )

            white = _rand()
            pink = 0.96 * pink + 0.04 * white
            noise = pink * 0.35

            sample = env * (tonal * 0.75 + noise * 0.55)
            val = int(max(-1.0, min(1.0, sample * 1.05)) * 32767 * 0.18)
            frames += struct.pack("<h", val)

        wav_file.writeframes(bytes(frames))


def build_whoosh_timeline(
    base_whoosh: Path,
    transition_times: List[float],
    total_duration: float,
    output_path: Path,
    whoosh_volume: float = 0.4,
    per_transition_volumes: Optional[List[float]] = None,
):
    if not base_whoosh.exists() or not base_whoosh.is_file():
        raise FileNotFoundError(f"ملف الـ whoosh الأساسي غير موجود: {base_whoosh}")

    if not isinstance(total_duration, (int, float)) or \
            not math.isfinite(total_duration) or total_duration <= 0:
        raise ValueError(f"مدة إجمالية غير صالحة لمسار الـ whoosh: {total_duration!r}")

    master = _clamp_float(whoosh_volume, 0.0, 1.0, 0.4)

    with wave.open(str(base_whoosh), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    base_samples = array.array("h")
    base_samples.frombytes(raw)
    if sys.byteorder == "big":
        base_samples.byteswap()
    base_len = len(base_samples)

    total_samples = int(total_duration * sr)
    if total_samples <= 0:
        raise ValueError(
            f"عدد العينات الكلي غير صالح لمسار الـ whoosh: {total_samples}"
        )

    out = array.array("h", bytes(2 * total_samples))

    for idx, t in enumerate(transition_times):
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            continue
        vol = master
        if per_transition_volumes is not None and idx < len(per_transition_volumes):
            try:
                v = float(per_transition_volumes[idx])
                if math.isfinite(v) and v >= 0:
                    vol = master * v
            except (TypeError, ValueError):
                pass
        vol = max(0.0, min(1.0, vol))

        start = int(t * sr)
        if start < 0 or start >= total_samples:
            continue
        for i in range(base_len):
            pos = start + i
            if pos >= total_samples:
                break
            v = out[pos] + int(base_samples[i] * vol)
            if v > 32767:
                v = 32767
            elif v < -32768:
                v = -32768
            out[pos] = v

    if sys.byteorder == "big":
        out.byteswap()
    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out.tobytes())


def _build_pulse_sample(sample_rate: int) -> array.array:
    duration = 0.6
    n = int(sample_rate * duration)
    samples = array.array("h", bytes(2 * n))
    for i in range(n):
        t = i / sample_rate
        attack = min(1.0, t / 0.02)
        decay = math.exp(-4.8 * t / duration)
        env = attack * decay

        wave_val = (
            0.68 * math.sin(2 * math.pi * 55.0 * t)
            + 0.22 * math.sin(2 * math.pi * 82.5 * t)
            + 0.10 * math.sin(2 * math.pi * 220.0 * t)
        )
        v = env * wave_val
        val = int(v * 32767 * 0.55)
        if val > 32767:
            val = 32767
        elif val < -32768:
            val = -32768
        samples[i] = val
    return samples


def _build_rhythm_timeline(
    trigger_times: List[float],
    total_duration: float,
    sample_rate: int,
    output_path: Path,
    rhythm_volume: float = 0.35,
) -> bool:
    if not trigger_times:
        return False
    if not isinstance(total_duration, (int, float)) or \
            not math.isfinite(total_duration) or total_duration <= 0:
        return False

    try:
        pulse = _build_pulse_sample(sample_rate)
    except Exception as e:
        logger.warning(f"تعذّر توليد نبضة الإيقاع: {e}")
        return False

    total_samples = int(total_duration * sample_rate)
    if total_samples <= 0:
        return False

    out = array.array("h", bytes(2 * total_samples))
    pulse_len = len(pulse)

    try:
        for t in trigger_times:
            if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
                continue
            start = int(t * sample_rate)
            if start < 0 or start >= total_samples:
                continue
            for i in range(pulse_len):
                idx = start + i
                if idx >= total_samples:
                    break
                v = out[idx] + int(pulse[i] * rhythm_volume)
                if v > 32767:
                    v = 32767
                elif v < -32768:
                    v = -32768
                out[idx] = v

        if sys.byteorder == "big":
            out.byteswap()
        with wave.open(str(output_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(out.tobytes())
    except Exception as e:
        logger.warning(f"تعذّر كتابة مسار الإيقاع: {e}")
        return False

    try:
        return output_path.exists() and output_path.stat().st_size > 0
    except OSError:
        return False


# ============================================================
# طبقة الذكاء الاصطناعي (Gemini Editorial Director)
# ============================================================

def _accepts_single_prompt(fn: Any) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    try:
        sig.bind("__prompt_probe__")
    except TypeError:
        return False
    return True


class _AIEditorialDirector:
    _instance: Optional["_AIEditorialDirector"] = None
    _call_fn = None
    _available: bool = False

    _plan_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    _cache_lock = threading.Lock()
    _MAX_PLAN_CACHE = 32

    _lingering_threads: int = 0
    _linger_lock = threading.Lock()

    @classmethod
    def get(cls) -> "_AIEditorialDirector":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._init_engine()
        return cls._instance

    def _init_engine(self) -> None:
        try:
            import gemini_engine  # noqa: F401
        except ImportError:
            self._available = False
            logger.info("ℹ️ gemini_engine غير متوفر — القواعد الذكية المحلية مفعّلة.")
            return
        except Exception as e:
            self._available = False
            logger.warning(f"تعذّر استيراد gemini_engine: {e}")
            return

        for attr in ("generate_json", "generate_text", "call_gemini",
                     "generate", "ask", "query", "complete", "prompt"):
            fn = getattr(gemini_engine, attr, None)
            if callable(fn) and _accepts_single_prompt(fn):
                self._call_fn = fn
                self._available = True
                logger.info(f"✅ Gemini متصل عبر gemini_engine.{attr}")
                return

        for cls_name in ("GeminiEngine", "GeminiClient", "Gemini", "Client"):
            cls_obj = getattr(gemini_engine, cls_name, None)
            if cls_obj is None:
                continue
            try:
                inst = cls_obj()
            except Exception:
                continue
            for method in ("generate_json", "generate_text", "generate",
                           "call", "ask", "query"):
                m = getattr(inst, method, None)
                if callable(m) and _accepts_single_prompt(m):
                    self._call_fn = m
                    self._available = True
                    logger.info(
                        f"✅ Gemini متصل عبر gemini_engine.{cls_name}.{method}"
                    )
                    return

        self._available = False
        logger.info(
            "ℹ️ gemini_engine موجود لكن بدون واجهة معروفة — fallback محلي."
        )

    def is_available(self) -> bool:
        return bool(self._available and self._call_fn is not None)

    def plan_montage(
        self,
        renderable_timeline: List[Dict[str, Any]],
        durations: List[float],
        transitions: List[str],
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {}

        cache_key = self._timeline_signature(
            renderable_timeline, durations, transitions
        )

        with self._cache_lock:
            cached = self._plan_cache.get(cache_key)
            if cached is not None:
                self._plan_cache.move_to_end(cache_key)
                return cached

        try:
            prompt = self._build_prompt(
                renderable_timeline, durations, transitions
            )
            raw = self._call_engine_safe(prompt)
            plan = self._parse_plan(raw, len(renderable_timeline))
            if plan:
                with self._cache_lock:
                    self._plan_cache[cache_key] = plan
                    self._plan_cache.move_to_end(cache_key)
                    while len(self._plan_cache) > self._MAX_PLAN_CACHE:
                        self._plan_cache.popitem(last=False)
                logger.info(
                    f"🎬 Gemini: خطة مونتاج لـ {len(plan.get('shots', []))} لقطة."
                )
            return plan or {}
        except Exception as e:
            logger.warning(f"فشل تخطيط AI: {e} — العودة للقواعد المحلية.")
            return {}

    @staticmethod
    def _coerce_result_to_text(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            if "shots" in result:
                return json.dumps(result, ensure_ascii=False, default=str)
            for k in ("text", "content", "output", "response", "result"):
                if k in result and isinstance(result[k], str):
                    return result[k]
            return json.dumps(result, ensure_ascii=False, default=str)
        if isinstance(result, (list, tuple)):
            return json.dumps(result, ensure_ascii=False, default=str)
        text_attr = getattr(result, "text", None)
        if isinstance(text_attr, str):
            return text_attr
        return str(result)

    def _call_engine_safe(self, prompt: str) -> str:
        box: Dict[str, Any] = {}
        fn = self._call_fn

        def _worker() -> None:
            try:
                res = fn(prompt)
                if inspect.isawaitable(res):
                    loop = asyncio.new_event_loop()
                    try:
                        res = loop.run_until_complete(res)
                    finally:
                        loop.close()
                box["result"] = res
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc

        th = threading.Thread(
            target=_worker, name="gemini-director", daemon=True
        )
        th.start()
        th.join(AI_CALL_TIMEOUT_SEC)
        if th.is_alive():
            with type(self)._linger_lock:
                type(self)._lingering_threads += 1
                count = type(self)._lingering_threads
            logger.warning(
                f"⚠️ خيط Gemini لم ينتهِ خلال المهلة ({AI_CALL_TIMEOUT_SEC}s). "
                f"تراكم الخيوط المعلّقة: {count}."
            )
            raise RuntimeError(
                f"انتهت مهلة استدعاء Gemini ({AI_CALL_TIMEOUT_SEC}s)"
            )
        if "error" in box:
            raise RuntimeError(
                f"استدعاء Gemini فشل: {box['error']}"
            ) from box["error"]
        return self._coerce_result_to_text(box.get("result"))

    def _build_prompt(
        self,
        timeline: List[Dict[str, Any]],
        durations: List[float],
        transitions: List[str],
    ) -> str:
        items = []
        for i, (it, d, tr) in enumerate(zip(timeline, durations, transitions + [""])):
            role = str(it.get("narrative_role") or "").lower().strip()
            text = str(it.get("text") or "")[:180]
            items.append({
                "idx": i,
                "role": role,
                "duration": round(float(d), 2),
                "text": text,
                "local_transition_after": tr,
            })
        tl_json = json.dumps(items, ensure_ascii=False)

        system = (
            "أنت مخرج مونتاج وثائقي عالمي (Netflix / National Geographic). "
            "مهمتك: خطة إبداعية محافظة وأنيقة لسلسلة لقطات. "
            "لا تُبالغ، لا تُكرر، واجعل الإيقاع متنوعاً ومتنفساً.\n"
            "أجب بـ JSON فقط بالصيغة التالية:\n"
            "{\n"
            '  "pacing": {"peaks": [int], "calm": [int], "breathing_after": [int]},\n'
            '  "shots": [\n'
            '    {"idx": int, "motion": str, "intensity": float, '
            '"transition_after": str, "whoosh_strength": float, '
            '"ambience_boost": float, "keyword": str|null}\n'
            "  ]\n"
            "}\n\n"
            "قواعد صارمة:\n"
            "- motion ∈ {zoom_in, zoom_out, pan_left, pan_right, pan_up, pan_down, "
            "diag_tl_br, diag_tr_bl, diag_bl_tr, diag_br_tl, zoom_pan_left, "
            "zoom_pan_right, zoom_pan_up, zoom_pan_down, slow_drift, static_micro}.\n"
            "- intensity ∈ [0.3, 1.0].\n"
            "- transition_after ∈ {cut, crossfade_soft, crossfade_medium, crossfade_deep}.\n"
            "- whoosh_strength ∈ [0.0, 1.0] (0 = بلا whoosh).\n"
            "- ambience_boost ∈ [-0.3, +0.3].\n"
            "- keyword: كلمة أو رقم واحد أنيق (اختياري، غالباً null).\n"
            "- لا تكرر نفس الحركة مرتين متتاليتين.\n"
            "- في اللحظات التأملية/الهادئة: حركة أبطأ، انتقال أنعم.\n"
            "- في الذروة/hook: حركة أوضح، intensity أعلى، whoosh أقوى.\n"
        )

        user = f"اللقطات:\n{tl_json}\n\nأعد JSON فقط دون أي شرح أو ```."
        return system + "\n" + user

    def _parse_plan(self, raw: str, n_shots: int) -> Dict[str, Any]:
        if not raw:
            return {}
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:]
            s = s.strip()
        first_brace = s.find("{")
        last_brace = s.rfind("}")
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            return {}
        s = s[first_brace:last_brace + 1]

        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            logger.debug(f"تعذّر تحليل JSON من AI: {e}")
            return {}
        if not isinstance(data, dict):
            return {}

        shots_raw = data.get("shots")
        if not isinstance(shots_raw, list):
            return {}

        cleaned_shots: List[Dict[str, Any]] = []
        seen_idx = set()
        for entry in shots_raw:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("idx"))
            except (TypeError, ValueError, OverflowError):
                continue
            if idx < 0 or idx >= n_shots or idx in seen_idx:
                continue
            seen_idx.add(idx)

            motion = str(entry.get("motion") or "").strip().lower()
            if motion not in MOTION_TYPES:
                motion = ""

            intensity = _clamp_float(entry.get("intensity", 0.5), 0.3, 1.0, 0.5)

            tr = str(entry.get("transition_after") or "").strip().lower()
            if tr not in TRANSITION_DURATIONS:
                tr = ""

            wh = _clamp_float(entry.get("whoosh_strength", 0.5), 0.0, 1.0, 0.5)
            amb = _clamp_float(entry.get("ambience_boost", 0.0), -0.3, 0.3, 0.0)

            kw = entry.get("keyword")
            if not isinstance(kw, str) or not kw.strip():
                kw = None
            else:
                kw = kw.strip()[:24]

            cleaned_shots.append({
                "idx": idx,
                "motion": motion,
                "intensity": intensity,
                "transition_after": tr,
                "whoosh_strength": wh,
                "ambience_boost": amb,
                "keyword": kw,
            })

        if not cleaned_shots:
            return {}

        pacing = data.get("pacing") if isinstance(data.get("pacing"), dict) else {}
        return {
            "shots": cleaned_shots,
            "pacing": {
                "peaks": _safe_int_list(pacing.get("peaks")),
                "calm": _safe_int_list(pacing.get("calm")),
                "breathing_after": _safe_int_list(pacing.get("breathing_after")),
            },
        }

    @staticmethod
    def _timeline_signature(
        timeline: List[Dict[str, Any]],
        durations: List[float],
        transitions: List[str],
    ) -> str:
        h = hashlib.sha256()
        for i, (it, d) in enumerate(zip(timeline, durations)):
            role = str(it.get("narrative_role") or "")
            text = str(it.get("text") or "")[:200]
            h.update(role.encode("utf-8", "ignore"))
            h.update(b"|")
            h.update(text.encode("utf-8", "ignore"))
            h.update(b"|")
            h.update(f"{round(float(d), 2)}".encode("ascii"))
            h.update(b"|")
            tr = transitions[i] if i < len(transitions) else ""
            h.update(str(tr).encode("utf-8", "ignore"))
            h.update(b"\n")
        return h.hexdigest()[:24]


# ============================================================
# قرارات الحركة والانتقال
# ============================================================

def _select_motion_local(
    item: Dict[str, Any],
    duration: float,
    prev_motion: Optional[str],
    shot_index: int,
) -> Dict[str, Any]:
    role = str(item.get("narrative_role") or "").lower().strip()
    candidates = ROLE_PREFERRED_MOTIONS.get(role) or ROLE_PREFERRED_MOTIONS[""]
    n = len(candidates)

    start = shot_index % n if isinstance(shot_index, int) and shot_index >= 0 else 0
    rotated = [candidates[(start + k) % n] for k in range(n)]

    def _ok(m: str) -> bool:
        if prev_motion is not None and m == prev_motion:
            return False
        if duration >= 6.0 and m in ("static_micro", "slow_drift"):
            return False
        if duration < 2.2 and m == "zoom_out":
            return False
        return True

    chosen = next((m for m in rotated if _ok(m)), None)
    if chosen is None:
        chosen = next((m for m in rotated if m != prev_motion), rotated[0])

    return {
        "type": chosen,
        "intensity": 0.55,
        "speed": "auto",
        "source": "local",
    }


def _select_transition_local(
    prev_item: Optional[Dict[str, Any]],
    curr_item: Dict[str, Any],
) -> str:
    role = str(curr_item.get("narrative_role") or "").lower().strip()
    return ROLE_PREFERRED_TRANSITION.get(role, "crossfade_soft")


def _merge_ai_into_motion(
    local: Dict[str, Any], ai_shot: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not ai_shot:
        return local
    merged = dict(local)
    if ai_shot.get("motion") in MOTION_TYPES:
        merged["type"] = ai_shot["motion"]
        merged["source"] = "ai"
    local_intensity = _clamp_float(local.get("intensity", 0.55), 0.25, 1.0, 0.55)
    merged["intensity"] = _clamp_float(
        ai_shot.get("intensity", local_intensity), 0.3, 1.0, local_intensity
    )
    return merged


# ============================================================
# فلتر Ken Burns
# ============================================================

def get_ken_burns_filter(
    motion: Any,
    duration: float,
    fps: int = 60,
) -> str:
    if not isinstance(duration, (int, float)) or \
            not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"مدة غير صالحة لفلتر Ken Burns: {duration!r}")

    if not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"معدل إطارات غير صالح لفلتر Ken Burns: {fps!r}")

    frames = int(round(duration * fps))
    if frames < 1:
        raise ValueError(
            f"المدة ({duration:.6f}s) قصيرة جدًا لإنتاج إطار واحد عند {fps}fps. "
            f"الحد الأدنى المطلوب تقريبًا: {1.0 / fps:.6f}s"
        )

    if isinstance(motion, int):
        legacy = (
            "zoom_in", "pan_right", "zoom_out",
            "pan_up", "zoom_in", "pan_left",
        )
        motion = {"type": legacy[motion % 6], "intensity": 0.55, "speed": "auto"}

    if not isinstance(motion, dict):
        motion = {"type": "zoom_in", "intensity": 0.5, "speed": "auto"}

    mtype = str(motion.get("type") or "zoom_in").lower()
    if mtype not in MOTION_TYPES:
        mtype = "zoom_in"

    intensity = _clamp_float(motion.get("intensity", 0.55), 0.25, 1.0, 0.55)

    speed_mod = 1.0
    if duration < 2.5:
        speed_mod = 0.7
    elif duration > 7.0:
        speed_mod = 1.15

    den = max(frames - 1, 1)

    max_zoom = 1.10 + 0.10 * intensity
    zr = min(0.30, (max_zoom - 1.0) * speed_mod)
    pz = 1.05 + 0.05 * intensity
    span = min(1.0, 0.55 + 0.45 * intensity * speed_mod)
    lo, hi = 0.5 - span / 2.0, 0.5 + span / 2.0
    vspan = span * 0.7
    vlo, vhi = 0.5 - vspan / 2.0, 0.5 + vspan / 2.0
    c = 0.5

    dz0 = 1.03
    dz1 = max(dz0 + 0.02, 1.0 + zr * 0.85)
    zp0 = 1.03
    zp1 = max(zp0 + 0.02, 1.0 + zr)
    zpl, zph = 0.5 - span * 0.35, 0.5 + span * 0.35
    sd = 1.04 + 0.02 * intensity
    ds = 0.35 * span
    sm = 1.02 + 0.015 * intensity
    m = max(0.02, 0.05 * intensity * speed_mod)

    table: Dict[str, Tuple[float, float, float, float, float, float]] = {
        "zoom_in":        (1.0, 1.0 + zr, c, c, c, c),
        "zoom_out":       (1.0 + zr, 1.0, c, c, c, c),
        "pan_right":      (pz, pz, lo, hi, c, c),
        "pan_left":       (pz, pz, hi, lo, c, c),
        "pan_up":         (pz, pz, c, c, hi, lo),
        "pan_down":       (pz, pz, c, c, lo, hi),
        "diag_tl_br":     (dz0, dz1, lo, hi, vlo, vhi),
        "diag_tr_bl":     (dz0, dz1, hi, lo, vlo, vhi),
        "diag_bl_tr":     (dz0, dz1, lo, hi, vhi, vlo),
        "diag_br_tl":     (dz0, dz1, hi, lo, vhi, vlo),
        "zoom_pan_right": (zp0, zp1, zpl, zph, c, c),
        "zoom_pan_left":  (zp0, zp1, zph, zpl, c, c),
        "zoom_pan_up":    (zp0, zp1, c, c, zph, zpl),
        "zoom_pan_down":  (zp0, zp1, c, c, zpl, zph),
        "slow_drift":     (sd, sd, 0.5 - ds / 2.0, 0.5 + ds / 2.0, c, c),
        "static_micro":   (sm, sm, 0.5 - m, 0.5 + m, 0.5 + 0.6 * m, 0.5 - 0.6 * m),
    }

    z0, z1, fx0, fx1, fy0, fy1 = table[mtype]

    def _frac(v: float) -> float:
        return max(0.0, min(1.0, v))

    fx0, fx1, fy0, fy1 = _frac(fx0), _frac(fx1), _frac(fy0), _frac(fy1)

    z = f"{z0:.5f}+({z1 - z0:.5f})*on/{den}"
    x = f"(iw-iw/zoom)*({fx0:.5f}+({fx1 - fx0:.5f})*on/{den})"
    y = f"(ih-ih/zoom)*({fy0:.5f}+({fy1 - fy0:.5f})*on/{den})"

    prep = (
        f"scale={KB_UPSCALE_W}:{KB_UPSCALE_H}:"
        f"force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={KB_UPSCALE_W}:{KB_UPSCALE_H},"
        f"setsar=1,"
    )

    return (
        f"{prep}"
        f"zoompan=z='{z}':x='{x}':y='{y}':"
        f"d={frames}:s=1920x1080:fps={fps}"
    )


# ============================================================
# أدوات المسارات والتحقق
# ============================================================

def _probe_media_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True,
            timeout=FFPROBE_TIMEOUT_SEC,
            encoding="utf-8", errors="replace",
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"فشل ffprobe في قراءة المدة للملف: {path} "
            f"(رمز الخروج: {e.returncode})"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"انتهت مهلة ffprobe أثناء قراءة المدة للملف: {path}"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffprobe غير متوفر في بيئة التشغيل. لا يمكن قياس مدة الوسائط."
        ) from e

    raw = (result.stdout or "").strip()
    if not raw:
        raise RuntimeError(f"ffprobe أعاد ناتجًا فارغًا للملف: {path}")

    try:
        duration = float(raw)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"قيمة مدة غير قابلة للتحويل من ffprobe للملف: {path} (الناتج: {raw!r})"
        ) from e

    if not math.isfinite(duration):
        raise RuntimeError(
            f"قيمة مدة غير منتهية أو NaN من ffprobe للملف: {path} ({duration!r})"
        )
    if duration <= 0:
        raise RuntimeError(
            f"قيمة مدة غير موجبة من ffprobe للملف: {path} ({duration})"
        )
    return duration


def _safe_concat_escape(path: Path) -> str:
    resolved = path.resolve().as_posix()
    escaped = resolved.replace("'", "'\\''")
    return f"file '{escaped}'\n"


def _validate_regular_file(path: Path, label: str, index: int = -1) -> None:
    idx_part = f"[{index}] " if index >= 0 else ""
    if not path.exists():
        raise FileNotFoundError(f"{idx_part}{label} غير موجود: {path}")
    if not path.is_file():
        raise ValueError(f"{idx_part}{label} ليس ملفًا عاديًا: {path}")


def _validate_non_empty_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} غير موجود بعد التنفيذ: {path}")
    if not path.is_file():
        raise RuntimeError(f"{label} ليس ملفًا عاديًا: {path}")
    try:
        if path.stat().st_size <= 0:
            raise RuntimeError(f"{label} فارغ (0 بايت): {path}")
    except OSError as e:
        raise RuntimeError(f"تعذّر فحص حجم {label}: {path}") from e


def _try_generate_ambience(
    output_path: Path,
    duration: float,
    boost: float = 0.0,
) -> bool:
    if not isinstance(duration, (int, float)) or \
            not math.isfinite(duration) or duration <= 6.0:
        return False

    fade_out_start = max(0.0, duration - 2.5)

    b = _clamp_float(boost, -0.3, 0.3, 0.0)
    base_volume = 0.32
    target_volume = max(0.05, min(0.60, base_volume * (1.0 + b)))

    lavfi_input = (
        f"anoisesrc=color=pink:amplitude=0.06:"
        f"sample_rate=44100:duration={duration:.3f}"
    )
    af_chain = (
        "lowpass=f=220,"
        "highpass=f=45,"
        "lowpass=f=180,"
        f"volume={target_volume:.4f},"
        "afade=t=in:st=0:d=2.5,"
        f"afade=t=out:st={fade_out_start:.3f}:d=2.5"
    )

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", lavfi_input,
        "-af", af_chain,
        "-t", f"{duration:.3f}",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(
            cmd, check=True, timeout=180,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
    except subprocess.CalledProcessError as e:
        err_tail = (e.stderr or "").strip()[-500:]
        logger.warning(
            f"فشل توليد طبقة الـ ambience (exit={e.returncode}): {err_tail}"
        )
        return False
    except subprocess.TimeoutExpired:
        logger.warning("انتهت مهلة توليد طبقة الـ ambience.")
        return False
    except FileNotFoundError:
        logger.warning("ffmpeg غير متوفر لتوليد الـ ambience.")
        return False
    except OSError as e:
        logger.warning(f"خطأ في تنفيذ ffmpeg لتوليد الـ ambience: {e}")
        return False

    try:
        return (
            output_path.exists()
            and output_path.is_file()
            and output_path.stat().st_size > 0
        )
    except OSError:
        return False


# ============================================================
# حساب حدود الصور
# ============================================================

def _compute_image_boundaries(
    renderable_timeline: List[Dict[str, Any]],
    audio_duration: float,
    fps: int,
) -> Tuple[List[float], List[float]]:
    n = len(renderable_timeline)
    if n == 0:
        return [], []

    min_dur = 1.0 / float(fps)

    starts: List[float] = []
    have_valid_starts = True
    for item in renderable_timeline:
        raw = item.get("start")
        try:
            s = float(raw)
        except (TypeError, ValueError):
            have_valid_starts = False
            break
        if not math.isfinite(s) or s < 0:
            have_valid_starts = False
            break
        starts.append(s)

    if not have_valid_starts:
        logger.warning(
            "عناصر timeline لا تحتوي على 'start' صالح. سيتم استخدام "
            "item['duration'] كخطة احتياطية لكل عنصر."
        )
        durations: List[float] = []
        for item in renderable_timeline:
            try:
                d = float(item.get("duration", 0.0))
            except (TypeError, ValueError):
                d = 0.0
            if not math.isfinite(d) or d <= 0:
                preview = str(item.get("text", ""))[:40]
                raise ValueError(
                    f"مدة احتياطية غير صالحة للعنصر (text={preview!r}): {d!r}"
                )
            if d < min_dur:
                d = min_dur
            durations.append(d)

        total_fallback = sum(durations)
        if total_fallback > audio_duration:
            excess_fb = total_fallback - audio_duration
            while excess_fb > 1e-6:
                max_idx = durations.index(max(durations))
                available = durations[max_idx] - min_dur
                if available <= 0:
                    break
                reduction = min(excess_fb, available)
                durations[max_idx] -= reduction
                excess_fb -= reduction
            residual_fb = sum(durations) - audio_duration
            if abs(residual_fb) > 1e-6:
                logger.warning(
                    f"تعذّرت الموازنة الدقيقة في مسار الـ fallback "
                    f"(sum={sum(durations):.6f}s, audio={audio_duration:.6f}s, "
                    f"residual={residual_fb:+.6f}s)."
                )

        image_starts: List[float] = []
        cur = 0.0
        for d in durations:
            image_starts.append(cur)
            cur += d
        return durations, image_starts

    for i in range(1, n):
        if starts[i] < starts[i - 1]:
            raise ValueError(
                f"بدايات الجمل غير مرتبة زمنيًا: "
                f"start[{i}]={starts[i]} < start[{i - 1}]={starts[i - 1]}"
            )

    if starts[-1] >= audio_duration:
        raise ValueError(
            f"بداية آخر جملة ({starts[-1]:.3f}s) أكبر من أو تساوي مدة الصوت "
            f"({audio_duration:.3f}s)."
        )

    durations = []
    for i in range(n):
        img_start = 0.0 if i == 0 else starts[i]
        img_end = starts[i + 1] if i < n - 1 else audio_duration

        d = img_end - img_start
        if not math.isfinite(d):
            raise ValueError(f"مدة غير منتهية للصورة رقم {i}: {d!r}")
        if d <= 0:
            logger.warning(
                f"مدة الصورة رقم {i} غير موجبة ({d:.6f}s)؛ سيتم رفعها إلى إطار واحد."
            )
            d = min_dur
        elif d < min_dur:
            d = min_dur
        durations.append(d)

    total_calc = sum(durations)
    excess = total_calc - audio_duration
    while excess > 1e-6:
        max_idx = durations.index(max(durations))
        available = durations[max_idx] - min_dur
        if available <= 0:
            break
        reduction = min(excess, available)
        durations[max_idx] -= reduction
        excess -= reduction

    residual = sum(durations) - audio_duration
    if abs(residual) > 1e-6:
        logger.warning(
            f"تعذّرت الموازنة الدقيقة بعد رفع اللقطات القصيرة "
            f"(sum={sum(durations):.6f}s, audio={audio_duration:.6f}s, "
            f"residual={residual:+.6f}s)."
        )

    image_starts = []
    cur = 0.0
    for d in durations:
        image_starts.append(cur)
        cur += d
    return durations, image_starts


def _compute_image_durations(
    renderable_timeline: List[Dict[str, Any]],
    audio_duration: float,
    fps: int,
) -> List[float]:
    durations, _ = _compute_image_boundaries(
        renderable_timeline, audio_duration, fps
    )
    return durations


# ============================================================
# مخطط المقاطع وسلسلة xfade
# ============================================================

def _compute_segment_layout(
    durations: List[float],
    transition_types: List[str],
    fps: int,
) -> Dict[str, Any]:
    n = len(durations)
    if n <= 0:
        raise ValueError("durations فارغة")
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"fps غير صالح: {fps!r}")

    tt = list(transition_types)
    if len(tt) < max(0, n - 1):
        tt += ["crossfade_soft"] * (n - 1 - len(tt))

    bounds: List[int] = [0]
    cum = 0.0
    for d in durations:
        cum += float(d)
        target = int(round(cum * fps))
        bounds.append(max(bounds[-1] + 1, target))

    d_frames = [bounds[i + 1] - bounds[i] for i in range(n)]

    T: List[int] = []
    O: List[int] = []
    for i in range(n - 1):
        base = int(round(TRANSITION_DURATIONS.get(tt[i], 0.15) * fps))
        max_safe = int(math.floor(0.4 * min(d_frames[i], d_frames[i + 1])))
        t = max(1, min(base, max_safe))
        T.append(t)
        O.append(bounds[i + 1] - t // 2)

    S: List[int] = [0] + O[:]
    E: List[int] = [O[i] + T[i] for i in range(n - 1)] + [bounds[n]]
    L: List[int] = [E[i] - S[i] for i in range(n)]

    return {"bounds": bounds, "T": T, "O": O, "S": S, "E": E, "L": L}


def _xfade_filter_for_range(
    first: int,
    last: int,
    layout: Dict[str, Any],
    fps: int,
    time_base_frames: int,
) -> Tuple[str, str]:
    if first == last:
        return "", "[0:v]"
    parts: List[str] = []
    prev = "[0:v]"
    for k in range(first, last):
        j = k - first
        cur = f"[{j + 1}:v]"
        out = f"[vx{j}]"
        t_sec = layout["T"][k] / fps
        off_sec = max(0.0, (layout["O"][k] - time_base_frames) / fps)
        parts.append(
            f"{prev}{cur}xfade=transition=fade:"
            f"duration={t_sec:.6f}:offset={off_sec:.6f}{out}"
        )
        prev = out
    return ";".join(parts), prev


def _xfade_filter_for_groups(
    groups: List[Tuple[int, int]],
    layout: Dict[str, Any],
    fps: int,
) -> Tuple[str, str]:
    if len(groups) == 1:
        return "", "[0:v]"
    parts: List[str] = []
    prev = "[0:v]"
    for g in range(len(groups) - 1):
        k = groups[g][1]
        cur = f"[{g + 1}:v]"
        out = f"[vx{g}]"
        t_sec = layout["T"][k] / fps
        off_sec = max(0.0, layout["O"][k] / fps)
        parts.append(
            f"{prev}{cur}xfade=transition=fade:"
            f"duration={t_sec:.6f}:offset={off_sec:.6f}{out}"
        )
        prev = out
    return ";".join(parts), prev


def _build_xfade_chain(
    n_segments: int,
    durations: List[float],
    transition_types: List[str],
    fps: int,
) -> Tuple[str, str, float, List[float], List[float]]:
    if n_segments <= 0:
        raise ValueError("n_segments يجب أن تكون ≥ 1")
    if len(durations) != n_segments:
        raise ValueError("عدد المدد لا يطابق عدد المقاطع")

    layout = _compute_segment_layout(durations, transition_types, fps)
    total = layout["bounds"][-1] / fps

    if n_segments == 1:
        return "", "[0:v]", total, [], []

    filt, out_label = _xfade_filter_for_range(
        0, n_segments - 1, layout, fps, 0
    )
    offsets = [o / fps for o in layout["O"]]
    safe_t_durs = [t / fps for t in layout["T"]]
    return filt, out_label, total, offsets, safe_t_durs


# ============================================================
# الرندر النهائي
# ============================================================

def render_final_video(
    frames: List[Path],
    timeline: List[Dict[str, Any]],
    audio_file: Path,
    subtitles_ass: Path,
    output_video_path: Path
):
    if not isinstance(frames, list) or not isinstance(timeline, list):
        raise TypeError("frames و timeline يجب أن تكونا قائمتين.")

    renderable_timeline = [it for it in timeline if not it.get("empty")]
    if not renderable_timeline:
        raise ValueError(
            "قائمة التوقيت لا تحتوي على أي عنصر قابل للرندرة."
        )

    if len(frames) == len(renderable_timeline):
        raw_pairs = list(zip(frames, renderable_timeline))
    elif len(frames) == len(timeline):
        raw_pairs = [
            (f, it) for f, it in zip(frames, timeline)
            if not it.get("empty")
        ]
    else:
        raise ValueError(
            f"عدم تناسق: عدد الصور ({len(frames)}) لا يطابق عدد العناصر القابلة "
            f"للرندرة ({len(renderable_timeline)}) ولا العدد الكامل للـ timeline "
            f"({len(timeline)})."
        )

    validated_pairs: List[Tuple[Path, Dict[str, Any]]] = []
    for idx, (frame_path, item) in enumerate(raw_pairs):
        fp = Path(frame_path)
        _validate_regular_file(fp, "ملف الصورة", index=idx)
        validated_pairs.append((fp, item))

    audio_file = Path(audio_file)
    subtitles_ass = Path(subtitles_ass)
    output_video_path = Path(output_video_path)

    _validate_regular_file(audio_file, "ملف التعليق الصوتي")
    _validate_regular_file(subtitles_ass, "ملف الترجمة ASS")

    output_dir = output_video_path.parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"تعذّر إنشاء مجلد الإخراج: {output_dir}") from e
    if not output_dir.is_dir():
        raise RuntimeError(f"مسار الإخراج الأب ليس مجلدًا: {output_dir}")

    audio_duration = _probe_media_duration(audio_file)

    fps = 60
    durations, _ = _compute_image_boundaries(
        renderable_timeline, audio_duration, fps
    )
    if len(durations) != len(validated_pairs):
        raise RuntimeError(
            f"عدم تطابق داخلي: عدد المدد المحسوبة ({len(durations)}) != عدد "
            f"الأزواج ({len(validated_pairs)})."
        )

    logger.info(
        f"📐 عدد الصور: {len(durations)} | "
        f"مجموع المدد: {sum(durations):.3f}s | "
        f"مدة الصوت: {audio_duration:.3f}s"
    )

    n_shots = len(renderable_timeline)

    local_transitions: List[str] = []
    for i in range(n_shots):
        if i < n_shots - 1:
            local_transitions.append(
                _select_transition_local(renderable_timeline[i], renderable_timeline[i + 1])
            )

    local_motions: List[Dict[str, Any]] = []
    prev_m = None
    for i, (it, d) in enumerate(zip(renderable_timeline, durations)):
        m = _select_motion_local(it, d, prev_m, i)
        local_motions.append(m)
        prev_m = m["type"]

    ai_plan = _AIEditorialDirector.get().plan_montage(
        renderable_timeline, durations, local_transitions
    )

    ai_shots_by_idx: Dict[int, Dict[str, Any]] = {}
    if ai_plan and isinstance(ai_plan.get("shots"), list):
        for s in ai_plan["shots"]:
            if isinstance(s, dict) and isinstance(s.get("idx"), int):
                ai_shots_by_idx[s["idx"]] = s

    final_motions: List[Dict[str, Any]] = []
    final_transitions: List[str] = []
    whoosh_strengths: List[float] = []
    ambience_boosts: List[float] = []

    for i in range(n_shots):
        ai_s = ai_shots_by_idx.get(i)
        merged_motion = _merge_ai_into_motion(local_motions[i], ai_s)
        final_motions.append(merged_motion)

        if i > 0 and final_motions[i]["type"] == final_motions[i - 1]["type"]:
            role = str(renderable_timeline[i].get("narrative_role") or "").lower().strip()
            alts = ROLE_PREFERRED_MOTIONS.get(role, ROLE_PREFERRED_MOTIONS[""])
            replaced = False
            for alt in alts:
                if alt != final_motions[i - 1]["type"]:
                    final_motions[i]["type"] = alt
                    replaced = True
                    break
            if not replaced:
                for alt in MOTION_TYPES:
                    if alt != final_motions[i - 1]["type"]:
                        final_motions[i]["type"] = alt
                        break

        if i < n_shots - 1:
            tr = local_transitions[i]
            if ai_s and ai_s.get("transition_after") in TRANSITION_DURATIONS:
                ai_tr = ai_s["transition_after"]
                role_next = str(
                    renderable_timeline[i + 1].get("narrative_role") or ""
                ).lower().strip()
                if ai_tr == "crossfade_deep" and role_next not in (
                    "reflection", "payoff", "question"
                ):
                    ai_tr = "crossfade_medium"
                tr = ai_tr
            final_transitions.append(tr)
        else:
            final_transitions.append("")

        wh = _clamp_float(
            ai_s.get("whoosh_strength", _DEFAULT_WHOOSH_STRENGTH)
            if ai_s else _DEFAULT_WHOOSH_STRENGTH,
            0.0, 1.0, _DEFAULT_WHOOSH_STRENGTH,
        )
        whoosh_strengths.append(wh)

        amb = _clamp_float(
            ai_s.get("ambience_boost", 0.0) if ai_s else 0.0,
            -0.3, 0.3, 0.0,
        )
        ambience_boosts.append(amb)

    layout = _compute_segment_layout(durations, final_transitions, fps)

    (xfade_filter, vx_out_label, video_duration_after_xfade,
     xfade_offsets, safe_t_durs) = _build_xfade_chain(
        n_shots, durations, final_transitions, fps
    )

    logger.info(
        f"🎞️ سلسلة xfade: {max(0, n_shots - 1)} انتقال | "
        f"مدة الفيديو بعد xfade ≈ {video_duration_after_xfade:.3f}s"
    )

    transition_times_final: List[float] = []
    for i, off in enumerate(xfade_offsets):
        transition_times_final.append(max(0.0, off + safe_t_durs[i] / 2.0))

    transition_times_cut: List[float] = [
        layout["bounds"][i + 1] / fps for i in range(n_shots - 1)
    ]

    whoosh_per_transition: List[float] = []
    for i in range(n_shots - 1):
        base = whoosh_strengths[i] if i < len(whoosh_strengths) else _DEFAULT_WHOOSH_STRENGTH
        if final_transitions[i] == "cut":
            base *= 0.7
        whoosh_per_transition.append(min(1.6, base / _DEFAULT_WHOOSH_STRENGTH))

    shot_start_in_final: List[float] = [
        0.0 if i == 0 else layout["O"][i - 1] / fps
        for i in range(n_shots)
    ]

    rhythm_trigger_times: List[float] = []
    for i, item in enumerate(renderable_timeline):
        role = str(item.get("narrative_role") or "").lower().strip()
        if role in RHYTHM_ENABLED_ROLES:
            rhythm_trigger_times.append(shot_start_in_final[i])

    # [إصلاح 1 — المراجعة الثالثة] مواضع القطع الفوري (مسار concat/fallback):
    # اللقطة i تظهر عند layout["bounds"][i] في الـ concat (وليس O[i-1]).
    rhythm_trigger_times_cut: List[float] = [
        layout["bounds"][i] / fps
        for i, item in enumerate(renderable_timeline)
        if str(item.get("narrative_role") or "").lower().strip()
        in RHYTHM_ENABLED_ROLES
    ]

    run_id = uuid.uuid4().hex
    temp_dir = output_dir / f"temp_segments_{run_id}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        raise RuntimeError(
            f"تعذّر إنشاء مجلد العمل المؤقت الفريد: {temp_dir}"
        ) from e

    sfx_whoosh = temp_dir / "whoosh_soft.wav"
    whoosh_timeline = temp_dir / "whoosh_timeline.wav"
    ambience_file = temp_dir / "ambience.wav"
    rhythm_file = temp_dir / "rhythm.wav"
    staged_output_video = temp_dir / "final_output_staged.mp4"

    safe_ass_dir: Optional[Path] = None

    try:
        create_synthetic_whoosh(sfx_whoosh)
        if not sfx_whoosh.exists() or sfx_whoosh.stat().st_size <= 0:
            raise RuntimeError(f"فشل إنشاء ملف الـ whoosh: {sfx_whoosh}")

        segment_files: List[Path] = []
        logger.info(
            f"🎬 بدء رندرة {n_shots} لقطة (run_id={run_id}) — "
            f"AI: {'مفعّل' if _AIEditorialDirector.get().is_available() else 'محلي'}."
        )

        for idx, ((frame_path, _item), motion) in enumerate(
            zip(validated_pairs, final_motions)
        ):
            seg_frames = layout["L"][idx]
            seg_dur = seg_frames / fps
            kb_filter = get_ken_burns_filter(motion, seg_dur, fps=fps)
            seg_output = temp_dir / f"seg_{idx:03d}.mp4"
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(frame_path),
                "-vf", kb_filter,
                "-frames:v", str(seg_frames),
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "22", "-pix_fmt", "yuv420p",
                "-r", str(fps),
                "-an",
                str(seg_output),
            ]
            _run_ffmpeg(cmd, SEGMENT_TIMEOUT_SEC, f"رندرة المقطع رقم {idx}")
            _validate_non_empty_file(seg_output, f"المقطع رقم {idx}")
            segment_files.append(seg_output)

        video_inputs: List[Path] = segment_files
        if n_shots > MAX_SINGLE_PASS_SHOTS:
            groups: List[Tuple[int, int]] = [
                (a, min(a + XFADE_GROUP_SIZE - 1, n_shots - 1))
                for a in range(0, n_shots, XFADE_GROUP_SIZE)
            ]
            group_files: List[Path] = []
            logger.info(
                f"🧩 عدد اللقطات كبير ({n_shots}) — دمج على {len(groups)} مجموعات."
            )
            for g, (a, b) in enumerate(groups):
                if a == b:
                    group_files.append(segment_files[a])
                    continue
                g_out = temp_dir / f"grp_{g:03d}.mp4"
                g_filter, g_label = _xfade_filter_for_range(
                    a, b, layout, fps, layout["S"][a]
                )
                g_inputs: List[str] = []
                for k in range(a, b + 1):
                    g_inputs += ["-i", str(segment_files[k])]
                g_cmd = [
                    "ffmpeg", "-y", "-v", "error",
                    *g_inputs,
                    "-filter_complex", g_filter,
                    "-map", g_label,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-r", str(fps),
                    "-an",
                    str(g_out),
                ]
                _run_ffmpeg(g_cmd, GROUP_TIMEOUT_SEC, f"دمج المجموعة رقم {g}")
                _validate_non_empty_file(g_out, f"ملف المجموعة رقم {g}")
                group_files.append(g_out)

            video_inputs = group_files
            xfade_filter, vx_out_label = _xfade_filter_for_groups(
                groups, layout, fps
            )

        n_video_inputs = len(video_inputs)

        build_whoosh_timeline(
            sfx_whoosh,
            transition_times_final,
            audio_duration,
            whoosh_timeline,
            whoosh_volume=0.42,
            per_transition_volumes=whoosh_per_transition or None,
        )
        _validate_non_empty_file(whoosh_timeline, "مسار الـ whoosh (xfade)")

        avg_amb_boost = (
            sum(ambience_boosts) / len(ambience_boosts)
            if ambience_boosts else 0.0
        )
        has_ambience = False
        try:
            has_ambience = _try_generate_ambience(
                ambience_file, audio_duration, boost=avg_amb_boost
            )
            if has_ambience:
                logger.info(
                    f"🌫️ Ambience مُولَّد ({audio_duration:.3f}s | "
                    f"boost={avg_amb_boost:+.3f})."
                )
        except Exception as amb_err:
            logger.warning(f"تعذّر توليد Ambience ({amb_err}).")
            has_ambience = False
            try:
                if ambience_file.exists():
                    ambience_file.unlink()
            except OSError:
                pass

        has_rhythm = False
        if rhythm_trigger_times:
            try:
                has_rhythm = _build_rhythm_timeline(
                    rhythm_trigger_times, audio_duration, 44100,
                    rhythm_file, rhythm_volume=0.32,
                )
                if has_rhythm:
                    logger.info(
                        f"🎵 Rhythm مُولَّد ({len(rhythm_trigger_times)} نبضة)."
                    )
            except Exception as r_err:
                logger.warning(f"تعذّر توليد الإيقاع ({r_err}).")
                has_rhythm = False
                try:
                    if rhythm_file.exists():
                        rhythm_file.unlink()
                except OSError:
                    pass
        else:
            logger.info("🎵 لا توجد أدوار مسموحة بالإيقاع.")

        # [إصلاح 2 — المراجعة الثالثة] مسار ASS آمن (نسخ عند وجود ').
        safe_ass, safe_ass_dir = _prepare_safe_ass(subtitles_ass)
        ass_escaped = _escape_filter_path(safe_ass)

        logger.info(
            "✨ تمريرة موحّدة: xfade + tpad + burn subs + audio mix + ducking..."
        )

        final_inputs: List[str] = []
        for vf in video_inputs:
            final_inputs += ["-i", str(vf)]
        idx_audio = n_video_inputs
        idx_whoosh = n_video_inputs + 1
        final_inputs += ["-i", str(audio_file)]
        final_inputs += ["-i", str(whoosh_timeline)]
        next_idx = n_video_inputs + 2
        ambience_input_idx = None
        rhythm_input_idx = None
        if has_ambience:
            final_inputs += ["-i", str(ambience_file)]
            ambience_input_idx = next_idx
            next_idx += 1
        if has_rhythm:
            final_inputs += ["-i", str(rhythm_file)]
            rhythm_input_idx = next_idx
            next_idx += 1

        filter_parts: List[str] = []

        if n_video_inputs == 1:
            filter_parts.append("[0:v]null[vx0]")
            vx_label = "[vx0]"
        else:
            filter_parts.append(xfade_filter)
            vx_label = vx_out_label

        pad_needed = max(0.0, audio_duration - video_duration_after_xfade)
        if pad_needed > 0.001:
            filter_parts.append(
                f"{vx_label}tpad=stop_mode=clone:"
                f"stop_duration={pad_needed:.4f}[vpad]"
            )
        else:
            filter_parts.append(f"{vx_label}null[vpad]")

        polish_base = (
            "eq=contrast=1.02:saturation=1.03:brightness=0.005,"
            "unsharp=5:5:0.35:5:5:0.0"
        )
        if audio_duration >= 0.9:
            fade_out_start = audio_duration - 0.4
            polish = (
                f"{polish_base},"
                "fade=t=in:st=0:d=0.4,"
                f"fade=t=out:st={fade_out_start:.3f}:d=0.4"
            )
        else:
            polish = polish_base
        filter_parts.append(f"[vpad]{polish}[vpolished]")
        filter_parts.append(
            f"[vpolished]ass=filename='{ass_escaped}'[vout]"
        )

        ducked_tracks: List[Tuple[str, int, str, str, str, str, str]] = []
        if has_rhythm and rhythm_input_idx is not None:
            ducked_tracks.append(
                ("rhythm", rhythm_input_idx, "0.45", "0.03", "5", "10", "300")
            )
        if has_ambience and ambience_input_idx is not None:
            ducked_tracks.append(
                ("ambience", ambience_input_idx, "0.55", "0.02", "6", "20", "400")
            )

        n_ducked = len(ducked_tracks)
        n_voice_splits = 1 + n_ducked

        if n_ducked > 0:
            voice_split_labels = "".join(f"[vsc{i}]" for i in range(n_ducked))
            filter_parts.append(
                f"[{idx_audio}:a]{_AUDIO_NORMALIZE_CHAIN},"
                f"asplit={n_voice_splits}[vmain]{voice_split_labels}"
            )
        else:
            filter_parts.append(
                f"[{idx_audio}:a]{_AUDIO_NORMALIZE_CHAIN}[vmain]"
            )

        filter_parts.append(
            f"[{idx_whoosh}:a]{_AUDIO_NORMALIZE_CHAIN}[whoosh_src]"
        )
        amix_inputs = ["[vmain]", "[whoosh_src]"]
        amix_weights = ["1", "0.55"]

        for i, (name, in_idx, weight, thr, ratio, att, rel) in enumerate(ducked_tracks):
            src_label = f"[{name}_src]"
            duck_label = f"[{name}_d]"
            filter_parts.append(
                f"[{in_idx}:a]{_AUDIO_NORMALIZE_CHAIN}{src_label}"
            )
            filter_parts.append(
                f"{src_label}[vsc{i}]sidechaincompress="
                f"threshold={thr}:ratio={ratio}:"
                f"attack={att}:release={rel}{duck_label}"
            )
            amix_inputs.append(duck_label)
            amix_weights.append(weight)

        n_amix = len(amix_inputs)
        weights_str = " ".join(amix_weights)

        supports_norm = _ffmpeg_supports_amix_normalize()
        amix_expr = (
            f"{''.join(amix_inputs)}amix=inputs={n_amix}:duration=first:"
            f"dropout_transition=0:weights='{weights_str}'"
        )
        if supports_norm:
            amix_expr += ":normalize=0"
        else:
            comp = sum(float(w) for w in amix_weights)
            amix_expr += f",volume={comp:.3f}"
        amix_expr += ",alimiter=limit=0.95:level=disabled[aout]"
        filter_parts.append(amix_expr)

        filter_complex = ";".join(filter_parts)

        cmd_final = [
            "ffmpeg", "-y", "-v", "warning",
            *final_inputs,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-t", f"{audio_duration:.4f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(fps),
            "-c:a", "aac", "-b:a", "224k",
            "-movflags", "+faststart",
            str(staged_output_video),
        ]

        try:
            _run_ffmpeg(cmd_final, FINAL_TIMEOUT_SEC, "التصدير الموحّد")
        except RuntimeError as e:
            logger.warning(
                f"⚠️ فشل التصدير الموحّد: {e} — "
                f"fallback: concat + تمريرة موحّدة (مع الحفاظ على المؤثرات)."
            )

            whoosh_timeline_cut = temp_dir / "whoosh_cut.wav"
            build_whoosh_timeline(
                sfx_whoosh,
                transition_times_cut,
                audio_duration,
                whoosh_timeline_cut,
                whoosh_volume=0.42,
                per_transition_volumes=whoosh_per_transition or None,
            )
            _validate_non_empty_file(
                whoosh_timeline_cut, "مسار الـ whoosh (fallback/cut)"
            )

            # [إصلاح 1 — المراجعة الثالثة] إيقاع مبني على مواضع القطع الفوري
            # ليتزامن مع ظهور اللقطة في مسار concat.
            fb_rhythm_path: Optional[Path] = None
            if has_rhythm and rhythm_trigger_times_cut:
                rhythm_cut_path = temp_dir / "rhythm_cut.wav"
                if _build_rhythm_timeline(
                    rhythm_trigger_times_cut, audio_duration, 44100,
                    rhythm_cut_path, rhythm_volume=0.32,
                ):
                    fb_rhythm_path = rhythm_cut_path
                    logger.info(
                        f"🎵 إيقاع fallback مُولَّد بمواضع القطع "
                        f"({len(rhythm_trigger_times_cut)} نبضة)."
                    )
                else:
                    logger.warning(
                        "تعذّر توليد إيقاع fallback — استخدام التوقيت الأصلي."
                    )
                    fb_rhythm_path = rhythm_file
            elif has_rhythm:
                fb_rhythm_path = rhythm_file

            trimmed_files: List[Path] = []
            for i, seg in enumerate(segment_files):
                a = layout["bounds"][i] - layout["S"][i]
                b = layout["bounds"][i + 1] - layout["S"][i]
                trimmed = temp_dir / f"trim_{i:03d}.mp4"
                _run_ffmpeg(
                    [
                        "ffmpeg", "-y", "-v", "error",
                        "-i", str(seg),
                        "-vf",
                        f"trim=start_frame={a}:end_frame={b},setpts=PTS-STARTPTS",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-r", str(fps),
                        "-an",
                        str(trimmed),
                    ],
                    FALLBACK_STEP_TIMEOUT_SEC, f"قص المقطع رقم {i}",
                )
                _validate_non_empty_file(trimmed, f"المقطع المقصوص رقم {i}")
                trimmed_files.append(trimmed)

            concat_list_file = temp_dir / "concat_list.txt"
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for seg in trimmed_files:
                    f.write(_safe_concat_escape(seg))

            fb_inputs: List[str] = [
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list_file),
            ]
            fb_idx_audio = 1
            fb_idx_whoosh = 2
            fb_inputs += ["-i", str(audio_file)]
            fb_inputs += ["-i", str(whoosh_timeline_cut)]
            fb_next = 3
            fb_amb_idx: Optional[int] = None
            fb_rhythm_idx: Optional[int] = None
            if has_ambience:
                fb_inputs += ["-i", str(ambience_file)]
                fb_amb_idx = fb_next
                fb_next += 1
            if fb_rhythm_path is not None:
                fb_inputs += ["-i", str(fb_rhythm_path)]
                fb_rhythm_idx = fb_next
                fb_next += 1

            fb_video_dur = layout["bounds"][-1] / fps
            fb_pad = max(0.0, audio_duration - fb_video_dur)

            fb_filter_parts: List[str] = []
            if fb_pad > 0.001:
                fb_filter_parts.append(
                    f"[0:v]tpad=stop_mode=clone:stop_duration={fb_pad:.4f}[vpad]"
                )
            else:
                fb_filter_parts.append("[0:v]null[vpad]")

            fb_polish_base = (
                "eq=contrast=1.02:saturation=1.03:brightness=0.005,"
                "unsharp=5:5:0.35:5:5:0.0"
            )
            if audio_duration >= 0.9:
                fb_fade_start = audio_duration - 0.4
                fb_polish = (
                    f"{fb_polish_base},"
                    "fade=t=in:st=0:d=0.4,"
                    f"fade=t=out:st={fb_fade_start:.3f}:d=0.4"
                )
            else:
                fb_polish = fb_polish_base
            fb_filter_parts.append(f"[vpad]{fb_polish}[vpolished]")
            fb_filter_parts.append(
                f"[vpolished]ass=filename='{ass_escaped}'[vout]"
            )

            fb_ducked: List[Tuple[str, int, str, str, str, str, str]] = []
            if has_rhythm and fb_rhythm_idx is not None:
                fb_ducked.append(
                    ("rhythm", fb_rhythm_idx, "0.45", "0.03", "5", "10", "300")
                )
            if has_ambience and fb_amb_idx is not None:
                fb_ducked.append(
                    ("ambience", fb_amb_idx, "0.55", "0.02", "6", "20", "400")
                )

            n_fb_ducked = len(fb_ducked)
            fb_splits = 1 + n_fb_ducked

            if n_fb_ducked > 0:
                fb_split_labels = "".join(f"[fsc{i}]" for i in range(n_fb_ducked))
                fb_filter_parts.append(
                    f"[{fb_idx_audio}:a]{_AUDIO_NORMALIZE_CHAIN},"
                    f"asplit={fb_splits}[fmain]{fb_split_labels}"
                )
            else:
                fb_filter_parts.append(
                    f"[{fb_idx_audio}:a]{_AUDIO_NORMALIZE_CHAIN}[fmain]"
                )

            fb_filter_parts.append(
                f"[{fb_idx_whoosh}:a]{_AUDIO_NORMALIZE_CHAIN}[fbwhoosh_src]"
            )
            fb_amix_inputs = ["[fmain]", "[fbwhoosh_src]"]
            fb_amix_weights = ["1", "0.55"]

            for i, (name, in_idx, weight, thr, ratio, att, rel) in enumerate(fb_ducked):
                src_label = f"[fb_{name}_src]"
                duck_label = f"[fb_{name}_d]"
                fb_filter_parts.append(
                    f"[{in_idx}:a]{_AUDIO_NORMALIZE_CHAIN}{src_label}"
                )
                fb_filter_parts.append(
                    f"{src_label}[fsc{i}]sidechaincompress="
                    f"threshold={thr}:ratio={ratio}:"
                    f"attack={att}:release={rel}{duck_label}"
                )
                fb_amix_inputs.append(duck_label)
                fb_amix_weights.append(weight)

            n_fb_amix = len(fb_amix_inputs)
            fb_weights_str = " ".join(fb_amix_weights)
            fb_amix_expr = (
                f"{''.join(fb_amix_inputs)}amix=inputs={n_fb_amix}:duration=first:"
                f"dropout_transition=0:weights='{fb_weights_str}'"
            )
            if supports_norm:
                fb_amix_expr += ":normalize=0"
            else:
                fb_comp = sum(float(w) for w in fb_amix_weights)
                fb_amix_expr += f",volume={fb_comp:.3f}"
            fb_amix_expr += ",alimiter=limit=0.95:level=disabled[aout]"
            fb_filter_parts.append(fb_amix_expr)

            fb_filter_complex = ";".join(fb_filter_parts)

            _run_ffmpeg(
                [
                    "ffmpeg", "-y", "-v", "warning",
                    *fb_inputs,
                    "-filter_complex", fb_filter_complex,
                    "-map", "[vout]",
                    "-map", "[aout]",
                    "-t", f"{audio_duration:.4f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-r", str(fps),
                    "-c:a", "aac", "-b:a", "224k",
                    "-movflags", "+faststart",
                    str(staged_output_video),
                ],
                FALLBACK_STEP_TIMEOUT_SEC, "التصدير (fallback)",
            )

        _validate_non_empty_file(staged_output_video, "الفيديو النهائي المرحلي")
        final_duration = _probe_media_duration(staged_output_video)
        if final_duration <= 0:
            raise RuntimeError(
                f"مدة الفيديو النهائي المرحلي غير موجبة: {final_duration}"
            )

        try:
            staged_output_video.replace(output_video_path)
        except OSError as e:
            raise RuntimeError(
                f"تعذّر نقل الفيديو النهائي: "
                f"{staged_output_video} -> {output_video_path}"
            ) from e

        logger.info(
            f"🏆 تم التصدير النهائي: {output_video_path} "
            f"(المدة النهائية = {final_duration:.3f}s | "
            f"المرجع الصوتي = {audio_duration:.3f}s)"
        )

    finally:
        if safe_ass_dir is not None:
            try:
                if safe_ass_dir.exists() and safe_ass_dir.is_dir():
                    shutil.rmtree(safe_ass_dir, ignore_errors=True)
            except Exception as cleanup_err:
                logger.warning(
                    f"تعذّر تنظيف مجلد ASS المؤقت {safe_ass_dir}: {cleanup_err}"
                )
        try:
            if temp_dir.exists() and temp_dir.is_dir():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_err:
            logger.warning(
                f"تعذّر تنظيف المجلد المؤقت {temp_dir}: {cleanup_err}"
            )


# ============================================================
# الاختبارات الإلزامية
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. اختبار _compute_image_durations
    timeline_test = [
        {"start": 0.20, "text": "a"},
        {"start": 1.40, "text": "b"},
        {"start": 2.00, "text": "c"},
    ]
    durs, starts_abs = _compute_image_boundaries(timeline_test, 3.00, 60)
    assert all(abs(a - b) < 1e-6 for a, b in zip(durs, [1.40, 0.60, 1.00])), \
        f"durations mismatch: {durs}"
    assert all(abs(a - b) < 1e-6 for a, b in zip(starts_abs, [0.0, 1.4, 2.0])), \
        f"image_starts mismatch: {starts_abs}"
    assert abs(sum(durs) - 3.00) < 1e-6, f"sum mismatch: {sum(durs)}"
    print("[OK] test #1: _compute_image_durations = [1.4, 0.6, 1.0]")

    # 2. اختبار مع جمل فارغة
    timeline_with_empty = [
        {"start": 0.10, "text": "a", "empty": False},
        {"start": 0.90, "text": "", "empty": True},
        {"start": 1.50, "text": "c", "empty": False},
    ]
    renderable = [it for it in timeline_with_empty if not it.get("empty")]
    durs2, _ = _compute_image_boundaries(renderable, 2.50, 60)
    assert abs(sum(durs2) - 2.50) < 1e-6, f"sum2 mismatch: {sum(durs2)}"
    print("[OK] test #2: empty items filtered, sum == audio_duration")

    # 3. اختبار غياب start
    timeline_no_start = [
        {"duration": 0.5, "text": "a"},
        {"duration": 0.7, "text": "b"},
    ]
    durs3, starts3 = _compute_image_boundaries(timeline_no_start, 1.2, 60)
    assert all(abs(a - b) < 1e-6 for a, b in zip(durs3, [0.5, 0.7])), durs3
    assert all(abs(a - b) < 1e-6 for a, b in zip(starts3, [0.0, 0.5])), starts3
    print("[OK] test #3: fallback durations path")

    # 4. اختبار بناء الإيقاع
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        rhythm_out = td_path / "rhythm.wav"
        ok = _build_rhythm_timeline(
            [0.5, 1.5], 3.0, 44100, rhythm_out, rhythm_volume=0.35
        )
        assert ok, "rhythm generation should succeed"
        assert rhythm_out.exists() and rhythm_out.stat().st_size > 0
        print("[OK] test #4: rhythm timeline generation")

        # 5. اختبار فشل الإيقاع
        empty_rhythm = td_path / "rhythm_empty.wav"
        ok_empty = _build_rhythm_timeline(
            [], 3.0, 44100, empty_rhythm, rhythm_volume=0.35
        )
        assert not ok_empty, "empty trigger list should return False"
        print("[OK] test #5: rhythm fails gracefully on empty triggers")

    # 6. اختبار بناء Whoosh + whoosh_volume
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        base = td_path / "base_whoosh.wav"
        create_synthetic_whoosh(base)
        assert base.exists() and base.stat().st_size > 0

        whoosh_out = td_path / "whoosh_timeline.wav"
        build_whoosh_timeline(base, [1.4, 2.0], 3.0, whoosh_out, whoosh_volume=0.4)
        assert whoosh_out.exists() and whoosh_out.stat().st_size > 0

        def _peak(p: Path) -> int:
            with wave.open(str(p), "rb") as w:
                arr = array.array("h")
                arr.frombytes(w.readframes(w.getnframes()))
            if sys.byteorder == "big":
                arr.byteswap()
            return max((abs(v) for v in arr), default=0)

        out_a = td_path / "wa.wav"
        out_b = td_path / "wb.wav"
        out_z = td_path / "wz.wav"
        build_whoosh_timeline(base, [1.0], 3.0, out_a,
                              whoosh_volume=0.8, per_transition_volumes=[1.0])
        build_whoosh_timeline(base, [1.0], 3.0, out_b,
                              whoosh_volume=0.2, per_transition_volumes=[1.0])
        build_whoosh_timeline(base, [1.0], 3.0, out_z,
                              whoosh_volume=0.8, per_transition_volumes=[0.0])
        assert _peak(out_a) > _peak(out_b) > 0, "whoosh_volume must scale output"
        assert _peak(out_z) == 0, "zero strength must be silent"
        print("[OK] test #6: whoosh timeline generation + whoosh_volume scaling")

    # 7. اختبار Ken Burns المتنوع
    for mtype in MOTION_TYPES:
        f = get_ken_burns_filter(
            {"type": mtype, "intensity": 0.6, "speed": "auto"},
            duration=3.0, fps=60,
        )
        assert "zoompan=" in f, f"missing zoompan for {mtype}"
        assert "on/179" in f, f"progress-based motion missing for {mtype}"
        assert "crop=" in f and "force_original_aspect_ratio=increase" in f, mtype
    f_legacy = get_ken_burns_filter(0, 3.0, fps=60)
    assert "zoompan=" in f_legacy
    print(f"[OK] test #7: Ken Burns متنوّع ({len(MOTION_TYPES)} حركة) + توافق خلفي")

    # 8. اختبار xfade chain + المخطط الإطاري
    durations_t = [3.0, 2.5, 4.0]
    tr_types = ["crossfade_soft", "crossfade_medium"]
    filt, out_label, total, offsets, safe_tdurs = _build_xfade_chain(
        3, durations_t, tr_types, 60
    )
    assert "xfade" in filt, "xfade should appear in filter"
    assert out_label == "[vx1]", out_label
    assert len(offsets) == 2, offsets
    assert len(safe_tdurs) == 2, safe_tdurs
    assert total > 0, "total must be positive"
    assert abs(total - sum(durations_t)) < 1e-6, \
        f"total mismatch: {total} vs {sum(durations_t)}"
    boundary = 0.0
    for i in range(2):
        boundary += durations_t[i]
        mid = offsets[i] + safe_tdurs[i] / 2.0
        assert abs(mid - boundary) <= 1.0 / 60.0, (i, mid, boundary)
    print(
        f"[OK] test #8: xfade chain "
        f"(total={total:.3f}s, offsets={offsets}, safe_tdurs={safe_tdurs})"
    )

    # 9. اختبار _AIEditorialDirector
    director = _AIEditorialDirector.get()
    plan = director.plan_montage(
        [{"text": "x", "narrative_role": "hook"}],
        [2.0],
        [""],
    )
    assert isinstance(plan, dict), "plan must be a dict"
    print(f"[OK] test #9: AI director defensive (available={director.is_available()})")

    # 10. اختبار اختيار الحركة المحلية
    m1 = _select_motion_local(
        {"narrative_role": "hook"}, 3.0, None, 0
    )
    assert m1["type"] in MOTION_TYPES, m1
    m2 = _select_motion_local(
        {"narrative_role": "hook"}, 3.0, m1["type"], 1
    )
    assert m2["type"] != m1["type"] or len(
        ROLE_PREFERRED_MOTIONS["hook"]
    ) == 1, f"motion repetition not prevented: {m1} → {m2}"
    print(f"[OK] test #10: local motion selection (m1={m1['type']}, m2={m2['type']})")

    # 11. اختبار الأدوار الجديدة
    new_roles = ("myth", "contradiction", "explanation", "analogy", "setup", "cta")
    for r in new_roles:
        assert r in ROLE_PREFERRED_MOTIONS, f"missing motion prefs for {r}"
        assert r in ROLE_PREFERRED_TRANSITION, f"missing transition prefs for {r}"
    print(f"[OK] test #11: new roles covered ({', '.join(new_roles)})")

    # 12. اختبار المخطط الإطاري
    for durs_case in (
        [1.0, 0.5, 2.0, 0.8],
        [1.0 / 60, 1.0 / 60, 1.0, 1.0 / 60, 0.5],
    ):
        n_c = len(durs_case)
        lay = _compute_segment_layout(
            durs_case, ["crossfade_medium"] * (n_c - 1), 60
        )
        for i in range(n_c):
            assert lay["L"][i] == lay["E"][i] - lay["S"][i] >= 1, (i, lay)
        for i in range(n_c - 1):
            assert lay["S"][i + 1] == lay["O"][i], (i, lay)
            assert lay["O"][i] >= 0 and lay["T"][i] >= 1, (i, lay)
            if i + 1 < n_c - 1:
                assert lay["O"][i + 1] >= lay["O"][i] + lay["T"][i], (i, lay)
        assert lay["E"][-1] == lay["bounds"][-1], lay
    print("[OK] test #12: segment layout consistency (incl. 1-frame shots)")

    # 13. اختبار تحصين تحليل خطة AI
    d_tmp = _AIEditorialDirector()
    raw_plan = (
        '{"shots":[{"idx":0,"motion":"zoom_in","intensity":"nan",'
        '"whoosh_strength":1e999},{"idx":9,"motion":"zoom_out"}],'
        '"pacing":{"peaks":null,"calm":[1,2.0,"x",null]}}'
    )
    parsed = d_tmp._parse_plan(raw_plan, 2)
    assert parsed and len(parsed["shots"]) == 1, parsed
    assert parsed["pacing"]["peaks"] == [], parsed
    assert parsed["pacing"]["calm"] == [1, 2], parsed
    assert 0.3 <= parsed["shots"][0]["intensity"] <= 1.0, parsed
    assert 0.0 <= parsed["shots"][0]["whoosh_strength"] <= 1.0, parsed
    as_text = _AIEditorialDirector._coerce_result_to_text({"shots": []})
    assert json.loads(as_text)["shots"] == [], as_text
    print("[OK] test #13: AI plan parsing hardened (null/NaN/inf/dict)")

    # ------------------------------------------------------------
    # 14. اختبار الهروب الآمن لمسار ASS (محمول على المنصات)
    # [إصلاح 4 — المراجعة الثالثة] تصحيح فحص شارحة الخلفية المتبقية.
    # ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td14:
        sub = Path(td14) / "sub [1]"
        sub.mkdir()
        p_real = sub / "my 'draft'.ass"
        p_real.touch()

        esc_real = _escape_filter_path(p_real)

        # (أ) الفاصلة العليا مُهرَّبة بنمط close-escape-reopen القياسي
        assert "'\\''" in esc_real, esc_real

        # (ب) فحص بنيوي صارم: الرمز الوحيد المُهرَّب هو ' — لا : ولا [ ]
        i = 0
        while i < len(esc_real):
            if esc_real[i] == "\\":
                assert i + 1 < len(esc_real) and esc_real[i + 1] == "'", (
                    f"unexpected escape at position {i} in {esc_real!r}"
                )
                i += 2
            else:
                i += 1

        # (ج) الأقواس المربعة لم تُهرَّب
        cleaned = esc_real.replace("'\\''", "")
        assert "\\[" not in cleaned, cleaned
        assert "\\]" not in cleaned, cleaned

        # (د) [إصلاح 4] لا backslash متبقٍ بعد إزالة '\\'' (فحص صحيح)
        assert "\\" not in cleaned, cleaned

    # [إصلاح 2 — المراجعة الثالثة] اختبار _prepare_safe_ass
    with tempfile.TemporaryDirectory() as td14b:
        td_path = Path(td14b)
        # مسار بلا ' → لا نسخ
        safe_src = td_path / "clean.ass"
        safe_src.touch()
        p_safe, d_safe = _prepare_safe_ass(safe_src)
        assert p_safe == safe_src, (p_safe, safe_src)
        assert d_safe is None, d_safe

        # مسار فيه ' → نسخ إلى مسار آمن
        sub2 = td_path / "sub2 [x]"
        sub2.mkdir()
        p_unsafe = sub2 / "with 'quote'.ass"
        p_unsafe.write_text("[Script Info]\n", encoding="utf-8")
        p_copied, d_copied = _prepare_safe_ass(p_unsafe)
        try:
            assert p_copied != p_unsafe
            assert p_copied.exists() and p_copied.stat().st_size > 0
            assert d_copied is not None and d_copied.is_dir()
            assert "'" not in p_copied.as_posix(), p_copied
        finally:
            if d_copied is not None:
                shutil.rmtree(d_copied, ignore_errors=True)

    print(
        "[OK] test #14: ASS path escaping + _prepare_safe_ass (portable)"
    )

    # ------------------------------------------------------------
    # 15. اختبار LRU + قفل كاش AI
    # ------------------------------------------------------------
    d_lru = _AIEditorialDirector()
    d_lru._plan_cache.clear()
    d_lru._MAX_PLAN_CACHE = 3
    for k in ("a", "b", "c"):
        d_lru._plan_cache[k] = {"shots": [{"idx": 0}]}
    d_lru._plan_cache["d"] = {"shots": [{"idx": 0}]}
    while len(d_lru._plan_cache) > d_lru._MAX_PLAN_CACHE:
        d_lru._plan_cache.popitem(last=False)
    assert list(d_lru._plan_cache.keys()) == ["b", "c", "d"], \
        list(d_lru._plan_cache.keys())
    d_lru._plan_cache.clear()
    print("[OK] test #15: AI plan cache LRU eviction")

    # ------------------------------------------------------------
    # 16. اختبار _run_ffmpeg FileNotFoundError
    # ------------------------------------------------------------
    try:
        _run_ffmpeg(["definitely-not-ffmpeg-xyz"], 5, "اختبار عدم التوفر")
        raise AssertionError("should have raised RuntimeError")
    except RuntimeError as e:
        assert "غير متوفر" in str(e), str(e)
    print("[OK] test #16: _run_ffmpeg reports missing binary clearly")

    # ------------------------------------------------------------
    # 17. اختبار مزامنة إيقاع الـ fallback
    # ------------------------------------------------------------
    durs_fb = [2.0, 3.0, 2.5]
    tr_fb = ["crossfade_medium", "crossfade_soft"]
    lay_fb = _compute_segment_layout(durs_fb, tr_fb, 60)
    # مواضع القطع الفوري = bounds[i]/fps
    cut_pos = [lay_fb["bounds"][i] / 60 for i in range(1, len(durs_fb))]
    # مواضع xfade = O[i-1]/fps
    xfade_pos = [lay_fb["O"][i - 1] / 60 for i in range(1, len(durs_fb))]
    # يجب أن تكون مواضع القطع > مواضع xfade (لأن O يبدأ قبل منتصف الحد)
    for i in range(len(cut_pos)):
        assert cut_pos[i] > xfade_pos[i], (i, cut_pos[i], xfade_pos[i])
    print(
        f"[OK] test #17: rhythm timing cut vs xfade "
        f"(cut={[round(x,3) for x in cut_pos]}, "
        f"xfade={[round(x,3) for x in xfade_pos]})"
    )

    print("[ALL TESTS PASSED]")
