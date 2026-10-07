"""
stage4_composer.py — محرك المونتاج السينمائي الآلي (نسخة إنتاج عالمية مُحصَّنة).
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
from typing import List, Dict, Any, Tuple, Optional, Set

try:
    from stage4_assets import get_asset_for_keyword
except ImportError:
    get_asset_for_keyword = None

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
    "punch_in", "slow_pan", "drift",
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
    "hook":          "crossfade_soft",
    "revelation":    "cut",
    "reflection":    "crossfade_medium",
    "question":      "crossfade_soft",
    "tension":       "cut",
    "payoff":        "crossfade_medium",
    "actionable":    "cut",
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

_AI_PROMPT_VERSION = "v3"  # رُفع بعد توضيح دلالة whoosh_strength.

_AUDIO_VOICE_NORMALIZE_CHAIN = (
    "aresample=44100,"
    "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo"
)

_AUDIO_SFX_NORMALIZE_CHAIN = (
    "aresample=44100,"
    "aformat=sample_fmts=fltp:sample_rates=44100,"
    "pan=stereo|c0=c0|c1=c0"
)

# للخلفية المتوافقة مع الإصدارات القديمة من الاستدعاءات الداخلية.
_AUDIO_NORMALIZE_CHAIN = _AUDIO_VOICE_NORMALIZE_CHAIN


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
        if isinstance(x, int):
            out.append(x)
        elif isinstance(x, float):
            if not math.isfinite(x) or not x.is_integer():
                continue
            out.append(int(x))
    return out


def _run_ffmpeg(
    cmd: List[str],
    timeout: float,
    label: str,
    cwd: Optional[Path] = None,
) -> None:
    """
    تشغيل ffmpeg مع مهلة + التقاط stderr + دعم cwd.
    """
    if cmd and Path(cmd[0]).name.lower() in ("ffmpeg", "ffmpeg.exe"):
        cmd = [cmd[0], "-nostdin", *cmd[1:]]

    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd) if cwd is not None else None,
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
    except OSError as e:
        raise RuntimeError(
            f"تعذّر تشغيل FFmpeg في {label}: {e}"
        ) from e


_FFMPEG_FILTERS: Optional[Set[str]] = None


def _get_ffmpeg_filters() -> Set[str]:
    """
    جلب أسماء الفلاتر المتاحة في ffmpeg مع caching.
    """
    global _FFMPEG_FILTERS
    if _FFMPEG_FILTERS is not None:
        return _FFMPEG_FILTERS

    filters: Set[str] = set()
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        out = (r.stdout or "") + (r.stderr or "")
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # الشكل المعتاد: flags filter_name description...
            name = parts[1]
            if name and name[0].isalpha():
                filters.add(name)
    except (OSError, subprocess.SubprocessError):
        filters = set()

    _FFMPEG_FILTERS = filters
    return filters


def _require_ffmpeg_filters(required: List[str]) -> None:
    available = _get_ffmpeg_filters()
    if not available:
        raise RuntimeError(
            "تعذّر فحص فلاتر FFmpeg. تأكد من أن ffmpeg مثبت وفي PATH."
        )
    missing = [f for f in required if f not in available]
    if missing:
        raise RuntimeError(
            "فلاتر FFmpeg مفقودة في هذه النسخة: "
            + ", ".join(missing)
            + ". يُوصى باستخدام FFmpeg حديث مدمج مع libass وعناصر الصوت القياسية."
        )


_AMIX_FEATURES: Optional[Tuple[bool, bool]] = None


def _ffmpeg_supports_amix_features() -> Tuple[bool, bool]:
    """
    يرجع (supports_weights, supports_normalize).

    الاستراتيجية:
    1) فحص وصف الفلتر عبر `ffmpeg -h filter=amix` (سريع).
    2) تأكيد باختبار فعلي قصير (يضمن عدم الاعتماد على نص وصفي قد يتغير).
    النتيجة مُخزَّنة مؤقتًا في _AMIX_FEATURES.
    """
    global _AMIX_FEATURES
    if _AMIX_FEATURES is not None:
        return _AMIX_FEATURES

    supports_weights = False
    supports_normalize = False

    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "filter=amix"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        out = (r.stdout or "") + (r.stderr or "")
        supports_weights = "weights" in out
        supports_normalize = "normalize" in out
    except (OSError, subprocess.SubprocessError):
        supports_weights = False
        supports_normalize = False

    # تأكيد باختبار فعلي عند الاشتباه بدعم weights.
    if supports_weights:
        try:
            probe = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-v", "error",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                    "-filter_complex",
                    "[0:a][1:a]amix=inputs=2:duration=first:"
                    "dropout_transition=0:weights='1 1'[aout]",
                    "-map", "[aout]", "-t", "0.05",
                    "-f", "null", "-",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                encoding="utf-8",
                errors="replace",
            )
            if probe.returncode != 0:
                supports_weights = False
                supports_normalize = False
        except (OSError, subprocess.SubprocessError):
            supports_weights = False
            supports_normalize = False

    _AMIX_FEATURES = (supports_weights, supports_normalize)
    return _AMIX_FEATURES


def _build_audio_mix_filter(
    mix_inputs: List[Tuple[str, float]],
    out_label: str = "[aout]",
) -> str:
    """
    بناء سلسلة مزج صوتية آمنة عبر إصدارات FFmpeg المختلفة.
    mix_inputs: قائمة (label, weight).

    ملاحظة حول alimiter=level=0:
      - limit=0.95 يمنع التجاوز فوق ‎-0.45 dBFS تقريبًا (سقف آمن قبل AAC).
      - level=0 يعطّل auto-leveling الداخلي لـ alimiter (وهو الافتراضي في معظم
        الإصدارات لكن نصرّح به للثبات)، فلا يُعاد رفع الإشارة تلقائيًا بعد القص.
      - الغرض: منع clipping صاخب دون تشويه ديناميكي غير متوقع.
    """
    if not mix_inputs:
        raise ValueError("mix_inputs فارغة")

    if len(mix_inputs) == 1:
        return f"{mix_inputs[0][0]}alimiter=limit=0.95:level=0{out_label}"

    supports_weights, supports_normalize = _ffmpeg_supports_amix_features()

    if supports_weights:
        labels = "".join(lbl for lbl, _ in mix_inputs)
        weights_str = " ".join(f"{float(w):.6f}" for _, w in mix_inputs)
        expr = (
            f"{labels}amix=inputs={len(mix_inputs)}:duration=first:"
            f"dropout_transition=0:weights='{weights_str}'"
        )
        if supports_normalize:
            expr += ":normalize=0"
        else:
            expr += f",volume={sum(float(w) for _, w in mix_inputs):.6f}"
    else:
        parts: List[str] = []
        labels: List[str] = []
        for i, (lbl, w) in enumerate(mix_inputs):
            vol_lbl = f"[mixvol{i}]"
            parts.append(f"{lbl}volume={float(w):.6f}{vol_lbl}")
            labels.append(vol_lbl)
        expr = (
            ";".join(parts)
            + ";"
            + "".join(labels)
            + f"amix=inputs={len(labels)}:duration=first:dropout_transition=0"
        )
        if supports_normalize:
            expr += ":normalize=0"
        else:
            # تعويض تقريبي لسلوك amix القديم الذي يميل إلى المتوسط.
            expr += f",volume={len(labels):.6f}"

    expr += f",alimiter=limit=0.95:level=0{out_label}"
    return expr


def _escape_filter_path(path: Path) -> str:
    """
    هروب مسار ملف للاستخدام داخل FFmpeg filtergraph بين اقتباسين مفردين:
        ass=filename='...'

    داخل الاقتباس الفردي في filtergraph:
    - الرمز \ يُهرَّب بـ \\
    - الرمز ' يُهرَّب بـ \'
    - لا حاجة لهروب : أو , أو [ ] داخل الاقتباس.
    """
    s = path.resolve().as_posix()
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    return s


def _prepare_safe_ass(
    subtitles_ass: Path,
    dest_dir: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    تجهيز ملف ASS لاستخدام آمن في فلتر ass.

    إذا تم تمرير dest_dir:
      - يُنسخ دائمًا إلى dest_dir/subs.ass (إلا إذا كان المصدر هو نفسه).
      - يُعاد (المسار الجديد, None).
      - الاستخدام الأمثل بعد ذلك: تشغيل ffmpeg بـ cwd=dest_dir ومرر filename=subs.ass.

    إذا لم يتم تمرير dest_dir:
      - إن كان المسار خاليًا من ' يُعاد كما هو.
      - وإلا يُنسخ إلى مجلد مؤقت آمن ويُعاد (المسار الجديد, مجلدTemp).
    """
    src = Path(subtitles_ass)

    if dest_dir is not None:
        dest_dir = Path(dest_dir)
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"تعذّر إنشاء مجلد ASS آمن: {dest_dir}") from e
        if not dest_dir.is_dir():
            raise RuntimeError(f"مسار ASS الآمن ليس مجلدًا: {dest_dir}")

        safe_path = dest_dir / "subs.ass"
        try:
            if src.resolve() != safe_path.resolve():
                shutil.copyfile(src, safe_path)
        except OSError as e:
            raise RuntimeError(
                f"تعذّر نسخ ASS إلى مسار آمن: {src} -> {safe_path}"
            ) from e
        return safe_path, None

    s = src.resolve().as_posix()
    if "'" not in s:
        return src, None

    try:
        safe_dir = Path(tempfile.mkdtemp(prefix="stage4_ass_"))
    except OSError as e:
        logger.warning(
            f"تعذّر إنشاء مجلد آمن لمسار ASS ({e}) — استخدام المسار الأصلي."
        )
        return src, None

    safe_path = safe_dir / "subs.ass"
    try:
        shutil.copyfile(src, safe_path)
    except OSError as e:
        logger.warning(
            f"تعذّر نسخ ASS إلى مسار آمن ({e}) — استخدام المسار الأصلي."
        )
        try:
            shutil.rmtree(safe_dir, ignore_errors=True)
        except Exception:
            pass
        return src, None

    logger.info(
        "🛡️ نُسخ ملف ASS إلى مسار آمن (المسار الأصلي يحتوي ')."
    )
    return safe_path, safe_dir


# ============================================================
# NumPy اختياري لتسريع المؤثرات الصوتية
# ============================================================

_NUMPY_MODULE: Optional[Any] = None
_NUMPY_CHECKED = False


def _get_numpy() -> Optional[Any]:
    global _NUMPY_MODULE, _NUMPY_CHECKED
    if not _NUMPY_CHECKED:
        try:
            import numpy as np  # type: ignore
            _NUMPY_MODULE = np
        except Exception:
            _NUMPY_MODULE = None
        _NUMPY_CHECKED = True
    return _NUMPY_MODULE


def _event_volume(
    master: float,
    per_transition_volumes: Optional[List[float]],
    idx: int,
) -> float:
    vol = master
    if per_transition_volumes is not None and idx < len(per_transition_volumes):
        try:
            v = float(per_transition_volumes[idx])
            if math.isfinite(v):
                vol = master * max(0.0, v)
        except (TypeError, ValueError):
            pass
    return max(0.0, min(1.0, vol))


# ============================================================
# المؤثرات الصوتية
# ============================================================

def create_synthetic_whoosh(output_path: Path) -> None:
    if output_path.is_file() and output_path.stat().st_size > 0:
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
            phase3 = 2 * math.pi * (90.0 * t + 110.0 * t * tn)

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
) -> None:
    if not base_whoosh.exists() or not base_whoosh.is_file():
        raise FileNotFoundError(f"ملف الـ whoosh الأساسي غير موجود: {base_whoosh}")

    if not isinstance(total_duration, (int, float)) or \
            not math.isfinite(total_duration) or total_duration <= 0:
        raise ValueError(f"مدة إجمالية غير صالحة لمسار الـ whoosh: {total_duration!r}")

    master = _clamp_float(whoosh_volume, 0.0, 1.0, 0.4)

    with wave.open(str(base_whoosh), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError("ملف whoosh الأساسي يجب أن يكون mono")
        if wf.getsampwidth() != 2:
            raise ValueError("ملف whoosh الأساسي يجب أن يكون 16-bit PCM")
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    total_samples = int(total_duration * sr)
    if total_samples <= 0:
        raise ValueError(
            f"عدد العينات الكلي غير صالح لمسار الـ whoosh: {total_samples}"
        )

    np = _get_numpy()
    if np is not None:
        base = np.frombuffer(raw, dtype="<i2").astype(np.int32)
        out = np.zeros(total_samples, dtype=np.int32)
        base_len = len(base)

        for idx, t in enumerate(transition_times):
            if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
                continue
            vol = _event_volume(master, per_transition_volumes, idx)
            if vol <= 0.0 or base_len == 0:
                continue

            start = int(t * sr)
            if start < 0 or start >= total_samples:
                continue

            end = min(start + base_len, total_samples)
            seg_len = end - start
            if seg_len <= 0:
                continue

            add = np.rint(
                base[:seg_len].astype(np.float32) * np.float32(vol)
            ).astype(np.int32)
            np.clip(out[start:end] + add, -32768, 32767, out=out[start:end])

        out16 = np.clip(out, -32768, 32767).astype("<i2")
        with wave.open(str(output_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(out16.tobytes())
        return

    base_samples = array.array("h")
    base_samples.frombytes(raw)
    if sys.byteorder == "big":
        base_samples.byteswap()
    base_len = len(base_samples)

    out = array.array("h", bytes(2 * total_samples))

    for idx, t in enumerate(transition_times):
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            continue
        vol = _event_volume(master, per_transition_volumes, idx)
        if vol <= 0.0:
            continue

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

    rhythm_volume = _clamp_float(rhythm_volume, 0.0, 1.0, 0.35)

    try:
        pulse = _build_pulse_sample(sample_rate)
    except Exception as e:
        logger.warning(f"تعذّر توليد نبضة الإيقاع: {e}")
        return False

    total_samples = int(total_duration * sample_rate)
    if total_samples <= 0:
        return False

    np = _get_numpy()
    if np is not None:
        try:
            pulse_np = np.frombuffer(pulse, dtype=np.int16).astype(np.int32)
            out = np.zeros(total_samples, dtype=np.int32)
            pulse_len = len(pulse_np)

            for t in trigger_times:
                if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
                    continue
                start = int(t * sample_rate)
                if start < 0 or start >= total_samples:
                    continue

                end = min(start + pulse_len, total_samples)
                seg_len = end - start
                if seg_len <= 0:
                    continue

                add = np.rint(
                    pulse_np[:seg_len].astype(np.float32) * np.float32(rhythm_volume)
                ).astype(np.int32)
                np.clip(out[start:end] + add, -32768, 32767, out=out[start:end])

            out16 = np.clip(out, -32768, 32767).astype("<i2")
            with wave.open(str(output_path), "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(out16.tobytes())
        except Exception as e:
            logger.warning(f"تعذّر كتابة مسار الإيقاع (NumPy): {e}")
            return False

        try:
            return output_path.is_file() and output_path.stat().st_size > 0
        except OSError:
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
        return output_path.is_file() and output_path.stat().st_size > 0
    except OSError:
        return False


# ============================================================
# طبقة الذكاء الاصطناعي (Gemini Editorial Director)
# ============================================================

def _accepts_single_prompt(fn: Any) -> bool:
    if isinstance(fn, type):
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    try:
        sig.bind("__prompt_probe__")
    except TypeError:
        return False
    return True


from stage4_director import AIEditorialDirector


# ============================================================
# قرارات الحركة والانتقال
# ============================================================

def _motion_allowed_for_duration(mtype: str, duration: float) -> bool:
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        return False
    if duration >= 6.0 and mtype in ("static_micro", "slow_drift"):
        return False
    if duration < 2.2 and mtype == "zoom_out":
        return False
    return True


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
        return _motion_allowed_for_duration(m, duration)

    chosen = next((m for m in rotated if _ok(m)), None)
    if chosen is None:
        chosen = next((m for m in rotated if _motion_allowed_for_duration(m, duration)), None)
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
    local: Dict[str, Any],
    ai_shot: Optional[Dict[str, Any]],
    duration: float,
) -> Dict[str, Any]:
    if not ai_shot:
        return local
    merged = dict(local)
    ai_motion = ai_shot.get("motion")
    if ai_motion in MOTION_TYPES and _motion_allowed_for_duration(ai_motion, duration):
        merged["type"] = ai_motion
        merged["source"] = "ai"
    elif ai_motion in MOTION_TYPES:
        # الحركة المقترحة من AI غير مسموحة للمدة، لكن نُبقي شدتها.
        merged["source"] = "ai_intensity_only"

    local_intensity = _clamp_float(local.get("intensity", 0.55), 0.25, 1.0, 0.55)
    merged["intensity"] = _clamp_float(
        ai_shot.get("intensity", local_intensity), 0.3, 1.0, local_intensity
    )
    return merged


def _choose_non_repeating_motion(
    role: str,
    duration: float,
    prev_type: str,
    current_motion: Dict[str, Any],
) -> Dict[str, Any]:
    cur_type = current_motion.get("type")
    if (
        cur_type in MOTION_TYPES
        and cur_type != prev_type
        and _motion_allowed_for_duration(cur_type, duration)
    ):
        return current_motion

    preferred = ROLE_PREFERRED_MOTIONS.get(role) or ROLE_PREFERRED_MOTIONS[""]
    for alt in preferred:
        if alt != prev_type and _motion_allowed_for_duration(alt, duration):
            new_motion = dict(current_motion)
            new_motion["type"] = alt
            new_motion["source"] = "local_adjusted"
            return new_motion

    for alt in MOTION_TYPES:
        if alt != prev_type and _motion_allowed_for_duration(alt, duration):
            new_motion = dict(current_motion)
            new_motion["type"] = alt
            new_motion["source"] = "local_adjusted"
            return new_motion

    return current_motion


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
        "punch_in":       (1.0, 1.15 + (intensity * 0.1), c, c, c, c),
        "slow_pan":       (pz, pz, lo, hi, c, c),
        "drift":          (sd, sd, 0.5 - ds / 2.0, 0.5 + ds / 2.0, c, c),
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
            capture_output=True,
            text=True,
            check=True,
            timeout=FFPROBE_TIMEOUT_SEC,
            encoding="utf-8",
            errors="replace",
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
    except OSError as e:
        raise RuntimeError(
            f"تعذّر تشغيل ffprobe للملف: {path} ({e})"
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
        "ffmpeg", "-nostdin", "-y", "-v", "error",
        "-f", "lavfi", "-i", lavfi_input,
        "-af", af_chain,
        "-t", f"{duration:.3f}",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=180,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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
            output_path.is_file()
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
    if not isinstance(audio_duration, (int, float)) or \
            not math.isfinite(audio_duration) or audio_duration <= 0:
        raise ValueError(f"مدة صوت غير صالحة: {audio_duration!r}")

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
        if total_fallback <= 0:
            raise ValueError("مجموع المدد الاحتياطية غير موجب")

        # توسيط نسبي إذا كان المجموع أقل من مدة الصوت.
        if abs(total_fallback - audio_duration) > 1e-6:
            factor = audio_duration / total_fallback
            durations = [max(min_dur, d * factor) for d in durations]

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

    for i, d in enumerate(durations):
        if not isinstance(d, (int, float)) or not math.isfinite(d) or d <= 0:
            raise ValueError(f"مدة غير صالحة للمقطع رقم {i}: {d!r}")

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
    """
    سلسلة xfade داخل نطاق مقاطع متتالي [first..last].

    المعامل time_base_frames يُطرح من O[k] ليصبح الإزاحة نسبية لأصل النطاق.
    - عند الاستخدام على كامل السلسلة: time_base_frames=0 (إزاحة مطلقة).
    - عند الاستخدام داخل مجموعة: time_base_frames=S[first] (إزاحة نسبية لبداية المجموعة).
    """
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
    """
    سلسلة xfade بين مجموعات مُدمجة مسبقًا (بعد رندرة كل مجموعة إلى ملف).

    المدخلات هي مخرجات المجموعات، وكل مجموعة مُخرَجها يحمل:
      - محتوى المجموعة على محورها المحلي الخاص (المجموعة g تبدأ محليًا من 0).

    عندما نُسلسل: vx0 = xfade(grp_0, grp_1)، ثم vx1 = xfade(vx0, grp_2)، ...
    يصبح محور vx{g} محاذيًا للمحور المطلق (بالزمن الحقيقي) للفيديو النهائي.

    ==> الإزاحة في كل xfade يجب أن تكون **مطلقة** (نسبةً لبداية التراكم)،
        أي O[k]/fps، وليس (O[k] - S[a])/fps.
        الأخيرة صحيحة فقط للمجموعة الأولى (a=0) حيث S[0]=0، وهي خطأ لما بعدها.
    """
    if len(groups) == 1:
        return "", "[0:v]"
    parts: List[str] = []
    prev = "[0:v]"
    for g in range(len(groups) - 1):
        _a, b = groups[g]
        k = b
        cur = f"[{g + 1}:v]"
        out = f"[vx{g}]"
        t_sec = layout["T"][k] / fps
        # التصحيح الحرج: offset مطلق زمنيًا (vx{g} مُراكم، محوره = المحور المطلق).
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
    layout: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, float, List[float], List[float]]:
    if n_segments <= 0:
        raise ValueError("n_segments يجب أن تكون ≥ 1")
    if len(durations) != n_segments:
        raise ValueError("عدد المدد لا يطابق عدد المقاطع")

    if layout is None:
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

    if output_video_path.exists() and output_video_path.is_dir():
        raise ValueError(f"مسار الإخراج مجلد وليس ملفًا: {output_video_path}")

    if output_video_path.suffix.lower() not in {".mp4", ".mov", ".m4v"}:
        logger.warning(
            f"امتداد الإخراج {output_video_path.suffix!r} غير مطابق لـ MP4؛ "
            f"سيتم تغييره إلى .mp4 لأن الترميز الناتج H.264/AAC في حاوية MP4."
        )
        output_video_path = output_video_path.with_suffix(".mp4")

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

    ai_plan = AIEditorialDirector.get().plan_montage(
        renderable_timeline, durations, local_transitions
    )

    ai_shots_by_idx: Dict[int, Dict[str, Any]] = {}
    visual_overlays_plan = []
    if ai_plan:
        if isinstance(ai_plan.get("shots"), list):
            for s in ai_plan["shots"]:
                if isinstance(s, dict) and isinstance(s.get("idx"), int):
                    ai_shots_by_idx[s["idx"]] = s
        if isinstance(ai_plan.get("visual_overlays"), list):
            visual_overlays_plan = ai_plan.get("visual_overlays")

    final_motions: List[Dict[str, Any]] = []
    final_transitions: List[str] = []
    whoosh_strengths: List[float] = []
    ambience_boosts: List[float] = []

    for i in range(n_shots):
        ai_s = ai_shots_by_idx.get(i)
        merged_motion = _merge_ai_into_motion(local_motions[i], ai_s, durations[i])
        if i > 0 and merged_motion["type"] == final_motions[i - 1]["type"]:
            role = str(renderable_timeline[i].get("narrative_role") or "").lower().strip()
            merged_motion = _choose_non_repeating_motion(
                role,
                durations[i],
                final_motions[i - 1]["type"],
                merged_motion,
            )
        final_motions.append(merged_motion)

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
        n_shots, durations, final_transitions, fps, layout=layout
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
        # whoosh_strengths[i] هو المطلوب للانتقال التالي للقطة i (بعد i).
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
        # Whoosh أصبح اختياريًا.
        has_whoosh_xfade = False
        if transition_times_final:
            try:
                create_synthetic_whoosh(sfx_whoosh)
                if sfx_whoosh.is_file() and sfx_whoosh.stat().st_size > 0:
                    build_whoosh_timeline(
                        sfx_whoosh,
                        transition_times_final,
                        audio_duration,
                        whoosh_timeline,
                        whoosh_volume=0.42,
                        per_transition_volumes=whoosh_per_transition or None,
                    )
                    has_whoosh_xfade = (
                        whoosh_timeline.is_file()
                        and whoosh_timeline.stat().st_size > 0
                    )
            except Exception as whoosh_err:
                logger.warning(f"تعذّر توليد Whoosh: {whoosh_err}")
                has_whoosh_xfade = False

            if not has_whoosh_xfade:
                try:
                    if whoosh_timeline.exists():
                        whoosh_timeline.unlink()
                except OSError:
                    pass
        else:
            logger.info("🌬️ لا انتقالات — تم تخطي Whoosh.")

        segment_files: List[Path] = []
        logger.info(
            f"🎬 بدء رندرة {n_shots} لقطة (run_id={run_id}) — "
            f"AI: {'مفعّل' if AIEditorialDirector.get().is_available() else 'محلي'}."
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
                "-i", str(frame_path.resolve()),
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
                    g_inputs += ["-i", str(segment_files[k].resolve())]
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

        # ASS آمن: نسخ إلى temp_dir واستخدام اسم نسبي.
        safe_ass, safe_ass_dir = _prepare_safe_ass(subtitles_ass, temp_dir)
        ass_filename = safe_ass.name

        final_inputs: List[str] = []
        for vf in video_inputs:
            final_inputs += ["-i", str(Path(vf).resolve())]

        final_inputs += ["-i", str(audio_file.resolve())]
        idx_audio = n_video_inputs
        next_idx = idx_audio + 1

        whoosh_idx: Optional[int] = None
        if has_whoosh_xfade:
            final_inputs += ["-i", str(whoosh_timeline.resolve())]
            whoosh_idx = next_idx
            next_idx += 1

        ambience_idx: Optional[int] = None
        if has_ambience:
            final_inputs += ["-i", str(ambience_file.resolve())]
            ambience_idx = next_idx
            next_idx += 1

        rhythm_idx: Optional[int] = None
        if has_rhythm:
            final_inputs += ["-i", str(rhythm_file.resolve())]
            rhythm_idx = next_idx
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

        # الترتيب الصحيح: تلوين/تحسين -> ترجمة -> fade.
        filter_parts.append(f"[vpad]{polish_base}[vcolor]")
        filter_parts.append(f"[vcolor]ass=filename={ass_filename}[vsub]")

        if audio_duration >= 0.9:
            fade_out_start = audio_duration - 0.4
            filter_parts.append(
                f"[vsub]fade=t=in:st=0:d=0.4,"
                f"fade=t=out:st={fade_out_start:.3f}:d=0.4[vout]"
            )
        else:
            filter_parts.append("[vsub]null[vout]")

        ducked_tracks: List[Tuple[str, int, float, str, str, str, str]] = []
        if has_rhythm and rhythm_idx is not None:
            ducked_tracks.append(
                ("rhythm", rhythm_idx, 0.45, "0.03", "5", "10", "300")
            )
        if has_ambience and ambience_idx is not None:
            ducked_tracks.append(
                ("ambience", ambience_idx, 0.55, "0.02", "6", "20", "400")
            )

        n_ducked = len(ducked_tracks)

        if n_ducked > 0:
            voice_split_labels = "".join(f"[vsc{i}]" for i in range(n_ducked))
            filter_parts.append(
                f"[{idx_audio}:a]{_AUDIO_VOICE_NORMALIZE_CHAIN},"
                f"asplit={n_ducked + 1}[vmain]{voice_split_labels}"
            )
        else:
            filter_parts.append(
                f"[{idx_audio}:a]{_AUDIO_VOICE_NORMALIZE_CHAIN}[vmain]"
            )

        mix_inputs: List[Tuple[str, float]] = [("[vmain]", 1.0)]

        if whoosh_idx is not None:
            filter_parts.append(
                f"[{whoosh_idx}:a]{_AUDIO_SFX_NORMALIZE_CHAIN}[whoosh_src]"
            )
            mix_inputs.append(("[whoosh_src]", 0.55))

        for i, (name, in_idx, weight, thr, ratio, att, rel) in enumerate(ducked_tracks):
            src_label = f"[{name}_src]"
            duck_label = f"[{name}_d]"
            filter_parts.append(
                f"[{in_idx}:a]{_AUDIO_SFX_NORMALIZE_CHAIN}{src_label}"
            )
            filter_parts.append(
                f"{src_label}[vsc{i}]sidechaincompress="
                f"threshold={thr}:ratio={ratio}:"
                f"attack={att}:release={rel}{duck_label}"
            )
            mix_inputs.append((duck_label, float(weight)))

        filter_parts.append(_build_audio_mix_filter(mix_inputs, "[aout]"))
        filter_complex = ";".join(p for p in filter_parts if p)

        cmd_final = [
            "ffmpeg", "-y", "-v", "warning",
            *final_inputs,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-t", f"{audio_duration:.4f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(staged_output_video),
        ]

        try:
            main_required = ["ass", "alimiter", "amix"]
            if n_video_inputs > 1:
                main_required.append("xfade")
            if pad_needed > 0.001:
                main_required.append("tpad")
            if overlay_inputs:
                main_required.append("overlay")
            if ducked_tracks:
                main_required.append("sidechaincompress")
            if whoosh_idx is not None or ducked_tracks:
                main_required.append("pan")
            _require_ffmpeg_filters(main_required)

            _run_ffmpeg(
                cmd_final,
                FINAL_TIMEOUT_SEC,
                "التصدير الموحّد",
                cwd=temp_dir,
            )
        except RuntimeError as e:
            logger.warning(
                f"⚠️ فشل التصدير الموحّد: {e} — "
                f"fallback: concat hard-cut + تمريرة موحّدة "
                f"(مع الحفاظ على الترجمة والمؤثرات الصوتية، بدون xfade)."
            )

            whoosh_timeline_cut = temp_dir / "whoosh_cut.wav"
            has_whoosh_cut = False
            if has_whoosh_xfade and transition_times_cut:
                try:
                    build_whoosh_timeline(
                        sfx_whoosh,
                        transition_times_cut,
                        audio_duration,
                        whoosh_timeline_cut,
                        whoosh_volume=0.42,
                        per_transition_volumes=whoosh_per_transition or None,
                    )
                    has_whoosh_cut = (
                        whoosh_timeline_cut.is_file()
                        and whoosh_timeline_cut.stat().st_size > 0
                    )
                except Exception as wh_err:
                    logger.warning(f"تعذّر توليد Whoosh للـ fallback: {wh_err}")
                    has_whoosh_cut = False

                if not has_whoosh_cut:
                    try:
                        if whoosh_timeline_cut.exists():
                            whoosh_timeline_cut.unlink()
                    except OSError:
                        pass

            # fallback rhythm: لا نستخدم توقيت xfade القديم إطلاقًا.
            fb_rhythm_path: Optional[Path] = None
            if has_rhythm and rhythm_trigger_times_cut:
                rhythm_cut_path = temp_dir / "rhythm_cut.wav"
                try:
                    if _build_rhythm_timeline(
                        rhythm_trigger_times_cut,
                        audio_duration,
                        44100,
                        rhythm_cut_path,
                        rhythm_volume=0.32,
                    ):
                        fb_rhythm_path = rhythm_cut_path
                        logger.info(
                            f"🎵 إيقاع fallback مُولَّد بمواضع القطع "
                            f"({len(rhythm_trigger_times_cut)} نبضة)."
                        )
                    else:
                        logger.warning(
                            "تعذّر توليد إيقاع fallback — سيتم تعطيل الإيقاع في fallback."
                        )
                        fb_rhythm_path = None
                except Exception as fr_err:
                    logger.warning(
                        f"خطأ في توليد إيقاع fallback ({fr_err}) — تعطيل الإيقاع."
                    )
                    fb_rhythm_path = None
            elif has_rhythm:
                logger.warning(
                    "لا توجد مواضع قطع صالحة للإيقاع في fallback — تعطيل الإيقاع."
                )
                fb_rhythm_path = None

            trimmed_files: List[Path] = []
            for i, seg in enumerate(segment_files):
                a = layout["bounds"][i] - layout["S"][i]
                b = layout["bounds"][i + 1] - layout["S"][i]
                a = max(0, int(a))
                b = max(a + 1, int(b))
                trimmed = temp_dir / f"trim_{i:03d}.mp4"
                _run_ffmpeg(
                    [
                        "ffmpeg", "-y", "-v", "error",
                        "-i", str(seg.resolve()),
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
                    # أسماء نسبية بسيطة داخل temp_dir.
                    f.write(f"file '{seg.name}'\n")

            fb_inputs: List[str] = [
                "-f", "concat", "-safe", "0",
                "-i", "concat_list.txt",
                "-i", str(audio_file.resolve()),
            ]
            fb_idx_audio = 1
            fb_next = 2

            fb_whoosh_idx: Optional[int] = None
            if has_whoosh_cut:
                fb_inputs += ["-i", str(whoosh_timeline_cut.resolve())]
                fb_whoosh_idx = fb_next
                fb_next += 1

            fb_amb_idx: Optional[int] = None
            if has_ambience:
                fb_inputs += ["-i", str(ambience_file.resolve())]
                fb_amb_idx = fb_next
                fb_next += 1

            fb_rhythm_idx: Optional[int] = None
            if fb_rhythm_path is not None:
                fb_inputs += ["-i", str(fb_rhythm_path.resolve())]
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

            fb_filter_parts.append(f"[vpad]{fb_polish_base}[fbcolor]")
            fb_filter_parts.append(f"[fbcolor]ass=filename={ass_filename}[fbsub]")

            if audio_duration >= 0.9:
                fb_fade_start = audio_duration - 0.4
                fb_filter_parts.append(
                    f"[fbsub]fade=t=in:st=0:d=0.4,"
                    f"fade=t=out:st={fb_fade_start:.3f}:d=0.4[vout]"
                )
            else:
                fb_filter_parts.append("[fbsub]null[vout]")

            fb_ducked: List[Tuple[str, int, float, str, str, str, str]] = []
            if fb_rhythm_idx is not None:
                fb_ducked.append(
                    ("rhythm", fb_rhythm_idx, 0.45, "0.03", "5", "10", "300")
                )
            if fb_amb_idx is not None:
                fb_ducked.append(
                    ("ambience", fb_amb_idx, 0.55, "0.02", "6", "20", "400")
                )

            n_fb_ducked = len(fb_ducked)

            if n_fb_ducked > 0:
                fb_split_labels = "".join(f"[fsc{i}]" for i in range(n_fb_ducked))
                fb_filter_parts.append(
                    f"[{fb_idx_audio}:a]{_AUDIO_VOICE_NORMALIZE_CHAIN},"
                    f"asplit={n_fb_ducked + 1}[fmain]{fb_split_labels}"
                )
            else:
                fb_filter_parts.append(
                    f"[{fb_idx_audio}:a]{_AUDIO_VOICE_NORMALIZE_CHAIN}[fmain]"
                )

            fb_mix_inputs: List[Tuple[str, float]] = [("[fmain]", 1.0)]

            if fb_whoosh_idx is not None:
                fb_filter_parts.append(
                    f"[{fb_whoosh_idx}:a]{_AUDIO_SFX_NORMALIZE_CHAIN}[fbwhoosh_src]"
                )
                fb_mix_inputs.append(("[fbwhoosh_src]", 0.55))

            for i, (name, in_idx, weight, thr, ratio, att, rel) in enumerate(fb_ducked):
                src_label = f"[fb_{name}_src]"
                duck_label = f"[fb_{name}_d]"
                fb_filter_parts.append(
                    f"[{in_idx}:a]{_AUDIO_SFX_NORMALIZE_CHAIN}{src_label}"
                )
                fb_filter_parts.append(
                    f"{src_label}[fsc{i}]sidechaincompress="
                    f"threshold={thr}:ratio={ratio}:"
                    f"attack={att}:release={rel}{duck_label}"
                )
                fb_mix_inputs.append((duck_label, float(weight)))

            fb_filter_parts.append(_build_audio_mix_filter(fb_mix_inputs, "[aout]"))
            fb_filter_complex = ";".join(p for p in fb_filter_parts if p)

            fb_required = ["ass", "alimiter", "amix"]
            if fb_pad > 0.001:
                fb_required.append("tpad")
            if overlay_inputs:
                fb_required.append("overlay")
            if fb_ducked:
                fb_required.append("sidechaincompress")
            if fb_whoosh_idx is not None or fb_ducked:
                fb_required.append("pan")
            _require_ffmpeg_filters(fb_required)

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
                    "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart",
                    str(staged_output_video),
                ],
                FALLBACK_STEP_TIMEOUT_SEC,
                "التصدير (fallback)",
                cwd=temp_dir,
            )

        _validate_non_empty_file(staged_output_video, "الفيديو النهائي المرحلي")
        final_duration = _probe_media_duration(staged_output_video)
        if final_duration <= 0:
            raise RuntimeError(
                f"مدة الفيديو النهائي المرحلي غير موجبة: {final_duration}"
            )

        drift = abs(final_duration - audio_duration)
        if drift > 0.15:
            logger.warning(
                f"انحراف مدة الفيديو النهائي عن الصوت: "
                f"video={final_duration:.3f}s, audio={audio_duration:.3f}s, "
                f"drift={drift:.3f}s"
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
        assert rhythm_out.is_file() and rhythm_out.stat().st_size > 0
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
        assert base.is_file() and base.stat().st_size > 0

        whoosh_out = td_path / "whoosh_timeline.wav"
        build_whoosh_timeline(base, [1.4, 2.0], 3.0, whoosh_out, whoosh_volume=0.4)
        assert whoosh_out.is_file() and whoosh_out.stat().st_size > 0

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

    # 9. اختبار AIEditorialDirector بدون شبكة
    director = AIEditorialDirector.get()
    director._available = False
    director._call_fn = None
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
    d_tmp = AIEditorialDirector()
    raw_plan = (
        '{"shots":[{"idx":0,"motion":"zoom_in","intensity":"nan",'
        '"whoosh_strength":1e999},{"idx":9,"motion":"zoom_out"}],'
        '"pacing":{"peaks":null,"calm":[1,2.0,"x",null]}}'
    )
    parsed = d_tmp._parse_plan(raw_plan, 2)
    assert parsed and len(parsed["shots"]) == 1, parsed
    print("[OK] test #13: AI plan parsing hardened (null/NaN/inf/dict/bytes)")

    # ------------------------------------------------------------
    # 14. اختبار الهروب الآمن لمسار ASS + _prepare_safe_ass
    # ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td14:
        sub = Path(td14) / "sub [1]"
        sub.mkdir()
        p_real = sub / "my 'draft'.ass"
        p_real.touch()

        esc_real = _escape_filter_path(p_real)

        # FFmpeg filtergraph escaping inside single quotes:
        # ' -> \' and \ -> \\
        assert "\\'" in esc_real, esc_real
        assert "'\\''" not in esc_real, esc_real

        i = 0
        while i < len(esc_real):
            if esc_real[i] == "\\":
                assert i + 1 < len(esc_real) and esc_real[i + 1] in ("\\", "'"), (
                    f"unexpected escape at position {i} in {esc_real!r}"
                )
                i += 2
            else:
                i += 1

        cleaned = esc_real.replace("\\'", "").replace("\\\\", "")
        assert "\\" not in cleaned, cleaned
        assert "\\[" not in cleaned, cleaned
        assert "\\]" not in cleaned, cleaned

    with tempfile.TemporaryDirectory() as td14b:
        td_path = Path(td14b)

        # بدون dest_dir: مسار نظيف لا يُنسخ
        clean = td_path / "clean.ass"
        clean.touch()
        p_safe, d_safe = _prepare_safe_ass(clean)
        assert p_safe == clean, (p_safe, clean)
        assert d_safe is None, d_safe

        # بدون dest_dir: مسار فيه ' يُنسخ
        sub2 = td_path / "sub2 [x]"
        sub2.mkdir()
        p_unsafe = sub2 / "with 'quote'.ass"
        p_unsafe.write_text("[Script Info]\n", encoding="utf-8")
        p_copied, d_copied = _prepare_safe_ass(p_unsafe)
        try:
            assert p_copied != p_unsafe
            assert p_copied.is_file() and p_copied.stat().st_size > 0
            assert d_copied is not None and d_copied.is_dir()
            assert "'" not in p_copied.name, p_copied
        finally:
            if d_copied is not None:
                shutil.rmtree(d_copied, ignore_errors=True)

        # مع dest_dir: دائمًا subs.ass نسبي آمن
        dest = td_path / "dest"
        dest.mkdir()
        p_dest, d_dest = _prepare_safe_ass(p_unsafe, dest)
        assert d_dest is None, d_dest
        assert p_dest.parent == dest, p_dest
        assert p_dest.name == "subs.ass", p_dest
        assert p_dest.is_file() and p_dest.stat().st_size > 0

    print("[OK] test #14: ASS path escaping + _prepare_safe_ass (portable)")

    # ------------------------------------------------------------
    # 15. اختبار LRU + قفل كاش AI
    # ------------------------------------------------------------
    d_lru = AIEditorialDirector()
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
    cut_pos = [lay_fb["bounds"][i] / 60 for i in range(1, len(durs_fb))]
    xfade_pos = [lay_fb["O"][i - 1] / 60 for i in range(1, len(durs_fb))]
    for i in range(len(cut_pos)):
        assert cut_pos[i] >= xfade_pos[i], (i, cut_pos[i], xfade_pos[i])
    print(
        f"[OK] test #17: rhythm timing cut vs xfade "
        f"(cut={[round(x, 3) for x in cut_pos]}, "
        f"xfade={[round(x, 3) for x in xfade_pos]})"
    )

    # ------------------------------------------------------------
    # 18. اختبار offset المطلق في xfade بين المجموعات (الإصلاح الحرج)
    # ------------------------------------------------------------
    durs_g = [2.0, 2.0, 2.0, 2.0]
    lay_g = _compute_segment_layout(durs_g, ["crossfade_medium"] * 3, 60)
    groups_g = [(0, 1), (2, 2), (3, 3)]
    filt_g, _ = _xfade_filter_for_groups(groups_g, lay_g, 60)

    # الإزاحتان يجب أن تكونا **مطلقتين** (نسبةً لبداية التراكم = الزمن المطلق).
    exp0 = lay_g["O"][1] / 60     # 232/60 = 3.866667
    exp1 = lay_g["O"][2] / 60     # 352/60 = 5.866667
    assert f"offset={exp0:.6f}" in filt_g, filt_g
    assert f"offset={exp1:.6f}" in filt_g, filt_g

    # نتأكد أن الخطأ القديم (الطرح النسبي) لم يعد موجودًا.
    wrong1 = (lay_g["O"][2] - lay_g["S"][2]) / 60
    if abs(exp1 - wrong1) > 1e-9:
        assert f"offset={wrong1:.6f}" not in filt_g, filt_g
    print(
        f"[OK] test #18: group xfade offsets are absolute "
        f"(exp0={exp0:.4f}, exp1={exp1:.4f})"
    )

    # ------------------------------------------------------------
    # 19. اختبار بناء مزج الصوت مع/بدون دعم amix features
    # ------------------------------------------------------------
    _AMIX_FEATURES = (True, True)
    mix = _build_audio_mix_filter([("[a]", 1.0), ("[b]", 0.5)], "[aout]")
    assert "weights='1.000000 0.500000'" in mix, mix
    assert "normalize=0" in mix, mix
    assert "level=0" in mix, mix

    _AMIX_FEATURES = (False, False)
    mix2 = _build_audio_mix_filter([("[a]", 1.0), ("[b]", 0.5)], "[aout]")
    assert "amix=inputs=2" in mix2, mix2
    assert "volume=" in mix2, mix2
    assert "level=0" in mix2, mix2

    one = _build_audio_mix_filter([("[a]", 1.0)], "[aout]")
    assert "amix" not in one, one
    assert "alimiter" in one, one
    _AMIX_FEATURES = None
    print("[OK] test #19: audio mix builder supports modern/legacy amix")

    # ------------------------------------------------------------
    # 20. اختبار قيود مدة الحركة ودمج AI
    # ------------------------------------------------------------
    assert not _motion_allowed_for_duration("zoom_out", 2.0)
    assert _motion_allowed_for_duration("zoom_out", 2.3)
    assert not _motion_allowed_for_duration("slow_drift", 6.0)
    assert _motion_allowed_for_duration("slow_drift", 5.9)

    local_motion = {"type": "zoom_in", "intensity": 0.5, "source": "local"}
    merged = _merge_ai_into_motion(
        local_motion,
        {"motion": "zoom_out", "intensity": 0.8},
        2.0,
    )
    assert merged["type"] == "zoom_in", merged
    assert abs(merged["intensity"] - 0.8) < 1e-9, merged
    assert merged["source"] == "ai_intensity_only", merged

    chosen = _choose_non_repeating_motion(
        "hook",
        2.0,
        "zoom_pan_right",
        {"type": "zoom_pan_right", "intensity": 0.5},
    )
    assert chosen["type"] != "zoom_pan_right", chosen
    assert _motion_allowed_for_duration(chosen["type"], 2.0), chosen
    print("[OK] test #20: motion duration constraints and AI merge hardened")

    print("[ALL TESTS PASSED]")
