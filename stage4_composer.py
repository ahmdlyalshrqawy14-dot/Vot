import os
import wave
import struct
import math
import shutil
import subprocess
import array
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("Stage4Composer")


# أدوار سردية يُسمح فيها بالإيقاع الوثائقي الخافت.
# أي دور آخر (reflection / question / tension / غير معروف) لا يُشغّل الإيقاع.
RHYTHM_ENABLED_ROLES = {"hook", "revelation", "payoff", "actionable"}


def create_synthetic_whoosh(output_path: Path):
    """
    إنشاء ملف مؤثر هوائي ناعم (Soft Air Whoosh) في الخلفية
    بصوت طبيعي وهادئ (-18dB) في حال لم يكن الملف متوفراً في السيرفر.
    """
    if output_path.exists():
        return

    sample_rate = 44100
    duration = 0.4  # ثانية
    n_samples = int(sample_rate * duration)

    with wave.open(str(output_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for i in range(n_samples):
            t = i / sample_rate
            envelope = math.sin(math.pi * t / duration) ** 2
            freq = 150 + 150 * (t / duration)
            sample = envelope * math.sin(2 * math.pi * freq * t)
            val = int(sample * 32767 * 0.12)
            wav_file.writeframes(struct.pack("<h", val))


def build_whoosh_timeline(
    base_whoosh: Path,
    transition_times: List[float],
    total_duration: float,
    output_path: Path,
    whoosh_volume: float = 0.4,
):
    """
    توليد مسار صوتي موحد (Unified SFX Track) في الذاكرة عبر بايثون الخالص:
    - يقرأ ملف whoosh الأساسي مرة واحدة.
    - ينشئ بُفر صوتي بطول الفيديو الكامل.
    - يزرع نسخة من الـ whoosh عند كل نقطة انتقال بشدة 0.4.
    """
    if not base_whoosh.exists() or not base_whoosh.is_file():
        raise FileNotFoundError(f"ملف الـ whoosh الأساسي غير موجود: {base_whoosh}")

    if not isinstance(total_duration, (int, float)) or \
            not math.isfinite(total_duration) or total_duration <= 0:
        raise ValueError(f"مدة إجمالية غير صالحة لمسار الـ whoosh: {total_duration!r}")

    with wave.open(str(base_whoosh), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    base_samples = array.array("h")
    base_samples.frombytes(raw)
    base_len = len(base_samples)

    total_samples = int(total_duration * sr)
    if total_samples <= 0:
        raise ValueError(
            f"عدد العينات الكلي غير صالح لمسار الـ whoosh: {total_samples}"
        )

    out = array.array("h", bytes(2 * total_samples))

    for t in transition_times:
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            continue
        start = int(t * sr)
        if start < 0 or start >= total_samples:
            continue
        for i in range(base_len):
            idx = start + i
            if idx >= total_samples:
                break
            v = out[idx] + int(base_samples[i] * whoosh_volume)
            if v > 32767:
                v = 32767
            elif v < -32768:
                v = -32768
            out[idx] = v

    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out.tobytes())


def _build_pulse_sample(sample_rate: int) -> array.array:
    """
    نبضة وثائقية ناعمة واحدة (sub-bass thump) بطول 0.5s.
    - ترددات منخفضة (55Hz + 82.5Hz) لإحساس سينمائي خافت.
    - Envelope هجومي سريع ثم تلاشٍ أسّي.
    - بلا Melody وبلا Beat حاد.
    """
    duration = 0.5
    n = int(sample_rate * duration)
    samples = array.array("h", bytes(2 * n))
    for i in range(n):
        t = i / sample_rate
        attack = min(1.0, t / 0.02)
        env = math.exp(-5.5 * t / duration) * attack
        wave_val = (
            0.75 * math.sin(2 * math.pi * 55.0 * t)
            + 0.25 * math.sin(2 * math.pi * 82.5 * t)
        )
        v = env * wave_val
        val = int(v * 32767 * 0.6)
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
    """
    بناء مسار الإيقاع الوثائقي الخافت:
    - يعيد False عند أي فشل (بدون رفع استثناء)، ليتابع الرندر بدونه.
    - نبضات وثائقية فقط عند اللحظات المسموح بها.
    """
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


def get_ken_burns_filter(pattern_index: int, duration: float, fps: int = 60) -> str:
    """
    تطبيق النمط الدوري الذكي لحركة كين بيرنز (6 حركات):
    1. Slow Zoom In
    2. Slow Pan Right
    3. Slow Zoom Out
    4. Slow Pan Up
    5. Slow Zoom In
    6. Slow Pan Left
    """
    if not isinstance(duration, (int, float)) or \
            not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"مدة غير صالحة لفلتر Ken Burns: {duration!r}")

    if not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"معدل إطارات غير صالح لفلتر Ken Burns: {fps!r}")

    frames = int(duration * fps)
    if frames < 1:
        raise ValueError(
            f"المدة ({duration:.6f}s) قصيرة جدًا لإنتاج إطار واحد عند {fps}fps. "
            f"الحد الأدنى المطلوب تقريبًا: {1.0 / fps:.6f}s"
        )

    pattern = pattern_index % 6

    if pattern == 0:
        return f"zoompan=z='min(zoom+0.0008,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 1:
        return f"zoompan=z='1.1':x='if(lte(on,1),0,x+1.2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 2:
        return f"zoompan=z='if(lte(on,1),1.15,max(1.0,zoom-0.0008))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 3:
        return f"zoompan=z='1.1':x='iw/2-(iw/zoom/2)':y='if(lte(on,1),ih*0.1,max(0,y-0.8))':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 4:
        return f"zoompan=z='min(zoom+0.0007,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    else:
        return f"zoompan=z='1.1':x='if(lte(on,1),iw*0.1,max(0,x-1.2))':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"


def _probe_media_duration(path: Path) -> float:
    """
    قراءة المدة الفعلية لملف وسائط (صوت أو فيديو) بالثواني عبر ffprobe.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"فشل ffprobe في قراءة المدة للملف: {path} "
            f"(رمز الخروج: {e.returncode})"
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
    """
    تهيئة مسار آمن لصيغة FFmpeg concat demuxer.
    """
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


def _try_generate_ambience(output_path: Path, duration: float) -> bool:
    """
    محاولة توليد طبقة Ambience داخلية شديدة الخفوت عبر ffmpeg lavfi.
    """
    if not isinstance(duration, (int, float)) or \
            not math.isfinite(duration) or duration <= 6.0:
        return False

    fade_out_start = max(0.0, duration - 2.0)

    lavfi_input = (
        f"anoisesrc=color=pink:amplitude=0.08:"
        f"sample_rate=44100:duration={duration:.3f}"
    )
    af_chain = (
        "lowpass=f=250,"
        "highpass=f=40,"
        "volume=0.35,"
        "afade=t=in:st=0:d=2,"
        f"afade=t=out:st={fade_out_start:.3f}:d=2"
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
        subprocess.run(cmd, check=True, timeout=180)
    except subprocess.CalledProcessError as e:
        logger.warning(f"فشل توليد طبقة الـ ambience (exit={e.returncode}).")
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


def _compute_image_boundaries(
    renderable_timeline: List[Dict[str, Any]],
    audio_duration: float,
    fps: int,
) -> Tuple[List[float], List[float]]:
    """
    حساب حدود كل صورة في الزمن المطلق للرندر.

    القاعدة الصحيحة:
    - الصورة رقم 0 تبدأ عند الزمن 0.0 في الرندر (وليس عند start[0]).
      هذا يضمن أن أي pause_before في أول جملة يظهر أثناء الصورة الأولى،
      ولا تُعرض صورة الجملة التالية قبل بدء صوتها.
    - الصورة i (i > 0) تبدأ عند start[i] الفعلي للجملة.
    - الصورة i تنتهي عند start[i+1]، أو عند audio_duration إن كانت الأخيرة.

    Returns:
        (durations, image_starts):
        - durations[i]: مدة العرض الفعلية للصورة i.
        - image_starts[i]: زمن البداية المطلق للصورة i في الرندر.
          image_starts[0] = 0.0 دائماً.
    """
    n = len(renderable_timeline)
    if n == 0:
        return [], []

    min_dur = 1.0 / float(fps)

    # محاولة قراءة start لكل عنصر
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

    # ============================================================
    # خطة احتياطية: item["duration"] لكل عنصر على حدة
    # ============================================================
    if not have_valid_starts:
        logger.warning(
            "عناصر timeline لا تحتوي على 'start' صالح. سيتم استخدام "
            "item['duration'] كخطة احتياطية لكل عنصر (بدون مدة ثابتة عمياء). "
            "التوقيت الناتج تقريبي وقد لا يعكس الفواصل بين الجمل."
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

        image_starts: List[float] = []
        cur = 0.0
        for d in durations:
            image_starts.append(cur)
            cur += d
        return durations, image_starts

    # ============================================================
    # التحقق من ترتيب البدء تصاعديًا
    # ============================================================
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

    # ============================================================
    # حساب الحدود المطلقة (الصورة 0 تبدأ عند 0.0)
    # ============================================================
    durations = []
    for i in range(n):
        if i == 0:
            img_start = 0.0
        else:
            img_start = starts[i]

        if i < n - 1:
            img_end = starts[i + 1]
        else:
            img_end = audio_duration

        d = img_end - img_start
        if not math.isfinite(d):
            raise ValueError(f"مدة غير منتهية للصورة رقم {i}: {d!r}")
        if d <= 0:
            logger.warning(
                f"مدة الصورة رقم {i} غير موجبة ({d:.6f}s) بسبب فرق تقريب "
                f"زمني صغير؛ سيتم رفعها إلى إطار واحد ({min_dur:.6f}s)."
            )
            d = min_dur
        elif d < min_dur:
            d = min_dur
        durations.append(d)

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
    """غلاف متوافق خلفيًا؛ يعيد المدد فقط."""
    durations, _ = _compute_image_boundaries(
        renderable_timeline, audio_duration, fps
    )
    return durations


def render_final_video(
    frames: List[Path],
    timeline: List[Dict[str, Any]],
    audio_file: Path,
    subtitles_ass: Path,
    output_video_path: Path
):
    """
    بناء خط المونتاج الآلي وتصدير الفيديو النهائي عبر FFmpeg:
    - 1080p Full HD / 60fps
    - مؤثرات Ken Burns فقط (بدون Vignette)
    - انتقالات صوتية Soft Whoosh عند كل قطع
    - طبقة Ambience خافتة اختيارية
    - إيقاع وثائقي خافت (Minimal Cinematic Rhythmic Underscore)
      يظهر فقط في hook/revelation/payoff/actionable
    - خلو تام 100% من الموسيقى الخلفية التقليدية
    """

    # ============================================================
    # [حماية 1] استبعاد العناصر الفارغة + التحقق من تناسق الأعداد
    # ============================================================
    if not isinstance(frames, list) or not isinstance(timeline, list):
        raise TypeError("frames و timeline يجب أن تكونا قائمتين.")

    renderable_timeline = [it for it in timeline if not it.get("empty")]

    if not renderable_timeline:
        raise ValueError(
            "قائمة التوقيت لا تحتوي على أي عنصر قابل للرندرة "
            "(كل العناصر فارغة أو القائمة فارغة أصلًا)."
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

    # ============================================================
    # [حماية 2] التحقق المبكر من سلامة ملفات الصور
    # ============================================================
    validated_pairs: List[Tuple[Path, Dict[str, Any]]] = []
    for idx, (frame_path, item) in enumerate(raw_pairs):
        fp = Path(frame_path)
        _validate_regular_file(fp, "ملف الصورة", index=idx)
        validated_pairs.append((fp, item))

    # ============================================================
    # [حماية 3] التحقق المبكر من مدخلات الصوت والترجمة ومجلد الإخراج
    # ============================================================
    audio_file = Path(audio_file)
    subtitles_ass = Path(subtitles_ass)
    output_video_path = Path(output_video_path)

    _validate_regular_file(audio_file, "ملف التعليق الصوتي")
    _validate_regular_file(subtitles_ass, "ملف الترجمة ASS")

    output_dir = output_video_path.parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"تعذّر إنشاء مجلد الإخراج: {output_dir}"
        ) from e

    if not output_dir.is_dir():
        raise RuntimeError(f"مسار الإخراج الأب ليس مجلدًا: {output_dir}")

    # ============================================================
    # [قياس مرجعي] مدة الصوت (المرجع الزمني الوحيد)
    # ============================================================
    audio_duration = _probe_media_duration(audio_file)

    # ============================================================
    # [حساب الحدود] من بدايات الجمل الفعلية (الصورة 0 تبدأ عند 0.0)
    # ============================================================
    fps = 60
    durations, image_starts = _compute_image_boundaries(
        renderable_timeline, audio_duration, fps
    )

    if len(durations) != len(validated_pairs):
        raise RuntimeError(
            f"عدم تطابق داخلي: عدد المدد المحسوبة ({len(durations)}) != عدد "
            f"الأزواج ({len(validated_pairs)})."
        )

    sum_durations = sum(durations)
    logger.info(
        f"📐 عدد الصور: {len(durations)} | "
        f"مجموع المدد المحسوبة: {sum_durations:.3f}s | "
        f"مدة الصوت: {audio_duration:.3f}s | "
        f"عينة المدد الأولى: {[round(d, 3) for d in durations[:6]]}"
    )

    # ============================================================
    # [حساب نقاط الانتقال الحقيقية] = الحدود المطلقة الفعلية للقطع
    # (بعد ضبط المدد لإطار واحد كحد أدنى، لضمان تطابق القطع مع نقاط
    # الـ Whoosh الفعلية في الفيديو المُنتَج)
    # ============================================================
    transition_times: List[float] = list(image_starts[1:])

    # ============================================================
    # [حساب نبضات الإيقاع] فقط للأدوار المسموح بها
    # ============================================================
    rhythm_trigger_times: List[float] = []
    for i, item in enumerate(renderable_timeline):
        role_raw = item.get("narrative_role")
        role = str(role_raw).lower().strip() if role_raw is not None else ""
        if role in RHYTHM_ENABLED_ROLES:
            rhythm_trigger_times.append(image_starts[i])

    # ============================================================
    # [مساحة عمل مؤقتة فريدة لكل استدعاء (uuid)]
    # ============================================================
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

    success = False
    try:
        create_synthetic_whoosh(sfx_whoosh)
        if not sfx_whoosh.exists() or sfx_whoosh.stat().st_size <= 0:
            raise RuntimeError(
                f"فشل إنشاء ملف الـ whoosh الأساسي: {sfx_whoosh}"
            )

        segment_files: List[Path] = []

        logger.info(
            f"🎬 بدء جلسة الرندرة (run_id={run_id}) لعدد "
            f"{len(validated_pairs)} مقطع Ken Burns..."
        )

        # ============================================================
        # 1. رندرة مقطع مستقل لكل لقطة
        # ============================================================
        for idx, ((frame_path, _item), duration) in enumerate(
            zip(validated_pairs, durations)
        ):
            kb_filter = get_ken_burns_filter(idx, duration, fps=fps)

            seg_output = temp_dir / f"seg_{idx:03d}.mp4"
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(frame_path),
                "-vf", kb_filter,
                "-t", str(duration),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
                "-an",
                str(seg_output),
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"فشل إنشاء المقطع رقم {idx} (stage=segment_render): "
                    f"exit={e.returncode}"
                ) from e

            _validate_non_empty_file(seg_output, f"المقطع رقم {idx}")
            segment_files.append(seg_output)

        # ============================================================
        # 2. قائمة تجميع المقاطع
        # ============================================================
        concat_list_file = temp_dir / "concat_list.txt"
        try:
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for seg in segment_files:
                    f.write(_safe_concat_escape(seg))
        except OSError as e:
            raise RuntimeError(
                f"تعذّر كتابة ملف قائمة الدمج: {concat_list_file}"
            ) from e

        if not concat_list_file.exists() or concat_list_file.stat().st_size <= 0:
            raise RuntimeError(
                f"ملف قائمة الدمج فارغ أو غير موجود: {concat_list_file}"
            )

        # ============================================================
        # [المرحلة A] دمج الفيديو فقط (بدون صوت)
        # ============================================================
        logger.info("🧩 [A] دمج مقاطع الفيديو (فيديو-فقط) باستخدام stream copy...")

        concat_video_only = temp_dir / "concat_video_only.mp4"
        cmd_concat_only = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list_file),
            "-c:v", "copy",
            "-an",
            str(concat_video_only),
        ]
        try:
            subprocess.run(cmd_concat_only, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"فشل دمج المقاطع (stage=concat_video_only): exit={e.returncode}"
            ) from e

        _validate_non_empty_file(concat_video_only, "الفيديو المُجمَّع (فيديو-فقط)")

        # ============================================================
        # [المرحلة B] قياس المدة الحقيقية + tpad + mux مع التعليق
        # ============================================================
        real_video_duration = _probe_media_duration(concat_video_only)
        pad_needed = max(0.0, audio_duration - real_video_duration)

        logger.info(
            f"🎧 مدة التعليق الصوتي المرجعية: {audio_duration:.3f}s | "
            f"🎞️ المدة الحقيقية للفيديو المُجمَّع: {real_video_duration:.3f}s | "
            f"🩹 الحشو المطلوب (tpad clone): {pad_needed:.3f}s"
        )

        unsubbed_video = temp_dir / "unsubbed_assembled.mp4"
        logger.info(
            "🎞️ [B] تطبيق tpad clone ودمج التعليق الصوتي (الصوت هو مرجع المدة)..."
        )

        tpad_filter = (
            f"[0:v]tpad=stop_mode=clone:stop_duration={pad_needed:.3f}[vpad]"
        )

        cmd_concat = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(concat_video_only),
            "-i", str(audio_file),
            "-filter_complex", tpad_filter,
            "-map", "[vpad]",
            "-map", "1:a",
            "-t", f"{audio_duration:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k",
            str(unsubbed_video),
        ]
        try:
            subprocess.run(cmd_concat, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"فشل دمج الفيديو مع الصوت (stage=pad_and_mux): exit={e.returncode}"
            ) from e

        _validate_non_empty_file(unsubbed_video, "الفيديو الوسيط (بدون ترجمة)")

        # ============================================================
        # 3. توليد مسار whoosh موحد بطول audio_duration
        #    نقاط الانتقال = الحدود الفعلية للقطع (image_starts[1:])
        # ============================================================
        build_whoosh_timeline(
            sfx_whoosh,
            transition_times,
            audio_duration,
            whoosh_timeline,
            whoosh_volume=0.4,
        )
        _validate_non_empty_file(whoosh_timeline, "مسار الـ whoosh الموحد")

        # ============================================================
        # 4. Ambience داخلي اختياري
        # ============================================================
        has_ambience = False
        try:
            has_ambience = _try_generate_ambience(ambience_file, audio_duration)
            if has_ambience:
                logger.info(
                    f"🌫️ تم توليد طبقة Ambience داخلية خافتة "
                    f"({audio_duration:.3f}s)."
                )
            else:
                logger.info("🌫️ لم يتم توليد طبقة Ambience؛ المتابعة بدونها.")
        except Exception as amb_err:
            logger.warning(
                f"تعذّر توليد Ambience ({amb_err}); المتابعة بدونها."
            )
            has_ambience = False
            try:
                if ambience_file.exists():
                    ambience_file.unlink()
            except OSError:
                pass

        # ============================================================
        # 5. إيقاع وثائقي خافت اختياري (Minimal Cinematic Rhythm)
        #    - فقط في الأدوار المسموح بها.
        #    - يُتجاهل إن لم توجد أي نبضة مسموحة.
        #    - فشله لا يمنع الرندر.
        # ============================================================
        has_rhythm = False
        if rhythm_trigger_times:
            try:
                has_rhythm = _build_rhythm_timeline(
                    rhythm_trigger_times,
                    audio_duration,
                    44100,
                    rhythm_file,
                    rhythm_volume=0.35,
                )
                if has_rhythm:
                    logger.info(
                        f"🎵 تم توليد الإيقاع الوثائقي الخافت "
                        f"({len(rhythm_trigger_times)} نبضة)."
                    )
                else:
                    logger.info("🎵 لم يتم توليد الإيقاع؛ المتابعة بدونه.")
            except Exception as r_err:
                logger.warning(
                    f"تعذّر توليد الإيقاع ({r_err}); المتابعة بدونه."
                )
                has_rhythm = False
                try:
                    if rhythm_file.exists():
                        rhythm_file.unlink()
                except OSError:
                    pass
        else:
            logger.info("🎵 لا توجد أدوار مسموحة بالإيقاع؛ المتابعة بدونه.")

        # ============================================================
        # 6. الميكس النهائي + حرق الترجمة داخل filter_complex موحد
        #    - Voice هو المدخل 0 (المرجع).
        #    - Whoosh (1) + Ambience (اختياري) + Rhythm (اختياري)
        #    - كل الطبقات الإضافية تمر عبر sidechaincompress بمفتاح Voice
        #      لتقليل تغطية الكلام.
        #    - alimiter نهائي يمنع الـ clipping.
        # ============================================================
        logger.info(
            "✨ حرق الترجمة الحركية وميكس الصوت النهائي "
            "(Voice > Whoosh > Rhythm > Ambience) مع ducking كامل..."
        )

        ass_escaped = (
            subtitles_ass.resolve().as_posix().replace(":", "\\:")
        )

        # تجميع المدخلات بحسب الطبقات المتاحة
        final_inputs = ["-i", str(unsubbed_video), "-i", str(whoosh_timeline)]
        next_input_idx = 2
        ambience_input_idx = None
        rhythm_input_idx = None
        if has_ambience:
            final_inputs += ["-i", str(ambience_file)]
            ambience_input_idx = next_input_idx
            next_input_idx += 1
        if has_rhythm:
            final_inputs += ["-i", str(rhythm_file)]
            rhythm_input_idx = next_input_idx
            next_input_idx += 1

        # الطبقات الإضافية وترتيبها في amix
        extra_tracks = []  # (label, input_idx, weight, thr, ratio, att, rel)
        extra_tracks.append(("whoosh", 1, "0.55", "0.03", "5", "10", "250"))
        if has_rhythm and rhythm_input_idx is not None:
            extra_tracks.append(
                ("rhythm", rhythm_input_idx, "0.45", "0.03", "5", "10", "300")
            )
        if has_ambience and ambience_input_idx is not None:
            extra_tracks.append(
                ("ambience", ambience_input_idx, "0.55", "0.02", "6", "20", "400")
            )

        # بناء سلسلة الفلتر
        n_extra = len(extra_tracks)
        n_voice_splits = 1 + n_extra

        filter_parts = []
        filter_parts.append(f"[0:v]ass='{ass_escaped}'[vout]")

        voice_split_labels = "".join(f"[vsc{i}]" for i in range(n_extra))
        filter_parts.append(
            f"[0:a]asplit={n_voice_splits}[vmain]{voice_split_labels}"
        )

        amix_inputs = ["[vmain]"]
        amix_weights = ["1"]
        for i, (name, in_idx, weight, thr, ratio, att, rel) in enumerate(extra_tracks):
            duck_label = f"[{name}_d]"
            filter_parts.append(
                f"[{in_idx}:a][vsc{i}]sidechaincompress="
                f"threshold={thr}:ratio={ratio}:"
                f"attack={att}:release={rel}{duck_label}"
            )
            amix_inputs.append(duck_label)
            amix_weights.append(weight)

        n_amix = len(amix_inputs)
        weights_str = " ".join(amix_weights)
        filter_parts.append(
            f"{''.join(amix_inputs)}amix=inputs={n_amix}:duration=first:"
            f"dropout_transition=0:weights={weights_str}:normalize=0,"
            f"alimiter=limit=0.95:level=disabled[aout]"
        )
        filter_complex = ";".join(filter_parts)

        cmd_final = [
            "ffmpeg", "-y", "-v", "info",
            *final_inputs,
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k",
            str(staged_output_video),
        ]

        try:
            subprocess.run(cmd_final, check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(
                f"⚠️ فشل التصدير النهائي الكامل (exit={e.returncode}). "
                f"سيتم المتابعة بفلتر مبسط: Voice + الترجمة فقط."
            )
            simple_filter = f"[0:v]ass='{ass_escaped}'[vout]"
            cmd_final_simple = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(unsubbed_video),
                "-filter_complex", simple_filter,
                "-map", "[vout]",
                "-map", "0:a",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac", "-b:a", "192k",
                str(staged_output_video),
            ]
            try:
                subprocess.run(cmd_final_simple, check=True)
            except subprocess.CalledProcessError as e2:
                raise RuntimeError(
                    f"فشل التصدير النهائي حتى مع الفلتر المبسط: "
                    f"exit={e2.returncode}"
                ) from e2

        # ============================================================
        # التحقق من الإخراج المرحلي قبل نقله إلى المسار النهائي
        # ============================================================
        _validate_non_empty_file(staged_output_video, "الفيديو النهائي المرحلي")
        final_duration = _probe_media_duration(staged_output_video)
        if final_duration <= 0:
            raise RuntimeError(
                f"مدة الفيديو النهائي المرحلي غير موجبة: {final_duration}"
            )

        # ============================================================
        # النقل الذري إلى المسار النهائي بعد نجاح كل التحققات.
        # ============================================================
        try:
            staged_output_video.replace(output_video_path)
        except OSError as e:
            raise RuntimeError(
                f"تعذّر نقل الفيديو النهائي المرحلي إلى المسار النهائي: "
                f"{staged_output_video} -> {output_video_path}"
            ) from e

        logger.info(
            f"🏆 تم تصدير الفيديو النهائي بنجاح بأعلى دقة: {output_video_path} "
            f"(المدة النهائية = {final_duration:.3f}s | "
            f"مرجع الصوت = {audio_duration:.3f}s)"
        )

        success = True

    finally:
        try:
            if temp_dir.exists() and temp_dir.is_dir():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_err:
            logger.warning(
                f"تعذّر تنظيف المجلد المؤقت {temp_dir}: {cleanup_err}"
            )

    if not success:
        raise RuntimeError("فشلت الرندرة دون استثناء صريح (حالة غير متوقعة).")


# ============================================================
# اختبارات الإلزامية (تُشغَّل فقط عند التنفيذ المباشر للملف)
# ============================================================
if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO)

    # ------------------------------------------------------------
    # 1. اختبار _compute_image_durations بالحالة المطلوبة
    # ------------------------------------------------------------
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
    transitions = list(starts_abs[1:])
    assert all(abs(a - b) < 1e-6 for a, b in zip(transitions, [1.40, 2.00])), \
        f"transitions mismatch: {transitions}"
    print("[OK] test #1: _compute_image_durations = [1.4, 0.6, 1.0]")

    # ------------------------------------------------------------
    # 2. اختبار مع جمل فارغة
    # ------------------------------------------------------------
    timeline_with_empty = [
        {"start": 0.10, "text": "a", "empty": False},
        {"start": 0.90, "text": "", "empty": True},
        {"start": 1.50, "text": "c", "empty": False},
    ]
    renderable = [it for it in timeline_with_empty if not it.get("empty")]
    durs2, _ = _compute_image_boundaries(renderable, 2.50, 60)
    assert abs(sum(durs2) - 2.50) < 1e-6, f"sum2 mismatch: {sum(durs2)}"
    print("[OK] test #2: empty items filtered, sum == audio_duration")

    # ------------------------------------------------------------
    # 3. اختبار غياب start → الخطة الاحتياطية
    # ------------------------------------------------------------
    timeline_no_start = [
        {"duration": 0.5, "text": "a"},
        {"duration": 0.7, "text": "b"},
    ]
    durs3, starts3 = _compute_image_boundaries(timeline_no_start, 1.2, 60)
    assert all(abs(a - b) < 1e-6 for a, b in zip(durs3, [0.5, 0.7])), durs3
    assert all(abs(a - b) < 1e-6 for a, b in zip(starts3, [0.0, 0.5])), starts3
    print("[OK] test #3: fallback durations path")

    # ------------------------------------------------------------
    # 4. اختبار بناء الإيقاع (نجاح متوقع)
    # ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        rhythm_out = td_path / "rhythm.wav"
        ok = _build_rhythm_timeline(
            [0.5, 1.5], 3.0, 44100, rhythm_out, rhythm_volume=0.35
        )
        assert ok, "rhythm generation should succeed"
        assert rhythm_out.exists() and rhythm_out.stat().st_size > 0
        print("[OK] test #4: rhythm timeline generation")

        # --------------------------------------------------------
        # 5. اختبار فشل الإيقاع (قائمة فارغة)
        # --------------------------------------------------------
        empty_rhythm = td_path / "rhythm_empty.wav"
        ok_empty = _build_rhythm_timeline(
            [], 3.0, 44100, empty_rhythm, rhythm_volume=0.35
        )
        assert not ok_empty, "empty trigger list should return False"
        print("[OK] test #5: rhythm fails gracefully on empty triggers")

    # ------------------------------------------------------------
    # 6. اختبار بناء Whoosh (نجاح متوقع)
    # ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        base = td_path / "base_whoosh.wav"
        create_synthetic_whoosh(base)
        assert base.exists() and base.stat().st_size > 0

        whoosh_out = td_path / "whoosh_timeline.wav"
        build_whoosh_timeline(base, [1.4, 2.0], 3.0, whoosh_out, whoosh_volume=0.4)
        assert whoosh_out.exists() and whoosh_out.stat().st_size > 0
        print("[OK] test #6: whoosh timeline generation")

    print("[ALL TESTS PASSED]")
