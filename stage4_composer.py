import os
import wave
import struct
import math
import subprocess
import array
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger("Stage4Composer")


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
    هذا يُغني عن تمرير عشرات المدخلات لـ FFmpeg ويختصر أمر التنفيذ النهائي.
    """
    with wave.open(str(base_whoosh), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    base_samples = array.array("h")
    base_samples.frombytes(raw)
    base_len = len(base_samples)

    total_samples = int(total_duration * sr)
    out = array.array("h", bytes(2 * total_samples))

    for t in transition_times:
        start = int(t * sr)
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
    frames = int(duration * fps)
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
    تُستخدم لجعل الصوت هو المرجع الزمني النهائي للمونتاج.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


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
    - خلو تام بنسبة 100% من الموسيقى الخلفية

    [سياسة المدة الجديدة - النسخة المُحصَّنة]
    - الصوت (التعليق الصوتي) هو المرجع الزمني الوحيد.
    - نقيس المدة الحقيقية للفيديو بعد الدمج الفعلي (ffprobe على الملف المُجمَّع)
      وليس عبر مجموع النظري للمقاطع، لتجنّب فروق الإطارات الناتجة عن تقريب
      الـ 60fps أثناء الترميز.
    - إذا كان الفيديو أقصر من الصوت: نُمدّد آخر إطار (tpad clone) بالفرق الحقيقي.
    - إذا كان الفيديو أطول: pad_needed=0 ويقتطع -t الزائد من الفيديو فقط.
    - لا نمسّ مسار التعليق الصوتي إطلاقًا.
    - حذف -shortest نهائيًا من خطوة الدمج.
    - amix duration=first آمن لأن المدخل الأول = التعليق بطول audio_duration.

    [البنية الجديدة]
    - المرحلة A: دمج المقاطع فيديو-فقط (بدون صوت) عبر -c:v copy.
    - المرحلة B: قياس المدة الحقيقية + tpad + mux مع التعليق الصوتي و-t audio_duration.
    """

    # ============================================================
    # [حماية 1] استبعاد العناصر الفارغة + التحقق من تناسق الأعداد
    # ============================================================
    renderable_timeline = [it for it in timeline if not it.get("empty")]

    if not renderable_timeline:
        raise ValueError(
            "قائمة التوقيت لا تحتوي على أي عنصر قابل للرندرة "
            "(كل العناصر فارغة أو القائمة فارغة أصلًا)."
        )

    # حالتان مسموحتان فقط:
    #   A) frames يطابق عدد العناصر غير الفارغة مباشرةً (الأكثر شيوعًا).
    #   B) frames يطابق الـ timeline الكامل (بما فيها placeholder للعناصر الفارغة).
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
    # [حماية 2 + 3] التحقق من المدة الموجبة ومن سلامة ملفات الصور
    # ============================================================
    validated_pairs: List[tuple] = []
    for idx, (frame_path, item) in enumerate(raw_pairs):
        duration = float(item.get("duration", 0.0) or 0.0)
        if duration <= 0:
            preview = str(item.get("text", ""))[:40]
            raise ValueError(
                f"مدة غير صالحة ({duration}) للعنصر رقم {idx} "
                f"(النص: {preview!r}). كل عنصر قابل للرندرة يجب أن تكون مدته > 0."
            )

        fp = Path(frame_path)
        if not fp.exists():
            raise FileNotFoundError(f"[{idx}] ملف الصورة غير موجود: {fp}")
        if not fp.is_file():
            raise ValueError(f"[{idx}] المسار ليس ملفًا عاديًا: {fp}")

        validated_pairs.append((fp, item, duration))

    # ============================================================
    # من هذه النقطة نعمل حصريًا على validated_pairs
    # ============================================================
    temp_dir = output_video_path.parent / "temp_segments"
    temp_dir.mkdir(parents=True, exist_ok=True)

    sfx_whoosh = output_video_path.parent / "whoosh_soft.wav"
    create_synthetic_whoosh(sfx_whoosh)

    segment_files = []
    fps = 60

    logger.info(f"🎬 جاري بناء {len(validated_pairs)} مقطع Ken Burns...")

    # 1. رندرة مقطع مستقل لكل لقطة
    for idx, (frame_path, _item, duration) in enumerate(validated_pairs):
        kb_filter = get_ken_burns_filter(idx, duration, fps=fps)

        seg_output = temp_dir / f"seg_{idx:03d}.mp4"
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(frame_path),
            "-vf", kb_filter,
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
            "-an",
            str(seg_output)
        ]
        subprocess.run(cmd, check=True)
        segment_files.append(seg_output)

    # 2. إنشاء قائمة تجميع المقاطع
    concat_list_file = temp_dir / "concat_list.txt"
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for seg in segment_files:
            f.write(f"file '{seg.resolve().as_posix()}'\n")

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
        str(concat_video_only)
    ]
    subprocess.run(cmd_concat_only, check=True)

    # ============================================================
    # [المرحلة B] قياس المدة الحقيقية + tpad + mux مع التعليق
    # ============================================================
    # الصوت هو المرجع الزمني الوحيد.
    audio_duration = _probe_media_duration(audio_file)
    real_video_duration = _probe_media_duration(concat_video_only)

    # الفرق الحقيقي المطلوب لحشو آخر إطار (clone) ليطابق الفيديو مدة الصوت.
    pad_needed = max(0.0, audio_duration - real_video_duration)

    logger.info(
        f"🎧 مدة التعليق الصوتي المرجعية: {audio_duration:.3f}s | "
        f"🎞️ المدة الحقيقية للفيديو المُجمَّع: {real_video_duration:.3f}s | "
        f"🩹 الحشو المطلوب (tpad clone): {pad_needed:.3f}s"
    )

    unsubbed_video = temp_dir / "unsubbed_assembled.mp4"
    logger.info("🎞️ [B] تطبيق tpad clone ودمج التعليق الصوتي (الصوت هو مرجع المدة)...")

    # ملاحظة: stop_duration=0 صالح في FFmpeg ولا يضيف أي إطار.
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
        str(unsubbed_video)
    ]
    subprocess.run(cmd_concat, check=True)

    # 4. توليد مسار whoosh موحد في الذاكرة.
    logger.info("✨ حرق الترجمة الحركية الصفراء الباهتة وتطبيق مؤثرات الانتقال الصوتية...")

    # بناء انتقالات whoosh من العناصر المُتحققة فقط (وليس من timeline الكامل)
    transition_times = []
    current_time = 0.0
    for _, _, duration in validated_pairs[:-1]:
        current_time += duration
        transition_times.append(current_time)

    # مسار الـ whoosh يُبنى بطول audio_duration ليطابق الفيديو النهائي تمامًا.
    whoosh_timeline = output_video_path.parent / "whoosh_timeline.wav"
    build_whoosh_timeline(
        sfx_whoosh,
        transition_times,
        audio_duration,
        whoosh_timeline,
        whoosh_volume=0.4,
    )

    # 5. دمج الترجمة وميكس الصوت داخل filter_complex واحد موحد.
    ass_escaped = subtitles_ass.resolve().as_posix().replace(":", "\\:")

    filter_complex = (
        f"[0:v]ass='{ass_escaped}'[vout];"
        f"[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0:weights=1 1[aout]"
    )

    cmd_final = [
        "ffmpeg", "-y", "-v", "info",
        "-i", str(unsubbed_video),
        "-i", str(whoosh_timeline),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        str(output_video_path)
    ]

    subprocess.run(cmd_final, check=True)

    logger.info(
        f"🏆 تم تصدير الفيديو النهائي بنجاح بأعلى دقة: {output_video_path} "
        f"(المدة النهائية = {audio_duration:.3f}s)"
    )
