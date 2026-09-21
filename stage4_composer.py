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
    - مؤثرات Ken Burns و Vignette
    - انتقالات صوتية Soft Whoosh عند كل قطع
    - خلو تام بنسبة 100% من الموسيقى الخلفية
    """
    temp_dir = output_video_path.parent / "temp_segments"
    temp_dir.mkdir(parents=True, exist_ok=True)

    sfx_whoosh = output_video_path.parent / "whoosh_soft.wav"
    create_synthetic_whoosh(sfx_whoosh)

    segment_files = []
    fps = 60

    logger.info("🎬 جاري بناء المقاطع الحركية للقطات (Ken Burns & Vignette)...")

    # 1. رندرة مقطع فيديو مستقل لكل لقطة بحسب مدتها وحركتها الخاصة
    #    [إصلاح 4]: حذف "-loop 1" لتفادي تقطيع zoompan واستهلاك الرام
    for idx, (frame_path, item) in enumerate(zip(frames, timeline)):
        duration = item["duration"]
        kb_filter = get_ken_burns_filter(idx, duration, fps=fps)
        full_filter = f"{kb_filter},vignette=angle=PI/4:aspect=16/9:eval=init"

        seg_output = temp_dir / f"seg_{idx:03d}.mp4"
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(frame_path),
            "-vf", full_filter,
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
            str(seg_output)
        ]
        subprocess.run(cmd, check=True)
        segment_files.append(seg_output)

    # 2. إنشاء قائمة تجميع المقاطع
    concat_list_file = temp_dir / "concat_list.txt"
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for seg in segment_files:
            f.write(f"file '{seg.resolve().as_posix()}'\n")

    unsubbed_video = temp_dir / "unsubbed_assembled.mp4"
    logger.info("🎞️ دمج مقاطع الفيديو مع المسار الصوتي الموحد...")

    # 3. دمج المقاطع مع ملف الصوت الموحد
    cmd_concat = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_list_file),
        "-i", str(audio_file),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(unsubbed_video)
    ]
    subprocess.run(cmd_concat, check=True)

    # 4. [إصلاح 1 + 3]: توليد مسار whoosh موحد في الذاكرة بدلاً من 80 مدخل FFmpeg
    logger.info("✨ حرق الترجمة الحركية الصفراء الباهتة وتطبيق مؤثرات الانتقال الصوتية...")

    transition_times = []
    current_time = 0.0
    for item in timeline[:-1]:
        current_time += item["duration"]
        transition_times.append(current_time)

    total_duration = sum(item["duration"] for item in timeline)

    whoosh_timeline = output_video_path.parent / "whoosh_timeline.wav"
    build_whoosh_timeline(
        sfx_whoosh,
        transition_times,
        total_duration,
        whoosh_timeline,
        whoosh_volume=0.4,
    )

    # 5. [إصلاح 5]: دمج الترجمة وميكس الصوت داخل filter_complex واحد موحد
    ass_escaped = subtitles_ass.resolve().as_posix().replace(":", "\\:")

    # [إصلاح 2]: مدخلان فقط + أوزان كاملة (1 1) لمنع كتم صوت المعلق
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

    logger.info(f"🏆 تم تصدير الفيديو النهائي بنجاح بأعلى دقة: {output_video_path}")
