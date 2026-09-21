import os
import wave
import struct
import math
import subprocess
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
            # مغلف انسيابي (Fade in / Fade out)
            envelope = math.sin(math.pi * t / duration) ** 2
            # تردد منخفض متدرج هادئ (150Hz إلى 300Hz)
            freq = 150 + 150 * (t / duration)
            sample = envelope * math.sin(2 * math.pi * freq * t)
            # خفض شدة الصوت إلى -18dB (حوالي 0.12 من المدى الأقصى)
            val = int(sample * 32767 * 0.12)
            wav_file.writeframes(struct.pack("<h", val))


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
        # الصورة 1: تقريب تدريجي بطيء نحو مركز الشخصية (Slow Zoom In)
        return f"zoompan=z='min(zoom+0.0008,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 1:
        # الصورة 2: تحريك أفقي ناعم لليمين (Slow Pan Right)
        return f"zoompan=z='1.1':x='if(lte(on,1),0,x+1.2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 2:
        # الصورة 3: ابتعاد تدريجي بطيء (Slow Zoom Out)
        return f"zoompan=z='if(lte(on,1),1.15,max(1.0,zoom-0.0008))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 3:
        # الصورة 4: تحريك عمودي ناعم للأعلى (Slow Pan Up)
        return f"zoompan=z='1.1':x='iw/2-(iw/zoom/2)':y='if(lte(on,1),ih*0.1,max(0,y-0.8))':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 4:
        # الصورة 5: تقريب تدريجي بطيء (Slow Zoom In)
        return f"zoompan=z='min(zoom+0.0007,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    else:
        # الصورة 6: تحريك أفقي ناعم لليسار (Slow Pan Left)
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

    # إنشاء ملف الـ whoosh الأساسي
    sfx_whoosh = output_video_path.parent / "whoosh_soft.wav"
    create_synthetic_whoosh(sfx_whoosh)

    segment_files = []
    fps = 60

    logger.info("🎬 جاري بناء المقاطع الحركية للقطات (Ken Burns & Vignette)...")

    # 1. رندرة مقطع فيديو مستقل لكل لقطة بحسب مدتها وحركتها الخاصة
    for idx, (frame_path, item) in enumerate(zip(frames, timeline)):
        duration = item["duration"]
        kb_filter = get_ken_burns_filter(idx, duration, fps=fps)
        # إضافة التظليل السينمائي الخفيف (12% Vignette)
        full_filter = f"{kb_filter},vignette=angle=PI/4:aspect=16/9:eval=init"

        seg_output = temp_dir / f"seg_{idx:03d}.mp4"
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-loop", "1", "-i", str(frame_path),
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
            # صياغة متوافقة مع مسارات ويندوز
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

    # 4. إضافة مؤثرات الـ Whooshes عند كل نقطة انتقال وحرق الترجمة الصفراء الباهتة
    logger.info("✨ حرق الترجمة الحركية الصفراء الباهتة وتطبيق مؤثرات الانتقال الصوتية...")

    # تجهيز مسار ملف الترجمة المتوافق مع ويندوز
    ass_escaped = subtitles_ass.resolve().as_posix().replace(":", "\\:")
    subtitle_filter = f"ass='{ass_escaped}'"

    # بناء فلتر دمج الـ whooshes مع الصوت الأصلي
    # نضيف الـ whoosh عند بداية كل مقطع (ما عدا الأول)
    total_segments = len(segment_files)
    
    # حساب أوقات الانتقالات من الـ timeline
    transition_times = []
    current_time = 0.0
    for item in timeline[:-1]:  # ما عندن آخر مقطع
        current_time += item["duration"]
        transition_times.append(current_time)
    
    # بناء فلتر معقد لدمج الـ whooshes
    # نستخدم adelay لتأخير كل whoosh لوقته المناسب
    audio_inputs = []
    filter_parts = []
    
    for idx, trans_time in enumerate(transition_times):
        delay_ms = int(trans_time * 1000)
        # كل whoosh يتأخر لوقت الانتقال
        filter_parts.append(
            f"[{idx+2}:a]adelay={delay_ms}|{delay_ms},volume=0.4[whoosh_{idx}]"
        )
    
    # دمج كل الـ whooshes مع الصوت الأصلي
    if filter_parts:
        # [0:a] هو الصوت الأصلي من unsubbed_video
        mix_inputs = "[0:a]"
        for idx in range(len(transition_times)):
            mix_inputs += f"[whoosh_{idx}]"
        
        filter_parts.append(
            f"{mix_inputs}amix=inputs={len(transition_times)+1}:duration=first:dropout_transition=0[aout]"
        )
        
        filter_complex = ";".join(filter_parts)
        
        cmd_final = [
            "ffmpeg", "-y", "-v", "info",
            "-i", str(unsubbed_video),
            "-i", str(sfx_whoosh),  # whoosh أساسي (سيتم تكراره بالتأخير)
            "-vf", subtitle_filter,
            "-filter_complex", filter_complex,
            "-map", "0:v",  # الفيديو من الملف الأول
            "-map", "[aout]",  # الصوت المدمج
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(output_video_path)
        ]
    else:
        # لو مفيش انتقالات (مقطع واحد بس)
        cmd_final = [
            "ffmpeg", "-y", "-v", "info",
            "-i", str(unsubbed_video),
            "-vf", subtitle_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy",
            str(output_video_path)
        ]

    subprocess.run(cmd_final, check=True)

    logger.info(f"🏆 تم تصدير الفيديو النهائي بنجاح بأعلى دقة: {output_video_path}")
