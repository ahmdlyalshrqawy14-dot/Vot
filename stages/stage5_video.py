import os
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel

def get_audio_duration(audio_path: Path) -> float:
    """استخراج مدة الصوت الكلية بدقة"""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def format_ass_time(seconds: float) -> str:
    """تحويل الثواني لصيغة توقيت ملف الترجمة ASS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def generate_subtitles(segments, ass_path: Path):
    """توليد نصوص متحركة بستايل نيون جذاب وظل أسود"""
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Kinetic,DejaVu Sans,50,&H0000FFFF,&H000000FF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,1,4,2,2,30,30,85,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for seg in segments:
            words = list(seg.words) if seg.words else []
            if not words:
                start = format_ass_time(seg.start)
                end = format_ass_time(seg.end)
                f.write(f"Dialogue: 0,{start},{end},Kinetic,,0,0,0,,{seg.text.strip().upper()}\n")
                continue

            # عرض الكلمات بمجموعات سريعة (3-4 كلمات)
            chunk_size = 3
            for i in range(0, len(words), chunk_size):
                chunk = words[i:i + chunk_size]
                c_start = format_ass_time(chunk[0].start)
                c_end = format_ass_time(chunk[-1].end)
                c_text = " ".join([w.word.strip().upper() for w in chunk])
                f.write(f"Dialogue: 0,{c_start},{c_end},Kinetic,,0,0,0,,{c_text}\n")

def run_stage_5(episode_dir: Path):
    print(f"\n--- [Stage 5] Starting Video Assembly Engine ---")
    audio_file = episode_dir / "voiceover.wav"
    images_dir = episode_dir / "images"
    output_video = episode_dir / "final_video.mp4"
    ass_file = episode_dir / "subtitles.ass"
    concat_file = episode_dir / "images_concat.txt"

    if not audio_file.exists():
        raise FileNotFoundError(f"Missing voiceover: {audio_file}")

    image_files = sorted(list(images_dir.glob("*.png")))
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    total_duration = get_audio_duration(audio_file)
    print(f"[VIDEO] Duration: {total_duration:.2f}s | Images: {len(image_files)}")

    # تفريغ الصوت واستخراج توقيت الكلمات
    print("[VIDEO] Extracting precision timestamps via faster-whisper...")
    whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    segments_gen, _ = whisper.transcribe(str(audio_file), word_timestamps=True)
    segments = list(segments_gen)

    # توليد ملف الترجمة الحركية
    generate_subtitles(segments, ass_file)

    # توزيع فترات عرض الصور بالتزامن مع الصوت
    durations = []
    num_imgs = len(image_files)
    num_segs = len(segments)

    if num_segs >= num_imgs:
        ratio = num_segs / num_imgs
        for i in range(num_imgs):
            start_idx = int(i * ratio)
            end_idx = int((i + 1) * ratio) if i < num_imgs - 1 else num_segs
            durations.append(max(1.0, segments[end_idx - 1].end - segments[start_idx].start))
    else:
        durations = [total_duration / num_imgs] * num_imgs

    # وزن التوقيت ليتطابق مع الصوت 100%
    scale = total_duration / sum(durations)
    durations = [d * scale for d in durations]

    # تجهيز قائمة صور FFmpeg
    with open(concat_file, "w", encoding="utf-8") as f:
        for img, dur in zip(image_files, durations):
            f.write(f"file '{img.name}'\n")
            f.write(f"duration {dur:.3f}\n")
        f.write(f"file '{image_files[-1].name}'\n")

    # الرندرة بـ FFmpeg: شاشة 1920x1080 بخلفية بيضاء + حرق الترجمة النيون + الصوت
    print("[VIDEO] Rendering 1080p Full HD video with FFmpeg...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", "images_concat.txt",
        "-i", "voiceover.wav",
        "-filter_complex",
        "[0:v]scale=1024:1024,pad=1920:1080:(1920-1024)/2:(1080-1024)/2:white,format=yuv420p,ass=subtitles.ass[v]",
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "final_video.mp4"
    ]

    res = subprocess.run(ffmpeg_cmd, cwd=str(episode_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[ERROR] FFmpeg failed: {res.stderr}")
        raise RuntimeError("Video rendering failed.")

    print(f"--- [Stage 5] Video successfully rendered: {output_video} ---")
    return output_video
