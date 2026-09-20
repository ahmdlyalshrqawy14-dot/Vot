import shutil
import subprocess
from pathlib import Path
from config.settings import VIDEO_WIDTH, VIDEO_HEIGHT, FPS

def get_audio_duration(file_path: Path) -> float:
    """استخراج مدة ملف الصوت بدقة بالثواني"""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(res.stdout.strip())

def format_ass_time(sec: float) -> str:
    """تحويل التوقيت إلى صيغة ملف الترجمة ASS"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def generate_subtitles(beats: list, durations: list, ass_path: Path):
    """
    توليد نصوص ترجمة واضحة ومقروءة بأسلوب حديث
    مع خلفية مظللة شبه شفافة تمنع تداخل النص مع الصور
    """
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {VIDEO_WIDTH}
PlayResY: {VIDEO_HEIGHT}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ModernSub,DejaVu Sans,48,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,8,0,2,60,60,75,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    curr_time = 0.0
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for beat, dur in zip(beats, durations):
            start = format_ass_time(curr_time)
            end = format_ass_time(curr_time + dur)
            text = beat["spoken_text"].strip().replace("\n", " ")
            f.write(f"Dialogue: 0,{start},{end},ModernSub,,0,0,0,,{text}\n")
            curr_time += dur

def run_stage_5(script_data: dict, episode_dir: Path) -> Path:
    """
    محرك المونتاج:
    - فحص الكادرات وسد أي نقص تلقائياً من الإطار السابق
    - دمج المقاطع الصوتية في تراك صوتي نقي بدون موسيقى
    - تحجيم الصور وضبطها على 1080p وحرق الترجمة
    """
    print(f"\n--- [Stage 5] Starting Voice-Only Montage Engine ---")
    audio_dir = episode_dir / "audio_beats"
    images_dir = episode_dir / "images"
    final_video = episode_dir / "final_video.mp4"
    ass_file = episode_dir / "subtitles.ass"

    beats = script_data.get("beats", [])

    # 1. نظام التكيف: سد أي كادر مفقود تلقائياً لضمان عدم توقف المونتاج
    last_valid_img = None
    for b in beats:
        b_id = b["id"]
        img_path = images_dir / f"beat_{b_id:03d}.png"
        if img_path.exists():
            last_valid_img = img_path
        elif last_valid_img is not None:
            print(f"[Adaptive Engine] Missing beat_{b_id:03d}.png -> Filled using previous frame.")
            shutil.copy(last_valid_img, img_path)

    # 2. حساب مدد الصوت والتزامن المطلق
    durations = []
    valid_beats = []
    for b in beats:
        b_id = b["id"]
        a_path = audio_dir / f"beat_{b_id:03d}.wav"
        i_path = images_dir / f"beat_{b_id:03d}.png"

        if a_path.exists() and i_path.exists():
            dur = get_audio_duration(a_path)
            durations.append(dur)
            valid_beats.append(b)

    if not valid_beats:
        raise RuntimeError("No valid audio/image pairs found for rendering.")

    # 3. توليد ملف الترجمة
    generate_subtitles(valid_beats, durations, ass_file)

    # 4. دمج المقاطع الصوتية في ملف رئيسي واحد (صوت المعلق الصافي)
    audio_concat_file = episode_dir / "audio_concat.txt"
    with open(audio_concat_file, "w", encoding="utf-8") as f:
        for b in valid_beats:
            f.write(f"file 'audio_beats/beat_{b['id']:03d}.wav'\n")

    master_voice = episode_dir / "master_voice.wav"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", "audio_concat.txt", "-c", "copy", "master_voice.wav"
    ], cwd=str(episode_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 5. تجهيز قائمة الصور مع أزمنة عرضها المطابقة للصوت بالمللي ثانية
    images_concat_file = episode_dir / "images_concat.txt"
    with open(images_concat_file, "w", encoding="utf-8") as f:
        for b, dur in zip(valid_beats, durations):
            f.write(f"file 'images/beat_{b['id']:03d}.png'\n")
            f.write(f"duration {dur:.3f}\n")
        f.write(f"file 'images/beat_{valid_beats[-1]['id']:03d}.png'\n")

    total_len = sum(durations)
    print(f"[VIDEO] Total duration: {total_len:.2f}s ({total_len/60:.2f} mins).")

    # 6. الرندرة بـ FFmpeg: كلام صافي + جودة 1080p + حرق الترجمة (بدون أي موسيقى خلفية)
    v_filter = (
        f"[0:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT},format=yuv420p,ass=subtitles.ass[v]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", "images_concat.txt",
        "-i", "master_voice.wav",
        "-filter_complex", v_filter,
        "-map", "[v]",
        "-map", "1:a",
        "-c:v", "libx264",
        "-preset", "faster",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", str(total_len),
        "final_video.mp4"
    ]

    res = subprocess.run(cmd, cwd=str(episode_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {res.stderr[-300:]}")

    print(f"--- [Stage 5] Clean Video rendered successfully: {final_video} ---")
    return final_video
