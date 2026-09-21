import os
import re
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger("Stage4Subtitles")


def format_ass_time(seconds: float) -> str:
    """تحويل الثواني إلى توقيت ASS بصيغة H:MM:SS.CC"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    centis = int((secs - int(secs)) * 100)
    return f"{hrs}:{mins:02d}:{int(secs):02d}.{centis:02d}"


def get_audio_duration_wave(audio_path: Path) -> float:
    """استخراج مدة ملف الصوت بدقة"""
    import wave
    try:
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        # كصمام أمان في حال كان mp3 نستخدم ffmpeg عبر ffprobe
        import subprocess
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(res.stdout.strip())


def align_audio_and_generate_ass(
    audio_path: Path,
    sentences: List[str],
    output_ass_path: Path
) -> List[Dict[str, Any]]:
    """
    المزامنة الدقيقة بالمللي ثانية وإنشاء ملف الترجمة الحركية:
    - اللون: أصفر باهت خفيف (#FDE047 = &H47E0FD& بالـ BGR)
    - بدون شريط أو صندوق خلفي
    - إضاءة الكلمات بالتزامن مع النطق (Word-by-Word Highlight)
    """
    duration = get_audio_duration_wave(audio_path)
    logger.info(f"🎙️ إجمالي مدة الصوت: {duration:.2f} ثانية لعدد {len(sentences)} جملة.")

    # محاولة استخدام faster-whisper للحصول على أدق توقيت للكلمات بالمللي ثانية
    word_timings = []
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
        for seg in segments:
            for w in seg.words:
                word_timings.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    except Exception as e:
        logger.warning(f"⚠️ تعذر تشغيل Whisper ({e}). تفعيل المزامنة التناسبية الذكية المبنية على أطوال الجمل...")
        # خوارزمية صمام الأمان: توزيع الزمن تناسبياً بناءً على عدد حروف كل جملة
        total_chars = sum(len(s) for s in sentences)
        curr_t = 0.0
        for s in sentences:
            s_dur = (len(s) / total_chars) * duration
            words = s.split()
            w_dur = s_dur / max(len(words), 1)
            for w in words:
                word_timings.append({"word": w, "start": curr_t, "end": curr_t + w_dur})
                curr_t += w_dur

    # ترويسة ملف الترجمة ASS المضبوطة بدقة
    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,64,&H0047E0FD,&H0000FFFF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,2.0,1.5,2,80,80,95,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # تقسيم الطوابع الزمنية للجمل
    sentence_timeline = []
    w_idx = 0
    total_words_stream = len(word_timings)

    ass_events = []

    for s_idx, sentence in enumerate(sentences):
        s_words = sentence.split()
        if not s_words:
            continue

        start_word = word_timings[min(w_idx, total_words_stream - 1)] if total_words_stream > 0 else {"start": 0}
        sent_start = start_word.get("start", 0.0)

        end_w_idx = min(w_idx + len(s_words) - 1, total_words_stream - 1)
        end_word = word_timings[end_w_idx] if total_words_stream > 0 else {"end": sent_start + 2.0}
        sent_end = max(end_word.get("end", sent_start + 1.5), sent_start + 0.8)

        # بناء وسم الكاريوكي الحركي {\k duration_in_centiseconds} لإضاءة الكلمات
        karaoke_text = ""
        for i in range(len(s_words)):
            curr_pos = min(w_idx + i, total_words_stream - 1)
            w_info = word_timings[curr_pos] if total_words_stream > 0 else {}
            w_start = w_info.get("start", sent_start)
            w_end = w_info.get("end", w_start + 0.3)
            w_duration_cs = max(int((w_end - w_start) * 100), 10)
            karaoke_text += f"{{\\k{w_duration_cs}}}{s_words[i]} "

        start_str = format_ass_time(sent_start)
        end_str = format_ass_time(sent_end)

        event_line = f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{karaoke_text.strip()}"
        ass_events.append(event_line)

        sentence_timeline.append({
            "index": s_idx + 1,
            "text": sentence,
            "start": sent_start,
            "end": sent_end,
            "duration": sent_end - sent_start
        })

        w_idx += len(s_words)

    # حفظ ملف الـ .ass
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "\n".join(ass_events) + "\n")

    logger.info(f"✅ تم توليد ملف الترجمة الحركية بنجاح: {output_ass_path}")
    return sentence_timeline
