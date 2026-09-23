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


def _build_proportional_timings(sentences: List[str], total_duration: float) -> List[Dict[str, Any]]:
    """
    خوارزمية الحل الاحتياطي: توزيع الزمن تناسبياً بناءً على عدد حروف كل جملة.
    تُستخدم عند فشل Whisper أو عند إرجاعه صفر كلمات.
    """
    non_empty = [s for s in sentences if s and s.strip()]
    total_chars = sum(len(s) for s in non_empty)
    if total_chars <= 0 or total_duration <= 0:
        return []
    timings: List[Dict[str, Any]] = []
    curr_t = 0.0
    for s in sentences:
        if not s or not s.strip():
            continue
        s_dur = (len(s) / total_chars) * total_duration
        words = s.split()
        w_dur = s_dur / max(len(words), 1)
        for w in words:
            timings.append({"word": w, "start": curr_t, "end": curr_t + w_dur})
            curr_t += w_dur
    return timings


def _validate_and_align_word_timings(
    sentences: List[str],
    word_timings: List[Dict[str, Any]],
    total_duration: float,
) -> List[Dict[str, Any]]:
    """
    يوائم بين كلمات النص الأصلي وكلمات Whisper بأمان.

    الحالات:
    1) Whisper أعاد عدد كلمات قريب من النص (±15%): نأخذ توقيتاته ونوزع الفرق.
    2) Whisper أعاد عدداً مختلفاً كثيراً: نرفضه ونستخدم المزامنة التناسبية.
    3) Whisper أعاد كلمات أكثر من النص: نقتصر على أول n_text كلمة.
    """
    # استخراج كل كلمات النص الأصلي كقائمة مسطحة
    text_words: List[str] = []
    for s in sentences:
        if s and s.strip():
            text_words.extend(s.split())

    n_text = len(text_words)
    if n_text == 0:
        return []

    n_whisper = len(word_timings)
    if n_whisper == 0:
        return _build_proportional_timings(sentences, total_duration)

    ratio = n_whisper / n_text

    # --- الحالة 2: اختلاف كبير ⇒ رفض Whisper كلياً ---
    if ratio < 0.85 or ratio > 1.15:
        logger.warning(
            f"⚠️ عدم تطابق كبير بين Whisper ({n_whisper}) "
            f"والنص ({n_text}) بنسبة {ratio:.2f}. "
            f"التحوّل إلى المزامنة التناسبية."
        )
        return _build_proportional_timings(sentences, total_duration)

    # --- الحالة 3: Whisper أكثر ⇒ نقتصر على النص ---
    if n_whisper >= n_text:
        logger.info(f"ℹ️ Whisper أعاد {n_whisper} كلمة، النص {n_text}. اقتطاع الفائض.")
        return [
            {
                "word": text_words[i],
                "start": float(word_timings[i]["start"]),
                "end": float(word_timings[i]["end"]),
            }
            for i in range(n_text)
        ]

    # --- الحالة 1: Whisper أقل ⇒ نوزّع الزمن المتبقي بعد آخر كلمة موثوقة ---
    logger.warning(
        f"⚠️ Whisper أعاد {n_whisper} كلمة مقابل {n_text} في النص. "
        f"توزيع {n_text - n_whisper} كلمة على الزمن المتبقي."
    )

    last_end = float(word_timings[-1]["end"])
    last_start = float(word_timings[-1]["start"])

    # نضمن أن لدينا وقتاً كافياً حتى لا نقع في per_word = 0
    remaining_time = max(total_duration - last_end, 0.0)
    remaining_words = n_text - n_whisper

    if remaining_time < 0.05 * remaining_words:
        # لا يوجد وقت كافٍ فعلي (قد يكون Whisper تجاوز المدة)
        # نمدّد بمعدل معقول: 0.35 ثانية/كلمة
        per_word = 0.35
        base_end = max(last_end, last_start + 0.1)
    else:
        per_word = remaining_time / remaining_words
        base_end = last_end

    aligned: List[Dict[str, Any]] = []

    # الكلمات المطابقة من Whisper
    for i in range(n_whisper):
        aligned.append({
            "word": text_words[i],
            "start": float(word_timings[i]["start"]),
            "end": float(word_timings[i]["end"]),
        })

    # الكلمات المتبقية بتوزيع تناسبي على الوقت المتبقي
    curr_t = base_end
    for i in range(n_whisper, n_text):
        aligned.append({
            "word": text_words[i],
            "start": curr_t,
            "end": curr_t + per_word,
        })
        curr_t += per_word

    return aligned


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
    # --- تحقق مبكر من المدخلات ---
    if not sentences:
        raise ValueError("قائمة الجمل فارغة.")

    valid_sentences = [s for s in sentences if s and s.strip()]
    if not valid_sentences:
        raise ValueError("كل الجمل فارغة، لا يوجد نص صالح للمزامنة.")

    duration = get_audio_duration_wave(audio_path)
    logger.info(f"🎙️ إجمالي مدة الصوت: {duration:.2f} ثانية لعدد {len(sentences)} جملة.")

    # --- محاولة استخدام faster-whisper للحصول على أدق توقيت للكلمات ---
    word_timings: List[Dict[str, Any]] = []
    whisper_ok = False
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
        for seg in segments:
            for w in seg.words:
                if w.word and w.word.strip():
                    word_timings.append({
                        "word": w.word.strip(),
                        "start": float(w.start),
                        "end": float(w.end),
                    })
        whisper_ok = len(word_timings) > 0
    except Exception as e:
        logger.warning(f"⚠️ تعذر تشغيل Whisper ({e}).")

    # --- تفعيل الحل الاحتياطي عند فشل Whisper أو إرجاعه صفر كلمات ---
    if not whisper_ok:
        logger.warning("⚠️ تفعيل المزامنة التناسبية الذكية المبنية على أطوال الجمل...")
        word_timings = _build_proportional_timings(sentences, duration)
    else:
        # ✅ الحماية الجديدة: مطابقة عدد كلمات Whisper مع النص قبل بناء الأحداث
        word_timings = _validate_and_align_word_timings(
            sentences, word_timings, duration
        )

    # --- صمام أمان أخير: لا يمكن إنتاج أي توقيت ---
    if not word_timings:
        logger.error("❌ لا يمكن إنتاج أي توقيت: لا Whisper ولا الخوارزمية التناسبية أنتجت بيانات.")
        raise RuntimeError("فشل توليد توقيتات الكلمات (word_timings فارغ).")

    # --- ترويسة ملف الترجمة ASS المضبوطة بدقة ---
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

    # --- تقسيم الطوابع الزمنية للجمل ---
    sentence_timeline: List[Dict[str, Any]] = []
    w_idx = 0
    total_words_stream = len(word_timings)

    ass_events: List[str] = []

    for s_idx, sentence in enumerate(sentences):
        s_words = (sentence or "").split()
        if not s_words:
            # نضيف مدخلاً فارغاً للحفاظ على تطابق الفهارس مع القائمة الأصلية
            sentence_timeline.append({
                "index": s_idx + 1,
                "text": sentence or "",
                "start": None,
                "end": None,
                "duration": 0.0,
                "empty": True,
            })
            continue

        start_word = word_timings[min(w_idx, total_words_stream - 1)]
        sent_start = start_word.get("start", 0.0)

        end_w_idx = min(w_idx + len(s_words) - 1, total_words_stream - 1)
        end_word = word_timings[end_w_idx]
        sent_end = max(end_word.get("end", sent_start + 1.5), sent_start + 0.8)

        # --- بناء وسم الكاريوكي الحركي {\k duration_in_centiseconds} لإضاءة الكلمات ---
        karaoke_text = ""
        for i in range(len(s_words)):
            curr_pos = min(w_idx + i, total_words_stream - 1)
            w_info = word_timings[curr_pos]
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
            "duration": sent_end - sent_start,
            "empty": False,
        })

        w_idx += len(s_words)

    # --- تحقق نهائي: عدد عناصر الـ timeline يجب أن يطابق عدد الجمل الأصلية ---
    if len(sentence_timeline) != len(sentences):
        logger.error(
            f"❌ عدم تطابق: sentences={len(sentences)} "
            f"لكن sentence_timeline={len(sentence_timeline)}"
        )
        raise RuntimeError("عدم تطابق بين الجمل الأصلية والخط الزمني الناتج.")

    if not ass_events:
        logger.error("❌ لم يتم توليد أي حدث ASS (ass_events فارغ).")
        raise RuntimeError("لا يمكن حفظ ملف ترجمة فارغ.")

    # --- حفظ ملف الـ .ass ---
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "\n".join(ass_events) + "\n")

    logger.info(f"✅ تم توليد ملف الترجمة الحركية بنجاح: {output_ass_path}")
    return sentence_timeline
