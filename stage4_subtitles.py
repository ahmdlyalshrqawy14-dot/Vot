import os
import re
import json
import difflib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("Stage4Subtitles")


# ===========================================================================
# Whisper model (lazy singleton - loaded once, reused for all clips/files)
# ===========================================================================

_WHISPER_MODEL = None


def _get_whisper_model():
    """تحميل نموذج faster-whisper مرة واحدة فقط وإعادة استخدامه."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    try:
        from faster_whisper import WhisperModel
        _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("✅ تم تحميل نموذج Whisper (base) مرة واحدة.")
        return _WHISPER_MODEL
    except Exception as e:
        logger.warning(f"⚠️ faster-whisper غير متاح: {e}")
        return None


def _extract_words_from_segments(segments) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    for seg in segments:
        seg_words = getattr(seg, "words", None)
        if not seg_words:
            continue
        for w in seg_words:
            txt = getattr(w, "word", "") or ""
            if txt.strip():
                words.append({
                    "word": txt.strip(),
                    "start": float(w.start),
                    "end": float(w.end),
                })
    return words


def _transcribe_clip(model, clip_path: Path) -> List[Dict[str, Any]]:
    """
    تفريغ مقطع صوتي منفصل. التوقيتات المُعادة نسبية (تبدأ من 0) داخل المقطع.
    """
    try:
        segments, _ = model.transcribe(
            str(clip_path), word_timestamps=True, language="en"
        )
        return _extract_words_from_segments(segments)
    except Exception as e:
        logger.warning(f"⚠️ فشل تفريغ المقطع {clip_path}: {e}")
        return []


def _transcribe_file(model, audio_path: Path) -> List[Dict[str, Any]]:
    """
    تفريغ ملف الصوت الكامل. التوقيتات المُعادة مطلقة (نسبة للملف الكامل).
    """
    try:
        segments, _ = model.transcribe(
            str(audio_path), word_timestamps=True, language="en"
        )
        return _extract_words_from_segments(segments)
    except Exception as e:
        logger.warning(f"⚠️ فشل تفريغ الملف {audio_path}: {e}")
        return []


# ===========================================================================
# واجهات عامة محفوظة (interface preserved)
# ===========================================================================

def format_ass_time(seconds: float) -> str:
    """
    تحويل الثواني إلى توقيت ASS بصيغة H:MM:SS.CC
    يعتمد على التحويل إلى centiseconds أولاً لتفادي أخطاء التقريب عند الحدود
    (مثل 59.999 أو 3599.999).
    """
    if seconds is None or seconds < 0:
        seconds = 0.0
    total_cs = int(round(float(seconds) * 100.0))
    if total_cs < 0:
        total_cs = 0
    hrs = total_cs // 360000
    rem = total_cs % 360000
    mins = rem // 6000
    rem = rem % 6000
    secs = rem // 100
    centis = rem % 100
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


def get_audio_duration_wave(audio_path: Path) -> float:
    """استخراج مدة ملف الصوت بدقة"""
    import wave
    try:
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        import subprocess
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return float(res.stdout.strip())


# ===========================================================================
# أدوات نصية مساعدة
# ===========================================================================

def _norm_word(w: str) -> str:
    """تطبيع كلمة للمقارنة فقط (بدون تغيير النص المعروض)."""
    if not w:
        return ""
    w = w.lower().strip()
    w = w.replace("’", "'").replace("‘", "'")
    w = re.sub(r"[^\w']", "", w)
    return w


def _escape_ass_text(text: str) -> str:
    """تهريب النص لمنع كسر تنسيق ASS."""
    if not text:
        return ""
    return (
        text.replace("\\", "\\\\")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("\n", "\\N")
    )


def _text_similar(a: str, b: str, threshold: float = 0.6) -> bool:
    if not a or not b:
        return False
    na = _norm_word(a)
    nb = _norm_word(b)
    if not na or not nb:
        return False
    try:
        ratio = difflib.SequenceMatcher(a=na, b=nb).ratio()
    except Exception:
        return False
    return ratio >= threshold


# ===========================================================================
# خوارزميات التوقيت
# ===========================================================================

def _build_proportional_timings(
    sentences: List[str], total_duration: float
) -> List[Dict[str, Any]]:
    """
    خوارزمية الحل الاحتياطي الأخير: توزيع الزمن تناسبياً بناءً على عدد حروف كل جملة.
    تُعيد قائمة مسطّحة من كلمات مع توقيتات مطلقة من 0 إلى total_duration.
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
            timings.append(
                {"word": w, "start": curr_t, "end": curr_t + w_dur}
            )
            curr_t += w_dur
    return timings


def _align_words_with_whisper(
    text_words: List[str],
    whisper_words: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
) -> List[Dict[str, Any]]:
    """
    محاذاة آمنة (SequenceMatcher) بين كلمات النص الأصلي وكلمات Whisper.

    ملاحظة مهمة:
    يُفترض أن تكون توقيتات whisper_words نسبية داخل المقطع (تبدأ من 0)
    عند استدعائها من مسار الـ clips. يتم تحويلها إلى توقيتات مطلقة
    بإضافة clip_start قبل أي معالجة.
    عند استدعائها من مسار الـ fallback للملف الكامل، يجب تمرير clip_start=0
    لأن توقيتات Whisper في هذه الحالة مطلقة أصلاً.

    المخرجات: قائمة بطول n_text بالضبط، كل كلمة لها start/end مطلق داخل
    [clip_start, clip_end]، مع منع التداخل.
    """
    n_text = len(text_words)
    if n_text == 0:
        return []
    if clip_end <= clip_start:
        clip_end = clip_start + 0.5

    # 1) تحويل توقيتات Whisper النسبية إلى توقيتات مطلقة + قصّ داخل الحدود
    abs_whisper: List[Dict[str, Any]] = []
    for w in whisper_words or []:
        try:
            rs = float(w.get("start", 0.0))
            re_ = float(w.get("end", 0.0))
        except Exception:
            continue
        s = clip_start + rs
        e = clip_start + re_
        if s < clip_start:
            s = clip_start
        if e > clip_end:
            e = clip_end
        if e <= s:
            e = min(clip_end, s + 0.01)
            if e <= s:
                s = max(clip_start, e - 0.01)
        abs_whisper.append({
            "word": w.get("word", ""),
            "start": s,
            "end": e,
        })

    # 2) لا Whisper؟ نوزّع داخل المقطع بالتساوي
    if not abs_whisper:
        dur = (clip_end - clip_start) / n_text
        return [
            {
                "word": w,
                "start": clip_start + i * dur,
                "end": clip_start + (i + 1) * dur,
            }
            for i, w in enumerate(text_words)
        ]

    a = [_norm_word(w) for w in text_words]
    b = [_norm_word(w["word"]) for w in abs_whisper]

    result: List[Optional[Dict[str, Any]]] = [None] * n_text
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                if j1 + k < len(abs_whisper):
                    ww = abs_whisper[j1 + k]
                    result[i1 + k] = {
                        "word": text_words[i1 + k],
                        "start": float(ww["start"]),
                        "end": float(ww["end"]),
                    }

    known = [i for i in range(n_text) if result[i] is not None]

    # 3) لا كلمة معروفة إطلاقاً؟ توزيع منتظم
    if not known:
        dur = (clip_end - clip_start) / n_text
        return [
            {
                "word": w,
                "start": clip_start + i * dur,
                "end": clip_start + (i + 1) * dur,
            }
            for i, w in enumerate(text_words)
        ]

    # 4) تعبئة الرأس
    first = known[0]
    if first > 0:
        head_end = float(result[first]["start"])
        if head_end < clip_start:
            head_end = clip_start
        if head_end > clip_end:
            head_end = clip_end
        span = max(head_end - clip_start, 0.05 * first)
        dur = span / first
        for i in range(first):
            result[i] = {
                "word": text_words[i],
                "start": clip_start + i * dur,
                "end": clip_start + (i + 1) * dur,
            }

    # 5) تعبئة الوسط
    for ki in range(len(known) - 1):
        i1 = known[ki]
        i2 = known[ki + 1]
        gap_count = i2 - i1 - 1
        if gap_count <= 0:
            continue
        gap_start = float(result[i1]["end"])
        gap_end = float(result[i2]["start"])
        if gap_end <= gap_start:
            gap_end = gap_start + 0.05 * gap_count
        dur = (gap_end - gap_start) / gap_count
        for k in range(1, gap_count + 1):
            result[i1 + k] = {
                "word": text_words[i1 + k],
                "start": gap_start + (k - 1) * dur,
                "end": gap_start + k * dur,
            }

    # 6) تعبئة الذيل
    last = known[-1]
    if last < n_text - 1:
        tail_count = n_text - last - 1
        tail_start = float(result[last]["end"])
        if tail_start < clip_start:
            tail_start = clip_start
        if tail_start >= clip_end:
            tail_start = max(clip_end - 0.05 * tail_count, clip_start)
        span = clip_end - tail_start
        if span < 0.05 * tail_count:
            span = 0.05 * tail_count
        dur = span / tail_count
        for k in range(1, tail_count + 1):
            result[last + k] = {
                "word": text_words[last + k],
                "start": tail_start + (k - 1) * dur,
                "end": tail_start + k * dur,
            }

    # 7) ضبط نهائي: منع التداخل + القصّ داخل [clip_start, clip_end]
    out: List[Dict[str, Any]] = []
    prev_end = clip_start
    for i in range(n_text):
        item = result[i]
        if item is None:
            item = {
                "word": text_words[i],
                "start": prev_end,
                "end": prev_end + 0.01,
            }
        s = float(item["start"])
        e = float(item["end"])
        if s < clip_start:
            s = clip_start
        if s < prev_end:
            s = prev_end
        if s > clip_end:
            s = clip_end
        if e <= s:
            e = s + 0.01
        if e > clip_end:
            e = clip_end
        if e <= s:
            # ضيق شديد: نمنح نافذة صغيرة عند الحد الأقصى
            s = max(clip_start, min(prev_end, clip_end - 0.001))
            e = min(clip_end, s + 0.001)
            if e <= s:
                s = max(clip_start, clip_end - 0.001)
                e = clip_end
        out.append({"word": text_words[i], "start": s, "end": e})
        prev_end = e
    return out


def _validate_and_align_word_timings(
    sentences: List[str],
    word_timings: List[Dict[str, Any]],
    total_duration: float,
) -> List[Dict[str, Any]]:
    """
    للملف الكامل: يوائم كلمات النص مع كلمات Whisper باستخدام نفس خوارزمية
    المحاذاة الآمنة (SequenceMatcher) بدلاً من pairing حسب الفهرس.

    يُفترض أن word_timings مُطلقة (نسبة للملف الكامل) لأنها من _transcribe_file.
    """
    text_words: List[str] = []
    for s in sentences:
        if s and s.strip():
            text_words.extend(s.split())

    if not text_words:
        return []

    if not word_timings:
        logger.warning(
            "⚠️ لا توجد كلمات Whisper للملف الكامل. توزيع تناسبي."
        )
        return _build_proportional_timings(sentences, total_duration)

    clip_end = max(total_duration, 0.5)
    return _align_words_with_whisper(
        text_words, word_timings, 0.0, clip_end
    )


def _enforce_monotonic(wlist: List[Dict[str, Any]]) -> None:
    """منع التداخل بين الكلمات."""
    prev_end = -1e9
    for item in wlist:
        if item["start"] < prev_end:
            item["start"] = prev_end
        if item["end"] <= item["start"]:
            item["end"] = item["start"] + 0.01
        prev_end = item["end"]


# ===========================================================================
# اكتشاف ملفات المرحلة الثالثة (metadata + clips)
# ===========================================================================

def _discover_metadata(audio_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    البحث عن ملف metadata ومجلد المقاطع بجانب ملف الصوت النهائي.
    يدعم عدة صيغ تسمية محتملة.
    """
    parent = audio_path.parent
    stem = audio_path.stem  # مثل: episode_201_audio

    meta_candidates: List[Path] = [
        parent / f"{stem}_metadata.json",
        parent / f"{stem}_audio_metadata.json",
    ]
    clip_candidates: List[Path] = []

    if stem.endswith("_audio"):
        base = stem[: -len("_audio")]  # episode_201
        meta_candidates.append(parent / f"{base}_audio_metadata.json")
        meta_candidates.append(parent / f"{base}_metadata.json")
        clip_candidates.append(parent / f"{base}_clips")
    else:
        base = stem

    clip_candidates.append(parent / f"{stem}_clips")
    clip_candidates.append(parent / f"{base}_clips")

    meta_path = next((p for p in meta_candidates if p.exists()), None)
    clips_dir = next((p for p in clip_candidates if p.is_dir()), None)
    return meta_path, clips_dir


def _resolve_sentence_idx(
    clip_meta: Dict[str, Any],
    sentences: List[str],
    used_idxs: set,
) -> Optional[int]:
    """
    تحديد فهرس الجملة (0-based) المرتبطة بهذا المقطع بأمان.
    يعتمد على sentence_index أولاً ثم النص.
    """
    si = clip_meta.get("sentence_index")
    st = clip_meta.get("sentence_text", "") or ""

    # 1) بواسطة sentence_index (1-based ثم 0-based)
    if isinstance(si, int):
        for candidate in (si - 1, si):
            if (
                0 <= candidate < len(sentences)
                and candidate not in used_idxs
                and sentences[candidate].strip()
            ):
                if not st or _text_similar(sentences[candidate], st):
                    return candidate

    # 2) بواسطة النص
    if st:
        best = None
        best_score = 0.0
        for i, s in enumerate(sentences):
            if i in used_idxs or not s.strip():
                continue
            if not _text_similar(s, st):
                continue
            score = difflib.SequenceMatcher(
                a=_norm_word(s), b=_norm_word(st)
            ).ratio()
            if score > best_score:
                best_score = score
                best = i
        if best is not None:
            return best

    return None


def _try_use_clips(
    clips: List[Dict[str, Any]],
    clips_dir: Path,
    sentences: List[str],
    sentence_word_timings: List[Optional[List[Dict[str, Any]]]],
    audio_duration: float,
) -> bool:
    """
    محاولة استخدام المقاطع المنفصلة كمصدر رئيسي للتوقيت.
    ترجع True عند النجاح، False للرجوع إلى fallback.

    ملاحظة: لا يُسمح بتجاوز audio_duration. إذا استمرّ cursor بعد آخر مقطع
    فوق المدة، يتم تسجيل تحذير فقط دون فشل كامل (المقص النهائي يتم في
    حلقة البناء الرئيسية).
    """
    try:
        ordered = sorted(clips, key=lambda c: c.get("sentence_index", 0))
    except Exception:
        ordered = list(clips)

    used_idxs: set = set()
    resolved: List[Optional[int]] = []
    for clip_meta in ordered:
        idx = _resolve_sentence_idx(clip_meta, sentences, used_idxs)
        resolved.append(idx)
        if idx is not None:
            used_idxs.add(idx)

    non_empty_idxs = [i for i, s in enumerate(sentences) if s and s.strip()]
    if not non_empty_idxs:
        return False

    # كل الجمل غير الفارغة يجب أن تُغطى
    missing = [i for i in non_empty_idxs if i not in used_idxs]
    if missing:
        logger.warning(
            f"⚠️ المقاطع لا تغطي كل الجمل (ناقص {len(missing)}). "
            f"سيتم الرجوع إلى fallback."
        )
        return False

    # كل مقطع يجب أن يُحل إلى جملة
    if any(r is None for r in resolved):
        logger.warning("⚠️ بعض المقاطع لم تُطابَق مع أي جملة. fallback.")
        return False

    model = _get_whisper_model()

    cursor = 0.0
    for pos, (clip_meta, s_idx) in enumerate(zip(ordered, resolved)):
        assert s_idx is not None
        filename = clip_meta.get("filename", "")
        clip_file = clips_dir / filename
        if not clip_file.exists():
            logger.warning(f"⚠️ ملف المقطع غير موجود: {clip_file}. fallback.")
            return False

        try:
            dur_s = float(clip_meta.get("duration_ms", 0)) / 1000.0
        except Exception:
            dur_s = 0.0
        try:
            pause_before = float(clip_meta.get("pause_before_ms", 0)) / 1000.0
        except Exception:
            pause_before = 0.0
        try:
            pause_after = float(clip_meta.get("pause_after_ms", 0)) / 1000.0
        except Exception:
            pause_after = 0.0

        # الوقفة قبل أول مقطع فقط تُضاف صراحة (لأن ما قبلها ليس له مقطع سابق)
        if pos == 0:
            cursor += max(pause_before, 0.0)

        clip_start = cursor
        clip_end = clip_start + max(dur_s, 0.05)

        if model is not None:
            wh_words = _transcribe_clip(model, clip_file)
        else:
            wh_words = []

        text_words = sentences[s_idx].split()
        aligned = _align_words_with_whisper(
            text_words, wh_words, clip_start, clip_end
        )
        sentence_word_timings[s_idx] = aligned

        # حساب الفجوة بين هذا المقطع والمقطع التالي.
        # نأخذ الأكبر بين pause_after للمقطع الحالي و pause_before للمقطع التالي
        # لتجنّب حساب نفس الوقفة مرتين.
        next_pb = 0.0
        if pos + 1 < len(ordered):
            try:
                next_pb = float(
                    ordered[pos + 1].get("pause_before_ms", 0)
                ) / 1000.0
            except Exception:
                next_pb = 0.0
        gap = max(pause_after, next_pb, 0.0)
        cursor = clip_end + gap

    if cursor > audio_duration + 0.5:
        logger.warning(
            f"⚠️ تقدير زمن المقاطع ({cursor:.2f}s) يتجاوز مدة الصوت "
            f"({audio_duration:.2f}s). سيتم الاقتصاص النهائي."
        )

    return True


def _fallback_full_audio(
    audio_path: Path,
    sentences: List[str],
    duration: float,
    sentence_word_timings: List[Optional[List[Dict[str, Any]]]],
) -> None:
    """
    Fallback: تفريغ الملف الكامل بـ Whisper، محاذاة النص باستخدام
    SequenceMatcher (عبر _validate_and_align_word_timings)، ثم توزيع
    الكلمات على الجمل حسب عدد كلمات كل جملة.
    """
    model = _get_whisper_model()
    wh_words: List[Dict[str, Any]] = []
    if model is not None:
        wh_words = _transcribe_file(model, audio_path)

    if wh_words:
        aligned_stream = _validate_and_align_word_timings(
            sentences, wh_words, duration
        )
    else:
        logger.warning(
            "⚠️ لا يوجد Whisper / لا كلمات. استخدام التوزيع التناسبي (توقيت تقريبي)."
        )
        aligned_stream = _build_proportional_timings(sentences, duration)

    if not aligned_stream:
        aligned_stream = _build_proportional_timings(sentences, duration)

    # توزيع stream على الجمل حسب عدد كلمات كل جملة
    w_idx = 0
    total = len(aligned_stream)
    for s_idx, s in enumerate(sentences):
        sw = (s or "").split()
        if not sw:
            continue
        chunk = aligned_stream[w_idx: w_idx + len(sw)]
        # padding إذا نقص
        while len(chunk) < len(sw):
            if chunk:
                last_end = chunk[-1]["end"]
            else:
                last_end = duration * (w_idx / max(total, 1))
            if last_end >= duration:
                last_end = max(duration - 0.05, 0.0)
            chunk.append({
                "word": sw[len(chunk)],
                "start": last_end,
                "end": min(last_end + 0.05, duration),
            })
        # قصّ نهائي إن تجاوز
        for item in chunk:
            if item["start"] < 0:
                item["start"] = 0.0
            if item["end"] > duration:
                item["end"] = duration
            if item["end"] <= item["start"]:
                item["end"] = min(item["start"] + 0.01, duration)
                if item["end"] <= item["start"]:
                    item["start"] = max(0.0, item["end"] - 0.01)
        sentence_word_timings[s_idx] = chunk
        w_idx += len(sw)


def _fill_missing_sentences(
    sentences: List[str],
    sentence_word_timings: List[Optional[List[Dict[str, Any]]]],
    duration: float,
) -> None:
    """
    تعبئة أي جملة غير فارغة لم تحصل على توقيتات، باستخدام الجيران.
    مع ضمان ألا تتجاوز النهاية مدة الصوت.
    """
    n = len(sentences)
    safe_duration = max(duration, 0.5)

    anchors: List[Tuple[int, float, float]] = []
    for i in range(n):
        wl = sentence_word_timings[i]
        if wl and sentences[i].strip():
            anchors.append((i, float(wl[0]["start"]), float(wl[-1]["end"])))

    if not anchors:
        non_empty = [i for i, s in enumerate(sentences) if s and s.strip()]
        if not non_empty:
            return
        total_chars = sum(len(sentences[i]) for i in non_empty)
        cursor = 0.0
        for i in non_empty:
            s = sentences[i]
            frac = len(s) / max(total_chars, 1)
            seg_dur = frac * safe_duration
            words = s.split()
            w_dur = seg_dur / max(len(words), 1)
            sentence_word_timings[i] = [
                {
                    "word": w,
                    "start": min(cursor + k * w_dur, safe_duration),
                    "end": min(cursor + (k + 1) * w_dur, safe_duration),
                }
                for k, w in enumerate(words)
            ]
            cursor += seg_dur
        return

    for i in range(n):
        if not sentences[i].strip():
            continue
        if sentence_word_timings[i]:
            continue

        prev = None
        nxt = None
        for a in anchors:
            if a[0] < i:
                prev = a
            elif a[0] > i and nxt is None:
                nxt = a
                break

        lo = prev[2] if prev else 0.0
        hi = nxt[1] if nxt else safe_duration
        if hi > safe_duration:
            hi = safe_duration
        if hi <= lo:
            hi = min(lo + 0.5, safe_duration)
        if hi <= lo:
            hi = lo + 0.1

        words = sentences[i].split()
        w_dur = (hi - lo) / max(len(words), 1)
        sentence_word_timings[i] = [
            {
                "word": w,
                "start": lo + k * w_dur,
                "end": lo + (k + 1) * w_dur,
            }
            for k, w in enumerate(words)
        ]


# ===========================================================================
# الدالة العامة الرئيسية
# ===========================================================================

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

    مصدر التوقيت الرئيسي:
    1) مقاطع الجمل المنفصلة (metadata + clips) إن وُجدت.
    2) fallback: تفريغ الملف الصوتي الكامل بـ Whisper.
    3) fallback أخير جداً: توزيع تناسبي تقريبي.
    """
    # --- تحقق مبكر من المدخلات ---
    if not sentences:
        raise ValueError("قائمة الجمل فارغة.")

    valid_sentences = [s for s in sentences if s and s.strip()]
    if not valid_sentences:
        raise ValueError("كل الجمل فارغة، لا يوجد نص صالح للمزامنة.")

    duration = get_audio_duration_wave(audio_path)
    logger.info(
        f"🎙️ إجمالي مدة الصوت: {duration:.2f} ثانية لعدد {len(sentences)} جملة."
    )

    # --- اكتشاف ملفات المرحلة الثالثة ---
    meta_path, clips_dir = _discover_metadata(audio_path)

    # sentence_word_timings[s_idx] = list of {word, start, end} أو None
    sentence_word_timings: List[Optional[List[Dict[str, Any]]]] = [None] * len(sentences)

    clips_used = False
    if meta_path and clips_dir:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            clips = meta.get("clips") or []
            if clips:
                logger.info(
                    f"📂 تم اكتشاف metadata ({meta_path.name}) و{len(clips)} مقطع."
                )
                clips_used = _try_use_clips(
                    clips, clips_dir, sentences,
                    sentence_word_timings, duration,
                )
            else:
                logger.warning("⚠️ metadata لا يحتوي على مقاطع.")
        except Exception as e:
            logger.warning(f"⚠️ فشل قراءة metadata: {e}")
    else:
        logger.info("ℹ️ لا توجد ملفات مقاطع/metadata بجانب الملف الصوتي.")

    if not clips_used:
        logger.info("ℹ️ Fallback: تفريغ الملف الصوتي الكامل بـ Whisper.")
        _fallback_full_audio(
            audio_path, sentences, duration, sentence_word_timings
        )

    # --- تعبئة أي جملة غير فارغة لم تحصل على توقيتات ---
    _fill_missing_sentences(sentences, sentence_word_timings, duration)

    # --- ترويسة ملف الترجمة ASS ---
    ass_header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,64,&H0047E0FD,&H0000FFFF,&H00101010,"
        "&H80000000,-1,0,0,0,100,100,0,0,1,2.0,1.5,2,80,80,95,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    ass_events: List[str] = []
    sentence_timeline: List[Dict[str, Any]] = []

    # --- بناء الأحداث والخط الزمني ---
    for s_idx, sentence in enumerate(sentences):
        s_words = (sentence or "").split()

        if not s_words:
            sentence_timeline.append({
                "index": s_idx + 1,
                "text": sentence or "",
                "start": None,
                "end": None,
                "duration": 0.0,
                "empty": True,
            })
            continue

        wlist = sentence_word_timings[s_idx]
        if not wlist or len(wlist) != len(s_words):
            # إعادة بناء آمنة داخل مدة الملف
            wlist = _align_words_with_whisper(
                s_words, [], 0.0, max(duration, 1.0)
            )
            sentence_word_timings[s_idx] = wlist

        # ضمان عدم التداخل
        _enforce_monotonic(wlist)

        # ضمان أن كل كلمة داخل حدود الصوت
        for item in wlist:
            if item["start"] < 0:
                item["start"] = 0.0
            if item["end"] > duration:
                item["end"] = duration
            if item["end"] <= item["start"]:
                item["end"] = min(item["start"] + 0.01, duration)
                if item["end"] <= item["start"]:
                    item["start"] = max(0.0, item["end"] - 0.01)

        sent_start = float(wlist[0]["start"])
        sent_end = float(wlist[-1]["end"])
        if sent_end <= sent_start:
            sent_end = min(sent_start + 0.5, duration)
        if sent_start < 0:
            sent_start = 0.0
        if sent_end > duration:
            logger.warning(
                f"⚠️ الجملة {s_idx+1} تتجاوز مدة الملف "
                f"({sent_end:.2f}s > {duration:.2f}s). سيتم تطبيعها."
            )
            sent_end = duration
        if sent_end <= sent_start:
            sent_end = min(sent_start + 0.01, duration)

        # --- بناء وسم الكاريوكي ---
        karaoke_text = ""
        for i, w in enumerate(s_words):
            w_start = float(wlist[i]["start"])
            w_end = float(wlist[i]["end"])
            w_dur = max(w_end - w_start, 0.01)
            dur_cs = max(int(round(w_dur * 100)), 5)
            karaoke_text += f"{{\\k{dur_cs}}}{_escape_ass_text(w)} "

        start_str = format_ass_time(sent_start)
        end_str = format_ass_time(sent_end)

        ass_events.append(
            f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,"
            f"{karaoke_text.strip()}"
        )

        sentence_timeline.append({
            "index": s_idx + 1,
            "text": sentence,
            "start": sent_start,
            "end": sent_end,
            "duration": sent_end - sent_start,
            "empty": False,
            "words": [
                {
                    "word": wlist[i]["word"],
                    "start": float(wlist[i]["start"]),
                    "end": float(wlist[i]["end"]),
                }
                for i in range(len(s_words))
            ],
        })

    # --- تحقق نهائي ---
    if len(sentence_timeline) != len(sentences):
        logger.error(
            f"❌ عدم تطابق: sentences={len(sentences)} "
            f"لكن sentence_timeline={len(sentence_timeline)}"
        )
        raise RuntimeError("عدم تطابق بين الجمل الأصلية والخط الزمني الناتج.")

    prev_end = -1.0
    for ev in sentence_timeline:
        if ev["empty"]:
            continue
        if ev["start"] is None or ev["end"] is None:
            raise RuntimeError(
                f"الجملة {ev['index']} بدون توقيت صالح (start/end = None)."
            )
        if ev["start"] < -0.001:
            raise RuntimeError(
                f"الجملة {ev['index']} لها بداية سالبة: {ev['start']}"
            )
        if ev["end"] <= ev["start"]:
            raise RuntimeError(
                f"الجملة {ev['index']} لها نهاية قبل البداية: "
                f"{ev['start']} -> {ev['end']}"
            )
        if ev["end"] > duration + 0.5:
            raise RuntimeError(
                f"الجملة {ev['index']} تتجاوز مدة الصوت: "
                f"{ev['end']:.3f} > {duration:.3f}"
            )
        # تحقق من عدد الكلمات المعروضة
        expected_words = len((ev["text"] or "").split())
        if len(ev.get("words", [])) != expected_words:
            raise RuntimeError(
                f"الجملة {ev['index']}: عدد الكلمات ({len(ev.get('words', []))}) "
                f"لا يطابق النص الأصلي ({expected_words})."
            )
        # تحقق أن كل كلمة داخل حدود الجملة والصوت
        for wi, witem in enumerate(ev["words"]):
            if witem["start"] < ev["start"] - 0.05:
                raise RuntimeError(
                    f"الجملة {ev['index']} كلمة {wi} تبدأ قبل الجملة."
                )
            if witem["end"] > ev["end"] + 0.05:
                raise RuntimeError(
                    f"الجملة {ev['index']} كلمة {wi} تنتهي بعد الجملة."
                )
            if witem["end"] > duration + 0.5:
                raise RuntimeError(
                    f"الجملة {ev['index']} كلمة {wi} تتجاوز مدة الصوت."
                )
            if witem["end"] <= witem["start"]:
                raise RuntimeError(
                    f"الجملة {ev['index']} كلمة {wi} لها نهاية <= بداية."
                )
        # تحقق من عدم التداخل بين الجمل
        if ev["start"] < prev_end - 0.001:
            raise RuntimeError(
                f"الجملة {ev['index']} تتداخل مع الجملة السابقة: "
                f"{ev['start']:.3f} < {prev_end:.3f}"
            )
        prev_end = ev["end"]

    if not ass_events:
        logger.error("❌ لم يتم توليد أي حدث ASS (ass_events فارغ).")
        raise RuntimeError("لا يمكن حفظ ملف ترجمة فارغ.")

    # --- حفظ ملف الـ .ass ---
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header + "\n".join(ass_events) + "\n")

    logger.info(
        f"✅ تم توليد ملف الترجمة الحركية بنجاح: {output_ass_path}"
    )
    return sentence_timeline
