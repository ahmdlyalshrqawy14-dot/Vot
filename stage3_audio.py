import os
import re
import json
import base64
import logging
import hashlib
import uuid
import shutil
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
from xml.sax.saxutils import escape as xml_escape

import requests

# =============================================================
# استيراد pydub بشكل آمن
# =============================================================
PYDUB_AVAILABLE = False
PYDUB_IMPORT_ERROR: Optional[str] = None

try:
    from pydub import AudioSegment  # type: ignore
    PYDUB_AVAILABLE = True
except ImportError as _imp_err:
    AudioSegment = None  # type: ignore
    PYDUB_IMPORT_ERROR = f"ImportError: {_imp_err}"
except Exception as _any_err:
    AudioSegment = None  # type: ignore
    PYDUB_IMPORT_ERROR = f"{type(_any_err).__name__}: {_any_err}"

from config import (
    GEMINI_KEYS,
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    GOOGLE_MALE_VOICES,
    AZURE_MALE_VOICES,
    REQUEST_TIMEOUT
)
from gemini_engine import call_gemini_with_fallback

logger = logging.getLogger("Stage3Audio")

if not PYDUB_AVAILABLE:
    logger.warning(
        "⚠️ [Dependency Missing] مكتبة pydub غير متوفرة أو غير قابلة للاستيراد. "
        f"التفاصيل: {PYDUB_IMPORT_ERROR or 'غير معروف'}. "
        "للتثبيت: python -m pip install pydub"
    )

# =============================================================
# VOT VOICE BIBLE (مقفل لقناة Vot) — v3 Human Storyteller
# =============================================================
VOT_VOICE_BIBLE = {
    "channel": "VOT",
    "series_frame": "The Rebuild",
    "identity": "intimate intelligent scientific narrator with quiet authority",
    "personality": [
        "confident",
        "human",
        "grounded",
        "controlled",
        "intellectually calm",
        "conversational"
    ],
    "default_delivery": "clear, warm, conversational storytelling, never theatrical",
    "never": [
        "announcer voice",
        "trailer hype",
        "advertisement tone",
        "hyper excitement",
        "shouting",
        "robotic perfection"
    ],
    "voice_consistency": {
        "locked_voice": "en-US-BrianMultilingualNeural",
        "male_voice_only": True,
        "language": "en-US"
    }
}

VOICE_BIBLE_VERSION = "v3"

APPROVED_AZURE_STYLES = {
    "whispering", "excited", "serious", "hopeful",
    "cheerful", "sad", "angry", "shouting", "calm"
}

APPROVED_NARRATIVE_ROLES = {
    "hook", "setup", "question", "myth", "contradiction", "explanation", "analogy",
    "transition", "tension", "revelation", "payoff", "actionable", "reflection", "cta"
}

APPROVED_DELIVERY_MODES = {
    "calm", "curious", "serious", "teaching", "suspense",
    "revelation", "encouraging", "reflective", "energetic"
}

_LIMITS = {
    "energy": (1, 5),
    "rate_percent": (-10, 4),
    "pitch_percent": (-4, 2),
    "pause_before_ms": (0, 700),
    "pause_after_ms": (0, 1000),
}

MAX_DIRECTOR_RETRIES = 3
MAX_CLIP_RETRIES = 3
MAX_MERGE_GAP_MS = 1500
CLEANUP_BACKUPS_AFTER_SUCCESS = False
VOICE_CONTEXT_FIELDS = ("creative_brief", "retention_plan", "scene_plan", "sections")
MAX_VOICE_CONTEXT_CHARS = 6000

# =============================================================
# نظام التوجيه الصوتي (Voice Director) المحدث
# =============================================================
STAGE_3_VOICE_DIRECTOR_SYSTEM_PROMPT = """You are the LEAD VOICE DIRECTOR for the documentary/science channel VOT.

Your ONLY job is to direct the delivery plan sentence-by-sentence.
You do NOT rewrite, add, or remove words.

VOT VOICE IDENTITY:
- A calm, intelligent friend and researcher sharing deep insights in a quiet room.
- Energy is firmly 5/10. Controlled, human, conversational storytelling.
- ABSOLUTELY NO trailer voice, NO hype announcer, NO loud cheerful pitch.

DELIVERY RULES:
1. Azure Style: Default to "calm" for 75%+ of the script. Use "serious" for critical facts or tension, "hopeful" for positive payoffs, and "whispering" for intimate secrets.
2. STRICTLY FORBIDDEN: Do NOT use "excited", "cheerful", "angry", or "shouting".
3. Rate & Pitch: Keep rate slightly slow (-5% to -2%) for deliberate clarity. Pitch should stay natural to slightly grounded (-3% to 0%).
4. Energy: Keep between 2 and 3. Reserve 4 strictly for major revelations. Never use 5.
5. Emphasize at most 1 key word per sentence.

OUTPUT SCHEMA (PURE JSON ONLY):
{
  "directions": [
    {
      "sentence_index": 1,
      "narrative_role": "hook",
      "delivery_mode": "curious",
      "azure_style": "calm",
      "energy": 3,
      "rate_percent": -4,
      "pitch_percent": -1,
      "pause_before_ms": 100,
      "pause_after_ms": 350,
      "emphasis_words": ["truth"],
      "breath_breaks": [],
      "reason": "grounded conversational hook"
    }
  ]
}
"""

def _clean_text_for_word_match(text: str) -> str:
    no_tags = re.sub(r"\[.*?\]", "", text)
    no_xml = re.sub(r"<.*?>", "", no_tags)
    clean = re.sub(r"[^\w\s]", "", no_xml).lower()
    return " ".join(clean.split())

def _find_express_elements(root):
    for ns in ("https://www.w3.org/2001/mstts", "http://www.w3.org/2001/mstts"):
        elems = root.findall(f".//{{{ns}}}express-as")
        if elems:
            return elems
    try:
        return root.findall(".//{*}express-as")
    except Exception:
        return []

def _validate_single_ssml(ssml_text: str, expected_voice: Optional[str] = None) -> None:
    if not ssml_text or "<speak" not in ssml_text:
        raise ValueError("SSML غير صالح: لا يحتوي على <speak>")

    emojis = re.findall(r"[\U00010000-\U0010ffff]", ssml_text)
    if emojis:
        raise ValueError(f"SSML يحتوي على إيموجي: {emojis}")

    try:
        root = ET.fromstring(ssml_text)
    except ET.ParseError as err:
        raise ValueError(f"SSML غير صالح بنيوياً: {err}")

    voice_elem = root.find(".//{http://www.w3.org/2001/10/synthesis}voice") or root.find(".//voice")
    if voice_elem is None:
        raise ValueError("SSML لا يحتوي على عنصر <voice>")

    if expected_voice is not None:
        actual = voice_elem.attrib.get("name", "")
        if actual != expected_voice:
            raise ValueError(f"الصوت في SSML '{actual}' لا يطابق المتوقع '{expected_voice}'")

    express_elements = _find_express_elements(root)
    if not express_elements:
        raise ValueError("SSML لا يحتوي على mstts:express-as")
    for elem in express_elements:
        style = elem.attrib.get("style", "").strip().lower()
        if style not in APPROVED_AZURE_STYLES:
            raise ValueError(f"النمط '{style}' غير معتمد في Azure!")

def _build_voice_context(episode_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(episode_context, dict):
        return {}

    extracted: Dict[str, Any] = {}
    for field in VOICE_CONTEXT_FIELDS:
        value = episode_context.get(field)
        if value is None:
            continue
        candidate = dict(extracted)
        candidate[field] = value
        try:
            serialized = json.dumps(candidate, ensure_ascii=False)
            if len(serialized) <= MAX_VOICE_CONTEXT_CHARS:
                extracted = candidate
        except Exception:
            continue
    return extracted

def _call_voice_director(
    sentences: List[str],
    episode_context: Optional[Dict[str, Any]] = None,
    previous_error: Optional[str] = None,
) -> List[Dict[str, Any]]:
    user_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    safe_context = _build_voice_context(episode_context)

    context_block = ""
    if safe_context:
        try:
            ctx_str = json.dumps(safe_context, ensure_ascii=False)
            context_block = f"\n\nEPISODE CONTEXT (READ-ONLY):\n{ctx_str}"
        except Exception:
            pass

    error_block = f"\n\nPREVIOUS ERROR: {previous_error}" if previous_error else ""

    user_prompt = (
        f"Sentences to direct ({len(sentences)} total):\n"
        f"{user_lines}"
        f"{context_block}"
        f"{error_block}\n\n"
        f'Return ONLY valid JSON with a "directions" array of exactly {len(sentences)} items.'
    )

    raw = call_gemini_with_fallback(
        system_instruction=STAGE_3_VOICE_DIRECTOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_mime_type="application/json",
    )

    cleaned = re.sub(r"^```(?:json)?\s*", "", str(raw).strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # ملاحظة أمنية: لا نقوم بتضمين أي جزء من الرد الخام لـ Gemini في رسالة
        # الخطأ لتفادي تسريب محتوى السيناريو أو سياق الحلقة الخاص.
        raise ValueError(
            "فشل تحويل رد Voice Director إلى JSON صالح (تفاصيل الرد الخام غير معروضة لأسباب أمنية)."
        )

    if isinstance(data, dict) and "directions" in data:
        return data["directions"]
    elif isinstance(data, list):
        return data
    raise ValueError("صيغة غير صالحة من Voice Director")

def generate_voice_direction_plan(
    sentences: List[str],
    episode_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("قائمة الجمل فارغة!")

    safe_context = _build_voice_context(episode_context)
    last_error = None

    for attempt in range(1, MAX_DIRECTOR_RETRIES + 1):
        try:
            directions = _call_voice_director(sentences, safe_context, previous_error=last_error)
            validate_voice_direction_plan(directions, sentences)
            return directions
        except Exception as e:
            last_error = str(e)
            logger.warning(f"محاولة {attempt} لتوليد خطة الصوت فشلت: {last_error}")

    return _build_fallback_plan(sentences)

def _build_fallback_plan(sentences: List[str]) -> List[Dict[str, Any]]:
    total = len(sentences)
    plan = []
    for i in range(1, total + 1):
        ratio = i / total
        style = "calm"
        energy = 2
        role = "explanation"
        mode = "teaching"

        if i <= 2:
            role, mode, style, energy = "hook", "curious", "calm", 3
        elif ratio >= 0.70 and ratio <= 0.85:
            role, mode, style, energy = "revelation", "serious", "serious", 3
        elif ratio > 0.85:
            role, mode, style, energy = "actionable", "encouraging", "hopeful", 3

        plan.append({
            "sentence_index": i,
            "narrative_role": role,
            "delivery_mode": mode,
            "azure_style": style,
            "energy": energy,
            "rate_percent": -4,
            "pitch_percent": -1,
            "pause_before_ms": 100,
            "pause_after_ms": 300,
            "emphasis_words": [],
            "breath_breaks": [],
            "reason": "fallback quiet storytelling plan",
        })
    return plan

def validate_voice_direction_plan(directions: List[Dict[str, Any]], sentences: List[str]) -> None:
    if len(directions) != len(sentences):
        raise ValueError(f"عدد التوجيهات ({len(directions)}) لا يطابق الجمل ({len(sentences)})")

    for i, d in enumerate(directions, start=1):
        if d.get("sentence_index") != i:
            raise ValueError(f"العنصر {i}: الفهرسة غير متطابقة")
        if d.get("azure_style") not in APPROVED_AZURE_STYLES:
            raise ValueError(f"العنصر {i}: النمط غير معتمد")

# =============================================================
# بناء SSML المطور مع خاصية StyleDegree المخففة للنبرة
# =============================================================
def _clamp(value: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(round(float(value)))
    except Exception:
        return default
    return max(lo, min(hi, v))

def build_sentence_ssml(
    sentence: str,
    direction: Dict[str, Any],
    voice_name: str,
) -> str:
    """بناء SSML مع ضبط styledegree تلقائياً لضمان النبرة الهادئة المتزنة"""
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("الجملة فارغة!")

    style = str(direction.get("azure_style", "calm")).strip().lower()
    if style not in APPROVED_AZURE_STYLES:
        style = "calm"

    rate = _clamp(direction.get("rate_percent", -4), -10, 4, -4)
    pitch = _clamp(direction.get("pitch_percent", -1), -4, 2, -1)
    energy = _clamp(direction.get("energy", 3), 1, 5, 3)

    # حساب الـ styledegree لتهدئة الأداء:
    # طاقة 1 تعطي 0.40، وطاقة 5 لا تتجاوز 0.75 لمنع التصنع الصوتي
    style_degree = round(0.40 + (energy - 1) * 0.08, 2)

    emphasis_words = direction.get("emphasis_words", []) or []
    working = sentence
    placeholders: List[str] = []

    for w in emphasis_words:
        if not isinstance(w, str) or not w.strip():
            continue
        word = w.strip()
        pattern = re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.IGNORECASE)
        def _repl(m):
            idx = len(placeholders)
            placeholders.append(m.group(0))
            return f"\x00EMPH{idx}\x00"
        working, n = pattern.subn(_repl, working, count=1)

    escaped = xml_escape(working)
    for idx, original in enumerate(placeholders):
        tag = f'<emphasis level="moderate">{xml_escape(original)}</emphasis>'
        escaped = escaped.replace(f"\x00EMPH{idx}\x00", tag)

    rate_str = f"{rate:+d}%"
    pitch_str = f"{pitch:+d}%"

    ssml = (
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="https://www.w3.org/2001/mstts" '
        'xml:lang="en-US">\n'
        f'<voice name="{voice_name}">\n'
        f'<mstts:express-as style="{style}" styledegree="{style_degree}">\n'
        f'<prosody rate="{rate_str}" pitch="{pitch_str}">\n'
        f'{escaped}\n'
        '</prosody>\n'
        '</mstts:express-as>\n'
        '</voice>\n'
        '</speak>'
    )
    return ssml

# =============================================================
# استدعاء Azure TTS API
# =============================================================
def call_azure_tts_api(ssml_payload: str, output_filepath: Path):
    if not AZURE_SPEECH_KEY:
        raise ValueError("AZURE_SPEECH_KEY مفقود في ملف الإعدادات!")

    url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-160kbitrate-mono-mp3",
        "User-Agent": "VotExpressiveAudioEngine"
    }

    response = requests.post(
        url, headers=headers, data=ssml_payload.encode("utf-8"), timeout=REQUEST_TIMEOUT
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"فشل استدعاء Azure TTS API (HTTP {response.status_code}): {response.text[:300]}"
        )

    # تحقق أن الرد الناجح (200) يحتوي فعلاً على بيانات صوتية وليس محتوى فارغ
    if not response.content or len(response.content) == 0:
        raise RuntimeError(
            "استجاب Azure TTS API بنجاح (HTTP 200) لكن بدون أي بيانات صوتية (محتوى فارغ)."
        )

    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "wb") as f:
        f.write(response.content)

    # تحقق بعد الكتابة أن الملف موجود فعلاً وحجمه أكبر من صفر
    if not output_filepath.exists() or output_filepath.stat().st_size == 0:
        raise RuntimeError(
            f"فشل حفظ المقطع الصوتي: الملف '{output_filepath.name}' غير موجود أو فارغ بعد الكتابة."
        )

# =============================================================
# الدمج والمعاينة
# =============================================================
def _estimate_mp3_duration_ms(file_path: Path, bitrate_kbps: int = 160) -> int:
    try:
        size = file_path.stat().st_size
    except OSError:
        return 0
    return int(size / (bitrate_kbps * 1000 / 8 / 1000)) if size > 0 else 0

def _get_mp3_duration_ms(file_path: Path) -> Optional[int]:
    if not PYDUB_AVAILABLE:
        return None
    try:
        seg = AudioSegment.from_mp3(str(file_path))
        return int(len(seg))
    except Exception:
        return None

def _merge_clips_with_pauses(
    clips_meta: List[Dict[str, Any]],
    clips_dir: Path,
    output_file: Path,
) -> None:
    if not PYDUB_AVAILABLE or AudioSegment is None:
        raise RuntimeError("pydub غير متوفرة لدمج المقاطع.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined = AudioSegment.empty()
    prev_meta = None

    for meta in clips_meta:
        clip_path = clips_dir / meta["filename"]
        segment = AudioSegment.from_mp3(str(clip_path))

        if prev_meta is None:
            gap = _clamp(meta.get("pause_before_ms", 0), 0, MAX_MERGE_GAP_MS, 0)
            if gap > 0:
                combined += AudioSegment.silent(duration=gap, frame_rate=segment.frame_rate)
        else:
            gap_raw = int(prev_meta.get("pause_after_ms", 0)) + int(meta.get("pause_before_ms", 0))
            gap = _clamp(gap_raw, 0, MAX_MERGE_GAP_MS, 0)
            if gap > 0:
                combined += AudioSegment.silent(duration=gap, frame_rate=segment.frame_rate)

        combined += segment
        prev_meta = meta

    if prev_meta is not None:
        tail = _clamp(prev_meta.get("pause_after_ms", 0), 0, MAX_MERGE_GAP_MS, 0)
        if tail > 0:
            combined += AudioSegment.silent(duration=tail, frame_rate=combined.frame_rate)

    combined.export(str(output_file), format="mp3")

def validate_audio_outputs(
    sentences: List[str],
    clips_dir: Path,
    final_audio_file: Path,
    metadata_file: Path,
    expected_voice: Optional[str] = None,
) -> None:
    # 1) الملف الصوتي النهائي
    if not final_audio_file.exists() or final_audio_file.stat().st_size == 0:
        raise ValueError("الملف الصوتي النهائي غير موجود أو فارغ!")

    # 2) ملف البيانات الوصفية موجود
    if not metadata_file.exists():
        raise ValueError("ملف البيانات الوصفية غير موجود!")

    # 3) البيانات الوصفية JSON صالح وهي dict
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"ملف البيانات الوصفية ليس JSON صالحاً: {e}")
    except OSError as e:
        raise ValueError(f"تعذر قراءة ملف البيانات الوصفية: {e}")

    if not isinstance(metadata, dict):
        raise ValueError("ملف البيانات الوصفية ليس كائن JSON (dict)!")

    # 4) مطابقة عدد الجمل
    sentence_count = metadata.get("sentence_count")
    if sentence_count is not None and int(sentence_count) != len(sentences):
        raise ValueError(
            f"عدد الجمل في البيانات الوصفية ({sentence_count}) "
            f"لا يطابق العدد الفعلي ({len(sentences)})"
        )

    # 5) قائمة المقاطع
    clips = metadata.get("clips")
    if not isinstance(clips, list):
        raise ValueError("حقل 'clips' مفقود أو ليس قائمة في البيانات الوصفية!")
    if len(clips) != len(sentences):
        raise ValueError(
            f"عدد المقاطع في البيانات الوصفية ({len(clips)}) "
            f"لا يطابق عدد الجمل ({len(sentences)})"
        )

    # 6) كل ملف مقطع موجود وحجمه > 0
    for idx, clip in enumerate(clips, start=1):
        if not isinstance(clip, dict):
            raise ValueError(f"المقطع رقم {idx} في البيانات الوصفية ليس كائن JSON!")
        filename = clip.get("filename")
        if not filename or not isinstance(filename, str):
            raise ValueError(f"المقطع رقم {idx}: اسم الملف مفقود أو غير صالح!")
        clip_path = clips_dir / filename
        if not clip_path.exists() or clip_path.stat().st_size == 0:
            raise ValueError(f"المقطع رقم {idx} ({filename}) غير موجود أو فارغ!")

    # 7) مطابقة الصوت
    if expected_voice is not None:
        meta_voice = metadata.get("voice_name")
        if meta_voice is not None and meta_voice != expected_voice:
            raise ValueError(
                f"الصوت في البيانات الوصفية '{meta_voice}' "
                f"لا يطابق الصوت المتوقع '{expected_voice}'"
            )

def _safe_cleanup_staging(staging_clips_dir: Path, staging_audio_file: Path, staging_metadata_file: Path):
    if staging_clips_dir.exists():
        shutil.rmtree(staging_clips_dir, ignore_errors=True)
    if staging_audio_file.exists():
        staging_audio_file.unlink(missing_ok=True)
    if staging_metadata_file.exists():
        staging_metadata_file.unlink(missing_ok=True)

# =============================================================
# الدالة الرئيسية لتوليد الصوت (Default Voice = Brian)
# =============================================================
def generate_stage3_audio(
    episode_id: str,
    sentences: List[str],
    engine: str = "azure",
    voice: str = "en-US-BrianMultilingualNeural",  # الصوت المعتمد الجديد لقناة Vot
    output_dir: Path = Path("outputs"),
    episode_context: Optional[Dict[str, Any]] = None,
) -> Path:
    # =========================================================
    # HARD GUARDS — قبل أي استدعاء مدفوع (Azure / Gemini)
    # =========================================================

    # A) المحرك: azure فقط
    engine_norm = str(engine).lower().strip()
    if engine_norm != "azure":
        raise ValueError(
            f"المحرك '{engine}' غير مدعوم. المحرك الوحيد المتاح هو 'azure'."
        )

    # B) الصوت: يجب أن يكون ضمن الأصوات الذكورية المعتمدة
    if not isinstance(AZURE_MALE_VOICES, (set, list, tuple, dict)) or voice not in AZURE_MALE_VOICES:
        try:
            allowed = ", ".join(sorted(AZURE_MALE_VOICES)) if AZURE_MALE_VOICES else "(لا يوجد)"
        except Exception:
            allowed = str(AZURE_MALE_VOICES)
        raise ValueError(
            f"الصوت '{voice}' غير مسموح. الأصوات المتاحة: {allowed}"
        )

    # C) الجمل: قائمة غير فارغة من نصوص غير فارغة
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("قائمة الجمل فارغة أو غير صالحة!")
    for i, s in enumerate(sentences, start=1):
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"الجملة رقم {i} فارغة أو ليست نصاً صالحاً!")

    # D) pydub إلزامي للدمج — قبل أي استدعاء Azure
    if not PYDUB_AVAILABLE or AudioSegment is None:
        raise RuntimeError(
            "مكتبة pydub مطلوبة لدمج المقاطع الصوتية ولا يمكن المتابعة بدونها. "
            f"التفاصيل: {PYDUB_IMPORT_ERROR or 'غير معروف'}. "
            "للتثبيت: python -m pip install pydub"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    final_audio_file = output_dir / f"episode_{episode_id}_audio.mp3"
    clips_dir = output_dir / f"episode_{episode_id}_clips"
    metadata_file = output_dir / f"episode_{episode_id}_audio_metadata.json"

    run_id = uuid.uuid4().hex[:10]
    staging_clips_dir = output_dir / f"episode_{episode_id}_clips__staging_{run_id}"
    staging_audio_file = output_dir / f"episode_{episode_id}_audio__staging_{run_id}.mp3"
    staging_metadata_file = output_dir / f"episode_{episode_id}_audio_metadata__staging_{run_id}.json"
    staging_clips_dir.mkdir(parents=True, exist_ok=True)

    try:
        directions = generate_voice_direction_plan(sentences, episode_context)
        clips_meta = []

        for i, (sentence, direction) in enumerate(zip(sentences, directions), start=1):
            clip_filename = f"sentence_{i:04d}.mp3"
            clip_path = staging_clips_dir / clip_filename
            sentence_hash = hashlib.sha256(sentence.encode("utf-8")).hexdigest()

            ssml = build_sentence_ssml(sentence, direction, voice)
            _validate_single_ssml(ssml, expected_voice=voice)

            for attempt in range(1, MAX_CLIP_RETRIES + 1):
                try:
                    call_azure_tts_api(ssml, clip_path)
                    if clip_path.exists() and clip_path.stat().st_size > 0:
                        break
                except Exception as e:
                    logger.warning(
                        f"[episode={episode_id}] محاولة {attempt} لتوليد المقطع رقم {i} فشلت: {e}"
                    )
                    if attempt == MAX_CLIP_RETRIES:
                        raise RuntimeError(f"فشل توليد المقطع {i}: {e}")

            duration_ms = _get_mp3_duration_ms(clip_path) or _estimate_mp3_duration_ms(clip_path)

            clips_meta.append({
                "sentence_index": i,
                "filename": clip_filename,
                "sentence_text": sentence,
                "sentence_hash": sentence_hash,
                "narrative_role": direction.get("narrative_role"),
                "delivery_mode": direction.get("delivery_mode"),
                "azure_style": direction.get("azure_style"),
                "energy": int(direction.get("energy", 3)),
                "rate_percent": int(direction.get("rate_percent", -4)),
                "pitch_percent": int(direction.get("pitch_percent", -1)),
                "pause_before_ms": int(direction.get("pause_before_ms", 100)),
                "pause_after_ms": int(direction.get("pause_after_ms", 300)),
                "emphasis_words": direction.get("emphasis_words", []),
                "breath_breaks": [],
                "reason": direction.get("reason", ""),
                "duration_ms": duration_ms,
            })

        _merge_clips_with_pauses(clips_meta, staging_clips_dir, staging_audio_file)

        metadata = {
            "episode_id": episode_id,
            "voice_name": voice,
            "language": "en-US",
            "sentence_count": len(sentences),
            "voice_bible_version": VOICE_BIBLE_VERSION,
            "clips": clips_meta,
        }
        with open(staging_metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # تحقق شامل على مسارات الـ staging قبل أي التزام نهائي
        validate_audio_outputs(
            sentences,
            staging_clips_dir,
            staging_audio_file,
            staging_metadata_file,
            expected_voice=voice,
        )

        # =====================================================
        # SAFE COMMIT — لا نحذف النهائيات إلا بعد نجاح التحقق
        # =====================================================

        # 1) الصوت والبيانات الوصفية: Path.replace ذرّي على نفس نظام الملفات
        staging_audio_file.replace(final_audio_file)
        staging_metadata_file.replace(metadata_file)

        # 2) مجلد المقاطع: نُزيح القديم جانباً، نُدخل الجديد، ثم نحذف القديم
        old_clips_backup = output_dir / f"episode_{episode_id}_clips__old_{run_id}"
        had_old_clips = clips_dir.exists()
        if had_old_clips:
            clips_dir.rename(old_clips_backup)

        try:
            staging_clips_dir.rename(clips_dir)
        except Exception:
            # استرجاع النسخة القديمة إن فشل نقل الـ staging
            if had_old_clips and old_clips_backup.exists() and not clips_dir.exists():
                try:
                    old_clips_backup.rename(clips_dir)
                except Exception:
                    logger.error("⚠️ فشل استرجاع مجلد المقاطع القديم بعد فشل الالتزام.")
            raise

        if old_clips_backup.exists():
            shutil.rmtree(old_clips_backup, ignore_errors=True)

        logger.info(f"✅ تم إنتاج صوت حلقة Vot بالهوية الجديدة بنجاح: {final_audio_file}")
        return final_audio_file

    except Exception as err:
        logger.error(f"❌ [episode={episode_id}] خطأ أثناء توليد الصوت: {err}")
        # تنظيف الـ staging فقط. لا نحذف ملفات نهائية من تشغيل سابق ناجح.
        _safe_cleanup_staging(staging_clips_dir, staging_audio_file, staging_metadata_file)
        raise err

def generate_voice_preview(
    voice: str = "en-US-BrianMultilingualNeural",
    voice_name: str = None,
    text: str = "Most people think collapse happens all at once. But it doesn't.",
    output_dir: Path = Path("outputs")
) -> Path:
    final_voice = voice_name or voice
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_file = output_dir / f"preview_{final_voice}.mp3"

    direction = {"azure_style": "calm", "energy": 2, "rate_percent": -4, "pitch_percent": -1}
    ssml_payload = build_sentence_ssml(text, direction, final_voice)
    call_azure_tts_api(ssml_payload, preview_file)
    return preview_file
