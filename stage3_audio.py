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
# استيراد pydub بشكل واضح وآمن مع تسجيل سبب الفشل
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
        "هذه مشكلة اعتمادية وليست خطأً من Azure Speech. "
        f"التفاصيل: {PYDUB_IMPORT_ERROR or 'غير معروف'}. "
        "للتثبيت: python -m pip install pydub"
    )

# =============================================================
# VOT VOICE BIBLE (ثابت على مستوى القناة)
# =============================================================
VOT_VOICE_BIBLE = {
    "identity": "calm scientific cinematic narrator",
    "personality": [
        "confident",
        "intelligent",
        "warm",
        "controlled",
        "slightly mysterious",
        "encouraging"
    ],
    "default_delivery": "clear, grounded, believable, never theatrical",
    "never": [
        "constant overacting",
        "advertisement tone",
        "trailer voice",
        "random emotional jumps",
        "unnatural speed changes"
    ],
    "voice_consistency": {
        "keep_one_selected_voice_across_episode": True,
        "male_voice_only": True,
        "language": "en-US"
    },
    "narrative_principle": "The voice should never compete with the story. It should make the story impossible to ignore."
}

VOICE_BIBLE_VERSION = "v1"

# قائمة الانفعالات المعتمدة لمسار Google (محفوظة لكن غير مستخدمة)
APPROVED_GOOGLE_EMOTIONS = {
    "admiration", "amusement", "anger", "anxiety", "apology", "approval", "awe",
    "aggression", "boredom", "calm", "celebration", "concern", "contempt", "contentment",
    "curiosity", "determination", "disapproval", "disbelief", "disgust", "embarrassment",
    "empathy", "enthusiasm", "excitement", "fear", "frustration", "gratitude", "hope",
    "humor", "interest", "joy", "longing", "love", "nervousness", "nostalgia", "pride",
    "relief", "sadness", "sarcasm", "satisfaction", "shock", "surprise", "suspense",
    "sympathy", "tenderness", "tiredness", "trust", "uncertainty", "urgency", "vulnerability",
    "warning", "whispers", "wonder", "gasps", "cries", "sighs", "laughs"
}

# الأنماط المعتمدة في Azure (9 أنماط فقط)
APPROVED_AZURE_STYLES = {
    "whispering", "excited", "serious", "hopeful",
    "cheerful", "sad", "angry", "shouting", "calm"
}

# الأدوار السردية المعتمدة
APPROVED_NARRATIVE_ROLES = {
    "hook", "setup", "question", "myth", "contradiction", "explanation", "analogy",
    "transition", "tension", "revelation", "payoff", "actionable", "reflection", "cta"
}

# أنماط الأداء (delivery_mode) المعتمدة
APPROVED_DELIVERY_MODES = {
    "calm", "curious", "serious", "teaching", "suspense",
    "revelation", "encouraging", "reflective", "energetic"
}

# الحدود الرقمية
_LIMITS = {
    "energy": (1, 5),
    "rate_percent": (-12, 8),
    "pitch_percent": (-5, 5),
    "pause_before_ms": (0, 700),
    "pause_after_ms": (0, 1000),
}

MAX_DIRECTOR_RETRIES = 3
MAX_CLIP_RETRIES = 3
MAX_MERGE_GAP_MS = 1500

# هل نحذف النسخ الاحتياطية بعد نجاح الاعتماد و Final QA؟
# False = الاحتفاظ بها كخط رجوع يدوي آمن (الافتراضي).
# True  = حذفها تلقائياً بعد التحقق الكامل.
CLEANUP_BACKUPS_AFTER_SUCCESS = False

# حقول سياق المرحلة الأولى المسموح بتمريرها إلى Voice Director
VOICE_CONTEXT_FIELDS = ("creative_brief", "retention_plan", "scene_plan", "sections")

# حد آمن لحجم السياق المرسل إلى Gemini
MAX_VOICE_CONTEXT_CHARS = 6000


# =============================================================
# برومبتات النظام
# =============================================================

STAGE_3_VOICE_DIRECTOR_SYSTEM_PROMPT = """You are the AI VOICE DIRECTOR for a cinematic science narration channel called VOT.

Your ONLY job is to design a delivery plan for an existing script, sentence by sentence.
You do NOT rewrite the script. You do NOT add words. You do NOT remove words. You do NOT paraphrase.

================ VOT VOICE BIBLE (LOCKED, NEVER CHANGE) ================
Identity: calm scientific cinematic narrator
Personality: confident, intelligent, warm, controlled, slightly mysterious, encouraging
Default delivery: clear, grounded, believable, never theatrical
NEVER: constant overacting, advertisement tone, trailer voice, random emotional jumps, unnatural speed changes
Voice consistency: one male en-US voice kept across the entire episode
Narrative principle: The voice should never compete with the story. It should make the story impossible to ignore.
=======================================================================

HARD RULES:
1. Return EXACTLY one direction object per input sentence. No more, no less.
2. sentence_index starts at 1 and increments by 1 with no gaps and no duplicates.
3. Do NOT rewrite, add, or remove any word of the sentences.
4. Delivery must vary narratively, NOT randomly. Adjacent sentences inside the same idea keep continuity.
5. Do NOT make every sentence dramatic. Default to calm / serious / hopeful.
6. "shouting" and "angry" are used VERY rarely and only when truly justified.
7. "whispering" is used only when it genuinely serves the meaning.
8. Emojis are strictly forbidden anywhere in the output.
9. Output PURE JSON only. No markdown, no commentary, no extra text.

ALLOWED VALUES:
- narrative_role: hook, setup, question, myth, contradiction, explanation, analogy, transition, tension, revelation, payoff, actionable, reflection, cta
- delivery_mode: calm, curious, serious, teaching, suspense, revelation, encouraging, reflective, energetic
- azure_style: whispering, excited, serious, hopeful, cheerful, sad, angry, shouting, calm

NUMERIC RANGES (integers):
- energy: 1 to 5
- rate_percent: -12 to 8
- pitch_percent: -5 to 5
- pause_before_ms: 0 to 700
- pause_after_ms: 0 to 1000

EMPHASIS:
- emphasis_words: array of words that MUST exist verbatim inside the sentence. Use 0 to 2 words. Prefer [] over guessing.

OUTPUT JSON SCHEMA:
{
  "directions": [
    {
      "sentence_index": 1,
      "narrative_role": "hook",
      "delivery_mode": "curious",
      "azure_style": "serious",
      "energy": 3,
      "rate_percent": -4,
      "pitch_percent": -1,
      "pause_before_ms": 120,
      "pause_after_ms": 350,
      "emphasis_words": ["important"],
      "breath_breaks": [],
      "reason": "brief internal direction"
    }
  ]
}
"""


# =============================================================
# أدوات مساعدة عامة
# =============================================================

def _clean_text_for_word_match(text: str) -> str:
    """تنظيف النص من الوسوم والإيموجي لمطابقة الكلمات حرفياً"""
    no_tags = re.sub(r"\[.*?\]", "", text)
    no_xml = re.sub(r"<.*?>", "", no_tags)
    clean = re.sub(r"[^\w\s]", "", no_xml).lower()
    return " ".join(clean.split())


def _find_express_elements(root):
    """يعثر على عناصر mstts:express-as مهما كانت namespace"""
    for ns in ("https://www.w3.org/2001/mstts", "http://www.w3.org/2001/mstts"):
        elems = root.findall(f".//{{{ns}}}express-as")
        if elems:
            return elems
    try:
        return root.findall(".//{*}express-as")
    except Exception:
        return []


# =============================================================
# التحقق من SSML الأحادي (يستخدم قبل الإرسال لكل Clip)
# =============================================================

def _validate_single_ssml(ssml_text: str, expected_voice: Optional[str] = None) -> None:
    """تحقق بنيوي سريع من صحة SSML جملة واحدة"""
    if not ssml_text or "<speak" not in ssml_text:
        raise ValueError("SSML غير صالح: لا يحتوي على <speak>")

    emojis = re.findall(r"[\U00010000-\U0010ffff]", ssml_text)
    if emojis:
        raise ValueError(f"SSML يحتوي على إيموجي: {emojis}")

    try:
        root = ET.fromstring(ssml_text)
    except ET.ParseError as err:
        raise ValueError(f"SSML غير صالح بنيوياً (XML Parse Error): {err}")

    voice_elem = root.find(".//{http://www.w3.org/2001/10/synthesis}voice")
    if voice_elem is None:
        voice_elem = root.find(".//voice")
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


def validate_azure_ssml(ssml_text: str, original_sentences: List[str], expected_voice: str):
    """فحص شامل لـSSML متعدد الجمل (محفوظ للتوافق مع الاستدعاءات القديمة)"""
    emojis = re.findall(r"[\U00010000-\U0010ffff]", ssml_text)
    if emojis:
        raise ValueError(f"مخالفة صارمة: مسار Azure يحتوي على رموز تعبيرية: {emojis}")

    try:
        root = ET.fromstring(ssml_text)
    except ET.ParseError as err:
        raise ValueError(f"كود SSML غير صالح بنيوياً: {err}")

    voice_elem = root.find(".//{http://www.w3.org/2001/10/synthesis}voice") or root.find(".//voice")
    if voice_elem is not None:
        voice_name = voice_elem.attrib.get("name", "")
        if voice_name != expected_voice:
            logger.warning(f"⚠️ الصوت داخل SSML '{voice_name}' لا يطابق المطلوب '{expected_voice}'")

    express_elements = _find_express_elements(root)
    if not express_elements:
        raise ValueError("SSML لا يحتوي على عنصر mstts:express-as واحد على الأقل!")

    for idx, elem in enumerate(express_elements, start=1):
        style = elem.attrib.get("style", "").strip().lower()
        if style not in APPROVED_AZURE_STYLES:
            raise ValueError(f"الجملة {idx}: النمط '{style}' غير معتمد!")

    logger.info("✅ فحص مسار Azure SSML: XML سليم، بلا إيموجي، والأنماط معتمدة.")


# =============================================================
# سياق المرحلة الأولى (READ-ONLY) لاستخدام Voice Director
# =============================================================

def _build_voice_context(
    episode_context: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    استخراج آمن لحقول سياق المرحلة الأولى فقط:
      creative_brief / retention_plan / scene_plan / sections
    """
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
        except Exception:
            logger.warning(f"⚠️ حقل السياق '{field}' غير قابل للتسلسل — سيتم تجاهله.")
            continue

        if len(serialized) > MAX_VOICE_CONTEXT_CHARS:
            logger.warning(
                f"⚠️ حقل السياق '{field}' تم تجاهله (تجاوز الحد الآمن "
                f"{MAX_VOICE_CONTEXT_CHARS} حرف)."
            )
            continue

        extracted = candidate

    return extracted


# =============================================================
# AI VOICE DIRECTOR
# =============================================================

def _call_voice_director(
    sentences: List[str],
    episode_context: Optional[Dict[str, Any]] = None,
    previous_error: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """استدعاء Gemini كمخرج صوتي، يُعيد قائمة directions فقط."""
    user_lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    safe_context = _build_voice_context(episode_context)

    context_block = ""
    if safe_context:
        try:
            ctx_str = json.dumps(safe_context, ensure_ascii=False)
            context_block = (
                "\n\nEPISODE CONTEXT (READ-ONLY):\n"
                "This context is provided ONLY to help you understand the narrative arc.\n"
                "HARD RULES FOR CONTEXT USAGE:\n"
                "- NEVER modify, rewrite, add, remove, or reorder any sentence.\n"
                "- NEVER return substitute text for the sentences; you output a directions array ONLY.\n"
                "- Use creative_brief to understand the core idea and angle.\n"
                "- Use retention_plan to identify hook, open loops, revelation, payoff.\n"
                "- Use scene_plan to understand scene progression and emotional build-up.\n"
                "- Use sections to understand the narrative function of each sentence.\n\n"
                "CONTEXT JSON:\n" + ctx_str
            )
        except Exception as e:
            logger.warning(f"⚠️ تعذر تحويل سياق المرحلة الأولى إلى JSON: {e}")
            context_block = ""

    error_block = ""
    if previous_error:
        error_block = (
            "\n\nPREVIOUS ATTEMPT FAILED WITH ERROR:\n"
            f"{previous_error}\n"
            "Fix this and return valid JSON only."
        )

    user_prompt = (
        f"Sentences to direct ({len(sentences)} total):\n"
        f"{user_lines}"
        f"{context_block}"
        f"{error_block}\n\n"
        f'Return ONLY the JSON object with a "directions" array of exactly {len(sentences)} items.'
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
        raise ValueError(f"JSON غير قابل للتحليل من Voice Director: {e} | raw[:200]={cleaned[:200]}")

    if isinstance(data, dict) and "directions" in data:
        directions = data["directions"]
    elif isinstance(data, list):
        directions = data
    else:
        raise ValueError(f"صيغة JSON غير متوقعة من Voice Director: {type(data)}")

    if not isinstance(directions, list):
        raise ValueError("directions ليست قائمة!")

    return directions


def generate_voice_direction_plan(
    sentences: List[str],
    episode_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    توليد خطة أداء صوتي منظمة (Voice Director).
    """
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("قائمة الجمل فارغة أو غير صالحة!")

    try:
        safe_context = _build_voice_context(episode_context)
    except Exception as e:
        logger.warning(f"⚠️ تعذر استخراج سياق المرحلة الأولى: {e} — سيتم الاستمرار بدون سياق.")
        safe_context = {}

    last_error: Optional[str] = None

    for attempt in range(1, MAX_DIRECTOR_RETRIES + 1):
        try:
            directions = _call_voice_director(
                sentences, safe_context, previous_error=last_error
            )
            validate_voice_direction_plan(directions, sentences)
            logger.info(f"✅ خطة الأداء الصوتي معتمدة بعد المحاولة {attempt}.")
            return directions
        except Exception as e:
            last_error = str(e)
            logger.warning(
                f"⚠️ محاولة {attempt}/{MAX_DIRECTOR_RETRIES} لتوليد خطة الأداء فشلت: {last_error}"
            )

    logger.error(
        "❌ فشل توليد خطة الأداء بعد الحد الأقصى. سيتم استخدام خطة fallback حتمية "
        "مبنية على موضع الجملة داخل الحلقة، بدون أي تغيير للنص."
    )
    return _build_fallback_plan(sentences)


def _build_fallback_plan(sentences: List[str]) -> List[Dict[str, Any]]:
    """خطة أداء حتمية احتياطية مبنية على موضع الجملة داخل الحلقة."""
    total = len(sentences)
    if total <= 0:
        return []

    plan: List[Dict[str, Any]] = []

    for i in range(1, total + 1):
        ratio = i / total

        if i <= 3:
            role, mode, style, energy = "hook", "curious", "serious", 4
        elif ratio <= 0.20:
            role, mode, style, energy = "setup", "teaching", "calm", 3
        elif ratio <= 0.50:
            role, mode, style, energy = "explanation", "teaching", "calm", 3
        elif ratio <= 0.75:
            role, mode, style, energy = "revelation", "serious", "serious", 4
        elif ratio <= 0.90:
            role, mode, style, energy = "actionable", "encouraging", "hopeful", 4
        else:
            role, mode, style, energy = "cta", "encouraging", "hopeful", 3

        plan.append({
            "sentence_index": i,
            "narrative_role": role,
            "delivery_mode": mode,
            "azure_style": style,
            "energy": energy,
            "rate_percent": 0,
            "pitch_percent": 0,
            "pause_before_ms": 80,
            "pause_after_ms": 200,
            "emphasis_words": [],
            "breath_breaks": [],
            "reason": "fallback deterministic plan (Voice Director failed)",
        })

    return plan


# =============================================================
# التحقق الصارم من خطة الأداء
# =============================================================

def validate_voice_direction_plan(
    directions: List[Dict[str, Any]],
    sentences: List[str],
) -> None:
    if not isinstance(directions, list):
        raise ValueError("خطة الأداء يجب أن تكون قائمة!")
    if len(directions) != len(sentences):
        raise ValueError(
            f"عدد التعليمات ({len(directions)}) لا يطابق عدد الجمل ({len(sentences)})!"
        )

    required_fields = (
        "sentence_index", "narrative_role", "delivery_mode", "azure_style",
        "energy", "rate_percent", "pitch_percent",
        "pause_before_ms", "pause_after_ms",
    )

    for i, d in enumerate(directions, start=1):
        if not isinstance(d, dict):
            raise ValueError(f"العنصر {i} ليس قاموسًا!")

        si = d.get("sentence_index")
        if si != i:
            raise ValueError(f"العنصر {i}: sentence_index يجب أن يكون {i} وليس {si}")

        for field in required_fields:
            if field not in d:
                raise ValueError(f"العنصر {i}: الحقل المطلوب '{field}' مفقود!")

        if d["narrative_role"] not in APPROVED_NARRATIVE_ROLES:
            raise ValueError(f"العنصر {i}: narrative_role '{d['narrative_role']}' غير مسموح!")
        if d["delivery_mode"] not in APPROVED_DELIVERY_MODES:
            raise ValueError(f"العنصر {i}: delivery_mode '{d['delivery_mode']}' غير مسموح!")
        if d["azure_style"] not in APPROVED_AZURE_STYLES:
            raise ValueError(f"العنصر {i}: azure_style '{d['azure_style']}' غير مسموح!")

        # energy
        energy = d["energy"]
        if not isinstance(energy, int) or isinstance(energy, bool):
            raise ValueError(f"العنصر {i}: energy يجب أن يكون عددًا صحيحًا!")
        if not (_LIMITS["energy"][0] <= energy <= _LIMITS["energy"][1]):
            raise ValueError(f"العنصر {i}: energy={energy} خارج النطاق {_LIMITS['energy']}")

        # rate
        rate = d["rate_percent"]
        if not isinstance(rate, (int, float)) or isinstance(rate, bool):
            raise ValueError(f"العنصر {i}: rate_percent يجب أن يكون رقميًا!")
        if not (_LIMITS["rate_percent"][0] <= rate <= _LIMITS["rate_percent"][1]):
            raise ValueError(f"العنصر {i}: rate_percent={rate} خارج النطاق {_LIMITS['rate_percent']}")

        # pitch
        pitch = d["pitch_percent"]
        if not isinstance(pitch, (int, float)) or isinstance(pitch, bool):
            raise ValueError(f"العنصر {i}: pitch_percent يجب أن يكون رقميًا!")
        if not (_LIMITS["pitch_percent"][0] <= pitch <= _LIMITS["pitch_percent"][1]):
            raise ValueError(f"العنصر {i}: pitch_percent={pitch} خارج النطاق {_LIMITS['pitch_percent']}")

        # pause before
        pb = d["pause_before_ms"]
        if not isinstance(pb, (int, float)) or isinstance(pb, bool):
            raise ValueError(f"العنصر {i}: pause_before_ms يجب أن يكون رقميًا!")
        if not (_LIMITS["pause_before_ms"][0] <= pb <= _LIMITS["pause_before_ms"][1]):
            raise ValueError(f"العنصر {i}: pause_before_ms={pb} خارج النطاق {_LIMITS['pause_before_ms']}")

        # pause after
        pa = d["pause_after_ms"]
        if not isinstance(pa, (int, float)) or isinstance(pa, bool):
            raise ValueError(f"العنصر {i}: pause_after_ms يجب أن يكون رقميًا!")
        if not (_LIMITS["pause_after_ms"][0] <= pa <= _LIMITS["pause_after_ms"][1]):
            raise ValueError(f"العنصر {i}: pause_after_ms={pa} خارج النطاق {_LIMITS['pause_after_ms']}")

        # emphasis_words
        ew = d.get("emphasis_words", [])
        if ew is None:
            d["emphasis_words"] = []
            ew = []
        if not isinstance(ew, list):
            raise ValueError(f"العنصر {i}: emphasis_words يجب أن تكون قائمة!")
        filtered: List[str] = []
        for w in ew:
            if not isinstance(w, str) or not w.strip():
                continue
            pattern = re.compile(r"(?<!\w)" + re.escape(w.strip()) + r"(?!\w)", re.IGNORECASE)
            if pattern.search(sentences[i - 1]):
                filtered.append(w.strip())
            else:
                logger.warning(
                    f"⚠️ العنصر {i}: emphasis_word '{w}' غير موجود في الجملة، سيتم تجاهله."
                )
        d["emphasis_words"] = filtered

        # breath_breaks
        bb = d.get("breath_breaks", [])
        if bb is None:
            d["breath_breaks"] = []
        elif not isinstance(bb, list):
            raise ValueError(f"العنصر {i}: breath_breaks يجب أن تكون قائمة!")

    total = len(directions)
    rare_count = sum(
        1 for d in directions if d["azure_style"] in ("shouting", "angry", "whispering")
    )
    if total >= 5 and rare_count > max(1, total // 5):
        raise ValueError(
            f"خطة الأداء تحتوي على {rare_count} استخدام للأنماط النادرة من أصل {total} جملة، وهذا مبالغ فيه!"
        )

    rate_values = [d["rate_percent"] for d in directions]
    if len(rate_values) >= 6:
        if all(r >= 6 for r in rate_values) or all(r <= -10 for r in rate_values):
            raise ValueError("خطة الأداء تسرّع/تُبطئ كل الجمل بشكل متطرف!")


# =============================================================
# بناء SSML على مستوى الجملة
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
    """بناء SSML لجملة واحدة مع الحفاظ الحرفي على النص وXML Escaping."""
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("الجملة فارغة أو غير صالحة لبناء SSML!")

    style = str(direction.get("azure_style", "calm")).strip().lower()
    if style not in APPROVED_AZURE_STYLES:
        raise ValueError(f"azure_style '{style}' غير معتمد!")

    rate = _clamp(direction.get("rate_percent", 0), -12, 8, 0)
    pitch = _clamp(direction.get("pitch_percent", 0), -5, 5, 0)

    emphasis_words = direction.get("emphasis_words", []) or []
    if not isinstance(emphasis_words, list):
        emphasis_words = []

    working = sentence
    placeholders: List[str] = []

    for w in emphasis_words:
        if not isinstance(w, str) or not w.strip():
            continue
        word = w.strip()
        pattern = re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)", re.IGNORECASE)
        def _repl(m, _word=word):
            idx = len(placeholders)
            placeholders.append(m.group(0))
            return f"\x00EMPH{idx}\x00"
        working, n = pattern.subn(_repl, working, count=1)
        if n == 0:
            continue

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
        f'<mstts:express-as style="{style}">\n'
        f'<prosody rate="{rate_str}" pitch="{pitch_str}">\n'
        f'{escaped}\n'
        '</prosody>\n'
        '</mstts:express-as>\n'
        '</voice>\n'
        '</speak>'
    )
    return ssml


# =============================================================
# Azure TTS API
# =============================================================

def call_azure_tts_api(ssml_payload: str, output_filepath: Path):
    """استدعاء REST API لمحرك Microsoft Azure Speech"""
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

    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "wb") as f:
        f.write(response.content)


# =============================================================
# دمج الصوت
# =============================================================

def _estimate_mp3_duration_ms(file_path: Path, bitrate_kbps: int = 160) -> int:
    try:
        size = file_path.stat().st_size
    except OSError:
        return 0
    if size <= 0:
        return 0
    return int(size / (bitrate_kbps * 1000 / 8 / 1000))


def _get_mp3_duration_ms(file_path: Path) -> Optional[int]:
    if not PYDUB_AVAILABLE:
        return None
    try:
        seg = AudioSegment.from_mp3(str(file_path))
        return int(len(seg))
    except Exception as e:
        logger.warning(f"⚠️ تعذر قياس مدة {file_path.name} عبر pydub: {e}")
        return None


def _merge_clips_with_pauses(
    clips_meta: List[Dict[str, Any]],
    clips_dir: Path,
    output_file: Path,
) -> None:
    """
    دمج الـClips بالترتيب مع احترام الفواصل الزمنية.

    - يتحقق أولاً من توفر مكتبة pydub (وإلا يرفض العمل برسالة تثبيت واضحة).
    - يتحقق من وجود كل ملف صوت وغير فارغ قبل بدء الدمج.
    - عند أي فشل: يقوم بـRollback ويحذف أي ملف ناتج ناقص/جزئي.
    """

    # ============ 1) فحص اعتمادية pydub ============
    if not PYDUB_AVAILABLE or AudioSegment is None:
        detail = PYDUB_IMPORT_ERROR or "سبب غير معروف"
        raise RuntimeError(
            "❌ [Dependency Missing] مكتبة pydub غير مثبتة أو غير قابلة للاستيراد، "
            "ولا يمكن دمج مقاطع الصوت دونها.\n"
            f"   التفاصيل: {detail}\n"
            "   هذا ليس خطأً من Azure Speech — بل مشكلة اعتمادية ناقصة.\n"
            "   للتثبيت، نفّذ الأمر التالي:\n"
            "       python -m pip install pydub\n"
            "   ملاحظة: pydub تحتاج أيضاً إلى ffmpeg على PATH لفك ترميز MP3."
        )

    # ============ 2) فحص المدخلات ============
    if not clips_meta:
        raise RuntimeError("لا توجد Clips للدمج!")

    if not clips_dir.exists() or not clips_dir.is_dir():
        raise RuntimeError(f"مجلد الـClips غير موجود أو ليس مجلداً: {clips_dir}")

    # ============ 3) فحص وجود وحجم كل ملف صوت قبل بدء الدمج ============
    missing: List[str] = []
    empty: List[str] = []
    for meta in clips_meta:
        clip_name = meta.get("filename")
        if not clip_name:
            raise RuntimeError(f"عنصر Clip بدون filename في Metadata: {meta}")
        clip_path = clips_dir / clip_name
        if not clip_path.exists() or not clip_path.is_file():
            missing.append(clip_name)
            continue
        if clip_path.stat().st_size == 0:
            empty.append(clip_name)

    if missing:
        raise RuntimeError(
            "❌ ملفات Clips مفقودة قبل الدمج، لا يمكن المتابعة:\n- " + "\n- ".join(missing)
        )
    if empty:
        raise RuntimeError(
            "❌ ملفات Clips فارغة قبل الدمج، لا يمكن المتابعة:\n- " + "\n- ".join(empty)
        )

    # ============ 4) الدمج الفعلي مع Rollback عند أي فشل ============
    output_file.parent.mkdir(parents=True, exist_ok=True)

    combined = AudioSegment.empty()
    prev_meta: Optional[Dict[str, Any]] = None

    try:
        for meta in clips_meta:
            clip_path = clips_dir / meta["filename"]

            try:
                segment = AudioSegment.from_mp3(str(clip_path))
            except Exception as read_err:
                raise RuntimeError(
                    f"فشل قراءة ملف الصوت '{clip_path.name}': {read_err}. "
                    "قد يكون الملف تالفاً أو أن ffmpeg غير مثبت/غير متاح على PATH."
                )

            if segment is None or len(segment) == 0:
                raise RuntimeError(
                    f"ملف الصوت '{clip_path.name}' مقروء لكن مدته صفر — غير صالح للدمج."
                )

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

        if len(combined) == 0:
            raise RuntimeError("ناتج الدمج فارغ — لا يمكن اعتماده كملف نهائي.")

        combined.export(str(output_file), format="mp3")

        if not output_file.exists() or output_file.stat().st_size == 0:
            raise RuntimeError(
                f"فشل التصدير: الملف الناتج مفقود أو فارغ بعد الدمج: {output_file}"
            )

    except Exception as merge_err:
        # ============ 5) Rollback: حذف أي ملف ناتج ناقص/جزئي ============
        if output_file.exists():
            try:
                output_file.unlink()
                logger.warning(
                    f"🧹 تم حذف الملف الصوتي الناتج غير المكتمل بعد فشل الدمج: "
                    f"{output_file.name}"
                )
            except Exception as cleanup_err:
                logger.error(
                    f"❌ فشل حذف الملف الناتج غير المكتمل '{output_file}' — "
                    f"يتطلب تنظيفاً يدوياً: {cleanup_err}"
                )

        if isinstance(merge_err, RuntimeError):
            raise
        raise RuntimeError(
            f"فشل دمج مقاطع الصوت (Dependency/Merge Error، وليس من Azure): {merge_err}"
        ) from merge_err

    logger.info(f"🎧 تم دمج {len(clips_meta)} Clips في الملف: {output_file}")


# =============================================================
# Audio QA
# =============================================================

def validate_audio_outputs(
    sentences: List[str],
    clips_dir: Path,
    final_audio_file: Path,
    metadata_file: Path,
    expected_voice: Optional[str] = None,
) -> None:
    """تحقق شامل وصارم من مخرجات المرحلة الثالثة"""
    if not clips_dir.exists() or not clips_dir.is_dir():
        raise ValueError(f"مجلد الـClips غير موجود: {clips_dir}")

    if not final_audio_file.exists() or final_audio_file.stat().st_size == 0:
        raise ValueError(f"الملف الصوتي النهائي مفقود أو فارغ: {final_audio_file}")

    if not metadata_file.exists() or metadata_file.stat().st_size == 0:
        raise ValueError(f"ملف الـMetadata مفقود أو فارغ: {metadata_file}")

    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        raise ValueError(f"تعذر قراءة ملف الـMetadata: {e}")

    if not isinstance(meta, dict):
        raise ValueError("بنية Metadata ليست قاموس JSON صالحاً!")

    if meta.get("sentence_count") != len(sentences):
        raise ValueError(
            f"عدد الجمل في Metadata ({meta.get('sentence_count')}) لا يطابق الأصل ({len(sentences)})"
        )

    if meta.get("language") != "en-US":
        raise ValueError(f"حقل language غير مطابق: {meta.get('language')}")

    if not meta.get("voice_bible_version"):
        raise ValueError("حقل voice_bible_version مفقود من Metadata!")

    if expected_voice is not None and meta.get("voice_name") != expected_voice:
        raise ValueError(
            f"الصوت في Metadata '{meta.get('voice_name')}' لا يطابق المتوقع '{expected_voice}'"
        )

    clips = meta.get("clips", [])
    if not isinstance(clips, list) or len(clips) != len(sentences):
        raise ValueError(
            f"عدد الـClips في Metadata ({len(clips) if isinstance(clips, list) else 'غير صالح'}) لا يطابق عدد الجمل ({len(sentences)})"
        )

    required_clip_fields = (
        "sentence_index", "filename", "sentence_text", "sentence_hash",
        "narrative_role", "delivery_mode", "azure_style", "energy",
        "rate_percent", "pitch_percent", "pause_before_ms", "pause_after_ms",
        "emphasis_words", "breath_breaks", "reason", "duration_ms"
    )

    seen_files = set()
    for i, c in enumerate(clips, start=1):
        if not isinstance(c, dict):
            raise ValueError(f"الـClip رقم {i} ليس قاموساً صالحاً!")

        if c.get("sentence_index") != i:
            raise ValueError(f"الـClip رقم {i}: الفهرسة غير متسلسلة (القيمة={c.get('sentence_index')})")

        for fld in required_clip_fields:
            if fld not in c:
                raise ValueError(f"الـClip {i}: الحقل المطلوب '{fld}' مفقود في Metadata!")

        fn = c.get("filename")
        if not fn or fn != f"sentence_{i:04d}.mp3":
            raise ValueError(f"الـClip {i}: اسم الملف غير صحيح أو غير متسلسل: {fn}")

        if fn in seen_files:
            raise ValueError(f"اسم ملف Clip مكرر: {fn}")
        seen_files.add(fn)

        p = clips_dir / fn
        if not p.exists() or p.stat().st_size == 0:
            raise ValueError(f"ملف الـClip مفقود أو فارغ على القرص: {p}")

        original_text = sentences[i - 1]
        if c.get("sentence_text") != original_text:
            raise ValueError(
                f"الـClip {i}: النص المحفوظ لا يطابق الجملة الأصلية حرفياً!\n"
                f"الأصل: {original_text}\nالمحفوظ: {c.get('sentence_text')}"
            )

        expected_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
        if c.get("sentence_hash") != expected_hash:
            raise ValueError(
                f"الـClip {i}: Hash الجملة غير صحيح! المتوقع={expected_hash}, المسجل={c.get('sentence_hash')}"
            )

        if not isinstance(c.get("emphasis_words"), list):
            raise ValueError(f"الـClip {i}: emphasis_words ليست قائمة!")

        if not isinstance(c.get("breath_breaks"), list):
            raise ValueError(f"الـClip {i}: breath_breaks ليست قائمة!")

        if c.get("duration_ms") is None or not isinstance(c.get("duration_ms"), (int, float)):
            raise ValueError(f"الـClip {i}: duration_ms غير محدد أو غير رقمي!")

    actual_files = set(f.name for f in clips_dir.glob("*.mp3"))
    if actual_files != seen_files:
        extra_files = actual_files - seen_files
        raise ValueError(f"تم اكتشاف ملفات clips إضافية أو مختلطة داخل المجلد: {extra_files}")

    logger.info("✅ Audio QA: جميع الـClips والـMetadata متوافقة تماماً وتطابق النصوص والـHashes.")


# =============================================================
# أدوات التنظيف والتحقق من الـRollback
# =============================================================

def _safe_cleanup_staging(
    staging_clips_dir: Path,
    staging_audio_file: Path,
    staging_metadata_file: Path,
) -> None:
    """
    تنظيف آمن لمسارات Staging دون إخفاء الأخطاء.
    يُسجّل تحذيراً صريحاً عند فشل أي حذف، ولا يرمي استثناءً.
    """
    if staging_clips_dir.exists():
        try:
            shutil.rmtree(staging_clips_dir)
            logger.info(f"🧹 تم حذف مجلد Staging: {staging_clips_dir.name}")
        except Exception as e:
            logger.warning(
                f"⚠️ فشل حذف مجلد Staging '{staging_clips_dir}' — "
                f"يتطلب تنظيفاً يدوياً: {e}"
            )

    if staging_audio_file.exists():
        try:
            staging_audio_file.unlink()
            logger.info(f"🧹 تم حذف ملف Staging: {staging_audio_file.name}")
        except Exception as e:
            logger.warning(
                f"⚠️ فشل حذف ملف Staging '{staging_audio_file}' — "
                f"يتطلب تنظيفاً يدوياً: {e}"
            )

    if staging_metadata_file.exists():
        try:
            staging_metadata_file.unlink()
            logger.info(f"🧹 تم حذف ملف Staging: {staging_metadata_file.name}")
        except Exception as e:
            logger.warning(
                f"⚠️ فشل حذف ملف Staging '{staging_metadata_file}' — "
                f"يتطلب تنظيفاً يدوياً: {e}"
            )


def _verify_rollback(
    clips_dir: Path,
    final_audio_file: Path,
    metadata_file: Path,
    had_clips: bool,
    had_audio: bool,
    had_metadata: bool,
) -> None:
    """
    التحقق الصريح من نجاح الاستعادة بعد الـRollback.
    يرمي RuntimeError إذا فشلت استعادة أي عنصر كان موجوداً سابقاً.
    """
    errors: List[str] = []

    if had_clips:
        if not clips_dir.exists() or not clips_dir.is_dir():
            errors.append(f"مجلد Clips القديم لم يُستعد: {clips_dir}")
        else:
            clips_count = len(list(clips_dir.glob("*.mp3")))
            if clips_count == 0:
                errors.append(f"مجلد Clips القديم عاد فارغاً: {clips_dir}")

    if had_audio:
        if not final_audio_file.exists() or final_audio_file.stat().st_size == 0:
            errors.append(f"ملف الصوت القديم لم يُستعد أو فارغ: {final_audio_file}")

    if had_metadata:
        if not metadata_file.exists() or metadata_file.stat().st_size == 0:
            errors.append(f"ملف Metadata القديم لم يُستعد أو فارغ: {metadata_file}")

    if errors:
        raise RuntimeError(
            "❌ فشل التحقق من نجاح الـRollback — الحالة السابقة غير مضمونة:\n- "
            + "\n- ".join(errors)
        )

    logger.info("✅ تم التحقق من نجاح الـRollback واستعادة الحالة السابقة بالكامل.")


def _rollback_committed_to_backups(
    *,
    clips_dir: Path,
    final_audio_file: Path,
    metadata_file: Path,
    staging_clips_dir: Path,
    staging_audio_file: Path,
    staging_metadata_file: Path,
    backup_clips_dir: Optional[Path],
    backup_audio_file: Optional[Path],
    backup_metadata_file: Optional[Path],
    had_clips: bool,
    had_audio: bool,
    had_metadata: bool,
) -> None:
    """
    تراجع كامل موحّد:
      1) إعادة الملفات المعتمدة حديثاً (المسارات النهائية) إلى Staging.
      2) استعادة النسخ القديمة من الـbackup إلى المسارات النهائية.
      3) التحقق الصريح من نجاح الاستعادة، ورمي RuntimeError عند أي فشل.
    يُستخدم في:
      - فشل مرحلة الاعتماد (Staging → Final).
      - فشل Final QA بعد الاعتماد.
    """
    # 1) إعادة الملفات الجديدة إلى Staging
    if clips_dir.exists() and not staging_clips_dir.exists():
        try:
            clips_dir.rename(staging_clips_dir)
        except Exception as e:
            logger.error(f"❌ فشل إرجاع clips الجديد إلى Staging: {e}")

    if final_audio_file.exists() and not staging_audio_file.exists():
        try:
            final_audio_file.replace(staging_audio_file)
        except Exception as e:
            logger.error(f"❌ فشل إرجاع الصوت الجديد إلى Staging: {e}")

    if metadata_file.exists() and not staging_metadata_file.exists():
        try:
            metadata_file.replace(staging_metadata_file)
        except Exception as e:
            logger.error(f"❌ فشل إرجاع metadata الجديد إلى Staging: {e}")

    # 2) استعادة النسخ القديمة من الـbackup
    if backup_clips_dir and backup_clips_dir.exists() and not clips_dir.exists():
        try:
            backup_clips_dir.rename(clips_dir)
        except Exception as e:
            logger.error(f"❌ فشل استعادة مجلد clips القديم: {e}")

    if backup_audio_file and backup_audio_file.exists() and not final_audio_file.exists():
        try:
            backup_audio_file.rename(final_audio_file)
        except Exception as e:
            logger.error(f"❌ فشل استعادة ملف الصوت القديم: {e}")

    if backup_metadata_file and backup_metadata_file.exists() and not metadata_file.exists():
        try:
            backup_metadata_file.rename(metadata_file)
        except Exception as e:
            logger.error(f"❌ فشل استعادة ملف metadata القديم: {e}")

    # 3) تحقق صريح من نجاح الاستعادة
    _verify_rollback(
        clips_dir=clips_dir,
        final_audio_file=final_audio_file,
        metadata_file=metadata_file,
        had_clips=had_clips,
        had_audio=had_audio,
        had_metadata=had_metadata,
    )


# =============================================================
# الدالة التنفيذية الشاملة للمرحلة الثالثة
# =============================================================

def generate_stage3_audio(
    episode_id: str,
    sentences: List[str],
    engine: str = "azure",
    voice: str = "en-US-GuyNeural",
    output_dir: Path = Path("outputs"),
    episode_context: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    المرحلة الثالثة (Azure فقط) — نظام إخراج صوتي من 4 طبقات:
      1) VOT Voice Bible
      2) AI Voice Director (خطة أداء من Gemini، JSON فقط)
      3) Sentence-Level Azure SSML (Clip معزول لكل جملة عبر Staging)
      4) Merge + Metadata + Audio QA + Safe Atomic Commit & Rollback + Final QA
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    final_audio_file = output_dir / f"episode_{episode_id}_audio.mp3"
    clips_dir = output_dir / f"episode_{episode_id}_clips"
    metadata_file = output_dir / f"episode_{episode_id}_audio_metadata.json"

    engine = engine.lower().strip()
    if engine != "azure":
        raise ValueError(f"محرك الصوت '{engine}' غير مدعوم! المحرك المتاح حالياً: 'azure' فقط.")

    if voice not in AZURE_MALE_VOICES:
        raise ValueError(
            f"الصوت '{voice}' غير مسموح به! الأصوات المعتمدة حصراً لـ Azure هي: {AZURE_MALE_VOICES}"
        )

    if not isinstance(sentences, list) or not sentences:
        raise ValueError("قائمة الجمل فارغة أو غير صالحة!")

    # --- فحص اعتمادية pydub قبل أي استدعاء Azure (Fail-Fast) ---
    if not PYDUB_AVAILABLE or AudioSegment is None:
        detail = PYDUB_IMPORT_ERROR or "سبب غير معروف"
        raise RuntimeError(
            "❌ لا يمكن بدء المرحلة الثالثة: مكتبة pydub غير مثبتة أو غير قابلة للاستيراد "
            "(Dependency Missing)، وهي ضرورية لدمج مقاطع الصوت.\n"
            f"   التفاصيل: {detail}\n"
            "   هذا ليس خطأً من Azure Speech.\n"
            "   للتثبيت: python -m pip install pydub"
        )

    logger.info(f"🎙️ بدء المرحلة الثالثة (Azure) | الحلقة={episode_id} | الصوت={voice}")

    # ---------------------------------------------------------
    # إعداد بيئة Staging معزولة لكل تشغيل
    # ---------------------------------------------------------
    run_id = uuid.uuid4().hex[:10]
    staging_clips_dir = output_dir / f"episode_{episode_id}_clips__staging_{run_id}"
    staging_audio_file = output_dir / f"episode_{episode_id}_audio__staging_{run_id}.mp3"
    staging_metadata_file = output_dir / f"episode_{episode_id}_audio_metadata__staging_{run_id}.json"

    staging_clips_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1) توليد خطة الأداء الصوتي
        directions = generate_voice_direction_plan(
            sentences=sentences,
            episode_context=episode_context,
        )

        # 2) توليد الـ Clips على مستوى الجملة داخل Staging Directory
        clips_meta: List[Dict[str, Any]] = []

        for i, (sentence, direction) in enumerate(zip(sentences, directions), start=1):
            if not isinstance(sentence, str) or not sentence.strip():
                raise ValueError(f"الجملة رقم {i} فارغة أو غير صالحة!")

            sentence_text = sentence
            sentence_hash = hashlib.sha256(sentence_text.encode("utf-8")).hexdigest()

            clip_filename = f"sentence_{i:04d}.mp3"
            clip_path = staging_clips_dir / clip_filename

            ssml = build_sentence_ssml(sentence_text, direction, voice)
            _validate_single_ssml(ssml, expected_voice=voice)

            ok = False
            last_err: Optional[str] = None

            for attempt in range(1, MAX_CLIP_RETRIES + 1):
                try:
                    logger.info(f"🌐 Azure TTS | الجملة {i}/{len(sentences)} | محاولة {attempt}")
                    call_azure_tts_api(ssml, clip_path)
                    if clip_path.exists() and clip_path.stat().st_size > 0:
                        ok = True
                        break
                    last_err = "الملف الناتج فارغ"
                except Exception as e:
                    last_err = str(e)
                    logger.warning(f"⚠️ فشل توليد الـClip للجملة {i} (محاولة {attempt}): {last_err}")

            if not ok:
                raise RuntimeError(
                    f"فشل توليد Clip الجملة {i} بعد {MAX_CLIP_RETRIES} محاولات. آخر خطأ: {last_err}"
                )

            duration_ms = _get_mp3_duration_ms(clip_path)
            if duration_ms is None:
                duration_ms = _estimate_mp3_duration_ms(clip_path)

            clips_meta.append({
                "sentence_index": i,
                "filename": clip_filename,
                "sentence_text": sentence_text,
                "sentence_hash": sentence_hash,
                "narrative_role": direction.get("narrative_role"),
                "delivery_mode": direction.get("delivery_mode"),
                "azure_style": direction.get("azure_style"),
                "energy": int(direction.get("energy", 3)),
                "rate_percent": int(direction.get("rate_percent", 0)),
                "pitch_percent": int(direction.get("pitch_percent", 0)),
                "pause_before_ms": int(direction.get("pause_before_ms", 0)),
                "pause_after_ms": int(direction.get("pause_after_ms", 0)),
                "emphasis_words": direction.get("emphasis_words") if isinstance(direction.get("emphasis_words"), list) else [],
                "breath_breaks": direction.get("breath_breaks") if isinstance(direction.get("breath_breaks"), list) else [],
                "reason": direction.get("reason") or "no reason provided by Voice Director",
                "duration_ms": duration_ms,
            })

        # 3) دمج الصوت إلى ملف Staging مؤقت
        _merge_clips_with_pauses(clips_meta, staging_clips_dir, staging_audio_file)

        # 4) كتابة الـ Metadata إلى ملف Staging مؤقت
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

        # 5) Audio QA الصارم على بيئة Staging قبل الاعتماد
        validate_audio_outputs(
            sentences=sentences,
            clips_dir=staging_clips_dir,
            final_audio_file=staging_audio_file,
            metadata_file=staging_metadata_file,
            expected_voice=voice,
        )

        # ---------------------------------------------------------
        # النقل الذري الآمن (Atomic Commit & Rollback)
        # ---------------------------------------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- التقاط الحالة السابقة قبل أي نقل ---
        had_clips = clips_dir.exists()
        had_audio = final_audio_file.exists()
        had_metadata = metadata_file.exists()

        backup_clips_dir: Optional[Path] = None
        backup_audio_file: Optional[Path] = None
        backup_metadata_file: Optional[Path] = None

        # المرحلة أ: نقل المخرجات القديمة المؤكدة إلى مسارات backup مؤقتة
        try:
            if had_clips:
                backup_clips_dir = output_dir / f"episode_{episode_id}_clips__backup_{ts}_{run_id}"
                clips_dir.rename(backup_clips_dir)
                logger.info(f"📦 تم تأمين مجلد clips القديم في: {backup_clips_dir.name}")

            if had_audio:
                backup_audio_file = output_dir / f"episode_{episode_id}_audio__backup_{ts}_{run_id}.mp3"
                final_audio_file.rename(backup_audio_file)
                logger.info(f"📦 تم تأمين ملف الصوت القديم في: {backup_audio_file.name}")

            if had_metadata:
                backup_metadata_file = output_dir / f"episode_{episode_id}_audio_metadata__backup_{ts}_{run_id}.json"
                metadata_file.rename(backup_metadata_file)
                logger.info(f"📦 تم تأمين ملف metadata القديم في: {backup_metadata_file.name}")

        except Exception as backup_err:
            logger.error(
                f"❌ فشل تأمين النسخ الاحتياطية القديمة: {backup_err}. جاري التراجع الفوري..."
            )
            # استعادة أي عنصر تم نقله قبل حدوث الخطأ
            if backup_clips_dir and backup_clips_dir.exists() and not clips_dir.exists():
                try:
                    backup_clips_dir.rename(clips_dir)
                except Exception as restore_err:
                    logger.error(f"❌ فشل إرجاع مجلد clips: {restore_err}")
            if backup_audio_file and backup_audio_file.exists() and not final_audio_file.exists():
                try:
                    backup_audio_file.rename(final_audio_file)
                except Exception as restore_err:
                    logger.error(f"❌ فشل إرجاع ملف الصوت: {restore_err}")
            if backup_metadata_file and backup_metadata_file.exists() and not metadata_file.exists():
                try:
                    backup_metadata_file.rename(metadata_file)
                except Exception as restore_err:
                    logger.error(f"❌ فشل إرجاع ملف metadata: {restore_err}")

            # التحقق الصريح من نجاح الاستعادة
            _verify_rollback(
                clips_dir=clips_dir,
                final_audio_file=final_audio_file,
                metadata_file=metadata_file,
                had_clips=had_clips,
                had_audio=had_audio,
                had_metadata=had_metadata,
            )

            raise RuntimeError(
                f"تعذر إنشاء نسخ احتياطية للمخرجات السابقة دون فقدان البيانات: {backup_err}"
            )

        # المرحلة ب: اعتماد ملفات Staging في المسارات النهائية
        try:
            staging_clips_dir.rename(clips_dir)
            staging_audio_file.replace(final_audio_file)
            staging_metadata_file.replace(metadata_file)
        except Exception as commit_err:
            logger.error(
                f"❌ فشل اعتماد ملفات Staging في المسارات النهائية: {commit_err}. "
                "جاري التراجع واستعادة النسخ السابقة بالكامل..."
            )
            _rollback_committed_to_backups(
                clips_dir=clips_dir,
                final_audio_file=final_audio_file,
                metadata_file=metadata_file,
                staging_clips_dir=staging_clips_dir,
                staging_audio_file=staging_audio_file,
                staging_metadata_file=staging_metadata_file,
                backup_clips_dir=backup_clips_dir,
                backup_audio_file=backup_audio_file,
                backup_metadata_file=backup_metadata_file,
                had_clips=had_clips,
                had_audio=had_audio,
                had_metadata=had_metadata,
            )
            raise RuntimeError(
                f"فشل الاعتماد النهائي لملفات Staging؛ تم التراجع وحفظ المخرجات السابقة سليمة: {commit_err}"
            )

        # ---------------------------------------------------------
        # 6) FINAL QA — تحقق صريح على المسارات النهائية بعد النقل
        # ---------------------------------------------------------
        logger.info("🔎 تشغيل Final QA على المسارات النهائية بعد الاعتماد الذري...")
        try:
            validate_audio_outputs(
                sentences=sentences,
                clips_dir=clips_dir,
                final_audio_file=final_audio_file,
                metadata_file=metadata_file,
                expected_voice=voice,
            )
        except Exception as final_qa_err:
            logger.error(
                f"❌ فشل Final QA بعد الاعتماد: {final_qa_err}. "
                "جاري التراجع الكامل واستعادة النسخ السابقة..."
            )
            _rollback_committed_to_backups(
                clips_dir=clips_dir,
                final_audio_file=final_audio_file,
                metadata_file=metadata_file,
                staging_clips_dir=staging_clips_dir,
                staging_audio_file=staging_audio_file,
                staging_metadata_file=staging_metadata_file,
                backup_clips_dir=backup_clips_dir,
                backup_audio_file=backup_audio_file,
                backup_metadata_file=backup_metadata_file,
                had_clips=had_clips,
                had_audio=had_audio,
                had_metadata=had_metadata,
            )
            raise RuntimeError(
                f"فشل Final QA بعد الاعتماد، وتم التراجع الكامل وحفظ المخرجات السابقة سليمة: {final_qa_err}"
            )

        logger.info("✅ Final QA نجح — المخرجات النهائية مطابقة تماماً للمتوقع.")

        # ---------------------------------------------------------
        # 7) تنظيف النسخ الاحتياطية القديمة (اختياري ومُتحكَّم به)
        # ---------------------------------------------------------
        if CLEANUP_BACKUPS_AFTER_SUCCESS:
            if backup_clips_dir and backup_clips_dir.exists():
                try:
                    shutil.rmtree(backup_clips_dir)
                    logger.info(f"🧹 تم حذف النسخة الاحتياطية القديمة: {backup_clips_dir.name}")
                except Exception as e:
                    logger.warning(
                        f"⚠️ فشل حذف النسخة الاحتياطية '{backup_clips_dir}' — "
                        f"يتطلب تنظيفاً يدوياً: {e}"
                    )

            if backup_audio_file and backup_audio_file.exists():
                try:
                    backup_audio_file.unlink()
                    logger.info(f"🧹 تم حذف النسخة الاحتياطية القديمة: {backup_audio_file.name}")
                except Exception as e:
                    logger.warning(
                        f"⚠️ فشل حذف النسخة الاحتياطية '{backup_audio_file}' — "
                        f"يتطلب تنظيفاً يدوياً: {e}"
                    )

            if backup_metadata_file and backup_metadata_file.exists():
                try:
                    backup_metadata_file.unlink()
                    logger.info(f"🧹 تم حذف النسخة الاحتياطية القديمة: {backup_metadata_file.name}")
                except Exception as e:
                    logger.warning(
                        f"⚠️ فشل حذف النسخة الاحتياطية '{backup_metadata_file}' — "
                        f"يتطلب تنظيفاً يدوياً: {e}"
                    )
        else:
            logger.info(
                "📦 الاحتفاظ بالنسخ الاحتياطية بعد نجاح Final QA "
                "(CLEANUP_BACKUPS_AFTER_SUCCESS=False)."
            )

        logger.info(f"✅ اكتملت المرحلة الثالثة بنجاح واعتماد المخرجات: {final_audio_file}")
        return final_audio_file

    except Exception as err:
        logger.error(f"❌ فشلت المرحلة الثالثة، جاري تنظيف Staging: {err}")

        # تنظيف Staging مع تسجيل صريح لأي فشل (بدون إخفاء الأخطاء)
        _safe_cleanup_staging(
            staging_clips_dir=staging_clips_dir,
            staging_audio_file=staging_audio_file,
            staging_metadata_file=staging_metadata_file,
        )

        raise err


# =============================================================
# Voice Preview (Azure فقط)
# =============================================================

def generate_voice_preview(
    voice: str = None,
    voice_name: str = None,
    text: str = "Hello, this is a sample of my voice. How do I sound to you?",
    output_dir: Path = Path("outputs")
) -> Path:
    """توليد عينة صوت قصيرة باستخدام Microsoft Azure فقط."""
    final_voice = voice or voice_name
    if not final_voice:
        raise ValueError("يجب تمرير اسم الصوت (voice أو voice_name)")

    if final_voice not in AZURE_MALE_VOICES:
        raise ValueError(
            f"الصوت '{final_voice}' غير مسموح به! الأصوات المعتمدة: {AZURE_MALE_VOICES}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = final_voice.replace("/", "_").replace("\\", "_")
    preview_file = output_dir / f"preview_{safe_name}.mp3"

    safe_text = xml_escape(text)

    ssml_payload = (
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="https://www.w3.org/2001/mstts" '
        'xml:lang="en-US">\n'
        f'<voice name="{final_voice}">\n'
        '<mstts:express-as style="calm">\n'
        f'{safe_text}\n'
        '</mstts:express-as>\n'
        '</voice>\n'
        '</speak>'
    )

    logger.info(f"🎧 توليد عينة صوتية (preview) للصوت: {final_voice}")
    call_azure_tts_api(ssml_payload, preview_file)
    return preview_file
