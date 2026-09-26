import os
import re
import json
import base64
import logging
import hashlib
import uuid
import shutil
import subprocess
from datetime import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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

# =============================================================
# التحقق من توفر FFmpeg (يُستخدم في المعالجة الصوتية النهائية)
# =============================================================
FFMPEG_PATH = shutil.which("ffmpeg")
FFMPEG_AVAILABLE = FFMPEG_PATH is not None

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

if not FFMPEG_AVAILABLE:
    logger.warning(
        "⚠️ [Dependency Missing] FFmpeg غير متوفر في البيئة الحالية. "
        "المعالجة الصوتية النهائية (Master Processing) لن تكون ممكنة. "
        "للتثبيت: قم بتثبيت ffmpeg وإضافته إلى PATH."
    )

# =============================================================
# VOT VOICE BIBLE — v3 Human Storyteller
# (يعرّف هوية الأداء فقط، وليس اسماً ثابتاً للصوت — الصوت يُختار من البوت)
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
        "voice_selection": "user_selected_from_AZURE_MALE_VOICES",
        "male_voice_only": True,
        "language": "en-US",
        "note": (
            "The voice is chosen by the user at runtime and passed into "
            "generate_stage3_audio(). This Bible defines performance identity, "
            "not a fixed voice name."
        )
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

# الحدود موحَّدة مع ما يطلبه الـ prompt من Voice Director
_LIMITS = {
    "energy": (1, 5),
    "rate_percent": (-8, 2),
    "pitch_percent": (-4, 2),
    "pause_before_ms": (0, 700),
    "pause_after_ms": (0, 1000),
}

# =============================================================
# سياسة إعادة المحاولة (Retry Policy)
# =============================================================
# MAX_CLIP_RETRIES: عدد المحاولات الشبكية الحقيقية لكل نمط.
# MAX_STYLE_FALLBACKS: عدد أنماط الـ fallback الإضافية بعد النمط المخطط.
#   - حالياً = 1 (calm فقط)، نمط آمن معروف الدعم لجميع أصوات Azure.
# MAX_PLAIN_FALLBACKS: عدد محاولات الـ Plain SSML النهائية (بدون mstts:express-as).
#   - حالياً = 1، وهو الحل الأخير المضمون لأي صوت موجود في AZURE_MALE_VOICES
#     حتى لو لم يدعم أي Style.
#
# السقف الكلي لاستدعاءات Azure لكل مقطع في أسوأ الحالات:
#   MAX_CLIP_RETRIES * (1 + MAX_STYLE_FALLBACKS + MAX_PLAIN_FALLBACKS) = 3 * 3 = 9.
# هذا سقف صريح ومحدود مسبقاً، ولا يمكن أن يتضاعف بلا داعٍ.
MAX_CLIP_RETRIES = 3
MAX_STYLE_FALLBACKS = 1
MAX_PLAIN_FALLBACKS = 1

# قيمة اعتبارية داخلية تعني "SSML بدون mstts:express-as"
PLAIN_STYLE_SENTINEL = "plain"

MAX_DIRECTOR_RETRIES = 3
MAX_MERGE_GAP_MS = 1500  # السقف المطلق الآمن لأي وقفة
VOICE_CONTEXT_FIELDS = ("creative_brief", "retention_plan", "scene_plan", "sections")
MAX_VOICE_CONTEXT_CHARS = 6000

# حدود فواصل التنفس داخل الجملة
_BREATH_MIN_MS = 100
_BREATH_MAX_MS = 500
_BREATH_MAX_COUNT = 2

# سقوف دورية لوقفات الحدود بين الجمل — تمنع التراكم المبالغ فيه
_BOUNDARY_GAP_CAPS: Dict[str, int] = {
    "hook": 400,
    "setup": 500,
    "question": 500,
    "myth": 400,
    "contradiction": 550,
    "explanation": 400,
    "analogy": 400,
    "transition": 500,
    "tension": 550,
    "revelation": 700,
    "payoff": 450,
    "actionable": 450,
    "reflection": 700,
    "cta": 400,
}

# خرائط الأدوار للخطة الاحتياطية
ROLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "hook":          {"style": "calm",    "energy": 3, "rate": -3, "pitch": 0,  "before": 100, "after": 380},
    "setup":         {"style": "calm",    "energy": 2, "rate": -4, "pitch": -1, "before": 120, "after": 260},
    "question":      {"style": "calm",    "energy": 3, "rate": -3, "pitch": 1,  "before": 150, "after": 450},
    "myth":          {"style": "serious", "energy": 3, "rate": -3, "pitch": -1, "before": 130, "after": 320},
    "contradiction": {"style": "serious", "energy": 3, "rate": -2, "pitch": -1, "before": 180, "after": 400},
    "explanation":   {"style": "calm",    "energy": 2, "rate": -3, "pitch": -1, "before": 100, "after": 250},
    "analogy":       {"style": "hopeful", "energy": 3, "rate": -2, "pitch": 0,  "before": 120, "after": 300},
    "transition":    {"style": "calm",    "energy": 2, "rate": -3, "pitch": -1, "before": 130, "after": 300},
    "tension":       {"style": "serious", "energy": 3, "rate": -5, "pitch": -2, "before": 220, "after": 480},
    "revelation":    {"style": "serious", "energy": 4, "rate": -3, "pitch": 0,  "before": 260, "after": 520},
    "payoff":        {"style": "hopeful", "energy": 4, "rate": -2, "pitch": 0,  "before": 160, "after": 420},
    "actionable":    {"style": "hopeful", "energy": 3, "rate": -1, "pitch": 0,  "before": 110, "after": 340},
    "reflection":    {"style": "calm",    "energy": 2, "rate": -6, "pitch": -1, "before": 220, "after": 520},
    "cta":           {"style": "hopeful", "energy": 3, "rate": -2, "pitch": 0,  "before": 140, "after": 400},
}

ROLE_TO_MODE: Dict[str, str] = {
    "hook": "curious",
    "setup": "calm",
    "question": "curious",
    "myth": "serious",
    "contradiction": "serious",
    "explanation": "teaching",
    "analogy": "teaching",
    "transition": "calm",
    "tension": "suspense",
    "revelation": "revelation",
    "payoff": "encouraging",
    "actionable": "encouraging",
    "reflection": "reflective",
    "cta": "encouraging",
}

# =============================================================
# أنماط لغوية لاكتشاف الدور السردي في الخطة الاحتياطية
# =============================================================
_CONTRADICTION_RE = re.compile(
    r"\b(but|however|yet|nevertheless|in reality|the truth is|actually|in fact|contrary to)\b",
    re.IGNORECASE,
)
_EXPLANATION_RE = re.compile(
    r"\b(because|which means|that's why|that is why|therefore|as a result|the reason|this is why)\b",
    re.IGNORECASE,
)
_ANALOGY_RE = re.compile(
    r"\b(imagine|think of it like|like a|similar to|just as|picture this)\b",
    re.IGNORECASE,
)
_TENSION_RE = re.compile(
    r"\b(beware|danger|risk|threat|warning|collapse|fail|breaks?|hidden)\b",
    re.IGNORECASE,
)
_REVELATION_RE = re.compile(
    r"\b(the key|the secret|what most people|the real|the truth|discovered|reveals?)\b",
    re.IGNORECASE,
)
_ACTIONABLE_RE = re.compile(
    r"\b(you can|you should|start|begin|try|do this|here's how|the first step)\b",
    re.IGNORECASE,
)

# =============================================================
# نظام التوجيه الصوتي (Voice Director)
# =============================================================
STAGE_3_VOICE_DIRECTOR_SYSTEM_PROMPT = """You are the LEAD VOICE DIRECTOR for the documentary/science channel VOT.

Your ONLY job is to direct the delivery plan sentence-by-sentence.
You do NOT rewrite, add, or remove words.

VOT VOICE IDENTITY:
- A calm, intelligent, human storyteller with quiet authority.
- Controlled, conversational documentary narration — never theatrical, never trailer-voice.
- Emotional range is human and nuanced: thoughtful, curious, tense, warm, quietly awed — but never over-acted.

ABSOLUTELY FORBIDDEN TONES:
- Trailer/hype voice, loud cheerful pitch, over-energetic reads, shouting.

PER-ROLE DIRECTION GUIDANCE (use narrative_role to drive variety — do NOT default everything to calm):

- hook:          curious, clear, inviting — pull the listener in without hype. energy 3, rate -3, pitch 0.
- setup:         warm, conversational, calm. energy 2, rate -4, pitch -1.
- question:      reflective with a natural slight upward inflection at the end; leave space. energy 3, rate -3, pitch +1.
- myth:          clear reporting tone, matter-of-fact. energy 3, rate -3, pitch -1.
- contradiction: mild natural emphasis on the contrast point. energy 3, rate -2, pitch -1.
- explanation:   warm teaching tone, clear articulation. energy 2, rate -3, pitch -1.
- analogy:       slightly more vivid, storytelling-inflected. energy 3, rate -2, pitch 0.
- transition:    smooth, low-key bridging. energy 2, rate -3, pitch -1.
- tension:       quiet, deliberate unease — no horror, no shouting. energy 3, rate -5, pitch -2.
- revelation:    clear rise in energy, controlled and confident — never announcer-style. energy 4, rate -3, pitch 0.
- payoff:        confident, satisfying resolution. energy 4, rate -2, pitch 0.
- actionable:    slightly higher energy, natural pace, practical. energy 3, rate -1, pitch 0.
- reflection:    slower, deeper, longer contemplative pauses. energy 2, rate -6, pitch -1.
- cta:           warm, clear, non-promotional. energy 3, rate -2, pitch 0.

AZURE STYLE RULES:
- Use styles that match the role: "calm", "serious", "hopeful", or "whispering".
- Do NOT use "excited", "cheerful", "angry", or "shouting".
- NEVER use "calm" for the majority of sentences — every sentence should match its role.

RATE & PITCH RULES:
- rate_percent between -8 and +2 (slower for reflection/tension, slightly faster for actionable).
- pitch_percent between -4 and +2 (natural human variation, avoid monotone).
- Do NOT keep rate and pitch identical across all sentences.

ENERGY RULES:
- energy between 2 and 4. Reserve 4 for revelation/payoff. Never use 5.
- Vary energy across sentences to avoid monotone delivery.

EMPHASIS:
- At most 1-2 key words per sentence. Leave the array empty if nothing stands out.

BREATH BREAKS:
- Optional. Insert 0-2 per sentence.
- Use only before a key reveal, after a question, or between two long clauses.
- Format: {"after_word_index": N, "duration_ms": M} where N is 1-based word index.
- duration_ms between 100 and 500.

OUTPUT SCHEMA (PURE JSON ONLY):
{
  "directions": [
    {
      "sentence_index": 1,
      "narrative_role": "hook",
      "delivery_mode": "curious",
      "azure_style": "calm",
      "energy": 3,
      "rate_percent": -3,
      "pitch_percent": 0,
      "pause_before_ms": 100,
      "pause_after_ms": 380,
      "emphasis_words": ["truth"],
      "breath_breaks": [{"after_word_index": 4, "duration_ms": 200}],
      "reason": "grounded inviting hook"
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

def _validate_single_ssml(
    ssml_text: str,
    expected_voice: Optional[str] = None,
    allow_no_express: bool = False,
) -> None:
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
        # مسموح فقط عندما نستخدم Plain SSML كـ fallback أخير
        if not allow_no_express:
            raise ValueError("SSML لا يحتوي على mstts:express-as")
        return

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
    except json.JSONDecodeError:
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

def _detect_narrative_role_from_text(sentence: str, position_ratio: float) -> str:
    """
    كشف الدور السردي من إشارات لغوية بسيطة داخل الجملة، بدل الاعتماد على
    موقع الجملة فقط. تُستخدم فقط في الخطة الاحتياطية عند فشل Voice Director.
    """
    text = sentence.strip()
    if not text:
        return "explanation"

    lowered = text.lower()
    word_count = len(re.findall(r"\S+", text))

    # 1) السؤال — إشارة قوية جداً
    if text.rstrip().endswith("?"):
        return "question"

    # 2) القياس / التشبيه
    if _ANALOGY_RE.search(lowered):
        return "analogy"

    # 3) التناقض
    if _CONTRADICTION_RE.search(lowered):
        return "contradiction"

    # 4) الشرح
    if _EXPLANATION_RE.search(lowered):
        return "explanation"

    # 5) الكشف
    if _REVELATION_RE.search(lowered):
        return "revelation"

    # 6) التوتر
    if _TENSION_RE.search(lowered):
        return "tension"

    # 7) القابل للتطبيق
    if _ACTIONABLE_RE.search(lowered):
        return "actionable"

    # 8) الجمل القصيرة جداً — غالباً hook أو payoff
    if word_count <= 5:
        return "hook" if position_ratio < 0.5 else "payoff"

    # 9) الجمل الطويلة في النهاية — تأمل
    if word_count >= 20 and position_ratio >= 0.65:
        return "reflection"

    # 10) الفولباك الموضعي الافتراضي
    if position_ratio < 0.15:
        return "setup"
    elif position_ratio < 0.35:
        return "explanation"
    elif position_ratio < 0.55:
        return "analogy"
    elif position_ratio < 0.70:
        return "transition"
    elif position_ratio < 0.85:
        return "payoff"
    else:
        return "reflection"

def _build_fallback_plan(sentences: List[str]) -> List[Dict[str, Any]]:
    total = len(sentences)
    plan: List[Dict[str, Any]] = []

    for i in range(1, total + 1):
        sentence = sentences[i - 1]
        ratio = (i - 1) / max(total - 1, 1) if total > 1 else 0.0

        # مواضع ثابتة: أول جملة hook، وقبل الأخيرة reflection، والأخيرة cta
        if i == 1:
            role = "hook"
        elif total >= 4 and i == total - 1:
            role = "reflection"
        elif i == total:
            role = "cta"
        else:
            role = _detect_narrative_role_from_text(sentence, ratio)

        profile = ROLE_PROFILES.get(role, ROLE_PROFILES["explanation"])
        mode = ROLE_TO_MODE.get(role, "teaching")

        plan.append({
            "sentence_index": i,
            "narrative_role": role,
            "delivery_mode": mode,
            "azure_style": profile["style"],
            "energy": profile["energy"],
            "rate_percent": profile["rate"],
            "pitch_percent": profile["pitch"],
            "pause_before_ms": profile["before"],
            "pause_after_ms": profile["after"],
            "emphasis_words": [],
            "breath_breaks": [],
            "reason": f"fallback {role} ({'linguistic' if role != 'explanation' else 'default'})",
        })

    return plan

def validate_voice_direction_plan(directions: List[Dict[str, Any]], sentences: List[str]) -> None:
    if not isinstance(directions, list):
        raise ValueError("خطة التوجيه ليست قائمة!")
    if len(directions) != len(sentences):
        raise ValueError(f"عدد التوجيهات ({len(directions)}) لا يطابق الجمل ({len(sentences)})")

    for i, d in enumerate(directions, start=1):
        if not isinstance(d, dict):
            raise ValueError(f"العنصر {i}: ليس كائن JSON (dict)")
        if d.get("sentence_index") != i:
            raise ValueError(f"العنصر {i}: الفهرسة غير متطابقة")
        if d.get("azure_style") not in APPROVED_AZURE_STYLES:
            raise ValueError(f"العنصر {i}: النمط غير معتمد")

# =============================================================
# أدوات التطبيع (Sanitization)
# =============================================================
def _clamp(value: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(round(float(value)))
    except Exception:
        return default
    return max(lo, min(hi, v))

def _sanitize_emphasis_words(sentence: str, raw_words: Any) -> List[str]:
    """تأكيد كلمة أو كلمتين كحد أقصى، موجودتين فعلاً في الجملة، بدون تكرار."""
    if not isinstance(raw_words, list):
        return []
    cleaned: List[str] = []
    seen = set()
    for w in raw_words:
        if not isinstance(w, str):
            continue
        candidate = w.strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)", re.IGNORECASE)
        if not pattern.search(sentence):
            continue
        cleaned.append(candidate)
        seen.add(key)
        if len(cleaned) >= 2:
            break
    return cleaned

def _sanitize_breath_breaks(sentence: str, raw_breaks: Any) -> List[Dict[str, int]]:
    """تحقق من فواصل التنفس: 0-2 فقط، داخل الجملة، بدون تكرار، ضمن الحدود."""
    if not isinstance(raw_breaks, list):
        return []
    word_tokens = re.findall(r"\S+", sentence)
    word_count = len(word_tokens)
    if word_count < 2:
        return []

    cleaned: List[Dict[str, int]] = []
    seen_idx = set()
    for item in raw_breaks:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("after_word_index"))
            dur = int(item.get("duration_ms", 200))
        except (TypeError, ValueError):
            continue
        if idx < 1 or idx >= word_count:
            continue
        if idx in seen_idx:
            continue
        dur = max(_BREATH_MIN_MS, min(_BREATH_MAX_MS, dur))
        cleaned.append({"after_word_index": idx, "duration_ms": dur})
        seen_idx.add(idx)
        if len(cleaned) >= _BREATH_MAX_COUNT:
            break

    cleaned.sort(key=lambda x: x["after_word_index"])
    return cleaned

def _sanitize_direction(sentence: str, direction: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(direction, dict):
        direction = {}

    cleaned: Dict[str, Any] = dict(direction)

    style = str(direction.get("azure_style", "calm")).strip().lower()
    if style not in APPROVED_AZURE_STYLES:
        style = "calm"
    cleaned["azure_style"] = style

    cleaned["energy"] = _clamp(direction.get("energy", 3), *_LIMITS["energy"], 3)
    cleaned["rate_percent"] = _clamp(
        direction.get("rate_percent", -4), *_LIMITS["rate_percent"], -4
    )
    cleaned["pitch_percent"] = _clamp(
        direction.get("pitch_percent", -1), *_LIMITS["pitch_percent"], -1
    )
    cleaned["pause_before_ms"] = _clamp(
        direction.get("pause_before_ms", 100), *_LIMITS["pause_before_ms"], 100
    )
    cleaned["pause_after_ms"] = _clamp(
        direction.get("pause_after_ms", 300), *_LIMITS["pause_after_ms"], 300
    )

    cleaned["emphasis_words"] = _sanitize_emphasis_words(
        sentence, direction.get("emphasis_words", [])
    )
    cleaned["breath_breaks"] = _sanitize_breath_breaks(
        sentence, direction.get("breath_breaks", [])
    )

    return cleaned

# =============================================================
# إدراج فواصل التنفس داخل الجملة
# =============================================================
def _insert_breath_markers(text: str, breaks: List[Dict[str, int]]) -> str:
    if not breaks:
        return text

    tokens = re.findall(r"\S+|\s+", text)
    word_token_indices = [i for i, t in enumerate(tokens) if not t.isspace()]
    if not word_token_indices:
        return text

    insert_map: Dict[int, int] = {}
    for b in breaks:
        idx = b["after_word_index"]
        dur = b["duration_ms"]
        if idx < 1 or idx > len(word_token_indices):
            continue
        token_pos = word_token_indices[idx - 1]
        if token_pos in insert_map:
            continue
        insert_map[token_pos] = dur

    if not insert_map:
        return text

    out: List[str] = []
    for i, t in enumerate(tokens):
        out.append(t)
        if i in insert_map:
            out.append(f"\x00BRK{insert_map[i]}\x00")
    return "".join(out)

# =============================================================
# بناء SSML المطور مع styledegree + emphasis + break
# دعم خاص للنمط PLAIN_STYLE_SENTINEL: بدون mstts:express-as نهائياً.
# =============================================================
def build_sentence_ssml(
    sentence: str,
    direction: Dict[str, Any],
    voice_name: str,
    style_override: Optional[str] = None,
) -> str:
    """بناء SSML مع ضبط styledegree، دعم emphasis وbreath_breaks.

    إذا كان style_override == PLAIN_STYLE_SENTINEL ("plain")، يتم إنتاج SSML
    بدون <mstts:express-as> مع الحفاظ على <prosody> و<emphasis> و<break>.
    """
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError("الجملة فارغة!")

    if style_override is not None:
        style = str(style_override).strip().lower()
    else:
        style = str(direction.get("azure_style", "calm")).strip().lower()

    use_plain = (style == PLAIN_STYLE_SENTINEL)

    if not use_plain and style not in APPROVED_AZURE_STYLES:
        style = "calm"

    rate = _clamp(direction.get("rate_percent", -4), *_LIMITS["rate_percent"], -4)
    pitch = _clamp(direction.get("pitch_percent", -1), *_LIMITS["pitch_percent"], -1)
    energy = _clamp(direction.get("energy", 3), *_LIMITS["energy"], 3)

    # styledegree متدرج حسب الطاقة — أدنى حد مقبول لتفادي الأداء المتصنّع
    style_degree = round(0.55 + (energy - 1) * 0.10, 2)
    style_degree = max(0.30, min(1.20, style_degree))

    emphasis_words = _sanitize_emphasis_words(sentence, direction.get("emphasis_words", []))
    breath_breaks = _sanitize_breath_breaks(sentence, direction.get("breath_breaks", []))

    # 1) أدخل فواصل التنفس بناءً على فهرس الكلمة (قبل التأكيد)
    working = _insert_breath_markers(sentence, breath_breaks)

    # 2) استبدل كلمات التأكيد بعلامات مؤقتة
    placeholders: List[str] = []
    for w in emphasis_words:
        pattern = re.compile(r"(?<!\w)" + re.escape(w) + r"(?!\w)", re.IGNORECASE)

        def _repl(m):
            idx = len(placeholders)
            placeholders.append(m.group(0))
            return f"\x00EMPH{idx}\x00"

        working, _ = pattern.subn(_repl, working, count=1)

    # 3) XML escape للنص الكامل (العلامات \x00 آمنة)
    escaped = xml_escape(working)

    # 4) استبدل علامات emphasis
    for idx, original in enumerate(placeholders):
        tag = f'<emphasis level="moderate">{xml_escape(original)}</emphasis>'
        escaped = escaped.replace(f"\x00EMPH{idx}\x00", tag)

    # 5) استبدل علامات break بالوسم الصحيح
    escaped = re.sub(
        r"\x00BRK(\d+)\x00",
        lambda m: f'<break time="{m.group(1)}ms"/>',
        escaped,
    )

    rate_str = f"{rate:+d}%"
    pitch_str = f"{pitch:+d}%"

    if use_plain:
        # Plain SSML — لا mstts:express-as. نحافظ على prosody فقط.
        ssml = (
            '<speak version="1.0" '
            'xmlns="http://www.w3.org/2001/10/synthesis" '
            'xml:lang="en-US">\n'
            f'<voice name="{voice_name}">\n'
            f'<prosody rate="{rate_str}" pitch="{pitch_str}">\n'
            f'{escaped}\n'
            '</prosody>\n'
            '</voice>\n'
            '</speak>'
        )
    else:
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

    if not response.content or len(response.content) == 0:
        raise RuntimeError(
            "استجاب Azure TTS API بنجاح (HTTP 200) لكن بدون أي بيانات صوتية (محتوى فارغ)."
        )

    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "wb") as f:
        f.write(response.content)

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

def _compute_boundary_gap_ms(prev_meta: Dict[str, Any], curr_meta: Dict[str, Any]) -> int:
    """
    حساب وقفة الحدود بين جملتين بناءً على الدور السردي للجملة التالية.
    يمنع تراكم الوقفات المبالغ فيه عبر تقليل النسبة + سقف دوري لكل دور.
    """
    try:
        prev_after = int(prev_meta.get("pause_after_ms", 0) or 0)
    except (TypeError, ValueError):
        prev_after = 0
    try:
        curr_before = int(curr_meta.get("pause_before_ms", 0) or 0)
    except (TypeError, ValueError):
        curr_before = 0

    role = str(curr_meta.get("narrative_role", "") or "")
    cap = _BOUNDARY_GAP_CAPS.get(role, 400)

    if role in ("revelation", "reflection"):
        # وقفة درامية / تأملية — 60% فقط من المجموع لتفادي المبالغة
        gap = int((prev_after + curr_before) * 0.6)
    elif role in ("tension", "contradiction"):
        # وقفة مشدودة محسوبة — 70% من كل قيمة
        gap = int(prev_after * 0.7 + curr_before * 0.7)
    elif role in ("transition", "setup"):
        # انتقال فقرة — الحد الأقصى بين القيمتين + هامش صغير
        gap = max(prev_after, curr_before) + 40
    elif role in ("payoff", "actionable"):
        # تسليم هادئ دون مبالغة
        gap = int((prev_after + curr_before) * 0.6)
    else:
        # جمل مترابطة — لا نجمع آلياً
        gap = max(prev_after, curr_before)

    return _clamp(gap, 0, min(cap, MAX_MERGE_GAP_MS), 0)

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
            gap = _compute_boundary_gap_ms(prev_meta, meta)
            if gap > 0:
                combined += AudioSegment.silent(duration=gap, frame_rate=segment.frame_rate)

        combined += segment
        prev_meta = meta

    if prev_meta is not None:
        tail = _clamp(prev_meta.get("pause_after_ms", 0), 0, MAX_MERGE_GAP_MS, 0)
        if tail > 0:
            combined += AudioSegment.silent(duration=tail, frame_rate=combined.frame_rate)

    combined.export(str(output_file), format="mp3")

# =============================================================
# معالجة صوتية نهائية احترافية (Master Processing)
# =============================================================
def _apply_master_audio_processing(input_file: Path, output_file: Path) -> None:
    """
    معالجة صوتية خفيفة واحترافية للصوت النهائي عبر FFmpeg:
    High-pass → Warmth EQ (120Hz) → De-mud EQ (250Hz) → De-harsh EQ (3.5kHz)
    → Compression → Loudness Normalization → Final Limiter.

    ترتيب المعالجة مهم:
    - Loudnorm قبل الـ limiter لتفادي تجاوز الـ True Peak بعد التطبيع.
    - الـ limiter النهائي يضمن عدم تجاوز -0.26 dBFS.
    """
    if not FFMPEG_AVAILABLE or not FFMPEG_PATH:
        raise RuntimeError(
            "FFmpeg غير متوفر في البيئة، ولا يمكن تنفيذ المعالجة الصوتية النهائية. "
            "قم بتثبيت ffmpeg وإضافته إلى PATH."
        )

    if not input_file.exists() or input_file.stat().st_size == 0:
        raise RuntimeError("ملف الصوت المُدمج (قبل المعالجة) مفقود أو فارغ.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink(missing_ok=True)

    filter_chain = (
        # 1) تنظيف الترددات المنخفضة جداً
        "highpass=f=80,"
        # 2) دفء عند صدر الرجل الطبيعي (120Hz)
        "equalizer=f=120:t=q:w=1:g=1.5,"
        # 3) تنظيف الطين / الـ mud (250Hz) بدل تعزيزه
        "equalizer=f=250:t=q:w=1.2:g=-0.8,"
        # 4) تهدئة قسوة الحضور (3.5kHz) — ليس de-esser حقيقي لكنه مضبوط
        "equalizer=f=3500:t=q:w=2:g=-1.5,"
        # 5) ضغط لطيف
        "acompressor=threshold=-18dB:ratio=2.5:attack=15:release=200,"
        # 6) تطبيع الجهارة (Loudness) أولاً
        "loudnorm=I=-16:TP=-1.5:LRA=11,"
        # 7) limiter نهائي بعد التطبيع لمنع القمم
        "alimiter=limit=0.97"
    )

    cmd = [
        str(FFMPEG_PATH),
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_file),
        "-af", filter_chain,
        "-ar", "24000",
        "-ac", "1",
        "-b:a", "160k",
        str(output_file),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("انتهت مهلة معالجة FFmpeg (600 ثانية) دون اكتمال.")
    except FileNotFoundError:
        raise RuntimeError("لم يتم العثور على ffmpeg في البيئة.")
    except Exception as e:
        raise RuntimeError(f"فشل تنفيذ FFmpeg: {e}")

    if proc.returncode != 0:
        err_tail = (proc.stderr or "")[:500]
        raise RuntimeError(f"فشلت معالجة FFmpeg (code={proc.returncode}): {err_tail}")

    if not output_file.exists() or output_file.stat().st_size == 0:
        raise RuntimeError("ملف الصوت النهائي بعد المعالجة غير موجود أو فارغ.")

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

def _safe_cleanup_staging(
    staging_clips_dir: Path,
    staging_audio_raw_file: Path,
    staging_audio_file: Path,
    staging_metadata_file: Path,
):
    if staging_clips_dir.exists():
        shutil.rmtree(staging_clips_dir, ignore_errors=True)
    for p in (staging_audio_raw_file, staging_audio_file, staging_metadata_file):
        try:
            if p.exists():
                p.unlink(missing_ok=True)
        except Exception:
            pass

# =============================================================
# بناء خطة إعادة المحاولة لكل مقطع
# =============================================================
def _build_attempt_plan(planned_style: str) -> List[Tuple[str, int]]:
    """
    يبني خطة المحاولات لكل مقطع على شكل قائمة (style, attempt_within_style).

    البنية (بالترتيب):
      1) النمط المخطط:          MAX_CLIP_RETRIES محاولات.
      2) calm (إن اختلف):       MAX_CLIP_RETRIES محاولات.
      3) Plain SSML (بدون
         mstts:express-as):    MAX_CLIP_RETRIES محاولات.

    السقف الكلي = MAX_CLIP_RETRIES * (1 + MAX_STYLE_FALLBACKS + MAX_PLAIN_FALLBACKS).
    حالياً: 3 * 3 = 9 استدعاءات كحد أقصى لكل مقطع.

    ملاحظات تصميمية:
      - هذا يعيد منطق إعادة المحاولة الأصلي (3 محاولات حقيقية لكل نمط)
        دون حذفه، ويضمن أن الفشل العابر (timeout/شبكة/rate limit) يُعاد
        قبل تبديل النمط.
      - لا يمكن للاستدعاءات أن تتضاعف بلا داعٍ (سقف واضح ومحدد مسبقاً).
      - calm نمط آمن معروف الدعم لجميع أصوات AZURE_MALE_VOICES.
      - Plain SSML هو الضمانة الأخيرة لأي صوت موجود في AZURE_MALE_VOICES
        حتى لو لم يدعم أي Style. لا يثبّت الصوت على أي اسم محدد.
    """
    plan: List[Tuple[str, int]] = []

    # 1) محاولات النمط المخطط — كلها أولاً
    for n in range(1, MAX_CLIP_RETRIES + 1):
        plan.append((planned_style, n))

    # 2) محاولات calm كـ fallback — فقط إذا كان مختلفاً
    if planned_style != "calm" and MAX_STYLE_FALLBACKS >= 1:
        for n in range(1, MAX_CLIP_RETRIES + 1):
            plan.append(("calm", n))

    # 3) محاولات Plain SSML كـ fallback أخير — دائماً متاحة كضمان
    if MAX_PLAIN_FALLBACKS >= 1:
        for n in range(1, MAX_CLIP_RETRIES + 1):
            plan.append((PLAIN_STYLE_SENTINEL, n))

    return plan

# =============================================================
# الدالة الرئيسية لتوليد الصوت
# (الصوت يُختار من البوت عبر المعامل voice)
# =============================================================
def generate_stage3_audio(
    episode_id: str,
    sentences: List[str],
    engine: str = "azure",
    voice: str = "en-US-BrianMultilingualNeural",  # افتراضي فقط، ويُستبدل بما يختاره البوت
    output_dir: Path = Path("outputs"),
    episode_context: Optional[Dict[str, Any]] = None,
) -> Path:
    # =========================================================
    # HARD GUARDS — قبل أي استدعاء مدفوع (Azure / Gemini / FFmpeg)
    # =========================================================

    # A) المحرك: azure فقط
    engine_norm = str(engine).lower().strip()
    if engine_norm != "azure":
        raise ValueError(
            f"المحرك '{engine}' غير مدعوم. المحرك الوحيد المتاح هو 'azure'."
        )

    # B) الصوت: يجب أن يكون ضمن الأصوات الذكرية المعتمدة
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

    # E) FFmpeg إلزامي للمعالجة النهائية — قبل أي استدعاء Azure
    #    (يمنع استدعاءات Azure مدفوعة قد تذهب هدراً إذا كانت البيئة غير جاهزة للمعالجة)
    if not FFMPEG_AVAILABLE:
        raise RuntimeError(
            "FFmpeg مطلوب للمعالجة الصوتية النهائية ولا يمكن المتابعة بدونه. "
            "قم بتثبيت ffmpeg وإضافته إلى PATH."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    final_audio_file = output_dir / f"episode_{episode_id}_audio.mp3"
    clips_dir = output_dir / f"episode_{episode_id}_clips"
    metadata_file = output_dir / f"episode_{episode_id}_audio_metadata.json"

    run_id = uuid.uuid4().hex[:10]
    staging_clips_dir = output_dir / f"episode_{episode_id}_clips__staging_{run_id}"
    staging_audio_raw_file = output_dir / f"episode_{episode_id}_audio_raw__staging_{run_id}.mp3"
    staging_audio_file = output_dir / f"episode_{episode_id}_audio__staging_{run_id}.mp3"
    staging_metadata_file = output_dir / f"episode_{episode_id}_audio_metadata__staging_{run_id}.json"
    staging_clips_dir.mkdir(parents=True, exist_ok=True)

    try:
        directions = generate_voice_direction_plan(sentences, episode_context)
        clips_meta: List[Dict[str, Any]] = []

        for i, (sentence, raw_direction) in enumerate(zip(sentences, directions), start=1):
            direction = _sanitize_direction(sentence, raw_direction)

            clip_filename = f"sentence_{i:04d}.mp3"
            clip_path = staging_clips_dir / clip_filename
            sentence_hash = hashlib.sha256(sentence.encode("utf-8")).hexdigest()

            planned_style = direction.get("azure_style", "calm")

            # =====================================================
            # خطة إعادة المحاولة الحقيقية
            # =====================================================
            # الترتيب (لكل مقطع):
            #   1. النمط المخطط:      MAX_CLIP_RETRIES محاولات.
            #   2. calm (إن اختلف):   MAX_CLIP_RETRIES محاولات.
            #   3. Plain SSML:        MAX_CLIP_RETRIES محاولات.
            # السقف الكلي = 3 * 3 = 9.
            #
            # لماذا هذا الترتيب؟
            #   - الفشل العابر (timeout / rate limit / خطأ شبكة) يجب أن يُعاد
            #     على نفس النمط قبل تبديله، وإلا فقدنا اتساق الأداء بلا داعٍ.
            #   - calm آمن ومضمون الدعم لجميع أصوات Azure المعروفة.
            #   - Plain SSML هو الخط الأخير الذي يضمن أن كل صوت في
            #     AZURE_MALE_VOICES يعمل حتى لو لم يدعم أي Style.
            #     لا يثبّت الصوت على اسم محدد — يبقى نفس voice المختار.
            attempt_plan: List[Tuple[str, int]] = _build_attempt_plan(planned_style)

            used_style = planned_style
            used_fallback = False
            succeeded = False
            last_error: Optional[Exception] = None
            total_attempts = 0

            for attempt_style, attempt_within_style in attempt_plan:
                total_attempts += 1
                is_plain_attempt = (attempt_style == PLAIN_STYLE_SENTINEL)
                try:
                    ssml = build_sentence_ssml(
                        sentence,
                        direction,
                        voice,
                        style_override=attempt_style,
                    )
                    _validate_single_ssml(
                        ssml,
                        expected_voice=voice,
                        allow_no_express=is_plain_attempt,
                    )
                    call_azure_tts_api(ssml, clip_path)
                    if clip_path.exists() and clip_path.stat().st_size > 0:
                        used_style = attempt_style
                        used_fallback = used_style != planned_style
                        succeeded = True
                        break
                    else:
                        last_error = RuntimeError("ملف المقطع فارغ بعد الاستدعاء")
                        logger.warning(
                            f"[episode={episode_id}] المقطع {i}: استجابة فارغة من Azure "
                            f"(النمط={attempt_style}، "
                            f"المحاولة {attempt_within_style}/{MAX_CLIP_RETRIES}، "
                            f"الإجمالي {total_attempts}/{len(attempt_plan)})"
                        )
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"[episode={episode_id}] المقطع {i}: فشل "
                        f"(النمط={attempt_style}، "
                        f"المحاولة {attempt_within_style}/{MAX_CLIP_RETRIES}، "
                        f"الإجمالي {total_attempts}/{len(attempt_plan)}): {e}"
                    )

            if not succeeded:
                raise RuntimeError(
                    f"فشل توليد المقطع {i} بعد {total_attempts} محاولات "
                    f"(النمط المخطط={planned_style}): {last_error}"
                )

            duration_ms = _get_mp3_duration_ms(clip_path) or _estimate_mp3_duration_ms(clip_path)

            clips_meta.append({
                "sentence_index": i,
                "filename": clip_filename,
                "sentence_text": sentence,
                "sentence_hash": sentence_hash,
                "narrative_role": direction.get("narrative_role"),
                "delivery_mode": direction.get("delivery_mode"),
                "azure_style": planned_style,
                # عند استخدام plain fallback، نُسجّل actual_azure_style="plain"
                # صراحةً للإشارة إلى أن SSML لم يحتوي على mstts:express-as.
                "actual_azure_style": used_style,
                "style_fallback_used": used_fallback,
                "total_attempts": total_attempts,
                "energy": int(direction.get("energy", 3)),
                "rate_percent": int(direction.get("rate_percent", -4)),
                "pitch_percent": int(direction.get("pitch_percent", -1)),
                "pause_before_ms": int(direction.get("pause_before_ms", 100)),
                "pause_after_ms": int(direction.get("pause_after_ms", 300)),
                "emphasis_words": direction.get("emphasis_words", []),
                "breath_breaks": direction.get("breath_breaks", []),
                "reason": direction.get("reason", ""),
                "duration_ms": duration_ms,
            })

        # 1) دمج المقاطع (قبل المعالجة)
        _merge_clips_with_pauses(clips_meta, staging_clips_dir, staging_audio_raw_file)

        # 2) المعالجة الصوتية النهائية (Master) — بعد الدمج
        #    عند الفشل: نرفع الاستثناء ولا نكتب metadata (لا حقل processing_error مضلل).
        _apply_master_audio_processing(staging_audio_raw_file, staging_audio_file)
        processing_status = "applied"

        # 3) البيانات الوصفية
        metadata = {
            "episode_id": episode_id,
            "voice_name": voice,
            "language": "en-US",
            "sentence_count": len(sentences),
            "voice_bible_version": VOICE_BIBLE_VERSION,
            "processing_status": processing_status,
            "retry_policy": {
                "max_clip_retries_per_style": MAX_CLIP_RETRIES,
                "max_style_fallbacks": MAX_STYLE_FALLBACKS,
                "max_plain_fallbacks": MAX_PLAIN_FALLBACKS,
                "max_total_attempts_per_clip": MAX_CLIP_RETRIES * (
                    1 + MAX_STYLE_FALLBACKS + MAX_PLAIN_FALLBACKS
                ),
            },
            "processing_chain": (
                "highpass=80,"
                "eq120+1.5dB,"
                "eq250-0.8dB,"
                "eq3500-1.5dB,"
                "acompressor(threshold=-18dB,ratio=2.5),"
                "loudnorm(I=-16,TP=-1.5,LRA=11),"
                "alimiter(0.97)"
            ),
            "clips": clips_meta,
        }
        with open(staging_metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # 4) تحقق شامل على مسارات الـ staging قبل أي التزام نهائي
        validate_audio_outputs(
            sentences,
            staging_clips_dir,
            staging_audio_file,
            staging_metadata_file,
            expected_voice=voice,
        )

        # =====================================================
        # SAFE COMMIT — التزام ذري كامل مع rollback شامل
        # =====================================================
        old_audio_backup = output_dir / f"episode_{episode_id}_audio__old_{run_id}.mp3"
        old_meta_backup = output_dir / f"episode_{episode_id}_audio_metadata__old_{run_id}.json"
        old_clips_backup = output_dir / f"episode_{episode_id}_clips__old_{run_id}"

        had_old_audio = final_audio_file.exists()
        had_old_meta = metadata_file.exists()
        had_old_clips = clips_dir.exists()

        moved_old = {"audio": False, "meta": False, "clips": False}
        placed_new = {"audio": False, "meta": False, "clips": False}

        try:
            # 1) انقل الملفات القديمة إلى backup (إن وُجدت)
            if had_old_audio:
                final_audio_file.replace(old_audio_backup)
                moved_old["audio"] = True
            if had_old_meta:
                metadata_file.replace(old_meta_backup)
                moved_old["meta"] = True
            if had_old_clips:
                clips_dir.replace(old_clips_backup)
                moved_old["clips"] = True

            # 2) انقل ملفات staging إلى مواقعها النهائية
            staging_audio_file.replace(final_audio_file)
            placed_new["audio"] = True
            staging_metadata_file.replace(metadata_file)
            placed_new["meta"] = True
            staging_clips_dir.replace(clips_dir)
            placed_new["clips"] = True

        except Exception as commit_err:
            logger.error(f"⚠️ فشل الالتزام الذري: {commit_err}")

            # إزالة أي ملف جديد وُضع جزئياً
            if placed_new["audio"]:
                try:
                    final_audio_file.unlink(missing_ok=True)
                except Exception:
                    pass
            if placed_new["meta"]:
                try:
                    metadata_file.unlink(missing_ok=True)
                except Exception:
                    pass
            if placed_new["clips"]:
                try:
                    shutil.rmtree(clips_dir, ignore_errors=True)
                except Exception:
                    pass

            # استرجاع الملفات القديمة من الـ backup
            if moved_old["audio"] and old_audio_backup.exists():
                try:
                    old_audio_backup.replace(final_audio_file)
                except Exception as e:
                    logger.error(f"⚠️ فشل استرجاع ملف الصوت القديم: {e}")
            if moved_old["meta"] and old_meta_backup.exists():
                try:
                    old_meta_backup.replace(metadata_file)
                except Exception as e:
                    logger.error(f"⚠️ فشل استرجاع ملف البيانات الوصفية القديم: {e}")
            if moved_old["clips"] and old_clips_backup.exists():
                try:
                    old_clips_backup.replace(clips_dir)
                except Exception as e:
                    logger.error(f"⚠️ فشل استرجاع مجلد المقاطع القديم: {e}")

            raise

        # 3) تنظيف الـ backups بعد نجاح الالتزام الكامل
        for _b in (old_audio_backup, old_meta_backup):
            try:
                if _b.exists():
                    _b.unlink(missing_ok=True)
            except Exception:
                pass
        if old_clips_backup.exists():
            shutil.rmtree(old_clips_backup, ignore_errors=True)

        # تنظيف الملف الوسيط (raw) بعد نجاح الالتزام
        try:
            if staging_audio_raw_file.exists():
                staging_audio_raw_file.unlink(missing_ok=True)
        except Exception:
            pass

        logger.info(f"✅ تم إنتاج صوت حلقة Vot بنجاح: {final_audio_file}")
        return final_audio_file

    except Exception as err:
        logger.error(f"❌ [episode={episode_id}] خطأ أثناء توليد الصوت: {err}")
        # تنظيف الـ staging فقط. لا نحذف ملفات نهائية من تشغيل سابق ناجح.
        _safe_cleanup_staging(
            staging_clips_dir,
            staging_audio_raw_file,
            staging_audio_file,
            staging_metadata_file,
        )
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
