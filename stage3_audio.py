import os
import re
import base64
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any
import requests

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

# قائمة الـ 58 انفعالاً المعتمدة حصراً لمسار Google
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

# قائمة أنماط مايكروسوفت أزور المعتمدة حصراً (9 أنماط)
APPROVED_AZURE_STYLES = {
    "whispering", "excited", "serious", "hopeful",
    "cheerful", "sad", "angry", "shouting", "calm"
}

# -------------------------------------------------------------
# برومبتات النظام
# -------------------------------------------------------------

# [تم تعطيل مسار Google] STAGE_3_GOOGLE_SYSTEM_PROMPT = """You are an expert Voiceover Director. Convert the provided array of English sentences into an expressive script ready for Google TTS.
#
# RULES:
# 1. Preserve every single word exactly. Do not add, omit, or paraphrase text.
# 2. Insert 1 to 3 emotion tags in square brackets before each sentence, picked ONLY from the approved emotions list:
# [admiration] [amusement] [anger] [anxiety] [apology] [approval] [awe] [aggression] [boredom] [calm] [celebration] [concern] [contempt] [contentment] [curiosity] [determination] [disapproval] [disbelief] [disgust] [embarrassment] [empathy] [enthusiasm] [excitement] [fear] [frustration] [gratitude] [hope] [humor] [interest] [joy] [longing] [love] [nervousness] [nostalgia] [pride] [relief] [sadness] [sarcasm] [satisfaction] [shock] [surprise] [suspense] [sympathy] [tenderness] [tiredness] [trust] [uncertainty] [urgency] [vulnerability] [warning] [whispers] [wonder] [gasps] [cries] [sighs] [laughs]
# 3. Vary the vocal energy constantly. Never repeat the exact same emotion tag three times consecutively.
# 4. Naturally insert exactly one contextual emoji inside each sentence (embedded in the middle context, NOT mechanically at the start or end).
# 5. Output ONLY the raw spoken lines. No titles, headers, markdown code wrappers, or line numbers.
#
# LINE FORMAT:
# [emotion1] [emotion2] "Spoken sentence text with an emoji inside." """


STAGE_3_AZURE_SYSTEM_PROMPT = """You are an expert Speech Synthesis Markup Language (SSML) Architect. Convert the provided array of English sentences into a fully compliant Microsoft Azure SSML payload.

PARAMETERS:
Voice: {voice_name}
Language: en-US

RULES:
1. Preserve every single word of the input sentences strictly.
2. Wrap each sentence in an appropriate mstts:express-as style="..." tag matching the emotional delivery.
3. Allowed styles ONLY: whispering, excited, serious, hopeful, cheerful, sad, angry, shouting, calm.
4. Alternate styles dynamically based on the narrative arc.
5. STRICT NO EMOJIS: Do NOT include raw emojis inside the spoken text under any circumstances.
6. Output valid, pure SSML markup only, without markdown wrappers or conversational filler.

FORMAT:
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
<voice name="{voice_name}">
<mstts:express-as style="excited">
Sentence one text.
</mstts:express-as>
<mstts:express-as style="whispering">
Sentence two text.
</mstts:express-as>
</voice>
</speak>"""

# -------------------------------------------------------------
# دوال التحقق البرمجي الصارم (Validation Functions)
# -------------------------------------------------------------

def _clean_text_for_word_match(text: str) -> str:
    """تنظيف النص من الوسوم والإيموجي لمطابقة الكلمات حرفياً"""
    no_tags = re.sub(r"\[.*?\]", "", text)
    no_xml = re.sub(r"<.*?>", "", no_tags)
    # إزالة الإيموجي وأي علامات ترقيم
    clean = re.sub(r"[^\w\s]", "", no_xml).lower()
    return " ".join(clean.split())


# [تم تعطيل مسار Google] def validate_google_tts_lines(lines: List[str], original_sentences: List[str]):
#     """فحص شروط مسار Google TTS"""
#     if len(lines) != len(original_sentences):
#         raise ValueError(f"عدد أسطر Google TTS ({len(lines)}) لا يطابق عدد الجمل الأصلية ({len(original_sentences)})!")
#
#     last_emotions = []
#     consecutive_counts = {}
#
#     for idx, (line, orig) in enumerate(zip(lines, original_sentences), start=1):
#         # 1. فحص وسوم الانفعالات
#         tags = re.findall(r"\[([a-zA-Z]+)\]", line)
#         if not (1 <= len(tags) <= 3):
#             raise ValueError(f"السطر {idx}: يجب أن يحتوي على 1 إلى 3 وسوم انفعال. الموجود: {tags}")
#
#         for t in tags:
#             t_lower = t.lower()
#             if t_lower not in APPROVED_GOOGLE_EMOTIONS:
#                 raise ValueError(f"السطر {idx}: الانفعال '[{t}]' غير موجود في قائمة الـ 58 انفعالاً المعتمدة!")
#
#         # 2. فحص حظر تكرار نفس الانفعال 3 مرات متتالية
#         first_tag = tags[0].lower()
#         if len(last_emotions) >= 2 and last_emotions[-1] == first_tag and last_emotions[-2] == first_tag:
#             raise ValueError(f"السطر {idx}: تم تكرار الانفعال [{first_tag}] ثلاث مرات متتالية، وهذا محظور!")
#         last_emotions.append(first_tag)
#
#         # 3. فحص وجود إيموجي سياقي
#         emojis = re.findall(r"[\U00010000-\U0010ffff]", line)
#         if len(emojis) == 0:
#             logger.warning(f"⚠️ السطر {idx}: لم يتم العثور على إيموجي سياقي!")
#
#         # 4. فحص التطابق الحرفي للكلمات مع النص الأصلي
#         if _clean_text_for_word_match(line) != _clean_text_for_word_match(orig):
#             logger.warning(f"⚠️ السطر {idx}: اختلاف طفيف في الكلمات مقارنة بالسكربت الأصلي.")
#
#     logger.info("✅ فحص مسار Google TTS: النص مطابق ومتقيد بقواعد الانفعالات والإيموجي بنجاح.")


def validate_azure_ssml(ssml_text: str, original_sentences: List[str], expected_voice: str):
    """فحص شروط مسار Azure SSML"""
    # 1. منع الإيموجي تماماً
    emojis = re.findall(r"[\U00010000-\U0010ffff]", ssml_text)
    if emojis:
        raise ValueError(f"مخالفة صارمة: مسار Azure يحتوي على رموز تعبيرية (Emojis): {emojis}")

    # 2. فحص سلامة بناء XML
    try:
        root = ET.fromstring(ssml_text)
    except ET.ParseError as err:
        raise ValueError(f"كود SSML غير صالح بنيوياً (XML Parse Error): {err}")

    # 3. فحص الصوت الرجالي المحدد
    voice_elem = root.find(".//{http://www.w3.org/2001/10/synthesis}voice") or root.find(".//voice")
    if voice_elem is not None:
        voice_name = voice_elem.attrib.get("name", "")
        if voice_name != expected_voice:
            logger.warning(f"⚠️ الصوت داخل SSML '{voice_name}' لا يطابق المطلوب '{expected_voice}'")

    # 4. فحص الأنماط المعتمدة (9 أنماط فقط)
    express_elements = root.findall(".//{https://www.w3.org/2001/mstts}express-as")
    if not express_elements:
        express_elements = root.findall(".//mstts:express-as", namespaces={"mstts": "https://www.w3.org/2001/mstts"})

    for idx, elem in enumerate(express_elements, start=1):
        style = elem.attrib.get("style", "").strip().lower()
        if style not in APPROVED_AZURE_STYLES:
            raise ValueError(f"الجملة {idx}: النمط '{style}' غير معتمد في قائمة مايكروسوفت أزور المحددة!")

    logger.info("✅ فحص مسار Azure SSML: كود XML سليم، وخالٍ تماماً من الإيموجي، وأنماط الأداء معتمدة.")


# -------------------------------------------------------------
# محركات توليد الصوت الحقيقية (APIs Execution)
# -------------------------------------------------------------

def call_azure_tts_api(ssml_payload: str, output_filepath: Path):
    """استدعاء REST API الرسمي لمحرك Microsoft Azure Speech"""
    if not AZURE_SPEECH_KEY:
        raise ValueError("AZURE_SPEECH_KEY مفقود في ملف الإعدادات!")

    url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-160kbitrate-mono-mp3",
        "User-Agent": "VotExpressiveAudioEngine"
    }

    logger.info("🌐 إرسال طلب SSML إلى Microsoft Azure TTS...")
    response = requests.post(url, headers=headers, data=ssml_payload.encode("utf-8"), timeout=REQUEST_TIMEOUT)

    if response.status_code == 200:
        with open(output_filepath, "wb") as f:
            f.write(response.content)
        logger.info(f"✅ تم حفظ الصوت النهائي بنجاح في: {output_filepath}")
    else:
        raise RuntimeError(f"فشل استدعاء Azure TTS API (HTTP {response.status_code}): {response.text}")


# [تم تعطيل مسار Google] def call_google_tts_api(text_content: str, voice_name: str, output_filepath: Path):
#     """استدعاء REST API الرسمي لمحرك Google Cloud TTS عبر المفاتيح المتاحة"""
#     if not GEMINI_KEYS:
#         raise ValueError("لا يوجد مفتاح Google API صالح لاستخدامه في Cloud TTS!")
#
#     payload = {
#         "input": {"text": text_content},
#         "voice": {
#             "languageCode": "en-US",
#             "name": voice_name
#         },
#         "audioConfig": {
#             "audioEncoding": "MP3",
#             "speakingRate": 1.0,
#             "sampleRateHertz": 24000
#         }
#     }
#
#     for key_idx, key in enumerate(GEMINI_KEYS, start=1):
#         url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={key}"
#         logger.info(f"🌐 إرسال الطلب إلى Google Cloud TTS بالمفتاح [{key_idx}]...")
#         try:
#             resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
#             if resp.status_code == 200:
#                 audio_base64 = resp.json().get("audioContent", "")
#                 if audio_base64:
#                     with open(output_filepath, "wb") as f:
#                         f.write(base64.b64decode(audio_base64))
#                     logger.info(f"✅ تم حفظ الصوت النهائي بنجاح في: {output_filepath}")
#                     return
#             else:
#                 logger.warning(f"⚠️ فشل مفتاح Google [{key_idx}] (HTTP {resp.status_code}): {resp.text[:120]}")
#         except Exception as e:
#             logger.warning(f"⚠️ استثناء أثناء الاتصال بـ Google TTS: {e}")
#
#     raise RuntimeError("فشلت كافة المحاولات لتوليد الصوت عبر Google Cloud TTS!")


# -------------------------------------------------------------
# الدالة التنفيذية الشاملة للمرحلة الثالثة
# -------------------------------------------------------------

def generate_stage3_audio(
    episode_id: str,
    sentences: List[str],
    engine: str = "azure",
    voice: str = "en-US-GuyNeural",
    output_dir: Path = Path("outputs")
) -> Path:
    """
    الدالة المحورية للمرحلة الثالثة (Azure فقط):
    1. التحقق من الصوت (رجالي فقط).
    2. صياغة النص التعبيري عبر Gemini.
    3. التحقق الصارم من القواعد (أنماط، بدون إيموجي).
    4. استدعاء Azure API لإنتاج ملف صوتي مدمج.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    final_audio_file = output_dir / f"episode_{episode_id}_audio.mp3"

    engine = engine.lower().strip()

    if engine != "azure":
        raise ValueError(f"محرك الصوت '{engine}' غير مدعوم! المحرك المتاح حالياً: 'azure' فقط.")

    if voice not in AZURE_MALE_VOICES:
        raise ValueError(f"الصوت '{voice}' غير مسموح به! الأصوات المعتمدة حصراً لـ Azure هي رجالية: {AZURE_MALE_VOICES}")

    logger.info(f"🎙️ تجهيز مسار Microsoft Azure SSML بالصوت الرجالي: {voice}")
    system_prompt = STAGE_3_AZURE_SYSTEM_PROMPT.format(voice_name=voice)
    user_prompt = f"Convert these {len(sentences)} sentences into Azure SSML:\n" + "\n".join(f"- {s}" for s in sentences)

    ssml_output = call_gemini_with_fallback(
        system_instruction=system_prompt,
        user_prompt=user_prompt,
        response_mime_type="text/plain"
    )

    # تنظيف من علامات الماركداون
    ssml_cleaned = re.sub(r"^```(?:xml|ssml)?\s*", "", ssml_output.strip())
    ssml_cleaned = re.sub(r"\s*```$", "", ssml_cleaned).strip()

    validate_azure_ssml(ssml_cleaned, sentences, voice)
    call_azure_tts_api(ssml_cleaned, final_audio_file)

    return final_audio_file


# -------------------------------------------------------------
# دالة توليد عينة صوت قصيرة (Voice Preview) - Azure فقط
# -------------------------------------------------------------

def generate_voice_preview(
    voice: str = None,
    voice_name: str = None,
    text: str = "Hello, this is a sample of my voice. How do I sound to you?",
    output_dir: Path = Path("outputs")
) -> Path:
    """
    تولد عينة صوت قصيرة باستخدام Microsoft Azure فقط.
    تدعم كلاً من voice و voice_name للتوافق مع الاستدعاءات المختلفة.
    """
    # توحيد اسم الصوت
    final_voice = voice or voice_name
    if not final_voice:
        raise ValueError("يجب تمرير اسم الصوت (voice أو voice_name)")

    output_dir.mkdir(parents=True, exist_ok=True)
    # تنظيف اسم الملف من أي رموز غريبة
    safe_name = final_voice.replace("/", "_").replace("\\", "_")
    preview_file = output_dir / f"preview_{safe_name}.mp3"

    ssml_payload = (
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="https://www.w3.org/2001/mstts" '
        'xml:lang="en-US">\n'
        f'<voice name="{final_voice}">\n'
        '<mstts:express-as style="calm">\n'
        f'{text}\n'
        '</mstts:express-as>\n'
        '</voice>\n'
        '</speak>'
    )

    logger.info(f"🎧 توليد عينة صوتية (preview) للصوت: {final_voice}")
    call_azure_tts_api(ssml_payload, preview_file)
    return preview_file
