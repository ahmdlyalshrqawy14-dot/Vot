import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from gemini_engine import call_gemini_with_fallback

logger = logging.getLogger("Stage5Metadata")

STAGE_5_SYSTEM_PROMPT = """You are a YouTube Metadata Strategist and Packaging Specialist. Your job is to create high-converting packaging assets (titles, thumbnail prompts, description, and tags) based on the episode's topic, script, and core takeaways.

INPUT:
You will receive the episode metadata and script summary.

STRICT GENERATION RULES:

1. TITLES:
- Provide exactly 3 high-CTR titles in clean, spoken English.
- Each title must represent a distinct marketing angle:
  * Angle 1: Curiosity (targets mystery or unconventional truth)
  * Angle 2: Pain Point (targets visceral frustration and physical suffering)
  * Angle 3: Outcome/Benefit (targets practical results and actionable clarity)
- Avoid cheap clickbait; keep them honest, punchy, and mobile-optimized.
- Format for titles_message:
🎯 Recommended Titles:

1. [Title 1] (Curiosity Angle)
2. [Title 2] (Pain Point Angle)
3. [Title 3] (Outcome Angle)

2. THUMBNAIL PROMPTS:
- Provide exactly 3 thumbnail prompts aligned with the 3 titles above (one for curiosity, one for pain, one for outcome).
- Use the FIXED character template:
Create a 2D muscular character illustration in the exact style of the provided example. The figure should be orange, with a smooth head, large white oval eyes, no mouth, and a defined muscular body wearing black shorts. Maintain the same proportions and facial features as in the reference. [DRAMATIC_THUMBNAIL_POSE]. Background must be plain grey. Clean, cel-shaded, and high-impact expressive style.
- [DRAMATIC_THUMBNAIL_POSE] must be exaggerated, dramatic, and instantly readable at a small mobile scale.
- STRICT NO-INDEX RULE: DO NOT include any subtle numbers, faint indices, or corner index numbers in thumbnail prompts.
- Separate each prompt ONLY by a blank line.

3. DESCRIPTION:
- First 2 lines must hook the reader and summarize the value before the 'Show More' fold.
- Concise bulleted breakdown of core insights.
- Authentic Call to Action linked to the niche.
- Conclude with the episode's Question of the Day to drive comments.
- Format for description_message:
📝 Video Description:

[Hook line 1]
[Hook line 2]

[Breakdown of points...]

[Call to action...]

Question of the Day: [Question]? Drop your experience in the comments below!

4. TAGS:
- High-volume, highly relevant English keywords (niche specific + broad).
- Strictly on a SINGLE LINE, separated by commas.
- Format for tags_message:
🏷️ Video Tags:
tag1, tag2, tag3, tag4, tag5

OUTPUT FORMAT:
Return a valid, parsable JSON object ONLY, with no markdown code blocks and no conversational filler:
{
  "titles_message": "string",
  "thumbnails_message": "string",
  "description_message": "string",
  "tags_message": "string"
}"""


def _clean_json_str(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def validate_stage5_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    التحقق الصارم من متطلبات المرحلة الخامسة:
    1. وجود الحقول الأربعة المخصصة لرسائل تليجرام.
    2. التحقق من ثوابت برومبتات الغلاف وخلوها التام من أي أرقام باهتة.
    3. التحقق من خلو سطر التاجز من فواصل أسطر متعددة.
    """
    required_keys = ["titles_message", "thumbnails_message", "description_message", "tags_message"]
    for k in required_keys:
        if k not in data or not str(data[k]).strip():
            raise ValueError(f"الحقل المطلوب '{k}' مفقود أو فارغ في مخرجات المرحلة الخامسة!")

    # 1. فحص برومبتات الثامبنيل
    thumbs = data["thumbnails_message"]
    # حظر شرط الأرقام الباهتة الخاص بالمرحلة الثانية
    prohibited_patterns = ["faint number", "subtle faint", "[INDEX]", "bottom-right corner, include"]
    for pattern in prohibited_patterns:
        if pattern.lower() in thumbs.lower():
            raise ValueError(f"مخالفة صارمة: برومبتات الغلاف تحتوي على شرط الرقم التسلسلي ({pattern})!")

    # فحص ثوابت الشخصية
    if "orange" not in thumbs.lower() or "plain grey" not in thumbs.lower() or "black shorts" not in thumbs.lower():
        logger.warning("⚠️ تنبيه: قد تكون بعض مواصفات الشخصية الثابتة غير مكتملة في أوامر الغلاف.")

    # 2. فحص التاجز
    tags_msg = data["tags_message"].strip()
    tags_body = tags_msg.replace("🏷️ Video Tags:", "").strip()
    if "\n" in tags_body:
        # إزالة أي فواصل أسطر في سطر التاجز لضمان بقائها في سطر واحد
        data["tags_message"] = "🏷️ Video Tags:\n" + ", ".join([t.strip() for t in tags_body.splitlines() if t.strip()])

    logger.info("✅ تم التحقق البرمجي بنجاح: حزمة النشر الرقمي مطابقة لكافة المعايير الصارمة.")
    return data


def generate_stage5_metadata(episode_data: Dict[str, Any], script_sentences: list) -> Dict[str, Any]:
    """
    توليد حزمة النشر الرقمي (العناوين، الغلاف، الوصف، والتاجز).
    """
    summary_text = (
        f"Topic: {episode_data.get('topic')}\n"
        f"The Myth: {episode_data.get('the_myth')}\n"
        f"The Truth: {episode_data.get('the_truth')}\n"
        f"Actionable Solution: {episode_data.get('actionable_solution')}\n"
        f"Core Takeaway: {episode_data.get('core_takeaway')}\n"
        f"Comment Question: {episode_data.get('comment_question')}\n"
        f"Total Sentences in Video: {len(script_sentences)}\n"
        f"Script Sample (First 3 sentences): {' '.join(script_sentences[:3])}"
    )

    user_prompt = f"Generate the YouTube packaging metadata for this episode based on the provided data:\n\n{summary_text}"

    logger.info("📦 بدء استدعاء Gemini لتوليد حزمة التغليف والنشر الرقمي...")
    raw_response = call_gemini_with_fallback(
        system_instruction=STAGE_5_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_mime_type="application/json"
    )

    cleaned_json = _clean_json_str(raw_response)
    try:
        parsed = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        logger.error(f"خطأ في فك تشفير JSON: {e}\nالرد الخام:\n{cleaned_json}")
        raise

    return validate_stage5_output(parsed)
