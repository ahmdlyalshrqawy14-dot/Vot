import re
import logging
from typing import List, Dict, Any
from gemini_engine import call_gemini_with_fallback

logger = logging.getLogger("Stage2Generator")

STAGE_2_SYSTEM_PROMPT = """You are an expert AI Art Director and Visual Storyboard Artist. Your task is to generate image prompts for an educational YouTube video based on a sequential list of script sentences.

BASE PROMPT TEMPLATE (FIXED)
Create a 2D muscular character illustration in the exact style of the provided example. The figure should be orange, with a smooth head, large white oval eyes, no mouth, and a defined muscular body wearing black shorts. Maintain the same proportions and facial features as in the reference. [DYNAMIC_POSE]. Background must be plain grey. In the bottom-right corner, include a very small, subtle, faint number "[INDEX]". Clean, cel-shaded, and expressive style. Keep the style consistent across all generated images.

STRICT GENERATION RULES:
- ONE PROMPT PER SENTENCE: Generate exactly one prompt for every sentence received in chronological order.
- INDEXING: Replace [INDEX] with the sequential image number provided in the instructions (e.g., "1", "2", "20").
- POSE CUSTOMIZATION:
  * Replace [DYNAMIC_POSE] with a clear action, posture, or physical gesture matching the emotional and physical context of that sentence.
  * NEVER alter character physical traits (orange skin, oval white eyes, no mouth, black shorts) or the plain grey background.
- OUTPUT FORMAT: Return the prompts separated ONLY by a single blank line. No quotation marks, no markdown code wrappers, no sentence text, and no headers."""

BATCH_SIZE = 24


def clean_and_parse_prompts(raw_text: str) -> List[str]:
    """تنظيف وتفكيك الأوامر النصية المفصولة بسطر فارغ"""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:text)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    
    # التقسيم بالسطر الفارغ
    blocks = [b.strip() for b in cleaned.split("\n\n") if b.strip()]
    return blocks


def validate_prompts_batch(prompts: List[str], expected_count: int, start_index: int) -> bool:
    """التحقق من مطابقة الدفعة لمعايير المرحلة الثانية"""
    if len(prompts) != expected_count:
        logger.warning(f"⚠️ عدم تطابق العدد في الدفعة: تم توليد {len(prompts)} والمتوقع {expected_count}")
        return False

    for idx, prompt in enumerate(prompts):
        current_index = start_index + idx
        expected_number_str = f'"{current_index}"'
        
        # فحص وجود الرقم الصحيح
        if expected_number_str not in prompt:
            logger.warning(f"⚠️ الرقم التسلسلي {expected_number_str} مفقود في البرومبت رقم {current_index}")
        
        # فحص ثوابت الشخصية
        if "orange" not in prompt or "plain grey" not in prompt or "black shorts" not in prompt:
            logger.warning(f"⚠️ الثوابت البصرية للشخصية أو الخلفية غير مكتملة في البرومبت {current_index}")
            
    return True


def generate_stage2_prompts_batches(sentences: List[str]) -> List[List[str]]:
    """
    توليد أوامر الصور وتجزئتها بدفعات لا تتجاوز 24 أمراً للدفعة الواحدة
    مع الحفاظ على تسلسل الـ [INDEX] من 1 حتى نهاية السكربت.
    """
    total_sentences = len(sentences)
    all_batches = []
    
    for batch_num, i in enumerate(range(0, total_sentences, BATCH_SIZE), start=1):
        batch_sentences = sentences[i : i + BATCH_SIZE]
        start_idx = i + 1
        end_idx = start_idx + len(batch_sentences) - 1
        
        logger.info(f"🎨 معالجة الدفعة [{batch_num}]: الجمل من {start_idx} إلى {end_idx} (إجمالي الدفعة: {len(batch_sentences)})")
        
        user_prompt = f"""Process the following sentences and generate exactly {len(batch_sentences)} prompts.
The index for this batch MUST start sequentially at {start_idx} and end at {end_idx}.

Sentences:
"""
        for s_idx, sent in enumerate(batch_sentences, start=start_idx):
            user_prompt += f"{s_idx}. {sent}\n"

        raw_output = call_gemini_with_fallback(
            system_instruction=STAGE_2_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_mime_type="text/plain"
        )
        
        prompts = clean_and_parse_prompts(raw_output)
        validate_prompts_batch(prompts, len(batch_sentences), start_idx)
        all_batches.append(prompts)

    return all_batches
