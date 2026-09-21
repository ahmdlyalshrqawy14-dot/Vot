import re
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:text)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return [b.strip() for b in cleaned.split("\n\n") if b.strip()]


def _process_single_batch(batch_tuple: tuple) -> tuple:
    """معالجة دفعة واحدة في مسار مستقل (Thread) لتأخذ مفتاحاً مستقلاً"""
    batch_idx, batch_sentences, start_idx, end_idx = batch_tuple

    logger.info(f"🚀 بدء معالجة الدفعة [{batch_idx}] بالتوازي: الجمل من {start_idx} إلى {end_idx}")

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
    return batch_idx, prompts


def generate_stage2_prompts_batches(sentences: List[str]) -> List[List[str]]:
    """
    توزيع كافة دفعات أوامر الصور على خيوط متوازية (Parallel Threads)
    حيث يحصل كل خيط تلقائياً على مفتاح مختلف من مصفوفة المفاتيح الأربعة!
    """
    total_sentences = len(sentences)
    batch_tasks = []
    
    # تجهيز بيانات الدفعات
    for batch_num, i in enumerate(range(0, total_sentences, BATCH_SIZE), start=1):
        batch_sentences = sentences[i : i + BATCH_SIZE]
        start_idx = i + 1
        end_idx = start_idx + len(batch_sentences) - 1
        batch_tasks.append((batch_num, batch_sentences, start_idx, end_idx))

    logger.info(f"⚡ تشغيل {len(batch_tasks)} دفعات بالتوازي عبر مفاتيح مختلفة في نفس اللحظة...")

    completed_results = {}
    # تشغيل الدفعات بالتوازي بعدد المهام (كل دفعة على مسار ومفتاح)
    with ThreadPoolExecutor(max_workers=min(len(batch_tasks), 4)) as executor:
        future_to_batch = {executor.submit(_process_single_batch, task): task[0] for task in batch_tasks}
        for future in as_completed(future_to_batch):
            batch_num, prompts = future.result()
            completed_results[batch_num] = prompts
            logger.info(f"✅ انتهت الدفعة [{batch_num}] بنجاح ({len(prompts)} برومبت)")

    # تجميع النتائج بالترتيب المتسلسل السليم (1, 2, 3...)
    sorted_batches = [completed_results[k] for k in sorted(completed_results.keys())]
    return sorted_batches
