import re
import logging
from typing import List, Dict, Any, Optional, Union
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


STAGE_2_ENRICHED_SYSTEM_PROMPT = """You are an expert AI Art Director, Visual Storyteller, and Storyboard Artist. Your task is to generate image prompts for an educational YouTube video based on a sequential list of script sentences, each mapped to a scene from an existing scene plan and visual bible.

=========================================
ABSOLUTE RULE: NARRATION ≠ LITERAL IMAGE
=========================================
NEVER translate a sentence into a literal illustration of its words.
Instead, extract:
- The core idea
- The mechanism or reason
- The conflict or tension
- The emotional state
- The result or transformation
Then translate THAT into a conceptual, symbolic, or story-driven visual scene.

Forbidden example:
Narration: "Your brain prefers immediate rewards."
Literal (FORBIDDEN): A brain with a reward icon next to it.

Required approach:
The orange character stands before two doors:
- A near, bright, easy door.
- A far, difficult door.
- The real goal lies behind the far door.
- Composition emphasizes the conflict of choice and instant reward.

Forbidden example:
Narration: "Weak knees are not caused by movement itself."
Literal (FORBIDDEN): Character just holding their knee.

Required approach:
A symbolic scene showing progressive loading strengthening the knee — e.g., a bridge that grows more stable as the character walks across it step by step.

=========================================
CHARACTER DNA (NEVER CHANGE)
=========================================
- Orange skin
- Muscular defined body
- Smooth head
- Two large white oval eyes
- NO mouth
- Black shorts
- Consistent proportions
- 2D cel-shaded illustration style

NEVER invent a new character. NEVER change clothing, head shape, or eyes. The character should stay on-screen unless absolutely necessary; even then, keep the visual identity consistent.

=========================================
FIXED VISUAL CONSTRAINTS (PRESERVE ALL)
=========================================
- Plain grey background (dominant). If symbolic environment/objects are needed, add them while keeping the grey backdrop dominant.
- Very small, subtle, faint number "[INDEX]" in the bottom-right corner.
- Numbering follows the sentence order.
- Clean, cel-shaded style.
- Consistent proportions.
- No mouth.
- Black shorts.
- ONE prompt per sentence.

=========================================
EACH PROMPT MUST INCLUDE
=========================================
1. Character constants.
2. The non-literal visual idea (conceptual / symbolic / story-driven).
3. Character action / dynamic pose.
4. Environment or symbolic visual element.
5. Emotional state.
6. Camera angle.
7. Composition.
8. Lighting / colors if relevant.
9. Continuity with previous and upcoming scenes.
10. Correct sequential index in the bottom-right.
11. Plain grey background.
12. No text inside the image except the index number.

=========================================
FORBIDDEN IN OUTPUT
=========================================
- Original sentence text as a title.
- Extra numbers beyond the image index.
- Headers.
- Markdown.
- Quotes around the prompt.
- Any explanation outside the prompts.
- Combining two sentences into one prompt.
- Skipping a sentence.
- Adding an extra prompt.

=========================================
SCENE HANDLING
=========================================
A scene may cover multiple sentences. Keep the same visual world within a scene, but give every sentence its own independent prompt.

Example progression inside one scene:
- Sentence 1: Establishing shot.
- Sentence 2: Continuation of the action.
- Sentence 3: Escalation.
- Sentence 4: Reveal or transformation.

Do NOT make all sentences in a scene identical images. Maintain:
- Same environment.
- Same key elements.
- Same character identity.
- Evolution of motion and composition from prompt to prompt.

=========================================
OUTPUT FORMAT
=========================================
Return prompts separated ONLY by a single blank line. No quotation marks, no markdown code wrappers, no sentence text, no headers."""


BATCH_SIZE = 24


def clean_and_parse_prompts(raw_text: str) -> List[str]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:text)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return [b.strip() for b in cleaned.split("\n\n") if b.strip()]


def _map_sentences_to_scenes(
    sentences: List[str],
    scene_plan: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]],
) -> Dict[int, Dict[str, Any]]:
    """
    يرجع قاموساً: index الجملة (1-based) -> بيانات المشهد الخاص بها.

    المرحلة الأولى تُنتج scene_plan كقائمة مباشرة:
        [
            {"scene_id": 1, "sentence_start": 0, "sentence_end": 3, ...},
            ...
        ]

    فهارس scene_plan هي 0-based (sentence_start / sentence_end)،
    بينما أرقام الجمل والـimage prompts في المرحلة الثانية هي 1-based.
    لذلك نقوم بتحويل one_based = zero_based + 1.

    للتوافق الخلفي فقط: إذا وصل scene_plan كقاموس يحتوي على مفتاح "scenes"
    وكانت قيمته قائمة، فسنستخرجها. لكن المسار الأساسي هو القائمة المباشرة.
    """
    mapping: Dict[int, Dict[str, Any]] = {}
    if not scene_plan:
        return mapping

    scenes: Optional[List[Dict[str, Any]]] = None

    # المسار الأساسي: قائمة مباشرة (كما تنتجها stage1_generator)
    if isinstance(scene_plan, list):
        scenes = scene_plan
    # توافق خلفي فقط: قاموس يحتوي "scenes"
    elif isinstance(scene_plan, dict):
        candidate = scene_plan.get("scenes")
        if isinstance(candidate, list):
            scenes = candidate

    if not scenes:
        return mapping

    total_sentences = len(sentences)

    for scene in scenes:
        if not isinstance(scene, dict):
            continue

        s_start = scene.get("sentence_start")
        s_end = scene.get("sentence_end")

        if not isinstance(s_start, int) or not isinstance(s_end, int):
            continue

        # فهارس المرحلة الأولى 0-based
        if s_start < 0 or s_end < s_start:
            continue

        for zero_based_idx in range(s_start, s_end + 1):
            one_based_idx = zero_based_idx + 1
            if 1 <= one_based_idx <= total_sentences:
                mapping[one_based_idx] = scene

    return mapping


def _format_scene_context(scene: Optional[Dict[str, Any]]) -> str:
    if not scene:
        return "No scene context available. Use pure narrative-driven visual storytelling."

    fields = [
        ("Scene ID", scene.get("scene_id")),
        ("Title", scene.get("title")),
        ("Narrative Purpose", scene.get("narrative_purpose")),
        ("Visual Concept", scene.get("visual_concept")),
        ("Emotion", scene.get("emotion")),
        ("Character Action", scene.get("character_action")),
        ("Environment", scene.get("environment")),
        ("Camera", scene.get("camera")),
        ("Composition", scene.get("composition")),
        ("Motion Potential", scene.get("motion_potential")),
        ("Transition", scene.get("transition")),
        ("Continuity Notes", scene.get("continuity_notes")),
    ]
    lines = [f"- {k}: {v}" for k, v in fields if v]
    if not lines:
        return "No detailed scene context available."
    return "\n".join(lines)


def _format_visual_bible(visual_bible: Optional[Dict[str, Any]]) -> str:
    if not visual_bible:
        return "No visual bible provided. Rely on the fixed character DNA and constraints."

    lines: List[str] = []
    for key, value in visual_bible.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            joined = "; ".join(str(v) for v in value if v)
            if joined:
                lines.append(f"- {key}: {joined}")
        elif isinstance(value, dict):
            sub = "; ".join(f"{k}={v}" for k, v in value.items() if v)
            if sub:
                lines.append(f"- {key}: {sub}")
        else:
            lines.append(f"- {key}: {value}")
    if not lines:
        return "No visual bible details available."
    return "\n".join(lines)


def _process_single_batch(batch_tuple: tuple) -> tuple:
    """
    معالجة دفعة واحدة في مسار مستقل (Thread) لتأخذ مفتاحاً مستقلاً.
    batch_tuple يحتوي:
      (batch_idx, batch_sentences, start_idx, end_idx,
       scene_map, visual_bible, has_stage1_context)
    """
    (
        batch_idx,
        batch_sentences,
        start_idx,
        end_idx,
        scene_map,
        visual_bible,
        has_stage1_context,
    ) = batch_tuple

    logger.info(
        f"🚀 بدء معالجة الدفعة [{batch_idx}] بالتوازي: الجمل من {start_idx} إلى {end_idx}"
    )

    if has_stage1_context:
        system_instruction = STAGE_2_ENRICHED_SYSTEM_PROMPT
        user_prompt = f"""Generate exactly {len(batch_sentences)} prompts.
The index for this batch MUST start sequentially at {start_idx} and end at {end_idx}.

Each sentence has its own scene context below. Every prompt MUST:
- Reflect the non-literal, conceptual visual storytelling for THAT sentence.
- Build on the scene's visual_concept, emotion, character_action, environment, camera, composition, continuity, and transitions.
- Preserve the character DNA and all fixed visual constraints.
- Evolve across the scene's sentences (establishing → continuation → escalation → reveal), not repeat identical images.

VISUAL BIBLE (global style anchor for the whole video):
{_format_visual_bible(visual_bible)}

SENTENCES AND THEIR SCENE CONTEXTS:
"""
        for s_idx, sent in enumerate(batch_sentences, start=start_idx):
            scene = scene_map.get(s_idx)
            user_prompt += f"\n[{s_idx}] Sentence: {sent}\nScene context:\n{_format_scene_context(scene)}\n"

        user_prompt += (
            "\nReturn the prompts separated ONLY by a single blank line, in order, "
            "with no headers, no labels, no markdown, and no quotes."
        )
    else:
        system_instruction = STAGE_2_SYSTEM_PROMPT
        user_prompt = f"""Process the following sentences and generate exactly {len(batch_sentences)} prompts.
The index for this batch MUST start sequentially at {start_idx} and end at {end_idx}.

Sentences:
"""
        for s_idx, sent in enumerate(batch_sentences, start=start_idx):
            user_prompt += f"{s_idx}. {sent}\n"

    raw_output = call_gemini_with_fallback(
        system_instruction=system_instruction,
        user_prompt=user_prompt,
        response_mime_type="text/plain",
    )

    prompts = clean_and_parse_prompts(raw_output)
    return batch_idx, prompts


def _validate_batch(batch_idx: int, expected_count: int, prompts: List[str]) -> None:
    if len(prompts) != expected_count:
        raise ValueError(
            f"Batch [{batch_idx}] returned {len(prompts)} prompts, "
            f"expected exactly {expected_count}. Refusing partial/inflated batch."
        )


def generate_stage2_prompts_batches(
    sentences: List[str],
    stage1_result: Optional[Dict[str, Any]] = None,
) -> List[List[str]]:
    """
    توزيع كافة دفعات أوامر الصور على خيوط متوازية (Parallel Threads).
    كل خيط يحصل تلقائياً على مفتاح مختلف من مصفوفة المفاتيح.

    إذا تم تمرير stage1_result (يحتوي scene_plan / visual_bible / full_script_sentences)
    يتم استخدام سياق المرحلة الأولى لتوليد Prompts مفاهيمية غير حرفية.

    إذا لم يُمرَّر stage1_result أو كان ناقصاً، يتم الرجوع للسلوك القديم (fallback)
    دون أي فشل.
    """
    total_sentences = len(sentences)

    # استخراج السياق من المرحلة الأولى (اختياري)
    scene_plan: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    visual_bible: Optional[Dict[str, Any]] = None
    has_stage1_context = False

    if isinstance(stage1_result, dict):
        candidate_plan = stage1_result.get("scene_plan")
        candidate_bible = stage1_result.get("visual_bible")
        if candidate_plan or candidate_bible:
            scene_plan = candidate_plan
            visual_bible = candidate_bible
            has_stage1_context = True

    if not has_stage1_context:
        logger.info(
            "ℹ️ لا يوجد scene_plan / visual_bible — سيتم استخدام السلوك القديم (fallback)."
        )

    # خريطة الجملة -> المشهد (1-based sentence index -> scene dict)
    scene_map = _map_sentences_to_scenes(sentences, scene_plan) if has_stage1_context else {}

    # تجهيز بيانات الدفعات
    batch_tasks = []
    for batch_num, i in enumerate(range(0, total_sentences, BATCH_SIZE), start=1):
        batch_sentences = sentences[i : i + BATCH_SIZE]
        start_idx = i + 1
        end_idx = start_idx + len(batch_sentences) - 1
        batch_tasks.append(
            (
                batch_num,
                batch_sentences,
                start_idx,
                end_idx,
                scene_map,
                visual_bible,
                has_stage1_context,
            )
        )

    logger.info(
        f"⚡ تشغيل {len(batch_tasks)} دفعات بالتوازي عبر مفاتيح مختلفة في نفس اللحظة..."
    )

    completed_results: Dict[int, List[str]] = {}

    with ThreadPoolExecutor(max_workers=min(len(batch_tasks), 4)) as executor:
        future_to_batch = {
            executor.submit(_process_single_batch, task): task[0] for task in batch_tasks
        }
        for future in as_completed(future_to_batch):
            batch_num, prompts = future.result()

            # التحقق الصارم: عدد الـPrompts في الدفعة = عدد الجمل في الدفعة
            expected = len(batch_tasks[batch_num - 1][1])
            _validate_batch(batch_num, expected, prompts)

            completed_results[batch_num] = prompts
            logger.info(
                f"✅ انتهت الدفعة [{batch_num}] بنجاح ({len(prompts)} برومبت)"
            )

    # تجميع النتائج بالترتيب المتسلسل السليم
    sorted_batches = [completed_results[k] for k in sorted(completed_results.keys())]

    # تحقق نهائي إضافي: الطول الكلي مطابق
    total_generated = sum(len(b) for b in sorted_batches)
    if total_generated != total_sentences:
        raise ValueError(
            f"Total prompts generated ({total_generated}) "
            f"does not match total sentences ({total_sentences})."
        )

    return sorted_batches
