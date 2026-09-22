import re
import logging
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from gemini_engine import call_gemini_with_fallback

logger = logging.getLogger("Stage2Generator")


STAGE_2_SYSTEM_PROMPT = """You are an expert AI Art Director and Visual Storyboard Artist. Your task is to generate explicit IMAGE GENERATION COMMANDS for an educational YouTube video based on a sequential list of script sentences.

=========================================
CORE PRINCIPLE: NARRATION ≠ LITERAL IMAGE
=========================================
You MUST NOT translate a sentence into a literal illustration of its words.
Instead, internally extract:
- The core idea
- The mechanism or reason
- The conflict or tension
- The emotional state
- The result or transformation
Then translate THAT into a conceptual, symbolic, or story-driven visual scene.

FORBIDDEN literal example:
Narration: "Your brain prefers immediate rewards."
Literal (FORBIDDEN): A brain with a reward icon next to it.

REQUIRED conceptual approach:
The orange character stands between two doors: a nearby easy door leading to a dead end, and a distant difficult door leading toward the real goal, body language showing the conflict of choice.

FORBIDDEN literal example:
Narration: "Weak knees are not caused by movement itself."
Literal (FORBIDDEN): Character just holding their knee.

REQUIRED symbolic approach:
A bridge that grows more stable as the character walks across it step by step, symbolizing progressive loading strengthening the knee.

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

NEVER invent a new character. NEVER change clothing, head shape, or eyes. Keep the same visual identity across every single image.

=========================================
ABSOLUTE VISUAL CONSTRAINTS (PRESERVE ALL)
=========================================
- Plain grey background (dominant). If symbolic environment or objects are needed, add them while keeping the grey backdrop dominant.
- Very small, subtle, faint image index number in the bottom-right corner.
- Numbering follows the sentence order, sequential, no gaps.
- Clean, cel-shaded style.
- Consistent proportions.
- No mouth.
- Black shorts.
- No text inside the image except the required small index number.

=========================================
EVERY PROMPT MUST BE AN EXPLICIT IMAGE GENERATION COMMAND
=========================================
Each prompt MUST start EXACTLY with:
"Create a 2D cel-shaded illustration showing ..."

It must read as a direct instruction to an image generation model, not a descriptive paragraph.
No prompt is allowed to begin with any other wording.

EACH PROMPT MUST CONTAIN
1. An explicit image creation command beginning EXACTLY with "Create a 2D cel-shaded illustration showing ...".
2. Fixed character DNA (orange, muscular, smooth head, large white oval eyes, no mouth, black shorts, consistent proportions, 2D cel-shaded).
3. The NON-LITERAL visual idea (conceptual / symbolic / story-driven).
4. Character action / dynamic pose.
5. Environment or symbolic visual element.
6. Emotional state.
7. Camera angle.
8. Composition.
9. Lighting / colors if relevant.
10. Continuity with previous and upcoming scenes.
11. Plain grey dominant background.
12. Very small subtle faint image index in the bottom-right corner.
13. No text inside the image except the required index.

=========================================
STRICT GENERATION RULES
=========================================
- ONE COMMAND PER SENTENCE: exactly one image prompt per sentence, in chronological order.
- INDEXING: replace [INDEX] with the sequential image number provided in the instructions.
- POSE CUSTOMIZATION: give a clear action, posture, or physical gesture matching the emotional and physical context of that sentence. NEVER alter character physical traits or the plain grey background.
- NO forbidden content: no literal restatement of the sentence, no extra numbers beyond the image index, no headers, no markdown, no quotes around the prompt, no explanation outside the prompts, no combining sentences, no skipping, no adding extra prompts.

=========================================
OUTPUT FORMAT
=========================================
Return the prompts separated ONLY by a single blank line. No quotation marks, no markdown code wrappers, no sentence text, no headers, no labels."""


STAGE_2_ENRICHED_SYSTEM_PROMPT = """You are an expert AI Art Director, Visual Storyteller, and Storyboard Artist. Your task is to generate explicit IMAGE GENERATION COMMANDS for an educational YouTube video based on a sequential list of script sentences, each mapped to a scene from an existing scene plan and visual bible.

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
- No text inside the image except the required index.

=========================================
EVERY PROMPT MUST BE AN EXPLICIT IMAGE GENERATION COMMAND
=========================================
Each prompt MUST start EXACTLY with:
"Create a 2D cel-shaded illustration showing ..."

The wording must read as a direct instruction to an image generation model, not a passive description.
No prompt is allowed to begin with any other wording.

EACH PROMPT MUST INCLUDE
1. An explicit image creation command starting EXACTLY with "Create a 2D cel-shaded illustration showing ...".
2. Character constants (orange, muscular, smooth head, large white oval eyes, no mouth, black shorts, consistent proportions, 2D cel-shaded).
3. The non-literal visual idea (conceptual / symbolic / story-driven).
4. Character action / dynamic pose.
5. Environment or symbolic visual element.
6. Emotional state.
7. Camera angle.
8. Composition.
9. Lighting / colors if relevant.
10. Continuity with previous and upcoming scenes.
11. Correct sequential index in the bottom-right.
12. Plain grey dominant background.
13. No text inside the image except the index number.

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
- Literal restatement of the narration.
- Any prompt that does NOT begin with "Create a 2D cel-shaded illustration showing ...".

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
        return "No scene context available. Use pure narrative-driven conceptual visual storytelling."

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


def _format_creative_brief(creative_brief: Optional[Any]) -> str:
    """
    تنسيق آمن للـcreative_brief القادم من المرحلة الأولى.
    يقبل dict / list / str / أي نوع آخر دون أن يفشل.
    """
    if not creative_brief:
        return "No creative brief provided. Anchor on the scene-level context and character DNA."

    if isinstance(creative_brief, dict):
        priority_keys = [
            "core_idea",
            "unique_angle",
            "central_conflict",
            "unexpected_insight",
            "episode_concept",
            "tone",
            "emotional_arc",
        ]
        lines: List[str] = []
        used_keys = set()

        for key in priority_keys:
            if key in creative_brief and creative_brief.get(key):
                value = creative_brief[key]
                if isinstance(value, (list, tuple)):
                    value = "; ".join(str(v) for v in value if v)
                elif isinstance(value, dict):
                    value = "; ".join(f"{k}={v}" for k, v in value.items() if v)
                lines.append(f"- {key}: {value}")
                used_keys.add(key)

        for key, value in creative_brief.items():
            if key in used_keys or value is None or value == "":
                continue
            if isinstance(value, (list, tuple)):
                value = "; ".join(str(v) for v in value if v)
            elif isinstance(value, dict):
                value = "; ".join(f"{k}={v}" for k, v in value.items() if v)
            lines.append(f"- {key}: {value}")

        if not lines:
            return "No creative brief details available."
        return "\n".join(lines)

    if isinstance(creative_brief, (list, tuple)):
        joined = "; ".join(str(v) for v in creative_brief if v)
        return joined or "No creative brief details available."

    return str(creative_brief)


def _process_single_batch(batch_tuple: tuple) -> tuple:
    """
    معالجة دفعة واحدة في مسار مستقل (Thread) لتأخذ مفتاحاً مستقلاً.
    batch_tuple يحتوي:
      (batch_idx, batch_sentences, start_idx, end_idx,
       scene_map, visual_bible, creative_brief, has_stage1_context)
    """
    (
        batch_idx,
        batch_sentences,
        start_idx,
        end_idx,
        scene_map,
        visual_bible,
        creative_brief,
        has_stage1_context,
    ) = batch_tuple

    logger.info(
        f"🚀 بدء معالجة الدفعة [{batch_idx}] بالتوازي: الجمل من {start_idx} إلى {end_idx}"
    )

    if has_stage1_context:
        system_instruction = STAGE_2_ENRICHED_SYSTEM_PROMPT
        user_prompt = f"""Generate exactly {len(batch_sentences)} explicit image generation commands.
The index for this batch MUST start sequentially at {start_idx} and end at {end_idx}.

MANDATORY PREFIX FOR EVERY COMMAND:
Every single command MUST start EXACTLY with the literal text:
"Create a 2D cel-shaded illustration showing ..."
No command is allowed to begin with any other wording. This prefix is non-negotiable.

Each sentence has its own scene context below. Every command MUST:
- Begin EXACTLY with "Create a 2D cel-shaded illustration showing ..." (literal, mandatory).
- Reflect the non-literal, conceptual visual storytelling for THAT sentence (never a literal restatement of the sentence).
- Build on the scene's visual_concept, emotion, character_action, environment, camera, composition, continuity, and transitions.
- Preserve the character DNA and all fixed visual constraints (orange skin, muscular body, smooth head, two large white oval eyes, no mouth, black shorts, consistent proportions, 2D cel-shaded, plain grey dominant background).
- Place a very small, subtle, faint image index in the bottom-right corner (no other text in the image).
- Evolve across the scene's sentences (establishing → continuation → escalation → reveal → transformation), not repeat identical images.

CREATIVE BRIEF (global narrative anchor):
{_format_creative_brief(creative_brief)}

Use the CREATIVE BRIEF to keep every command aligned with the video's core_idea, unique_angle, central_conflict, unexpected_insight, episode_concept, tone, and emotional_arc.

VISUAL BIBLE (global style anchor for the whole video):
{_format_visual_bible(visual_bible)}

SENTENCES AND THEIR SCENE CONTEXTS:
"""
        for s_idx, sent in enumerate(batch_sentences, start=start_idx):
            scene = scene_map.get(s_idx)
            user_prompt += f"\n[{s_idx}] Sentence: {sent}\nScene context:\n{_format_scene_context(scene)}\n"

        user_prompt += (
            "\nReturn the image generation commands separated ONLY by a single blank line, in order, "
            "with no headers, no labels, no markdown, no quotes, and no extra commentary. "
            "Remember: every command MUST begin with the literal prefix "
            "\"Create a 2D cel-shaded illustration showing ...\"."
        )
    else:
        system_instruction = STAGE_2_SYSTEM_PROMPT
        user_prompt = f"""Process the following sentences and generate exactly {len(batch_sentences)} explicit image generation commands.
The index for this batch MUST start sequentially at {start_idx} and end at {end_idx}.

MANDATORY PREFIX FOR EVERY COMMAND:
Every single command MUST start EXACTLY with the literal text:
"Create a 2D cel-shaded illustration showing ..."
No command is allowed to begin with any other wording. This prefix is non-negotiable.

Reminder:
- Every command MUST begin EXACTLY with "Create a 2D cel-shaded illustration showing ..." (literal, mandatory).
- NEVER restate the sentence literally. Translate it into a conceptual, symbolic, or story-driven visual.
- Always preserve character DNA (orange skin, muscular body, smooth head, two large white oval eyes, no mouth, black shorts, consistent proportions, 2D cel-shaded).
- Always keep a plain grey dominant background.
- Always include a very small, subtle, faint image index in the bottom-right corner; no other text in the image.

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

    إذا تم تمرير stage1_result (يحتوي scene_plan / visual_bible / creative_brief)
    يتم استخدام سياق المرحلة الأولى لتوليد Prompts مفاهيمية غير حرفية
    تبدأ حرفياً بأمر إنشاء صورة صريح.

    إذا لم يُمرَّر stage1_result أو كان ناقصاً، يتم الرجوع للسلوك القديم (fallback)
    دون أي فشل، مع الحفاظ على نفس القواعد البصرية والصياغة كأمر إنشاء صورة.
    """
    total_sentences = len(sentences)

    # استخراج السياق من المرحلة الأولى (اختياري)
    scene_plan: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    visual_bible: Optional[Dict[str, Any]] = None
    creative_brief: Optional[Any] = None
    has_stage1_context = False

    if isinstance(stage1_result, dict):
        candidate_plan = stage1_result.get("scene_plan")
        candidate_bible = stage1_result.get("visual_bible")
        candidate_brief = stage1_result.get("creative_brief")
        if candidate_plan or candidate_bible or candidate_brief:
            scene_plan = candidate_plan
            visual_bible = candidate_bible
            creative_brief = candidate_brief
            has_stage1_context = True

    if not has_stage1_context:
        logger.info(
            "ℹ️ لا يوجد scene_plan / visual_bible / creative_brief — سيتم استخدام السلوك القديم (fallback)."
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
                creative_brief,
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
