import re
import logging
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from gemini_engine import call_gemini_with_fallback

logger = logging.getLogger("Stage2Generator")


# =========================================================
# STRICT VALIDATION CONSTANTS
# =========================================================
MANDATORY_PREFIX = "Create a 2D cel-shaded illustration showing"

REQUIRED_CHARACTER_DNA_TOKENS = [
    "orange",
    "muscular",
    "smooth head",
    "large white oval eyes",
    "no mouth",
    "black shorts",
]

REQUIRED_BACKGROUND_PHRASE = "plain grey background"
REQUIRED_CORNER_PHRASE = "bottom-right corner"
REQUIRED_SUBTLE_TOKENS = ("faint", "subtle")
REQUIRED_ANTI_TEXT_TOKENS = ("no written text", "no text")


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

FORBIDDEN literal example:
Narration: "Distraction makes it hard to move forward."
Literal (FORBIDDEN): a brain, a confused person holding their head, or a question mark.

REQUIRED conceptual approach:
The orange character standing between a nearby easy door leading to a dead end and a distant difficult door leading toward the real goal, with hesitant body language showing the conflict of choice.

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
- Plain grey background (dominant). If symbolic environment or objects are needed, add them ON/OVER the grey backdrop without replacing it.
- Very small, subtle, faint image index number placed INSIDE the image in the bottom-right corner.
- Numbering follows the sentence order, sequential, no gaps.
- Clean, cel-shaded style.
- Consistent proportions.
- No mouth.
- Black shorts.
- No written text inside the image except the required faint index number.

=========================================
EVERY PROMPT MUST BE AN EXPLICIT, SELF-CONTAINED IMAGE GENERATION COMMAND
=========================================
Each prompt MUST start EXACTLY with the literal text:
"Create a 2D cel-shaded illustration showing ..."

Every prompt will be COPIED INDEPENDENTLY into Google Flow. Therefore, each prompt MUST be COMPLETE ON ITS OWN and MUST literally spell out every required element below — even if some are repeated in these instructions.

=========================================
MANDATORY LITERAL CONTENT INSIDE EVERY PROMPT
=========================================
Every single prompt MUST include, verbatim:

1. Start with EXACTLY:
   Create a 2D cel-shaded illustration showing

2. The full character DNA spelled out literally:
   consistent orange muscular character, smooth head, two large white oval eyes, no mouth, black shorts, consistent proportions, clean 2D cel-shaded illustration style

3. The literal background phrase:
   plain grey background as the dominant background
   (symbolic environment/objects must be described as placed ON/OVER it, not replacing it)

4. A clear in-image number instruction, using the correct sequential index for that prompt, in this exact style:
   include the very small, subtle, faint number "N" inside the image, placed in the bottom-right corner
   where N is replaced by the correct image index for that prompt.

5. The anti-text clause (verbatim or clearly equivalent):
   no written text, labels, captions, symbols containing letters, or extra numbers inside the image; only the required faint image index is allowed

6. The prompt MUST NOT end with a bare standalone number.
   FORBIDDEN ending: "..., plain grey background, 1"
   REQUIRED ending: "... plain grey background as the dominant background. Include the very small, subtle, faint number "1" inside the image, placed in the bottom-right corner. No written text, labels, captions, symbols containing letters, or extra numbers inside the image; only the required faint image index is allowed."

7. When a visual symbol like an X is needed, describe it NON-TEXTUALLY, e.g.:
   a red cross-shaped visual symbol with no written text
   NEVER write red "X" text.

=========================================
EACH PROMPT MUST CONTAIN
=========================================
1. The explicit command beginning EXACTLY with "Create a 2D cel-shaded illustration showing ...".
2. Full character DNA (listed above) written literally.
3. The NON-LITERAL visual idea (conceptual / symbolic / story-driven).
4. Character action / dynamic pose.
5. Environment or symbolic visual element, placed ON/OVER the grey background.
6. Emotional state.
7. Camera angle.
8. Composition.
9. Lighting / colors if relevant.
10. Continuity with previous and upcoming scenes.
11. Literal phrase "plain grey background as the dominant background".
12. Anti-text clause.
13. Instruction to place the correct faint index number inside the image in the bottom-right corner.

=========================================
STRICT GENERATION RULES
=========================================
- ONE COMMAND PER SENTENCE: exactly one image prompt per sentence, in chronological order.
- INDEXING: use the correct sequential image number for each prompt, and place it inside the in-image number instruction — never as a bare trailing number.
- POSE CUSTOMIZATION: give a clear action, posture, or physical gesture matching the emotional and physical context of that sentence. NEVER alter character physical traits or the plain grey background.
- NO forbidden content: no literal restatement of the sentence, no extra numbers beyond the image index, no headers, no markdown, no quotes around the prompt, no explanation outside the prompts, no combining sentences, no skipping, no adding extra prompts.
- EVERY PROMPT MUST BE SELF-CONTAINED: it must include all required literal phrases above even if that means repetition across prompts.

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

Forbidden example:
Narration: "Distraction makes it hard to move forward."
Literal (FORBIDDEN): a brain, a confused person holding their head, or a question mark.

Required approach:
The orange character standing between a nearby easy door leading to a dead end and a distant difficult door leading toward the real goal, with hesitant body language showing the conflict of choice.

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
- Plain grey background (dominant). If symbolic environment/objects are needed, add them ON/OVER the grey backdrop without replacing it.
- Very small, subtle, faint image index number placed INSIDE the image in the bottom-right corner.
- Numbering follows the sentence order.
- Clean, cel-shaded style.
- Consistent proportions.
- No mouth.
- Black shorts.
- ONE prompt per sentence.
- No written text inside the image except the required faint index number.

=========================================
EVERY PROMPT MUST BE AN EXPLICIT, SELF-CONTAINED IMAGE GENERATION COMMAND
=========================================
Each prompt MUST start EXACTLY with the literal text:
"Create a 2D cel-shaded illustration showing ..."

Every prompt will be COPIED INDEPENDENTLY into Google Flow. Therefore, each prompt MUST be COMPLETE ON ITS OWN and MUST literally spell out every required element below — even if some are repeated in these instructions.

=========================================
MANDATORY LITERAL CONTENT INSIDE EVERY PROMPT
=========================================
Every single prompt MUST include, verbatim:

1. Start with EXACTLY:
   Create a 2D cel-shaded illustration showing

2. The full character DNA spelled out literally:
   consistent orange muscular character, smooth head, two large white oval eyes, no mouth, black shorts, consistent proportions, clean 2D cel-shaded illustration style

3. The literal background phrase:
   plain grey background as the dominant background
   (symbolic environment/objects must be described as placed ON/OVER it, not replacing it)

4. A clear in-image number instruction, using the correct sequential index for that prompt, in this exact style:
   include the very small, subtle, faint number "N" inside the image, placed in the bottom-right corner
   where N is replaced by the correct image index for that prompt.

5. The anti-text clause (verbatim or clearly equivalent):
   no written text, labels, captions, symbols containing letters, or extra numbers inside the image; only the required faint image index is allowed

6. The prompt MUST NOT end with a bare standalone number.
   FORBIDDEN ending: "..., plain grey background, 1"
   REQUIRED ending: "... plain grey background as the dominant background. Include the very small, subtle, faint number "1" inside the image, placed in the bottom-right corner. No written text, labels, captions, symbols containing letters, or extra numbers inside the image; only the required faint image index is allowed."

7. When a visual symbol like an X is needed, describe it NON-TEXTUALLY, e.g.:
   a red cross-shaped visual symbol with no written text
   NEVER write red "X" text.

=========================================
EACH PROMPT MUST INCLUDE
=========================================
1. The explicit command starting EXACTLY with "Create a 2D cel-shaded illustration showing ...".
2. Full character DNA (listed above) written literally.
3. The non-literal visual idea (conceptual / symbolic / story-driven).
4. Character action / dynamic pose.
5. Environment or symbolic visual element, placed ON/OVER the grey background.
6. Emotional state.
7. Camera angle.
8. Composition.
9. Lighting / colors if relevant.
10. Continuity with previous and upcoming scenes.
11. Literal phrase "plain grey background as the dominant background".
12. Anti-text clause.
13. Instruction to place the correct faint index number inside the image in the bottom-right corner.

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
- Any prompt that does NOT spell out the full character DNA, the plain grey background phrase, the anti-text clause, and the in-image number instruction.
- Any prompt that ends with a bare standalone number instead of an in-image number instruction.

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


# =========================================================
# STRICT PER-PROMPT VALIDATION
# =========================================================
def _validate_single_prompt(
    prompt: str,
    expected_index: int,
    position_in_batch: int,
    batch_num: int,
) -> None:
    """
    يتحقق من أن الـPrompt:
    - يبدأ بالبادئة الإلزامية حرفيًا.
    - يحتوي على كل عناصر Character DNA صراحةً.
    - يحتوي على عبارة الخلفية الرمادية.
    - يحتوي على إشارة الركن السفلي الأيمن.
    - يحتوي على faint أو subtle.
    - يحتوي على رقم الصورة الصحيح داخل تعليمات الرقم.
    - لا ينتهي برقم مجرد منفصل.
    - يحتوي على شرط منع النصوص الأخرى.
    يرفع ValueError واضحًا عند أول فشل، ويرفض الدفعة كاملة.
    """
    issues: List[str] = []
    lower = prompt.lower()

    # 1) البادئة الإلزامية حرفيًا
    if not prompt.startswith(MANDATORY_PREFIX):
        issues.append(
            f"does not start with the mandatory literal prefix '{MANDATORY_PREFIX}'"
        )

    # 2) عناصر Character DNA
    for token in REQUIRED_CHARACTER_DNA_TOKENS:
        if token.lower() not in lower:
            issues.append(f"missing required character DNA token '{token}'")

    # 3) عبارة الخلفية الرمادية
    if REQUIRED_BACKGROUND_PHRASE.lower() not in lower:
        issues.append(
            f"missing required background phrase '{REQUIRED_BACKGROUND_PHRASE}'"
        )

    # 4) عبارة الركن السفلي الأيمن
    if REQUIRED_CORNER_PHRASE.lower() not in lower:
        issues.append(
            f"missing required corner phrase '{REQUIRED_CORNER_PHRASE}'"
        )

    # 5) faint أو subtle لوصف الرقم
    if not any(tok in lower for tok in REQUIRED_SUBTLE_TOKENS):
        issues.append(
            "missing 'faint' or 'subtle' descriptor for the in-image index number"
        )

    # 6) رقم الصورة الصحيح داخل تعليمات الرقم
    if f'"{expected_index}"' not in prompt:
        issues.append(
            f"does not contain the correct image index '\"{expected_index}\"' "
            "inside the in-image number instruction"
        )

    # 7) ممنوع الانتهاء برقم مجرد منفصل
    if re.search(r"[\s,;]\d+\s*[.!]?\s*$", prompt):
        issues.append(
            "ends with a bare standalone number instead of an in-image number instruction"
        )

    # 8) شرط منع النصوص الأخرى
    if not any(tok in lower for tok in REQUIRED_ANTI_TEXT_TOKENS):
        issues.append(
            "missing explicit anti-text clause "
            "('no written text, labels, captions, symbols containing letters, or extra numbers inside the image; "
            "only the required faint image index is allowed')"
        )

    if issues:
        raise ValueError(
            f"Batch [{batch_num}] prompt #{position_in_batch} "
            f"(expected image index {expected_index}) failed strict validation: "
            + "; ".join(issues)
            + "\n--- INVALID PROMPT ---\n"
            + prompt
            + "\n----------------------"
        )


def _validate_batch_prompts(
    batch_num: int,
    start_idx: int,
    prompts: List[str],
) -> None:
    """
    تحقق كامل لكل prompt داخل دفعة واحدة مع تمرير رقم الصورة الصحيح لكل موضع.
    """
    for offset, prompt in enumerate(prompts):
        expected_index = start_idx + offset
        position_in_batch = offset + 1
        _validate_single_prompt(
            prompt=prompt,
            expected_index=expected_index,
            position_in_batch=position_in_batch,
            batch_num=batch_num,
        )


def _validate_batch(batch_idx: int, expected_count: int, prompts: List[str]) -> None:
    if len(prompts) != expected_count:
        raise ValueError(
            f"Batch [{batch_idx}] returned {len(prompts)} prompts, "
            f"expected exactly {expected_count}. Refusing partial/inflated batch."
        )


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

MANDATORY LITERAL CONTENT INSIDE EVERY COMMAND:
Each command MUST literally contain ALL of the following (do NOT omit any, even if it means repeating across commands):
- The full character DNA: consistent orange muscular character, smooth head, two large white oval eyes, no mouth, black shorts, consistent proportions, clean 2D cel-shaded illustration style.
- The literal background phrase: plain grey background as the dominant background (symbolic elements must be placed ON/OVER it, not replacing it).
- A clear in-image number instruction using the CORRECT sequential image index for that command, in this style:
  include the very small, subtle, faint number "N" inside the image, placed in the bottom-right corner
  (replace N with the correct index: {start_idx} for the first command, {start_idx + 1} for the second, ..., {end_idx} for the last).
- The anti-text clause: no written text, labels, captions, symbols containing letters, or extra numbers inside the image; only the required faint image index is allowed.
- The command MUST NOT end with a bare standalone number. Never append the index as a standalone trailing digit.

Each sentence has its own scene context below. Every command MUST:
- Begin EXACTLY with "Create a 2D cel-shaded illustration showing ..." (literal, mandatory).
- Reflect the non-literal, conceptual visual storytelling for THAT sentence (never a literal restatement of the sentence).
- Build on the scene's visual_concept, emotion, character_action, environment, camera, composition, continuity, and transitions.
- Spell out character DNA and background phrase literally inside the text.
- Include a correct in-image number instruction with the correct index for that position.
- Include the anti-text clause literally.
- Evolve across the scene's sentences (establishing → continuation → escalation → reveal → transformation), not repeat identical images.

CREATIVE BRIEF (global narrative anchor):
{_format_creative_brief(creative_brief)}

Use the CREATIVE BRIEF to keep every command aligned with the video's core_idea, unique_angle, central_conflict, unexpected_insight, episode_concept, tone, and emotional_arc.

VISUAL BIBLE (global style anchor for the whole video):
{_format_visual_bible(visual_bible)}

SENTENCES AND THEIR SCENE CONTEXTS (the [N] is the correct image index to place inside the in-image number instruction for that command):
"""
        for s_idx, sent in enumerate(batch_sentences, start=start_idx):
            scene = scene_map.get(s_idx)
            user_prompt += f"\n[{s_idx}] Sentence: {sent}\nScene context:\n{_format_scene_context(scene)}\n"

        user_prompt += (
            "\nReturn the image generation commands separated ONLY by a single blank line, in order, "
            "with no headers, no labels, no markdown, no quotes, and no extra commentary. "
            "Every command MUST begin with the literal prefix "
            "\"Create a 2D cel-shaded illustration showing ...\", MUST spell out the full character DNA, "
            "the plain grey background phrase, the anti-text clause, and MUST place the correct "
            "in-image index number inside the bottom-right corner (never as a bare trailing number)."
        )
    else:
        system_instruction = STAGE_2_SYSTEM_PROMPT
        user_prompt = f"""Process the following sentences and generate exactly {len(batch_sentences)} explicit image generation commands.
The index for this batch MUST start sequentially at {start_idx} and end at {end_idx}.

MANDATORY PREFIX FOR EVERY COMMAND:
Every single command MUST start EXACTLY with the literal text:
"Create a 2D cel-shaded illustration showing ..."
No command is allowed to begin with any other wording. This prefix is non-negotiable.

MANDATORY LITERAL CONTENT INSIDE EVERY COMMAND:
Each command MUST literally contain ALL of the following (do NOT omit any, even if it means repeating across commands):
- The full character DNA: consistent orange muscular character, smooth head, two large white oval eyes, no mouth, black shorts, consistent proportions, clean 2D cel-shaded illustration style.
- The literal background phrase: plain grey background as the dominant background (symbolic elements must be placed ON/OVER it, not replacing it).
- A clear in-image number instruction using the CORRECT sequential image index for that command, in this style:
  include the very small, subtle, faint number "N" inside the image, placed in the bottom-right corner
  (replace N with the correct index: {start_idx} for the first command, {start_idx + 1} for the second, ..., {end_idx} for the last).
- The anti-text clause: no written text, labels, captions, symbols containing letters, or extra numbers inside the image; only the required faint image index is allowed.
- The command MUST NOT end with a bare standalone number. Never append the index as a standalone trailing digit.

Reminder:
- Every command MUST begin EXACTLY with "Create a 2D cel-shaded illustration showing ..." (literal, mandatory).
- NEVER restate the sentence literally. Translate it into a conceptual, symbolic, or story-driven visual.
- Always spell out character DNA literally inside the prompt text.
- Always include "plain grey background as the dominant background" literally.
- Always include a correct in-image number instruction: include the very small, subtle, faint number "N" inside the image, placed in the bottom-right corner.
- Always include the anti-text clause literally.
- Never end with a bare standalone number.

Sentences (the [N] is the correct image index to place inside the in-image number instruction for that command):
"""
        for s_idx, sent in enumerate(batch_sentences, start=start_idx):
            user_prompt += f"[{s_idx}] {sent}\n"

    raw_output = call_gemini_with_fallback(
        system_instruction=system_instruction,
        user_prompt=user_prompt,
        response_mime_type="text/plain",
    )

    prompts = clean_and_parse_prompts(raw_output)

    # تحقق صارم: عدد الـPrompts مطابق لعدد جمل الدفعة
    _validate_batch(batch_idx, len(batch_sentences), prompts)

    # تحقق صارم لكل Prompt على حدة
    _validate_batch_prompts(batch_idx, start_idx, prompts)

    return batch_idx, prompts


def generate_stage2_prompts_batches(
    sentences: List[str],
    stage1_result: Optional[Dict[str, Any]] = None,
) -> List[List[str]]:
    """
    توزيع كافة دفعات أوامر الصور على خيوط متوازية (Parallel Threads).
    كل خيط يحصل تلقائياً على مفتاح مختلف من مصفوفة المفاتيح.

    إذا تم تمرير stage1_result (يحتوي scene_plan / visual_bible / creative_brief)
    يتم استخدام سياق المرحلة الأولى لتوليد Prompts مفاهيمية غير حرفية
    تبدأ حرفياً بأمر إنشاء صورة صريح وتحتوي كل العناصر الإلزامية داخل النص.

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

            # تحقق نهائي على عدد الـPrompts في الدفعة (تكرار أمان)
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
