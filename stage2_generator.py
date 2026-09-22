import re
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from gemini_engine import call_gemini_with_fallback

logger = logging.getLogger("Stage2Generator")


# =========================================================
# HARD CONSTRAINTS (STRICT — NEVER RELAXED)
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
REQUIRED_SUBTLE_TOKENS = ("subtle", "faint")
REQUIRED_SMALL_TOKENS = ("very small", "small", "tiny")
REQUIRED_ANTI_TEXT_TOKENS = ("no written text", "no text")


# =========================================================
# LEGACY FLEXIBLE GROUPS (KEPT FOR DIAGNOSTICS ONLY).
# These must NEVER cause a prompt to be rejected or repaired.
# They are used only to log a rough "shape" of the prompt.
# =========================================================
FLEXIBLE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "conceptual": (
        "symbol", "symbolic", "symbolizing", "symbolising", "symbolizes",
        "symbolises", "metaphor", "metaphorical", "representing", "represents",
        "conceptual", "conceptually", "allegory", "allegorical",
        "as if", "as though", "visually conveys", "evokes", "embodies",
        "serves as", "stands for", "story-driven", "illustrates the idea",
        "communicates", "conveys", "transformation", "progression", "contrast",
    ),
    "action": (
        "standing", "stands", "walking", "walks", "running", "runs",
        "stepping", "steps", "climbing", "climbs", "holding", "holds",
        "reaching", "reaches", "gazing", "gazes", "looking", "looks",
        "facing", "faces", "turning", "turns", "raising", "raises",
        "lowering", "lowers", "moving", "moves", "crouching", "crouches",
        "leaning", "leans", "pushing", "pushes", "pulling", "pulls",
        "gesture", "gesturing", "pointing", "performing", "squatting",
        "lifting", "observing", "watching", "touching", "pose", "posture",
        "body language", "arms", "hands", "shoulders",
    ),
    "environment": (
        "on the grey background", "over the grey background",
        "on the background", "over the background",
        "placed on", "placed over", "resting on", "emerging from",
        "rising from", "in front of",
        "door", "doors", "bridge", "path", "pathway", "stairs", "steps",
        "wall", "floor", "ground", "platform", "structure", "barrier",
        "threshold", "gateway", "tunnel", "ladder", "rope", "box",
        "cage", "mirror", "clock", "mask", "chain", "chains", "rock",
        "mountain", "river", "weight", "stone", "block", "pillar",
        "element", "prop", "object", "surface", "environment",
        "landscape", "scene", "hologram", "joint", "knee", "vault",
        "statue", "room", "gym", "studio", "background", "visual element",
    ),
    "emotion": (
        "determined", "confused", "tired", "hopeful", "worried", "calm",
        "frustrated", "curious", "focused", "confident", "anxious",
        "relieved", "surprised", "thoughtful", "hesitant", "resigned",
        "eager", "defeated", "proud", "ashamed", "doubtful", "serene",
        "tense", "relaxed", "emotion", "emotional", "expression",
        "mood", "feeling", "stance",
    ),
    "camera": (
        "close-up", "close up", "medium shot", "wide shot", "long shot",
        "side-profile", "side profile", "low-angle", "low angle",
        "high-angle", "high angle", "overhead", "dutch angle",
        "camera", "shot", "framing", "angle", "view",
    ),
    "composition": (
        "composition", "centered", "centred", "symmetrical",
        "character on the left", "character on the right",
        "foreground", "background depth", "depth", "leading lines",
        "balanced composition", "balanced", "rule of thirds",
        "framed", "framing",
    ),
    "lighting": (
        "lighting", "light", "rim light", "soft light", "directional light",
        "high contrast", "cinematic lighting", "muted", "warm light",
        "cool light", "color palette", "colour palette", "palette",
        "gold", "shadow", "shadows", "glow", "glowing", "tones",
        "color", "colour", "blue", "red", "warm", "cool", "bright",
        "dark", "contrast", "illuminated",
    ),
    "continuity": (
        "continuing", "continuation", "continuity", "matching the previous",
        "evolving from", "preparing for the next", "preserving the same",
        "previous shot", "previous image", "next shot", "next image",
        "same visual world", "same environment", "next transformation",
        "same symbol", "same setting", "same world", "preserving",
        "evolving", "previous", "next", "transition", "transformation",
        "progression", "sequence",
    ),
}


# =========================================================
# AI CREATIVE REVIEWER CONFIG
# =========================================================
MAX_AI_REVIEW_ATTEMPTS = 2
CREATIVE_REVIEW_REPAIR_THRESHOLD = 80   # score < 80  -> add to repair list
CREATIVE_REVIEW_ACCEPT_THRESHOLD = 90   # score >= 90 -> auto-accept

BATCH_SIZE = 24
MAX_PROMPT_REPAIR_ATTEMPTS = 4


# =========================================================
# SYSTEM PROMPTS
# =========================================================
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


CREATIVE_REVIEWER_SYSTEM_PROMPT = """You are a senior Visual Quality Reviewer for AI image-generation prompts used in an educational YouTube video.

The recurring character is:
- consistent orange muscular figure
- smooth head
- two large white oval eyes
- no mouth
- black shorts
- clean 2D cel-shaded illustration style
- plain grey background as the dominant background
- a very small, subtle, faint in-image index number in the bottom-right corner
- no written text inside the image except that index

Your job is to score each prompt's CREATIVE AND VISUAL QUALITY, not to check word presence.

IMPORTANT RULES FOR YOUR REVIEW
================================
- Do NOT penalize a prompt for repeating the Character DNA, the plain grey background phrase, or the in-image index number. This repetition is MANDATORY.
- Do NOT require the literal words "composition", "continuity", "emotion", "lighting", or "camera". Judge the actual visual meaning, not the vocabulary.
- If composition, continuity, emotion, or lighting are clearly conveyed in meaning (even without those exact words), treat them as present.
- Judge whether the visual idea is CONCEPTUAL / SYMBOLIC rather than a literal restatement of the sentence.
- Judge whether the action / pose is clear and drawable.
- Judge whether the prompt is executable in Google Flow (self-contained, explicit, no contradictions).
- Judge whether character identity and background constraints are preserved.
- Judge camera, composition, lighting, and continuity between adjacent prompts.

SCORING (TOTAL 100)
===================
1. Conceptual clarity / non-literal visual idea — 25 points.
2. Action, scene, and image-convertibility — 20 points.
3. Executability in Google Flow — 20 points.
4. Character identity and visual constraints preserved — 20 points.
5. Camera, composition, lighting, continuity — 15 points.

DECISION RULES
==============
- score >= 90 -> "accept"
- 80 <= score < 90 -> "accept" unless you identify a MATERIAL visual defect that would clearly harm the image -> then "repair"
- score < 80 -> "repair"

OUTPUT FORMAT
=============
Return ONLY valid JSON. No markdown, no code fences, no commentary outside JSON.

{
  "reviews": [
    {
      "image_index": 1,
      "score": 94,
      "decision": "accept",
      "strengths": ["Clear conceptual metaphor", "Strong visual action"],
      "issues": []
    }
  ]
}

Rules:
- One review element per prompt.
- Same order as prompts given.
- image_index must be the integer index provided with each prompt.
- score must be an integer 0-100.
- decision must be "accept" or "repair".
- issues must be a list of strings (empty if none).
- Do NOT rewrite the prompts.
- Do NOT add any text outside the JSON.
"""


# =========================================================
# UTILITIES
# =========================================================
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
    """
    mapping: Dict[int, Dict[str, Any]] = {}
    if not scene_plan:
        return mapping

    scenes: Optional[List[Dict[str, Any]]] = None

    if isinstance(scene_plan, list):
        scenes = scene_plan
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
# HARD CONSTRAINT CHECK (STRICT — 100%, NO RELAXATION)
# =========================================================
def _check_hard_constraints(prompt: str, expected_index: int) -> List[str]:
    """
    فحص الشروط الحاكمة الصارمة فقط.
    أي فشل هنا يعني رفض الـPrompt مباشرة (بدون مساحة إبداعية).
    """
    issues: List[str] = []
    lower = prompt.lower()

    # 1) Mandatory literal prefix
    if not prompt.startswith(MANDATORY_PREFIX):
        issues.append(
            f"does not start with the mandatory literal prefix '{MANDATORY_PREFIX}'"
        )

    # 2) Character DNA tokens
    for token in REQUIRED_CHARACTER_DNA_TOKENS:
        if token.lower() not in lower:
            issues.append(f"missing required character DNA token '{token}'")

    # 3) Background phrase
    if REQUIRED_BACKGROUND_PHRASE.lower() not in lower:
        issues.append(
            f"missing required background phrase '{REQUIRED_BACKGROUND_PHRASE}'"
        )

    # 4) Corner phrase
    if REQUIRED_CORNER_PHRASE.lower() not in lower:
        issues.append(
            f"missing required corner phrase '{REQUIRED_CORNER_PHRASE}'"
        )

    # 5) Small descriptor for the in-image index ("very small")
    if not any(tok in lower for tok in REQUIRED_SMALL_TOKENS):
        issues.append(
            "missing 'very small' descriptor for the in-image index number"
        )

    # 6) Subtle / faint descriptor
    if not any(tok in lower for tok in REQUIRED_SUBTLE_TOKENS):
        issues.append(
            "missing 'faint' or 'subtle' descriptor for the in-image index number"
        )

    # 7) Correct index inside the in-image number instruction
    if f'"{expected_index}"' not in prompt:
        issues.append(
            f"does not contain the correct image index '\"{expected_index}\"' "
            "inside the in-image number instruction"
        )

    # 8) Forbidden bare trailing number
    if re.search(r"[\s,;]\d+\s*[.!]?\s*$", prompt):
        issues.append(
            "ends with a bare standalone number instead of an in-image number instruction"
        )

    # 9) Anti-text clause
    if not any(tok in lower for tok in REQUIRED_ANTI_TEXT_TOKENS):
        issues.append(
            "missing explicit anti-text clause "
            "('no written text, labels, captions, symbols containing letters, or extra numbers inside the image; "
            "only the required faint image index is allowed')"
        )

    return issues


# =========================================================
# DIAGNOSTIC ONLY: LEGACY FLEXIBLE GROUPS (never used to reject)
# =========================================================
def _diagnose_flexible_groups(prompt: str) -> Tuple[int, List[str], List[str]]:
    """
    تُرجع فقط معلومات تشخيصية لعدد المجموعات الإبداعية المكتشفة بالكلمات.
    لا تُستخدم أبدًا لقبول أو رفض.
    """
    lower = prompt.lower()
    matched: List[str] = []
    missing: List[str] = []
    for group_name, keywords in FLEXIBLE_GROUPS.items():
        if any(kw in lower for kw in keywords):
            matched.append(group_name)
        else:
            missing.append(group_name)
    return len(matched), matched, missing


# =========================================================
# AI CREATIVE REVIEWER
# =========================================================
def _build_creative_review_request(
    prompts: List[str],
    sentences: List[str],
    prompt_indices: List[int],
    scene_map: Dict[int, Dict[str, Any]],
    visual_bible: Any,
    creative_brief: Any,
) -> str:
    body = "Review the following batch of image-generation prompts.\n\n"
    body += "CREATIVE BRIEF (narrative anchor):\n"
    body += _format_creative_brief(creative_brief) + "\n\n"
    body += "VISUAL BIBLE (style anchor):\n"
    body += _format_visual_bible(visual_bible) + "\n\n"
    body += "PROMPTS TO REVIEW (each is self-contained):\n"

    for i, prompt in enumerate(prompts):
        abs_idx = prompt_indices[i]
        sentence = sentences[i] if i < len(sentences) else ""
        scene = scene_map.get(abs_idx) if scene_map else None
        body += "\n---\n"
        body += f"image_index: {abs_idx}\n"
        body += f"narration sentence: {sentence}\n"
        body += f"scene context:\n{_format_scene_context(scene)}\n"
        body += f"prompt:\n{prompt}\n"

    body += "\nReturn ONLY the JSON object described in the system instructions."
    return body


def _parse_creative_review_response(
    raw: str,
    expected_count: int,
    expected_indices: List[int],
) -> Optional[List[Dict[str, Any]]]:
    if not raw:
        return None

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    data: Optional[Any] = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    reviews = data.get("reviews")
    if not isinstance(reviews, list):
        return None
    if len(reviews) != expected_count:
        return None

    validated: List[Dict[str, Any]] = []
    for i, r in enumerate(reviews):
        if not isinstance(r, dict):
            return None
        idx = r.get("image_index")
        score = r.get("score")
        decision = r.get("decision")
        # image_index must be a plain int (not bool)
        if not isinstance(idx, int) or isinstance(idx, bool):
            return None
        if idx != expected_indices[i]:
            return None
        # score must be a real number in [0, 100] — reject bool explicitly
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        if not (0 <= score <= 100):
            return None
        if decision not in ("accept", "repair"):
            return None

        strengths = r.get("strengths", [])
        issues = r.get("issues", [])
        if not isinstance(strengths, list):
            strengths = []
        if not isinstance(issues, list):
            issues = []

        validated.append({
            "image_index": int(idx),
            "score": int(score),
            "decision": decision,
            "strengths": [str(s) for s in strengths],
            "issues": [str(s) for s in issues],
        })

    return validated


def _review_batch_creatively(
    prompts: List[str],
    sentences: List[str],
    prompt_indices: List[int],
    scene_map: Dict[int, Dict[str, Any]],
    visual_bible: Any,
    creative_brief: Any,
) -> Optional[List[Dict[str, Any]]]:
    """
    يستدعي مراجع الذكاء الاصطناعي على الدفعة كاملة في طلب واحد.
    يرجع قائمة reviews بنفس ترتيب prompts، أو None إذا فشل المراجع
    بعد MAX_AI_REVIEW_ATTEMPTS (وهذا لا يُعد فشلًا للمرحلة).

    prompt_indices: قائمة الفهارس الحقيقية (absolute indices) المقابلة لكل prompt
                    بنفس الترتيب. قد تكون غير متتابعة (مثل [1,3,4,7]).
    """
    if not prompts:
        return []

    expected_count = len(prompts)
    expected_indices = list(prompt_indices)

    if len(expected_indices) != expected_count:
        raise ValueError(
            "_review_batch_creatively: prompt_indices length "
            f"({len(expected_indices)}) must match prompts length ({expected_count})."
        )

    user_prompt = _build_creative_review_request(
        prompts=prompts,
        sentences=sentences,
        prompt_indices=prompt_indices,
        scene_map=scene_map,
        visual_bible=visual_bible,
        creative_brief=creative_brief,
    )

    for attempt in range(1, MAX_AI_REVIEW_ATTEMPTS + 1):
        try:
            raw = call_gemini_with_fallback(
                system_instruction=CREATIVE_REVIEWER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                response_mime_type="application/json",
            )
        except Exception as exc:
            logger.warning(
                f"⚠️ AI creative review attempt {attempt}/{MAX_AI_REVIEW_ATTEMPTS} "
                f"failed to call Gemini: {exc}"
            )
            continue

        parsed = _parse_creative_review_response(
            raw=raw,
            expected_count=expected_count,
            expected_indices=expected_indices,
        )
        if parsed is not None:
            return parsed

        logger.warning(
            f"⚠️ AI creative review attempt {attempt}/{MAX_AI_REVIEW_ATTEMPTS} "
            "returned invalid JSON. Retrying."
        )

    logger.warning(
        "⚠️ AI creative review unavailable after "
        f"{MAX_AI_REVIEW_ATTEMPTS} attempts. "
        "Falling back to hard-constraints-only acceptance for this batch."
    )
    return None


# =========================================================
# BATCH VALIDATION HELPERS
# =========================================================
def _validate_single_prompt(
    prompt: str,
    expected_index: int,
    position_in_batch: int,
    batch_num: int,
) -> None:
    issues = _check_hard_constraints(prompt, expected_index)
    if issues:
        raise ValueError(
            f"Batch [{batch_num}] prompt #{position_in_batch} "
            f"(expected image index {expected_index}) failed strict validation: "
            + " | ".join(issues)
            + "\n--- INVALID PROMPT ---\n"
            + prompt
            + "\n----------------------"
        )


def _validate_batch_prompts(
    batch_num: int,
    start_idx: int,
    prompts: List[str],
) -> None:
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


# =========================================================
# PARTIAL REPAIR FOR INVALID PROMPTS ONLY
# =========================================================
def _build_repair_request(
    invalid_items: List[Dict[str, Any]],
    visual_bible: Any,
    creative_brief: Any,
    start_idx: int,
    end_idx: int,
) -> str:
    header = (
        "This is a fresh independent repair request.\n\n"
        "Repair ONLY the invalid image-generation prompts listed below.\n"
        "Do not rewrite, replace, or return any valid prompt.\n"
        "Return exactly one repaired prompt for each invalid item.\n"
        "Return the repaired prompts in the same order.\n"
        "Return prompts separated ONLY by a single blank line.\n"
        "No JSON.\n"
        "No markdown.\n"
        "No headers.\n"
        "No explanations.\n\n"
        "Every repaired prompt MUST start EXACTLY with:\n"
        "Create a 2D cel-shaded illustration showing\n\n"
    )

    body = f"Batch index range: {start_idx} .. {end_idx}\n\n"

    body += "CREATIVE BRIEF (global narrative anchor):\n"
    body += _format_creative_brief(creative_brief) + "\n\n"

    body += "VISUAL BIBLE (global style anchor):\n"
    body += _format_visual_bible(visual_bible) + "\n\n"

    body += "MANDATORY LITERAL CONTENT inside every repaired prompt:\n"
    body += "- Start with EXACTLY: Create a 2D cel-shaded illustration showing\n"
    body += (
        "- Full character DNA spelled out literally: consistent orange muscular character, "
        "smooth head, two large white oval eyes, no mouth, black shorts, consistent proportions, "
        "clean 2D cel-shaded illustration style\n"
    )
    body += (
        "- Literal background phrase: plain grey background as the dominant background "
        "(symbolic elements placed ON/OVER it, not replacing it)\n"
    )
    body += (
        "- In-image number instruction with the CORRECT index for that prompt: "
        "include the very small, subtle, faint number \"N\" inside the image, "
        "placed in the bottom-right corner\n"
    )
    body += (
        "- Anti-text clause: no written text, labels, captions, symbols containing letters, "
        "or extra numbers inside the image; only the required faint image index is allowed\n"
    )
    body += "- Conceptual / symbolic (NON-literal) visual idea\n"
    body += "- Clear character action / pose / body language\n"
    body += "- Symbolic environment or visual element placed ON/OVER the grey background\n"
    body += "- Emotional state / expression\n"
    body += "- Camera angle / shot type\n"
    body += "- Composition\n"
    body += "- Lighting / colors when relevant\n"
    body += "- Continuity with the previous and next scenes\n"
    body += "- Do NOT end with a bare standalone number.\n\n"

    body += "INVALID PROMPTS TO REPAIR (in order):\n"

    for item in invalid_items:
        body += "\n---\n"
        body += f"Image index: {item['absolute_index']}\n"
        body += f"Sentence: {item['sentence']}\n"
        body += f"Scene context:\n{item['scene_context']}\n"
        if item.get("previous_prompt"):
            body += f"Previous valid prompt (for continuity): {item['previous_prompt']}\n"
        if item.get("next_prompt"):
            body += f"Next valid prompt (for continuity): {item['next_prompt']}\n"
        body += "Issues to fix: " + " | ".join(item["issues"]) + "\n"
        body += f"Rejected prompt:\n{item['invalid_prompt']}\n"

    body += (
        "\nReturn ONLY the repaired prompts separated by a single blank line, in order, "
        "with no headers, no labels, no markdown, no quotes, no JSON, and no commentary."
    )
    return header + body


def _repair_invalid_prompts(
    invalid_items: List[Dict[str, Any]],
    valid_prompts: Dict[int, str],
    batch_sentences: List[str],
    start_idx: int,
    end_idx: int,
    scene_map: Dict[int, Dict[str, Any]],
    visual_bible: Any,
    creative_brief: Any,
    batch_num: int,
) -> Dict[int, str]:
    """
    تصلح فقط الـPrompts المخالفة عبر طلبات مستقلة، مع إعادة المحاولة حتى
    MAX_PROMPT_REPAIR_ATTEMPTS. تُرجع قاموساً: absolute_index -> repaired prompt.

    الفحص داخل الحلقة: الشروط الحاكمة فقط.
    المراجعة الإبداعية بعد الإصلاح تُنفَّذ على مستوى _process_single_batch.
    """
    if not invalid_items:
        return {}

    repaired: Dict[int, str] = {}
    remaining: List[Dict[str, Any]] = list(invalid_items)
    attempts = 0

    while remaining and attempts < MAX_PROMPT_REPAIR_ATTEMPTS:
        attempts += 1
        logger.info(
            f"🔧 الدفعة [{batch_num}] محاولة إصلاح {attempts}/{MAX_PROMPT_REPAIR_ATTEMPTS} "
            f"لـ {len(remaining)} برومبت مخالف."
        )

        enriched_items: List[Dict[str, Any]] = []
        for item in remaining:
            new_item = dict(item)
            idx = new_item["absolute_index"]
            if idx - 1 in valid_prompts:
                new_item["previous_prompt"] = valid_prompts[idx - 1]
            if idx + 1 in valid_prompts:
                new_item["next_prompt"] = valid_prompts[idx + 1]
            enriched_items.append(new_item)

        user_prompt = _build_repair_request(
            invalid_items=enriched_items,
            visual_bible=visual_bible,
            creative_brief=creative_brief,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        raw = call_gemini_with_fallback(
            system_instruction=STAGE_2_ENRICHED_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_mime_type="text/plain",
        )
        parsed = clean_and_parse_prompts(raw)

        if len(parsed) != len(enriched_items):
            logger.warning(
                f"⚠️ الدفعة [{batch_num}] محاولة إصلاح {attempts}: "
                f"عدد البرومبتات المُعادة ({len(parsed)}) لا يطابق المطلوب "
                f"({len(enriched_items)}). سيتم إعادة المحاولة."
            )
            continue

        still_invalid: List[Dict[str, Any]] = []
        for i, item in enumerate(enriched_items):
            candidate = parsed[i]
            expected_index = item["absolute_index"]
            hard_issues = _check_hard_constraints(candidate, expected_index)
            if hard_issues:
                new_item = dict(item)
                new_item["issues"] = [
                    "Hard constraint failure after repair: " + "; ".join(hard_issues)
                ]
                new_item["invalid_prompt"] = candidate
                still_invalid.append(new_item)
            else:
                repaired[expected_index] = candidate

        if not still_invalid:
            remaining = []
        else:
            remaining = still_invalid

    if remaining:
        failed_indices = [item["absolute_index"] for item in remaining]
        details = "\n".join(
            f"  - index {item['absolute_index']}: " + " | ".join(item["issues"])
            for item in remaining
        )
        raise ValueError(
            f"Batch [{batch_num}] failed to repair {len(remaining)} prompt(s) "
            f"after {MAX_PROMPT_REPAIR_ATTEMPTS} attempts. "
            f"Unrepaired image indices: {failed_indices}.\nDetails:\n{details}"
        )

    return repaired


# =========================================================
# BATCH PROCESSING
# =========================================================
def _process_single_batch(batch_tuple: tuple) -> tuple:
    """
    معالجة دفعة واحدة:
      1) توليد البرومبتات الأساسية.
      2) فحص الشروط الحاكمة الصارمة فقط.
      3) مراجعة إبداعية ذكية عبر AI على الدفعة كاملة (طلب واحد)
         مع تمرير الفهارس الحقيقية للبرومبتات الناجحة.
      4) تصنيف: صحيح / يحتاج إصلاح
         (فشل hard OR score < 80 OR قرار reviewer = "repair").
      5) إصلاح المخالف فقط (فحص hard constraints داخل الحلقة).
      6) بعد الإصلاح: إعادة مراجعة إبداعية سريعة للبرومبتات المُصلَحة (للتوثيق فقط).
      7) إعادة بناء الدفعة بترتيبها الأصلي، مع فحص نهائي صارم.
    لا تُرجَع الدفعة إلا بعد أن يصبح كل برومبت صحيحًا بالشروط الحاكمة.
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

    # ---- بناء الـUser Prompt ----
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
- Include a clear character action / pose, a symbolic visual element placed on/over the grey background, an emotional state, a camera angle, a clear composition, lighting / color direction, and continuity with adjacent scenes.
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
            user_prompt += (
                f"\n[{s_idx}] Sentence: {sent}\n"
                f"Scene context:\n{_format_scene_context(scene)}\n"
            )

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
- Always include a clear character action / pose, a symbolic visual element placed on/over the grey background, an emotional state, a camera angle, a clear composition, lighting / color direction, and continuity with adjacent scenes.
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
    _validate_batch(batch_idx, len(batch_sentences), prompts)

    # ---- 1) فحص الشروط الحاكمة الصارمة ----
    valid_prompts: Dict[int, str] = {}
    invalid_items: List[Dict[str, Any]] = []

    for offset, prompt in enumerate(prompts):
        expected_index = start_idx + offset
        hard_issues = _check_hard_constraints(prompt, expected_index)

        # تشخيصي فقط (لا يؤثر على القبول)
        diag_matched, diag_matched_groups, _ = _diagnose_flexible_groups(prompt)
        logger.debug(
            f"🔎 [idx={expected_index}] hard_issues={len(hard_issues)} "
            f"flexible_matched={diag_matched}/8 groups={diag_matched_groups}"
        )

        if hard_issues:
            sentence = batch_sentences[offset]
            scene = scene_map.get(expected_index) if scene_map else None
            invalid_items.append({
                "absolute_index": expected_index,
                "sentence": sentence,
                "scene_context": _format_scene_context(scene),
                "invalid_prompt": prompt,
                "issues": ["Hard constraint failure: " + "; ".join(hard_issues)],
                "origin": "hard",
            })
        else:
            valid_prompts[expected_index] = prompt

    # ---- 2) مراجعة إبداعية ذكية على البرومبتات التي نجحت في الشروط الحاكمة ----
    if valid_prompts:
        ordered_indices = sorted(valid_prompts.keys())
        review_prompts = [valid_prompts[i] for i in ordered_indices]
        review_sentences = [batch_sentences[i - start_idx] for i in ordered_indices]

        reviews = _review_batch_creatively(
            prompts=review_prompts,
            sentences=review_sentences,
            prompt_indices=ordered_indices,
            scene_map=scene_map,
            visual_bible=visual_bible,
            creative_brief=creative_brief,
        )

        if reviews is not None:
            for i, review in enumerate(reviews):
                idx = ordered_indices[i]
                score = review["score"]
                decision = review["decision"]
                review_issues = review.get("issues", []) or []

                should_repair = (
                    decision == "repair"
                    or score < CREATIVE_REVIEW_REPAIR_THRESHOLD
                )

                if should_repair:
                    sentence = batch_sentences[idx - start_idx]
                    scene = scene_map.get(idx) if scene_map else None
                    issue_lines = [
                        f"AI reviewer score {score}/100 (decision={decision})"
                    ]
                    if review_issues:
                        issue_lines.append(
                            "Reviewer issues: "
                            + "; ".join(str(x) for x in review_issues)
                        )
                    invalid_items.append({
                        "absolute_index": idx,
                        "sentence": sentence,
                        "scene_context": _format_scene_context(scene),
                        "invalid_prompt": valid_prompts[idx],
                        "issues": issue_lines,
                        "origin": "creative",
                    })
                    del valid_prompts[idx]
                else:
                    logger.info(
                        f"✅ [idx={idx}] creative score {score}/100 accepted "
                        f"(decision={decision})."
                    )
        else:
            logger.warning(
                f"⚠️ Batch [{batch_idx}] AI creative review unavailable — "
                "accepting hard-constraint-valid prompts as-is."
            )

    logger.info(
        f"📊 الدفعة [{batch_idx}]: {len(valid_prompts)} صحيح، "
        f"{len(invalid_items)} مخالف يحتاج إصلاحًا."
    )

    # ---- 3) إصلاح المخالف فقط (فحص hard constraints داخل الحلقة) ----
    if invalid_items:
        repaired = _repair_invalid_prompts(
            invalid_items=invalid_items,
            valid_prompts=valid_prompts,
            batch_sentences=batch_sentences,
            start_idx=start_idx,
            end_idx=end_idx,
            scene_map=scene_map,
            visual_bible=visual_bible,
            creative_brief=creative_brief,
            batch_num=batch_idx,
        )

        # ---- 4) إعادة مراجعة إبداعية للبرومبتات المُصلَحة (توثيق فقط) ----
        creative_origins = {
            item["absolute_index"] for item in invalid_items
            if item.get("origin") == "creative"
        }
        repaired_for_review = sorted(
            idx for idx in repaired.keys() if idx in creative_origins
        )
        if repaired_for_review:
            review_prompts = [repaired[i] for i in repaired_for_review]
            review_sentences = [
                batch_sentences[i - start_idx] for i in repaired_for_review
            ]
            post_reviews = _review_batch_creatively(
                prompts=review_prompts,
                sentences=review_sentences,
                prompt_indices=repaired_for_review,
                scene_map=scene_map,
                visual_bible=visual_bible,
                creative_brief=creative_brief,
            )
            if post_reviews is not None:
                for i, r in enumerate(post_reviews):
                    logger.info(
                        f"🔁 post-repair review [idx={repaired_for_review[i]}]: "
                        f"score {r['score']}/100 decision={r['decision']}"
                    )
            else:
                logger.info(
                    "ℹ️ post-repair creative review unavailable — "
                    "repaired prompts accepted on hard constraints."
                )

        valid_prompts.update(repaired)

    # ---- 5) إعادة بناء الدفعة بالترتيب الأصلي ----
    ordered_prompts: List[str] = []
    for offset in range(len(batch_sentences)):
        idx = start_idx + offset
        if idx not in valid_prompts:
            raise ValueError(
                f"Batch [{batch_idx}] is missing a valid prompt for image index {idx} "
                "after repair. Refusing to return an incomplete batch."
            )
        ordered_prompts.append(valid_prompts[idx])

    # ---- 6) فحص نهائي صارم قبل الإرجاع ----
    _validate_batch_prompts(batch_idx, start_idx, ordered_prompts)

    logger.info(
        f"✅ الدفعة [{batch_idx}] جاهزة ({len(ordered_prompts)} برومبت صحيح)."
    )
    return batch_idx, ordered_prompts


def generate_stage2_prompts_batches(
    sentences: List[str],
    stage1_result: Optional[Dict[str, Any]] = None,
) -> List[List[str]]:
    """
    توزيع كافة دفعات أوامر الصور على خيوط متوازية (Parallel Threads).
    كل خيط يحصل تلقائياً على مفتاح مختلف من مصفوفة المفاتيح.

    كل دفعة:
    - تُولَّد.
    - تُفحص بالشروط الحاكمة الصارمة.
    - تُراجع إبداعياً بواسطة AI على دفعة كاملة (طلب واحد) مع فهارس حقيقية.
    - تُصنَّف إلى صحيحة ومخالفة (فشل hard OR score < 80 OR decision=repair).
    - تُصلَح المخالفة فقط في طلبات مستقلة (حتى MAX_PROMPT_REPAIR_ATTEMPTS).
    - لا تُرجَع إلا بعد نجاح كل برومبت في الشروط الحاكمة.

    أي دفعة يفشل إصلاحها بالكامل تُرفع كـValueError ولا تُحفظ.
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
            "ℹ️ لا يوجد scene_plan / visual_bible / creative_brief — "
            "سيتم استخدام السلوك القديم (fallback)."
        )

    # خريطة الجملة -> المشهد (1-based sentence index -> scene dict)
    scene_map = (
        _map_sentences_to_scenes(sentences, scene_plan)
        if has_stage1_context else {}
    )

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
            executor.submit(_process_single_batch, task): task[0]
            for task in batch_tasks
        }
        for future in as_completed(future_to_batch):
            batch_num, prompts = future.result()

            expected = len(batch_tasks[batch_num - 1][1])
            _validate_batch(batch_num, expected, prompts)

            completed_results[batch_num] = prompts
            logger.info(
                f"✅ انتهت الدفعة [{batch_num}] بنجاح ({len(prompts)} برومبت)"
            )

    # تجميع النتائج بالترتيب المتسلسل السليم
    sorted_batches = [completed_results[k] for k in sorted(completed_results.keys())]

    total_generated = sum(len(b) for b in sorted_batches)
    if total_generated != total_sentences:
        raise ValueError(
            f"Total prompts generated ({total_generated}) "
            f"does not match total sentences ({total_sentences})."
        )

    return sorted_batches
