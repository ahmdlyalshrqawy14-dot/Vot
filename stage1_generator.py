import json
import re
import copy
import logging
from typing import Dict, Any, List
from gemini_engine import call_gemini_with_fallback

logger = logging.getLogger("Stage1Generator")


# ===========================================================================
# MAIN SYSTEM PROMPT (unchanged)
# ===========================================================================

STAGE_1_SYSTEM_PROMPT = """You are an elite AI Episode Director for a high-retention YouTube channel focused on self-development, fitness psychology, and human behavior.
You are NOT a generic scriptwriter. You are a director who thinks in terms of narrative architecture, emotional pacing, visual storytelling, and viewer retention.

Your task: take a raw input JSON containing episode metadata and produce a fully directed 5-to-6-minute YouTube episode (target: 800 - 900 words at ~150 words per minute), along with a complete creative brief, retention plan, scene plan, visual bible, and self-critique report.

INPUT DATA STRUCTURE
You will receive a JSON object with:
"id": Episode ID
"topic": The title/subject
"the_myth": The common misconception to debunk
"the_truth": The scientific/practical reality
"the_analogy": The relatable mental model
"key_points": Bullet points to expand
"actionable_solution": Clear, practical action steps
"core_takeaway": The punchy summary
"comment_question": High-engagement closing question

INTERNAL DIRECTING WORKFLOW (execute this reasoning silently BEFORE writing):
1. Analyze the episode idea deeply.
2. Extract the essential information from episode_data (the_myth, the_truth, the_analogy, key_points, actionable_solution, core_takeaway).
3. Define a UNIQUE ANGLE that is specific, non-generic, and different from typical content on this topic.
4. Define the CORE IDEA in one sharp sentence.
5. Define the CENTRAL CONFLICT (the tension the viewer feels: desire vs. obstacle, belief vs. reality).
6. Define an UNEXPECTED INSIGHT that reframes the topic.
7. Define the EPISODE CONCEPT (the overarching creative frame).
8. Choose a NARRATIVE ARCHITECTURE that fits the topic (not always the same order).
9. Build an EMOTIONAL PROGRESSION (e.g., curiosity → frustration → clarity → motivation → resolve).
10. Define the PAYOFF that directly connects back to the HOOK.
11. THEN write the script based on that plan.
12. Divide the script into SCENES with clear boundaries.
13. Create VISUAL DIRECTION for each scene (character action, environment, camera, composition, motion potential, transition, continuity).
14. Run an INTERNAL CRITICAL REVIEW before outputting: check hook strength, originality, retention, coherence, visual storytelling, repetition, and factual caution. Fix issues before final output.

NARRATIVE ARCHITECTURE (NOT a fixed template):
Do NOT just do Hook → Intro → Myth → Truth → Actions. Instead, build a real narrative using functional beats such as:
- Hook
- Open Loop
- Pattern Interrupt
- Problem
- Mechanism
- Story or Analogy
- Escalation
- Solution
- Application
- Payoff
- Ending Callback
You may reorder these beats to fit the topic, but the episode MUST feel connected, escalating, and payoff-driven. Each beat must serve a purpose; never pad.

SCRIPTWRITING & PACING RULES
TONE & STYLE:
- Write like a professional YouTube creator speaking directly to the viewer.
- Conversational, authoritative, warm, and relatable spoken English.
- The entire script must feel like ONE continuous, cohesive piece — not disconnected lines.
- Use natural transitions so narration flows smoothly from sentence to sentence.
- Avoid robotic or staccato delivery.

SENTENCE CADENCE:
- Keep sentences relatively short and easy to speak in one breath.
- Every sentence must end with a period, question mark, or exclamation point.
- Prefer clear, direct sentences. Avoid extremely long compound sentences, semicolons, or em-dashes.
- Allow natural connecting words (And, But, So, That's why, Here's the thing) so the script is not choppy.

SENTENCE COUNT RULES (CRITICAL):
- The "full_script_sentences" array MUST contain between 60 and 80 sentences, inclusive.
- The PREFERRED range is 60 to 70 sentences.
- Do NOT exceed 80 sentences under any circumstance.
- Do NOT go below 60 sentences under any circumstance.
- Each sentence must be short, natural, and easy to speak in one breath.
- Each sentence must be independently visualizable as its own shot.
- Do NOT use overly long sentences just to reduce the count.
- Do NOT add filler just to reach the count.
- Preserve narrative flow. The script must NOT feel choppy or mechanical even though there are many sentences.
- Each sentence must end with a period, question mark, or exclamation point.

FULL_SCRIPT_SENTENCES INTEGRITY RULES (CRITICAL):
- "full_script_sentences" MUST be a JSON array of NON-EMPTY strings.
- Every element MUST contain real, speakable text after stripping whitespace.
- NEVER include empty strings, whitespace-only strings, or null entries.
- NEVER include placeholders or filler tokens such as "...", "-", or similar.
- If a sentence is meant to exist, it MUST contain actual words.
- The number of elements in "full_script_sentences" defines N (the total sentence count).
- The final element of "full_script_sentences" is at index N - 1.
- Indices are 0-based, NOT 1-based.

SCENE PLAN INDEXING RULES (CRITICAL — READ CAREFULLY):
- Build the "scene_plan" ONLY AFTER the final "full_script_sentences" list is fully written and frozen.
- Count the final sentences yourself. Let N be that count.
- The last valid 0-based index is N - 1.
- sentence_start and sentence_end MUST both be between 0 and N - 1 inclusive.
- NEVER use N as a value for sentence_end. N is OUT OF RANGE and will be rejected.
- The first scene MUST start at sentence_start = 0.
- Scenes MUST be contiguous: each next scene's sentence_start MUST equal the previous scene's sentence_end + 1.
- There MUST be NO gaps between scenes and NO overlaps between scenes.
- The final scene in scene_plan MUST end exactly at index N - 1.
- Concretely: if "full_script_sentences" has 74 sentences, the last scene must end at sentence_end = 73, NOT 74.
- The number of scenes is NOT fixed and MUST adapt to the actual number of sentences (60 - 80).

STRUCTURE & WORD BUDGET (800 - 900 words total):
Distribute words according to the narrative architecture you chose — not a rigid template. Ensure the total stays in 800 - 900 words.

HARD RULES:
- No filler to hit word count.
- No recycled general introductions.
- No absolute or exaggerated medical claims.
- No invented facts beyond episode_data.
- Preserve the meaning of the_truth, key_points, and actionable_solution.

CHARACTER DNA (MUST NOT CHANGE):
- Orange character, muscular body, smooth head, two big white eyes, no mouth, black shorts.
- Character identity must remain consistent across all scenes.

VISUAL RULES FOR SCENES:
- Every scene must add a VISUAL IDEA, not just illustrate the sentence literally.
- Each scene must include character_action AND visual_concept.
- Visual progression must exist from the first scene to the last.
- Do NOT make every scene just a different pose of the character in an empty background.
- Do NOT add text inside images unless absolutely essential to the idea.

RETENTION_PLAN STRUCTURE RULES (CRITICAL — READ CAREFULLY):
The "retention_plan" object MUST follow this exact JSON shape, and the three list fields MUST ALWAYS be real JSON arrays of strings:

"retention_plan": {
  "hook_strategy": "string",
  "open_loops": ["string", "string"],
  "pattern_interrupts": ["string", "string"],
  "escalation_points": ["string", "string"],
  "main_reveal": "string",
  "payoff": "string",
  "ending_callback": "string"
}

Mandatory rules for these three fields:
- retention_plan.open_loops
- retention_plan.pattern_interrupts
- retention_plan.escalation_points

They MUST be JSON arrays (Lists) of strings, always.
- These three fields MUST be Lists/Arrays of strings.
- NEVER return them as a single string.
- Even if there is only ONE item, you MUST still use an array, e.g. ["single item"].
- Do NOT use a direct string such as: "open_loops": "single item".
- Do NOT write the list as Markdown, and do NOT use a comma-separated string; the required format is a real JSON array.
- Every item inside each of these lists MUST be a non-empty string.
- This applies to EVERY item of the retention_plan, with no exceptions.

OUTPUT FORMAT
Return a single valid, parsable JSON object ONLY. No markdown, no backticks, no conversational filler.
Required JSON schema (all fields mandatory):

{
  "id": 0,
  "topic": "string",
  "total_word_count": 0,
  "hook": "string",
  "sections": [
    { "section_name": "hook", "sentences": ["..."] },
    { "section_name": "intro", "sentences": ["..."] },
    { "section_name": "myth_and_truth", "sentences": ["..."] },
    { "section_name": "analogy", "sentences": ["..."] },
    { "section_name": "key_points", "sentences": ["..."] },
    { "section_name": "actionable_blueprint", "sentences": ["..."] },
    { "section_name": "takeaway_and_cta", "sentences": ["..."] },
    { "section_name": "comment_question", "sentences": ["..."] }
  ],
  "full_script_sentences": ["Chronological list of every single sentence ending with punctuation."],

  "creative_brief": {
    "core_idea": "string",
    "unique_angle": "string",
    "central_conflict": "string",
    "unexpected_insight": "string",
    "episode_concept": "string",
    "episode_format": "string",
    "target_audience": "string",
    "viewer_problem": "string",
    "viewer_outcome": "string",
    "tone": "string",
    "emotional_arc": "string"
  },

  "retention_plan": {
    "hook_strategy": "string",
    "open_loops": ["string", "string"],
    "pattern_interrupts": ["string", "string"],
    "escalation_points": ["string", "string"],
    "main_reveal": "string",
    "payoff": "string",
    "ending_callback": "string"
  },

  "scene_plan": [
    {
      "scene_id": 1,
      "sentence_start": 0,
      "sentence_end": 3,
      "script_segment": "string",
      "narrative_purpose": "string",
      "emotion": "string",
      "visual_concept": "string",
      "character_action": "string",
      "environment": "string",
      "camera": "string",
      "composition": "string",
      "motion_potential": "string",
      "transition": "string",
      "continuity_notes": "string"
    }
  ],

  "visual_bible": {
    "character_dna": "Orange character, muscular body, smooth head, two big white eyes, no mouth, black shorts.",
    "environment_style": "string",
    "color_logic": "string",
    "lighting_style": "string",
    "camera_language": "string",
    "composition_rules": "string",
    "recurring_symbols": ["string"],
    "visual_progression": "string"
  },

  "quality_report": {
    "hook_score": 0,
    "originality_score": 0,
    "retention_score": 0,
    "narrative_coherence_score": 0,
    "visual_storytelling_score": 0,
    "repetition_check": "string",
    "factual_caution_check": "string",
    "final_issues": ["string"],
    "approved": true
  }
}

SCENE PLAN RULES:
- scene_id MUST start at 1 and increase sequentially (1, 2, 3, ...) matching the order in the list.
- scene_plan MUST cover the entire script from sentence index 0 to len(full_script_sentences)-1 with no gaps and no out-of-range indices.
- sentence_start and sentence_end are 0-based indices into full_script_sentences.
- The number of scenes is NOT fixed and MUST adapt to the actual number of sentences (60 - 80).
- Each scene must add a visual idea, not just repeat the sentence.
- Character identity must remain consistent across scenes.
- Every scene must include character_action and visual_concept.
- Visual progression must exist from start to end of the episode.
- Do not make every scene the same pose against an empty background.
"""


# ===========================================================================
# PARTIAL-REPAIR SYSTEM PROMPTS
# ===========================================================================

SCENE_PLAN_REPAIR_SYSTEM_PROMPT = """You are a scene-plan repair specialist for a YouTube script pipeline.
You will receive a FROZEN list of sentences. You must NOT modify, reorder, paraphrase, merge, or split them.
Your ONLY job is to output a valid "scene_plan" JSON array for those sentences.

HARD RULES (CRITICAL):
- Indexing is 0-based.
- N = number of sentences. The last valid index is N - 1.
- First scene: sentence_start MUST be 0.
- Contiguity: each next scene's sentence_start MUST equal the previous scene's sentence_end + 1.
- No gaps. No overlaps.
- The final scene's sentence_end MUST equal N - 1.
- NEVER use N as sentence_end. N is OUT OF RANGE.
- scene_id starts at 1 and increments by 1 in list order.
- Every scene MUST include these exact keys:
  scene_id, sentence_start, sentence_end, script_segment, narrative_purpose,
  emotion, visual_concept, character_action, environment, camera,
  composition, motion_potential, transition, continuity_notes.
- Every scene must add a real VISUAL IDEA (not just literal illustration).
- Character DNA must be consistent: Orange character, muscular body, smooth head, two big white eyes, no mouth, black shorts.
- Do NOT add text inside images unless essential.
- Output ONLY the JSON array. No markdown, no backticks, no explanation.
"""


RETENTION_PLAN_REPAIR_SYSTEM_PROMPT = """You are a retention-plan repair specialist.
Return ONLY a valid "retention_plan" JSON object with EXACTLY these fields:
- hook_strategy (non-empty string)
- open_loops (JSON array of non-empty strings)
- pattern_interrupts (JSON array of non-empty strings)
- escalation_points (JSON array of non-empty strings)
- main_reveal (non-empty string)
- payoff (non-empty string)
- ending_callback (non-empty string)

CRITICAL:
- open_loops, pattern_interrupts, and escalation_points MUST be real JSON arrays of strings.
- Even a single item MUST be wrapped in an array, e.g. ["one item"].
- NEVER return them as a single string.
- No markdown. No backticks. No explanation.
Return only the JSON object.
"""


VISUAL_BIBLE_REPAIR_SYSTEM_PROMPT = """You are a visual-bible repair specialist.
Return ONLY a valid "visual_bible" JSON object with EXACTLY these fields:
- character_dna (string) — MUST be exactly: "Orange character, muscular body, smooth head, two big white eyes, no mouth, black shorts."
- environment_style (non-empty string)
- color_logic (non-empty string)
- lighting_style (non-empty string)
- camera_language (non-empty string)
- composition_rules (non-empty string)
- recurring_symbols (JSON array of strings)
- visual_progression (non-empty string)

No markdown. No backticks. No explanation.
Return only the JSON object.
"""


QUALITY_REPORT_REPAIR_SYSTEM_PROMPT = """You are a quality-report repair specialist.
Return ONLY a valid "quality_report" JSON object with EXACTLY these fields:
- hook_score (number between 0 and 10)
- originality_score (number between 0 and 10)
- retention_score (number between 0 and 10)
- narrative_coherence_score (number between 0 and 10)
- visual_storytelling_score (number between 0 and 10)
- repetition_check (non-empty string)
- factual_caution_check (non-empty string)
- final_issues (JSON array of strings — may be empty)
- approved (boolean true/false)

No markdown. No backticks. No explanation.
Return only the JSON object.
"""


# ===========================================================================
# CONSTANTS
# ===========================================================================

MIN_SENTENCES = 60
MAX_SENTENCES = 80
PREFERRED_MIN_SENTENCES = 60
PREFERRED_MAX_SENTENCES = 70

MAX_FULL_GENERATION_ATTEMPTS = 4
MAX_FIELD_REPAIR_ATTEMPTS = 4


# ===========================================================================
# JSON helpers
# ===========================================================================

def _clean_json_string(raw_text: str) -> str:
    if raw_text is None:
        raise ValueError("استجابة النموذج فارغة (None).")
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json_safe(raw_text: str) -> Any:
    cleaned = _clean_json_string(raw_text)
    if not cleaned:
        raise ValueError("استجابة النموذج فارغة بعد التنظيف.")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"استجابة النموذج ليست JSON صالحًا: {e}") from e


# ===========================================================================
# Base validation helpers
# ===========================================================================

def _require_non_empty_string(container: Dict[str, Any], key: str, context: str) -> str:
    if key not in container:
        raise ValueError(f"الحقل المطلوب '{key}' مفقود داخل {context}!")
    value = container[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"الحقل '{key}' داخل {context} يجب أن يكون نصًا غير فارغ.")
    return value


def _require_list(container: Dict[str, Any], key: str, context: str) -> List[Any]:
    if key not in container:
        raise ValueError(f"الحقل المطلوب '{key}' مفقود داخل {context}!")
    value = container[key]
    if not isinstance(value, list):
        raise ValueError(f"الحقل '{key}' داخل {context} يجب أن يكون قائمة.")
    return value


def _require_bool(container: Dict[str, Any], key: str, context: str) -> bool:
    if key not in container:
        raise ValueError(f"الحقل المطلوب '{key}' مفقود داخل {context}!")
    value = container[key]
    if not isinstance(value, bool):
        raise ValueError(f"الحقل '{key}' داخل {context} يجب أن يكون قيمة Boolean حقيقية.")
    return value


def _require_score_0_10(container: Dict[str, Any], key: str, context: str) -> float:
    if key not in container:
        raise ValueError(f"الحقل المطلوب '{key}' مفقود داخل {context}!")
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"الحقل '{key}' داخل {context} يجب أن يكون رقمًا.")
    if value < 0 or value > 10:
        raise ValueError(f"الحقل '{key}' داخل {context} يجب أن يكون بين 0 و10 (القيمة: {value}).")
    return float(value)


def _require_int(container: Dict[str, Any], key: str, context: str) -> int:
    if key not in container:
        raise ValueError(f"الحقل المطلوب '{key}' مفقود داخل {context}!")
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"الحقل '{key}' داخل {context} يجب أن يكون رقمًا صحيحًا.")
    return value


# ===========================================================================
# Field-level validators
# ===========================================================================

_CREATIVE_BRIEF_STRING_FIELDS = [
    "core_idea",
    "unique_angle",
    "central_conflict",
    "unexpected_insight",
    "episode_concept",
    "episode_format",
    "target_audience",
    "viewer_problem",
    "viewer_outcome",
    "tone",
    "emotional_arc",
]

_RETENTION_PLAN_STRING_FIELDS = [
    "hook_strategy",
    "main_reveal",
    "payoff",
    "ending_callback",
]

_RETENTION_PLAN_LIST_FIELDS = [
    "open_loops",
    "pattern_interrupts",
    "escalation_points",
]

_VISUAL_BIBLE_STRING_FIELDS = [
    "character_dna",
    "environment_style",
    "color_logic",
    "lighting_style",
    "camera_language",
    "composition_rules",
    "visual_progression",
]

_QUALITY_REPORT_SCORE_FIELDS = [
    "hook_score",
    "originality_score",
    "retention_score",
    "narrative_coherence_score",
    "visual_storytelling_score",
]

_QUALITY_REPORT_STRING_FIELDS = [
    "repetition_check",
    "factual_caution_check",
]

_SCENE_REQUIRED_STRING_FIELDS = [
    "script_segment",
    "narrative_purpose",
    "emotion",
    "visual_concept",
    "character_action",
    "environment",
    "camera",
    "composition",
    "motion_potential",
    "transition",
    "continuity_notes",
]


def _validate_creative_brief(data: Dict[str, Any]) -> None:
    cb = data.get("creative_brief")
    if not isinstance(cb, dict):
        raise ValueError("'creative_brief' يجب أن يكون كائنًا صحيحًا.")
    for field in _CREATIVE_BRIEF_STRING_FIELDS:
        _require_non_empty_string(cb, field, "creative_brief")


def _validate_retention_plan(data: Dict[str, Any]) -> None:
    rp = data.get("retention_plan")
    if not isinstance(rp, dict):
        raise ValueError("'retention_plan' يجب أن يكون كائنًا صحيحًا.")
    for field in _RETENTION_PLAN_STRING_FIELDS:
        _require_non_empty_string(rp, field, "retention_plan")
    for field in _RETENTION_PLAN_LIST_FIELDS:
        items = _require_list(rp, field, "retention_plan")
        for i, item in enumerate(items):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"retention_plan.{field}[{i}] يجب أن يكون نصًا غير فارغ."
                )


def _validate_visual_bible(data: Dict[str, Any]) -> None:
    vb = data.get("visual_bible")
    if not isinstance(vb, dict):
        raise ValueError("'visual_bible' يجب أن يكون كائنًا صحيحًا.")
    for field in _VISUAL_BIBLE_STRING_FIELDS:
        _require_non_empty_string(vb, field, "visual_bible")
    _require_list(vb, "recurring_symbols", "visual_bible")


def _validate_quality_report(data: Dict[str, Any]) -> None:
    qr = data.get("quality_report")
    if not isinstance(qr, dict):
        raise ValueError("'quality_report' يجب أن يكون كائنًا صحيحًا.")
    for field in _QUALITY_REPORT_SCORE_FIELDS:
        _require_score_0_10(qr, field, "quality_report")
    for field in _QUALITY_REPORT_STRING_FIELDS:
        _require_non_empty_string(qr, field, "quality_report")
    _require_list(qr, "final_issues", "quality_report")
    _require_bool(qr, "approved", "quality_report")


# ===========================================================================
# Sentence validation (extracted, reusable)
# ===========================================================================

def _extract_and_validate_full_script_sentences(
    data: Dict[str, Any],
) -> (List[str], int):
    """Normalize and validate full_script_sentences. Returns (cleaned, total_words)."""
    sentences = data.get("full_script_sentences", [])
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("مصفوفة 'full_script_sentences' فارغة!")

    cleaned_sentences: List[str] = []
    total_words = 0
    for index, s in enumerate(sentences):
        if not isinstance(s, str):
            raise ValueError(f"full_script_sentences[{index}] يجب أن يكون نصًا.")
        s_clean = s.strip()
        if not s_clean:
            raise ValueError(
                f"full_script_sentences[{index}] فارغة. "
                "ممنوع وجود عناصر فارغة لأن scene_plan يعتمد على الفهرسة نفسها."
            )
        if not (
            s_clean.endswith(".")
            or s_clean.endswith("?")
            or s_clean.endswith("!")
        ):
            s_clean += "."
        cleaned_sentences.append(s_clean)
        total_words += len(s_clean.split())

    if not cleaned_sentences:
        raise ValueError("مصفوفة 'full_script_sentences' فارغة بعد التنظيف.")

    count = len(cleaned_sentences)
    if count < MIN_SENTENCES:
        raise ValueError(
            f"عدد الجمل في 'full_script_sentences' غير كافٍ. "
            f"العدد الفعلي: {count}، الحد الأدنى المطلوب: {MIN_SENTENCES}."
        )
    if count > MAX_SENTENCES:
        raise ValueError(
            f"عدد الجمل في 'full_script_sentences' تجاوز الحد الأقصى. "
            f"العدد الفعلي: {count}، الحد الأقصى المسموح: {MAX_SENTENCES}."
        )

    return cleaned_sentences, total_words


# ===========================================================================
# Scene plan validation (strict, no auto-correction)
# ===========================================================================

def _validate_scene_plan(data: Dict[str, Any], total_sentences: int) -> None:
    scene_plan = data.get("scene_plan")
    if not isinstance(scene_plan, list) or not scene_plan:
        raise ValueError("'scene_plan' يجب أن يكون قائمة غير فارغة.")

    covered = [False] * total_sentences
    previous_end = -1

    for idx, scene in enumerate(scene_plan):
        if not isinstance(scene, dict):
            raise ValueError(f"عنصر scene_plan رقم {idx} ليس كائنًا صحيحًا.")

        scene_id = _require_int(scene, "scene_id", f"scene_plan[{idx}]")
        s_start = _require_int(scene, "sentence_start", f"scene_plan[{idx}]")
        s_end = _require_int(scene, "sentence_end", f"scene_plan[{idx}]")

        expected_scene_id = idx + 1
        if scene_id != expected_scene_id:
            raise ValueError(
                f"scene_plan[{idx}]: scene_id يجب أن يكون {expected_scene_id} "
                f"(متسلسلًا بدءًا من 1 حسب ترتيب المشهد)، لكنه كان {scene_id}."
            )
        if scene_id <= 0:
            raise ValueError(
                f"scene_plan[{idx}]: scene_id يجب أن يكون رقمًا صحيحًا موجبًا (وجدنا {scene_id})."
            )

        for field in _SCENE_REQUIRED_STRING_FIELDS:
            _require_non_empty_string(scene, field, f"scene_plan[{idx}]")

        if s_start < 0 or s_end < 0:
            raise ValueError(f"scene_plan[{idx}]: لا يمكن أن تكون الفهارس سالبة.")
        if s_start > s_end:
            raise ValueError(f"scene_plan[{idx}]: sentence_start أكبر من sentence_end.")
        if s_end >= total_sentences:
            raise ValueError(
                f"scene_plan[{idx}]: sentence_end={s_end} خارج نطاق full_script_sentences "
                f"(الحد الأقصى {total_sentences - 1}). "
                f"عدد الجمل الفعلي N = {total_sentences}، لذا آخر فهرس مسموح هو {total_sentences - 1}."
            )

        if idx == 0:
            if s_start != 0:
                raise ValueError(
                    f"scene_plan[0]: يجب أن يبدأ أول مشهد عند sentence_start=0 (وجدنا {s_start})."
                )
        else:
            expected_start = previous_end + 1
            if s_start < expected_start:
                raise ValueError(
                    f"scene_plan[{idx}]: تداخل مع المشهد السابق. "
                    f"يجب أن يبدأ عند {expected_start} لكنه بدأ عند {s_start}."
                )
            if s_start > expected_start:
                raise ValueError(
                    f"scene_plan[{idx}]: توجد فجوة بين المشهد السابق وهذا المشهد. "
                    f"يجب أن يبدأ عند {expected_start} لكنه بدأ عند {s_start}."
                )

        for i in range(s_start, s_end + 1):
            if covered[i]:
                raise ValueError(
                    f"scene_plan[{idx}]: الجملة رقم {i} مُغطاة في أكثر من مشهد (تكرار)."
                )
            covered[i] = True

        previous_end = s_end

    if previous_end != total_sentences - 1:
        raise ValueError(
            f"آخر مشهد يجب أن ينتهي عند {total_sentences - 1} لكنه انتهى عند {previous_end}. "
            f"عدد الجمل الفعلي N = {total_sentences}."
        )

    if not all(covered):
        missing = [i for i, c in enumerate(covered) if not c]
        raise ValueError(
            f"scene_plan لا يغطي كل الجمل. الجمل غير المغطاة (0-based): {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )


# ===========================================================================
# Required top-level keys
# ===========================================================================

_REQUIRED_TOP_LEVEL_KEYS = [
    "id",
    "topic",
    "hook",
    "sections",
    "full_script_sentences",
    "creative_brief",
    "retention_plan",
    "scene_plan",
    "visual_bible",
    "quality_report",
]


def _check_required_top_level_keys(data: Dict[str, Any]) -> None:
    for k in _REQUIRED_TOP_LEVEL_KEYS:
        if k not in data:
            raise ValueError(f"الحقل المطلوب '{k}' مفقود من مخرجات JSON!")


# ===========================================================================
# Public validator
# ===========================================================================

def validate_script_output(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("مخرجات النموذج يجب أن تكون كائن JSON.")

    _check_required_top_level_keys(data)

    # ---- full_script_sentences ----
    cleaned_sentences, total_words = _extract_and_validate_full_script_sentences(data)
    data["full_script_sentences"] = cleaned_sentences
    data["total_word_count"] = total_words

    sentence_count = len(cleaned_sentences)

    if PREFERRED_MIN_SENTENCES <= sentence_count <= PREFERRED_MAX_SENTENCES:
        logger.info(
            f"🎯 عدد الجمل: {sentence_count} (ضمن النطاق المفضل "
            f"{PREFERRED_MIN_SENTENCES}-{PREFERRED_MAX_SENTENCES})."
        )
    else:
        logger.warning(
            f"ℹ️ عدد الجمل: {sentence_count} (مقبول لكنه أعلى من النطاق المفضل "
            f"{PREFERRED_MIN_SENTENCES}-{PREFERRED_MAX_SENTENCES})."
        )

    if 800 <= total_words <= 900:
        logger.info(f"🎯 حجم السكربت مثالي: {total_words} كلمة.")
    else:
        logger.warning(f"ℹ️ حجم السكربت: {total_words} كلمة (المستهدف: 800 - 900 كلمة).")

    # ---- validate sub-objects ----
    _validate_creative_brief(data)
    _validate_retention_plan(data)
    _validate_visual_bible(data)
    _validate_quality_report(data)

    # ---- scene_plan validation (strict, no auto-correction) ----
    _validate_scene_plan(data, sentence_count)

    return data


# ===========================================================================
# Partial repair: field checkers (silent)
# ===========================================================================

def _field_is_valid(validator, *args) -> bool:
    try:
        validator(*args)
        return True
    except ValueError:
        return False


# ===========================================================================
# Partial repair: individual field repair calls
# ===========================================================================

def _repair_scene_plan(
    full_script_sentences: List[str],
    episode_data: Dict[str, Any],
    error_msg: Any,
) -> List[Dict[str, Any]]:
    N = len(full_script_sentences)
    sentences_json = json.dumps(full_script_sentences, ensure_ascii=False, indent=2)
    ep_json = json.dumps(episode_data, ensure_ascii=False, indent=2)

    user_prompt = (
        "This is a fresh independent repair request.\n"
        "The provided full_script_sentences is frozen and must not be changed.\n"
        "Return only the requested JSON field (scene_plan).\n"
        "Do not return the full episode object.\n"
        "Do not use markdown.\n\n"
        f"Number of sentences N = {N}.\n"
        f"Valid 0-based indices are 0 through {N - 1}.\n"
        f"The first scene MUST start at sentence_start = 0.\n"
        f"The final scene MUST end at sentence_end = {N - 1}.\n"
        f"sentence_end = {N} is OUT OF RANGE and will be rejected.\n"
        "Scenes must be contiguous: each next scene's sentence_start = previous sentence_end + 1.\n"
        "No gaps. No overlaps. scene_id must start at 1 and increment by 1.\n\n"
        f"Previous validation failure:\n{error_msg}\n\n"
        f"Episode data (context only):\n{ep_json}\n\n"
        f"Frozen full_script_sentences (DO NOT MODIFY):\n{sentences_json}\n\n"
        "Return ONLY the scene_plan JSON array now."
    )

    raw = call_gemini_with_fallback(
        system_instruction=SCENE_PLAN_REPAIR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_json_safe(raw)

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if isinstance(parsed.get("scene_plan"), list):
            return parsed["scene_plan"]
        raise ValueError(
            "استجابة إصلاح scene_plan يجب أن تكون JSON array أو كائنًا يحتوي على scene_plan كقائمة."
        )
    raise ValueError("استجابة إصلاح scene_plan ليست قائمة JSON صالحة.")


def _repair_retention_plan(
    full_script_sentences: List[str],
    current_retention_plan: Any,
    episode_data: Dict[str, Any],
    error_msg: Any,
) -> Dict[str, Any]:
    sentences_json = json.dumps(full_script_sentences, ensure_ascii=False, indent=2)
    current_json = json.dumps(current_retention_plan, ensure_ascii=False, indent=2) \
        if current_retention_plan is not None else "null"
    ep_json = json.dumps(episode_data, ensure_ascii=False, indent=2)

    user_prompt = (
        "This is a fresh independent repair request.\n"
        "The provided full_script_sentences is frozen and must not be changed.\n"
        "Return only the requested JSON field (retention_plan).\n"
        "Do not return the full episode object.\n"
        "Do not use markdown.\n\n"
        "Reminder: open_loops, pattern_interrupts, and escalation_points MUST be real JSON arrays of non-empty strings.\n"
        "Even a single item MUST be wrapped in an array, e.g. [\"one item\"].\n\n"
        f"Previous validation failure:\n{error_msg}\n\n"
        f"Episode data (context only):\n{ep_json}\n\n"
        f"Frozen full_script_sentences (DO NOT MODIFY):\n{sentences_json}\n\n"
        f"Current (possibly broken) retention_plan:\n{current_json}\n\n"
        "Return ONLY the retention_plan JSON object now."
    )

    raw = call_gemini_with_fallback(
        system_instruction=RETENTION_PLAN_REPAIR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_json_safe(raw)

    if not isinstance(parsed, dict):
        raise ValueError("استجابة إصلاح retention_plan يجب أن تكون كائن JSON.")
    if isinstance(parsed.get("retention_plan"), dict):
        return parsed["retention_plan"]
    return parsed


def _repair_visual_bible(
    full_script_sentences: List[str],
    current_visual_bible: Any,
    episode_data: Dict[str, Any],
    error_msg: Any,
) -> Dict[str, Any]:
    sentences_json = json.dumps(full_script_sentences, ensure_ascii=False, indent=2)
    current_json = json.dumps(current_visual_bible, ensure_ascii=False, indent=2) \
        if current_visual_bible is not None else "null"
    ep_json = json.dumps(episode_data, ensure_ascii=False, indent=2)

    user_prompt = (
        "This is a fresh independent repair request.\n"
        "The provided full_script_sentences is frozen and must not be changed.\n"
        "Return only the requested JSON field (visual_bible).\n"
        "Do not return the full episode object.\n"
        "Do not use markdown.\n\n"
        "Reminder: character_dna MUST be exactly: "
        "\"Orange character, muscular body, smooth head, two big white eyes, no mouth, black shorts.\"\n\n"
        f"Previous validation failure:\n{error_msg}\n\n"
        f"Episode data (context only):\n{ep_json}\n\n"
        f"Frozen full_script_sentences (DO NOT MODIFY):\n{sentences_json}\n\n"
        f"Current (possibly broken) visual_bible:\n{current_json}\n\n"
        "Return ONLY the visual_bible JSON object now."
    )

    raw = call_gemini_with_fallback(
        system_instruction=VISUAL_BIBLE_REPAIR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_json_safe(raw)

    if not isinstance(parsed, dict):
        raise ValueError("استجابة إصلاح visual_bible يجب أن تكون كائن JSON.")
    if isinstance(parsed.get("visual_bible"), dict):
        return parsed["visual_bible"]
    return parsed


def _repair_quality_report(
    full_script_sentences: List[str],
    creative_brief: Any,
    retention_plan: Any,
    scene_plan: Any,
    visual_bible: Any,
    current_quality_report: Any,
    error_msg: Any,
) -> Dict[str, Any]:
    sentences_json = json.dumps(full_script_sentences, ensure_ascii=False, indent=2)
    cb_json = json.dumps(creative_brief, ensure_ascii=False, indent=2)
    rp_json = json.dumps(retention_plan, ensure_ascii=False, indent=2)
    sp_json = json.dumps(scene_plan, ensure_ascii=False, indent=2)
    vb_json = json.dumps(visual_bible, ensure_ascii=False, indent=2)
    current_json = json.dumps(current_quality_report, ensure_ascii=False, indent=2) \
        if current_quality_report is not None else "null"

    user_prompt = (
        "This is a fresh independent repair request.\n"
        "Return only the requested JSON field (quality_report).\n"
        "Do not return the full episode object.\n"
        "Do not use markdown.\n\n"
        "Scores must be numbers between 0 and 10. approved must be a real boolean.\n\n"
        f"Previous validation failure:\n{error_msg}\n\n"
        f"Frozen full_script_sentences:\n{sentences_json}\n\n"
        f"creative_brief:\n{cb_json}\n\n"
        f"retention_plan:\n{rp_json}\n\n"
        f"scene_plan:\n{sp_json}\n\n"
        f"visual_bible:\n{vb_json}\n\n"
        f"Current (possibly broken) quality_report:\n{current_json}\n\n"
        "Return ONLY the quality_report JSON object now."
    )

    raw = call_gemini_with_fallback(
        system_instruction=QUALITY_REPORT_REPAIR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_json_safe(raw)

    if not isinstance(parsed, dict):
        raise ValueError("استجابة إصلاح quality_report يجب أن تكون كائن JSON.")
    if isinstance(parsed.get("quality_report"), dict):
        return parsed["quality_report"]
    return parsed


# ===========================================================================
# Partial repair: per-field retry wrappers
# ===========================================================================

def _repair_scene_plan_with_attempts(
    full_script_sentences: List[str],
    data: Dict[str, Any],
    episode_data: Dict[str, Any],
    first_error: Any,
) -> List[Dict[str, Any]]:
    N = len(full_script_sentences)
    last_err: Any = first_error

    for attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
        try:
            sp = _repair_scene_plan(full_script_sentences, episode_data, last_err)
            test_data = copy.deepcopy(data)
            test_data["scene_plan"] = sp
            _validate_scene_plan(test_data, N)
            return sp
        except Exception as e:
            last_err = e
            logger.warning(
                f"scene_plan repair attempt {attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} "
                f"failed: {e}"
            )

    raise ValueError(
        f"فشل إصلاح scene_plan بعد {MAX_FIELD_REPAIR_ATTEMPTS} محاولات. آخر خطأ: {last_err}"
    )


def _repair_retention_plan_with_attempts(
    full_script_sentences: List[str],
    data: Dict[str, Any],
    episode_data: Dict[str, Any],
    first_error: Any,
) -> Dict[str, Any]:
    last_err: Any = first_error
    current = data.get("retention_plan")

    for attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
        try:
            rp = _repair_retention_plan(
                full_script_sentences, current, episode_data, last_err
            )
            test_data = copy.deepcopy(data)
            test_data["retention_plan"] = rp
            _validate_retention_plan(test_data)
            return rp
        except Exception as e:
            last_err = e
            logger.warning(
                f"retention_plan repair attempt {attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} "
                f"failed: {e}"
            )

    raise ValueError(
        f"فشل إصلاح retention_plan بعد {MAX_FIELD_REPAIR_ATTEMPTS} محاولات. آخر خطأ: {last_err}"
    )


def _repair_visual_bible_with_attempts(
    full_script_sentences: List[str],
    data: Dict[str, Any],
    episode_data: Dict[str, Any],
    first_error: Any,
) -> Dict[str, Any]:
    last_err: Any = first_error
    current = data.get("visual_bible")

    for attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
        try:
            vb = _repair_visual_bible(
                full_script_sentences, current, episode_data, last_err
            )
            test_data = copy.deepcopy(data)
            test_data["visual_bible"] = vb
            _validate_visual_bible(test_data)
            return vb
        except Exception as e:
            last_err = e
            logger.warning(
                f"visual_bible repair attempt {attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} "
                f"failed: {e}"
            )

    raise ValueError(
        f"فشل إصلاح visual_bible بعد {MAX_FIELD_REPAIR_ATTEMPTS} محاولات. آخر خطأ: {last_err}"
    )


def _repair_quality_report_with_attempts(
    data: Dict[str, Any],
    episode_data: Dict[str, Any],
    first_error: Any,
) -> Dict[str, Any]:
    last_err: Any = first_error
    current = data.get("quality_report")

    for attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
        try:
            qr = _repair_quality_report(
                full_script_sentences=data["full_script_sentences"],
                creative_brief=data.get("creative_brief"),
                retention_plan=data.get("retention_plan"),
                scene_plan=data.get("scene_plan"),
                visual_bible=data.get("visual_bible"),
                current_quality_report=current,
                error_msg=last_err,
            )
            test_data = copy.deepcopy(data)
            test_data["quality_report"] = qr
            _validate_quality_report(test_data)
            return qr
        except Exception as e:
            last_err = e
            logger.warning(
                f"quality_report repair attempt {attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} "
                f"failed: {e}"
            )

    raise ValueError(
        f"فشل إصلاح quality_report بعد {MAX_FIELD_REPAIR_ATTEMPTS} محاولات. آخر خطأ: {last_err}"
    )


# ===========================================================================
# Merge + partial repair orchestrator
# ===========================================================================

def _merge_and_validate_partial_repairs(
    base_data: Dict[str, Any],
    episode_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Given base_data whose full_script_sentences is already normalized and valid,
    attempt to repair only the auxiliary fields that fail validation:
      - retention_plan
      - visual_bible
      - scene_plan
      - quality_report

    Non-repairable field failures (creative_brief) bubble up so the caller can
    trigger a full regeneration.
    """
    data = copy.deepcopy(base_data)

    # Sentences must be frozen and valid.
    sentences = data.get("full_script_sentences")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError(
            "_merge_and_validate_partial_repairs: full_script_sentences missing or empty."
        )
    N = len(sentences)

    # ---- creative_brief: NOT repairable here ----
    if not _field_is_valid(_validate_creative_brief, data):
        # Force the caller to regenerate the whole JSON.
        _validate_creative_brief(data)

    # ---- retention_plan ----
    if not _field_is_valid(_validate_retention_plan, data):
        logger.info("Detected invalid retention_plan. Attempting partial repair...")
        rp = _repair_retention_plan_with_attempts(
            full_script_sentences=sentences,
            data=data,
            episode_data=episode_data,
            first_error="retention_plan validation failed.",
        )
        data["retention_plan"] = rp
        _validate_retention_plan(data)

    # ---- visual_bible ----
    if not _field_is_valid(_validate_visual_bible, data):
        logger.info("Detected invalid visual_bible. Attempting partial repair...")
        vb = _repair_visual_bible_with_attempts(
            full_script_sentences=sentences,
            data=data,
            episode_data=episode_data,
            first_error="visual_bible validation failed.",
        )
        data["visual_bible"] = vb
        _validate_visual_bible(data)

    # ---- scene_plan ----
    if not _field_is_valid(_validate_scene_plan, data, N):
        logger.info("Detected invalid scene_plan. Attempting partial repair...")
        # Capture first error message for the repair prompt.
        first_err: Any = "scene_plan validation failed."
        try:
            _validate_scene_plan(data, N)
        except ValueError as e:
            first_err = e

        sp = _repair_scene_plan_with_attempts(
            full_script_sentences=sentences,
            data=data,
            episode_data=episode_data,
            first_error=first_err,
        )
        data["scene_plan"] = sp
        _validate_scene_plan(data, N)

    # ---- quality_report ----
    if not _field_is_valid(_validate_quality_report, data):
        logger.info("Detected invalid quality_report. Attempting partial repair...")
        qr = _repair_quality_report_with_attempts(
            data=data,
            episode_data=episode_data,
            first_error="quality_report validation failed.",
        )
        data["quality_report"] = qr
        _validate_quality_report(data)

    # Final full validation on the merged result.
    return validate_script_output(data)


# ===========================================================================
# Public generator
# ===========================================================================

def generate_stage1_script(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate the Stage-1 script with a partial-repair strategy:

    1. Ask the model for a full JSON.
    2. If full_script_sentences is invalid -> discard and regenerate full JSON.
    3. If full_script_sentences is valid -> freeze it and try to repair only the
       auxiliary fields that fail (retention_plan / visual_bible / scene_plan /
       quality_report). Never mutate the frozen sentences.
    4. Only return after a successful full validate_script_output().
    """
    last_error: Any = None

    for full_attempt in range(MAX_FULL_GENERATION_ATTEMPTS):
        logger.info(
            f"Full generation attempt {full_attempt + 1}/{MAX_FULL_GENERATION_ATTEMPTS}"
        )

        user_prompt = (
            "Here is the raw input JSON for the episode:\n"
            f"{json.dumps(episode_data, ensure_ascii=False, indent=2)}"
        )

        # ---------- call model ----------
        try:
            raw_response = call_gemini_with_fallback(
                system_instruction=STAGE_1_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            parsed_json = _parse_json_safe(raw_response)
        except Exception as e:
            last_error = e
            logger.warning(
                f"Attempt {full_attempt + 1}: model call / JSON parse failed: {e}"
            )
            continue

        if not isinstance(parsed_json, dict):
            last_error = ValueError("استجابة النموذج ليست كائن JSON.")
            logger.warning(f"Attempt {full_attempt + 1}: response is not a JSON object.")
            continue

        # ---------- core checks (top-level keys + sentences) ----------
        try:
            _check_required_top_level_keys(parsed_json)
            cleaned_sentences, total_words = _extract_and_validate_full_script_sentences(
                parsed_json
            )
        except ValueError as e:
            last_error = e
            logger.warning(
                f"Attempt {full_attempt + 1}: core validation failed (sentences / keys). "
                f"Regenerating full JSON. Reason: {e}"
            )
            continue

        # Freeze sentences as the source of truth for this attempt.
        parsed_json["full_script_sentences"] = cleaned_sentences
        parsed_json["total_word_count"] = total_words

        # ---------- try full validation first ----------
        try:
            return validate_script_output(parsed_json)
        except ValueError as e:
            last_error = e
            logger.info(
                f"Attempt {full_attempt + 1}: full validation failed. "
                f"Trying partial repairs. Reason: {e}"
            )

        # ---------- try partial repairs ----------
        try:
            repaired = _merge_and_validate_partial_repairs(parsed_json, episode_data)
            return repaired
        except Exception as e:
            last_error = e
            logger.warning(
                f"Attempt {full_attempt + 1}: partial repairs did not produce a valid "
                f"result. Regenerating full JSON. Reason: {e}"
            )
            continue

    raise ValueError(
        f"فشل توليد سكربت المرحلة الأولى بعد {MAX_FULL_GENERATION_ATTEMPTS} محاولات كاملة. "
        f"آخر خطأ: {last_error}"
    )
