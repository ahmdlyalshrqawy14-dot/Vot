import json
import re
import logging
from typing import Dict, Any, List
from gemini_engine import call_gemini_with_fallback

logger = logging.getLogger("Stage1Generator")

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

SCENE PLAN INDEXING RULES (CRITICAL — READ CAREFULLY):
- If full_script_sentences contains N sentences, the last valid index is N - 1.
- sentence_start and sentence_end MUST both be between 0 and N - 1 inclusive.
- NEVER use N as a value for sentence_end. N is OUT OF RANGE and will be rejected.
- Build the scene_plan ONLY AFTER the final full_script_sentences list is frozen and its count N is known.
- The final scene in scene_plan MUST end exactly at index len(full_script_sentences) - 1.
- Concretely: if the final full_script_sentences has 74 sentences, the last scene must end at sentence_end = 73, NOT 74.
- Indices are 0-based, never 1-based. The first scene must start at sentence_start = 0.
- Scenes must be contiguous: each next scene's sentence_start must equal the previous scene's sentence_end + 1.
- Do NOT leave gaps between scenes and do NOT overlap scenes.
"""


# ---------------------------------------------------------------------------
# Sentence count bounds (new)
# ---------------------------------------------------------------------------

MIN_SENTENCES = 60
MAX_SENTENCES = 80
PREFERRED_MIN_SENTENCES = 60
PREFERRED_MAX_SENTENCES = 70


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _clean_json_string(raw_text: str) -> str:
    if raw_text is None:
        raise ValueError("استجابة النموذج فارغة (None).")
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json_safe(raw_text: str) -> Dict[str, Any]:
    cleaned = _clean_json_string(raw_text)
    if not cleaned:
        raise ValueError("استجابة النموذج فارغة بعد التنظيف.")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"استجابة النموذج ليست JSON صالحًا: {e}") from e


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Field-level validators
# ---------------------------------------------------------------------------

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
        _require_list(rp, field, "retention_plan")


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


# ---------------------------------------------------------------------------
# Safe off-by-one fix for the LAST scene's sentence_end
# ---------------------------------------------------------------------------

def _attempt_last_scene_off_by_one_fix(data: Dict[str, Any], total_sentences: int) -> bool:
    """
    محاولة إصلاح آمنة لحالة واحدة فقط:
    - جميع المشاهد السابقة صحيحة ومتسلسلة بلا فجوات أو تداخل.
    - آخر مشهد sentence_end يساوي بالضبط len(cleaned_sentences) (خطأ off-by-one بمقدار +1).

    لا يُطبَّق الإصلاح إذا كان هناك أي خطأ آخر (فجوة، تداخل، فهرس سالب،
    sentence_end أكبر من len، أو أي مشكلة في مشهد سابق).

    يُعيد True إذا تم التصحيح، False إذا لم ينطبق شرط التصحيح الآمن.
    """
    scene_plan = data.get("scene_plan")
    if not isinstance(scene_plan, list) or not scene_plan:
        return False

    previous_end = -1
    last_index = len(scene_plan) - 1

    for idx, scene in enumerate(scene_plan):
        if not isinstance(scene, dict):
            return False

        s_start = scene.get("sentence_start")
        s_end = scene.get("sentence_end")

        # يجب أن يكونا أعدادًا صحيحة حقيقية (لا bool)
        if isinstance(s_start, bool) or not isinstance(s_start, int):
            return False
        if isinstance(s_end, bool) or not isinstance(s_end, int):
            return False

        # لا نقبل أي فهرس سالب
        if s_start < 0 or s_end < 0:
            return False

        # لا نقبل أي sentence_end أكبر من total_sentences
        if s_end > total_sentences:
            return False

        if s_start > s_end:
            return False

        is_last = (idx == last_index)

        if idx == 0:
            if s_start != 0:
                return False
        else:
            # لا فجوات ولا تداخل
            if s_start != previous_end + 1:
                return False

        if is_last:
            # الحالة الوحيدة المسموح إصلاحها: off-by-one بمقدار +1 في النهاية
            if s_end == total_sentences:
                scene["sentence_end"] = total_sentences - 1
                logger.warning(
                    "⚠️ تم إصلاح خطأ off-by-one في آخر مشهد داخل scene_plan: "
                    "sentence_end كان %d (خارج النطاق) وتم تصحيحه إلى %d "
                    "(آخر فهرس مسموح لـ full_script_sentences). "
                    "لم يتم تغيير أي مشهد سابق، ولم تُضف أو تُحذف مشاهد.",
                    total_sentences,
                    total_sentences - 1,
                )
                return True
            # أي قيمة أخرى: لا نصلح هنا، نترك المدقق الرئيسي يرمي خطأه
            return False
        else:
            # المشاهد غير الأخيرة يجب أن تنتهي قبل آخر فهرس
            if s_end >= total_sentences:
                return False
            previous_end = s_end

    return False


# ---------------------------------------------------------------------------
# Scene plan validation
# ---------------------------------------------------------------------------

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
                f"(الحد الأقصى {total_sentences - 1})."
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
            f"آخر مشهد يجب أن ينتهي عند {total_sentences - 1} لكنه انتهى عند {previous_end}."
        )

    if not all(covered):
        missing = [i for i, c in enumerate(covered) if not c]
        raise ValueError(
            f"scene_plan لا يغطي كل الجمل. الجمل غير المغطاة (0-based): {missing[:10]}"
            f"{' ...' if len(missing) > 10 else ''}"
        )


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------

def validate_script_output(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("مخرجات النموذج يجب أن تكون كائن JSON.")

    # ---- top-level required keys ----
    required_keys = ["id", "topic", "hook", "sections", "full_script_sentences"]
    for k in required_keys:
        if k not in data:
            raise ValueError(f"الحقل المطلوب '{k}' مفقود من مخرجات JSON!")

    required_new_keys = [
        "creative_brief",
        "retention_plan",
        "scene_plan",
        "visual_bible",
        "quality_report",
    ]
    for k in required_new_keys:
        if k not in data:
            raise ValueError(f"الحقل المطلوب '{k}' مفقود من مخرجات JSON!")

    # ---- full_script_sentences ----
    sentences = data.get("full_script_sentences", [])
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("مصفوفة 'full_script_sentences' فارغة!")

    cleaned_sentences: List[str] = []
    total_words = 0
    for s in sentences:
        if not isinstance(s, str):
            raise ValueError("كل عنصر في 'full_script_sentences' يجب أن يكون نصًا.")
        s_clean = s.strip()
        if not s_clean:
            continue
        if not (s_clean.endswith(".") or s_clean.endswith("?") or s_clean.endswith("!")):
            s_clean += "."
        cleaned_sentences.append(s_clean)
        total_words += len(s_clean.split())

    if not cleaned_sentences:
        raise ValueError("مصفوفة 'full_script_sentences' فارغة بعد التنظيف.")

    data["full_script_sentences"] = cleaned_sentences
    data["total_word_count"] = total_words

    # ---- NEW: sentence count validation ----
    sentence_count = len(cleaned_sentences)

    if sentence_count < MIN_SENTENCES:
        raise ValueError(
            f"عدد الجمل في 'full_script_sentences' غير كافٍ. "
            f"العدد الفعلي: {sentence_count}، الحد الأدنى المطلوب: {MIN_SENTENCES}."
        )

    if sentence_count > MAX_SENTENCES:
        raise ValueError(
            f"عدد الجمل في 'full_script_sentences' تجاوز الحد الأقصى. "
            f"العدد الفعلي: {sentence_count}، الحد الأقصى المسموح: {MAX_SENTENCES}."
        )

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

    # ---- word count log ----
    if 800 <= total_words <= 900:
        logger.info(f"🎯 حجم السكربت مثالي: {total_words} كلمة.")
    else:
        logger.warning(f"ℹ️ حجم السكربت: {total_words} كلمة (المستهدف: 800 - 900 كلمة).")

    # ---- validate sub-objects ----
    _validate_creative_brief(data)
    _validate_retention_plan(data)
    _validate_visual_bible(data)
    _validate_quality_report(data)

    # ---- scene_plan validation ----
    total_sentences = len(cleaned_sentences)

    # إصلاح آمن لحالة واحدة فقط:
    # off-by-one بمقدار +1 في sentence_end الخاص بآخر مشهد فقط،
    # مع بقاء جميع المشاهد السابقة صحيحة ومتسلسلة بلا فجوات أو تداخل.
    # لا يُطبَّق أي إصلاح آخر، وأي خطأ مختلف سيُرفع كـ ValueError من المدقق الرئيسي.
    _attempt_last_scene_off_by_one_fix(data, total_sentences)

    _validate_scene_plan(data, total_sentences)

    return data


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_stage1_script(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = f"Here is the raw input JSON for the episode:\n{json.dumps(episode_data, ensure_ascii=False, indent=2)}"

    raw_response = call_gemini_with_fallback(
        system_instruction=STAGE_1_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )

    parsed_json = _parse_json_safe(raw_response)
    return validate_script_output(parsed_json)
