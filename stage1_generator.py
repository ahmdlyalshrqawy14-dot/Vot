import json
import re
import logging
from typing import Dict, Any, List
from gemini_engine import call_gemini_with_fallback

logger = logging.getLogger("Stage1Generator")

STAGE_1_SYSTEM_PROMPT = """You are an elite YouTube Episode Producer, Narrative Architect, and Visual Director specializing in self-development, fitness psychology, and audience retention.

Your job is NOT to simply rewrite the input data into a script.
Your job is to think like a producer: design an EPISODE, then a SCRIPT, then a VISUAL PLAN.

You must follow this production pipeline internally, in this exact order:

Raw Episode
→ Episode Strategy
→ Unique Angle
→ Narrative Blueprint
→ Script
→ Scene Blueprint
→ Visual Direction
→ Image Prompts
→ Quality Control

==================================================
INPUT DATA STRUCTURE
==================================================
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

==================================================
STEP 1 — EPISODE STRATEGY
==================================================
Before writing anything, decide the strategic intent of the episode.
Produce:
- central_promise: the single value the viewer will walk away with.
- target_viewer: a precise description of who this episode is for.
- viewer_problem: the real problem or frustration the viewer feels.
- desired_viewer_response: what you want the viewer to think, feel, or do after watching.
- content_goal: what this episode is trying to accomplish (retention, authority, transformation, curiosity, etc.).

==================================================
STEP 2 — UNIQUE ANGLE
==================================================
Choose a specific angle the episode will be told from.
Do NOT use the same narrative template every time.
The angle defines HOW the topic is presented to the viewer.
Produce:
- angle: the specific lens, framing, or entry point.
- why_this_angle: why this angle fits this topic and this viewer.
- curiosity_question: the question that pulls the viewer through the episode.

==================================================
STEP 3 — NARRATIVE BLUEPRINT
==================================================
Design the narrative map BEFORE writing the script.
Choose the structure that fits the topic. Do NOT force a fixed template.
Possible beats include (but are not limited to):
Hook, Problem, Contradiction, Explanation, Story, Mechanism, Solution, Payoff, Twist, Reframe, Demonstration, Warning, Transformation.
Produce:
- structure_type: a short label for the chosen structure.
- beats: list of beats, each with:
  - beat_id: short identifier (e.g. "beat_01")
  - purpose: why this beat exists in the narrative.
  - summary: what happens in this beat.
  - attention_function: how this beat maintains or resets attention.

The Script's sections must follow the beats, not a fixed outline.

==================================================
STEP 4 — SCRIPT
==================================================
Target: 5–6 minutes spoken English, roughly 800–900 words at ~150 wpm.

TONE & STYLE:
- Spoken, natural English — as a professional YouTube creator speaking directly to the viewer.
- Conversational, authoritative yet warm.
- The script must feel like ONE continuous cohesive piece, not disconnected lines.
- Smooth transitions between sections and sentences.
- Strong hook, strong retention, practical value, meaningful payoff.
- Natural CTA and a strong comment question.

CADENCE:
- Short, breathable sentences.
- Every sentence ends with ".", "?", or "!".
- Prefer clear, direct sentences. Avoid semicolons, em-dashes, and overly long compound sentences.
- Natural connectors are allowed: "And", "But", "So", "That's why", "Here's the thing", etc.

The sections in the script must reflect the Narrative Blueprint beats.
Do NOT force the old fixed outline (HOOK / INTRO / MYTH / ANALOGY / KEY POINTS / ACTIONABLE / OUTRO).
The blueprint decides the order.

ABSOLUTELY NO SHORTCUTS: do not summarize or skip key points. Expand every idea into concrete, vivid, spoken material.

Produce:
- hook: the opening spoken hook (string).
- sections: list of sections, each with:
  - section_name
  - purpose
  - sentences (list of strings)
- full_script_sentences: chronological flat list of every sentence in the entire script, in spoken order.

==================================================
STEP 5 — SCENE BLUEPRINT
==================================================
Convert the script into scenes for visual production.
A scene is a narrative unit, NOT a single sentence.
NEVER assume "one sentence = one image".
A scene may cover several sentences when that makes narrative sense.

Every scene must reference the script through indexes into script.full_script_sentences (0-based).

Produce scene_blueprint: list of scenes, each with:
- scene_id: e.g. "scene_01"
- narration_sentence_indexes: list of integers, 0-based indexes into script.full_script_sentences
- purpose: narrative purpose of the scene.
- visual_role: the role the visuals play here (e.g. pattern interrupt, explanation, metaphor, emotion, contrast, context, storytelling, mechanism, demonstration, attention reset).
- visual_description: what the viewer actually sees.
- transition_note: how this scene transitions into the next.

==================================================
STEP 6 — VISUAL DIRECTION
==================================================
Define the visual language for the whole episode so every image feels like it belongs to the same film.
Produce visual_direction with:
- visual_style: overall style (palette, lighting, tone, medium).
- character_direction: how characters appear, repeat, and evolve.
- environment_direction: how environments look and behave.
- continuity_rules: list of rules that must stay consistent across scenes.

==================================================
STEP 7 — IMAGE PROMPTS
==================================================
For every scene, produce an image prompt.
The prompt must be CONSTRUCTED from:
Scene Purpose + Visual Role + Visual Description + Visual Direction.
NEVER just copy a script sentence into a prompt.
Each image must ADD value: explanation, metaphor, emotion, contrast, context, storytelling, mechanism, visual demonstration, or attention reset.

Produce image_prompts: list of items, each with:
- scene_id: must match a scene_id from scene_blueprint.
- prompt: the full image prompt string.
- visual_purpose: what this image contributes to the episode.

==================================================
STEP 8 — QUALITY REPORT
==================================================
Review the whole episode before outputting.
Produce quality_report with:
- narrative_coherence: how well the story holds together.
- visual_coherence: how well the visuals form one consistent language.
- script_visual_alignment: how well images match the narration.
- issues_found: list of any problems or weaknesses.
- final_status: short final verdict (e.g. "approved", "approved_with_notes").

==================================================
OUTPUT FORMAT
==================================================
Return ONLY a valid, parsable JSON object.
No markdown, no backticks, no commentary, no text outside the JSON.

The JSON MUST follow this exact shape:

{
  "id": 0,
  "topic": "string",

  "episode_strategy": {
    "central_promise": "string",
    "target_viewer": "string",
    "viewer_problem": "string",
    "desired_viewer_response": "string",
    "content_goal": "string"
  },

  "unique_angle": {
    "angle": "string",
    "why_this_angle": "string",
    "curiosity_question": "string"
  },

  "narrative_blueprint": {
    "structure_type": "string",
    "beats": [
      {
        "beat_id": "string",
        "purpose": "string",
        "summary": "string",
        "attention_function": "string"
      }
    ]
  },

  "script": {
    "hook": "string",
    "sections": [
      {
        "section_name": "string",
        "purpose": "string",
        "sentences": []
      }
    ],
    "full_script_sentences": []
  },

  "scene_blueprint": [
    {
      "scene_id": "scene_01",
      "narration_sentence_indexes": [],
      "purpose": "string",
      "visual_role": "string",
      "visual_description": "string",
      "transition_note": "string"
    }
  ],

  "visual_direction": {
    "visual_style": "string",
    "character_direction": "string",
    "environment_direction": "string",
    "continuity_rules": []
  },

  "image_prompts": [
    {
      "scene_id": "scene_01",
      "prompt": "string",
      "visual_purpose": "string"
    }
  ],

  "quality_report": {
    "narrative_coherence": "string",
    "visual_coherence": "string",
    "script_visual_alignment": "string",
    "issues_found": [],
    "final_status": "string"
  },

  "total_word_count": 0
}"""


def _clean_json_string(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _clean_sentence(sentence: str) -> str:
    s_clean = sentence.strip()
    if not s_clean:
        return ""
    if not (s_clean.endswith(".") or s_clean.endswith("?") or s_clean.endswith("!")):
        s_clean += "."
    return s_clean


def _require_non_empty_string(container: Dict[str, Any], key: str, where: str) -> str:
    if key not in container:
        raise ValueError(f"الحقل المطلوب '{key}' مفقود في {where}!")
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"الحقل '{key}' في {where} يجب أن يكون string غير فارغ.")
    return value


def validate_script_output(data: Dict[str, Any]) -> Dict[str, Any]:
    required_top_level = [
        "id",
        "topic",
        "episode_strategy",
        "unique_angle",
        "narrative_blueprint",
        "script",
        "scene_blueprint",
        "visual_direction",
        "image_prompts",
        "quality_report",
    ]
    for k in required_top_level:
        if k not in data:
            raise ValueError(f"الحقل المطلوب '{k}' مفقود من مخرجات JSON!")

    # ----- script -----
    script = data.get("script")
    if not isinstance(script, dict):
        raise ValueError("الحقل 'script' يجب أن يكون كائنًا (object).")

    # script.hook
    _require_non_empty_string(script, "hook", "'script'")

    # script.sections
    sections = script.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("الحقل 'script.sections' يجب أن يكون قائمة غير فارغة.")
    for s_idx, section in enumerate(sections):
        if not isinstance(section, dict):
            raise ValueError(f"script.sections[{s_idx}] يجب أن يكون كائنًا (object).")
        _require_non_empty_string(section, "section_name", f"script.sections[{s_idx}]")
        _require_non_empty_string(section, "purpose", f"script.sections[{s_idx}]")

        sentences = section.get("sentences")
        if not isinstance(sentences, list) or not sentences:
            raise ValueError(
                f"script.sections[{s_idx}].sentences يجب أن يكون قائمة غير فارغة."
            )
        for sent_idx, sent in enumerate(sentences):
            if not isinstance(sent, str) or not sent.strip():
                raise ValueError(
                    f"script.sections[{s_idx}].sentences[{sent_idx}] يجب أن يكون string غير فارغ."
                )

    # script.full_script_sentences
    if "full_script_sentences" not in script:
        raise ValueError("الحقل 'script.full_script_sentences' مفقود!")

    raw_sentences = script.get("full_script_sentences", [])
    if not isinstance(raw_sentences, list) or not raw_sentences:
        raise ValueError("مصفوفة 'script.full_script_sentences' فارغة!")

    cleaned_sentences: List[str] = []
    total_words = 0
    for s in raw_sentences:
        if not isinstance(s, str):
            continue
        s_clean = _clean_sentence(s)
        if not s_clean:
            continue
        cleaned_sentences.append(s_clean)
        total_words += len(s_clean.split())

    if not cleaned_sentences:
        raise ValueError("لا توجد جمل صالحة في 'script.full_script_sentences'!")

    script["full_script_sentences"] = cleaned_sentences
    data["script"] = script

    # ----- backward compatibility -----
    data["full_script_sentences"] = list(cleaned_sentences)
    data["total_word_count"] = total_words

    # ----- episode_strategy -----
    episode_strategy = data.get("episode_strategy")
    if not isinstance(episode_strategy, dict):
        raise ValueError("'episode_strategy' يجب أن يكون كائنًا (object).")
    for key in [
        "central_promise",
        "target_viewer",
        "viewer_problem",
        "desired_viewer_response",
        "content_goal",
    ]:
        _require_non_empty_string(episode_strategy, key, "'episode_strategy'")

    # ----- unique_angle -----
    unique_angle = data.get("unique_angle")
    if not isinstance(unique_angle, dict):
        raise ValueError("'unique_angle' يجب أن يكون كائنًا (object).")
    for key in ["angle", "why_this_angle", "curiosity_question"]:
        _require_non_empty_string(unique_angle, key, "'unique_angle'")

    # ----- narrative_blueprint -----
    narrative_blueprint = data.get("narrative_blueprint")
    if not isinstance(narrative_blueprint, dict):
        raise ValueError("'narrative_blueprint' يجب أن يكون كائنًا (object).")
    _require_non_empty_string(narrative_blueprint, "structure_type", "'narrative_blueprint'")

    beats = narrative_blueprint.get("beats")
    if not isinstance(beats, list) or not beats:
        raise ValueError("'narrative_blueprint.beats' يجب أن يكون قائمة غير فارغة.")

    seen_beat_ids = set()
    for b_idx, beat in enumerate(beats):
        if not isinstance(beat, dict):
            raise ValueError(f"narrative_blueprint.beats[{b_idx}] يجب أن يكون كائنًا (object).")
        for key in ["beat_id", "purpose", "summary", "attention_function"]:
            _require_non_empty_string(beat, key, f"narrative_blueprint.beats[{b_idx}]")
        beat_id = beat.get("beat_id")
        if beat_id in seen_beat_ids:
            raise ValueError(f"تكرار في beat_id داخل narrative_blueprint: '{beat_id}'.")
        seen_beat_ids.add(beat_id)

    # ----- quality_report -----
    quality_report = data.get("quality_report")
    if not isinstance(quality_report, dict):
        raise ValueError("'quality_report' يجب أن يكون كائنًا (object).")
    for key in ["narrative_coherence", "visual_coherence", "script_visual_alignment", "final_status"]:
        _require_non_empty_string(quality_report, key, "'quality_report'")
    issues_found = quality_report.get("issues_found")
    if not isinstance(issues_found, list):
        raise ValueError("'quality_report.issues_found' يجب أن يكون قائمة (list).")

    # ----- scene_blueprint -----
    scene_blueprint = data.get("scene_blueprint")
    if not isinstance(scene_blueprint, list) or not scene_blueprint:
        raise ValueError("'scene_blueprint' يجب أن يكون قائمة غير فارغة!")

    num_sentences = len(cleaned_sentences)
    valid_scene_ids = set()
    required_scene_keys = [
        "scene_id",
        "narration_sentence_indexes",
        "purpose",
        "visual_role",
        "visual_description",
        "transition_note",
    ]
    used_sentence_indexes = set()
    for idx, scene in enumerate(scene_blueprint):
        if not isinstance(scene, dict):
            raise ValueError(f"scene_blueprint[{idx}] ليس كائنًا (object).")
        for key in required_scene_keys:
            if key not in scene:
                raise ValueError(f"scene_blueprint[{idx}] ينقصه الحقل '{key}'.")

        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError(f"scene_blueprint[{idx}].scene_id غير صالح.")
        if scene_id in valid_scene_ids:
            raise ValueError(f"scene_id مكرر في scene_blueprint: '{scene_id}'.")
        valid_scene_ids.add(scene_id)

        # non-empty string fields on scene
        for str_key in ["purpose", "visual_role", "visual_description", "transition_note"]:
            val = scene.get(str_key)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"scene_blueprint[{idx}].{str_key} يجب أن يكون string غير فارغ."
                )

        idxs = scene.get("narration_sentence_indexes")
        if not isinstance(idxs, list) or not idxs:
            raise ValueError(
                f"scene_blueprint[{idx}].narration_sentence_indexes يجب أن يكون قائمة غير فارغة."
            )
        for i in idxs:
            if isinstance(i, bool) or not isinstance(i, int):
                raise ValueError(
                    f"scene_blueprint[{idx}].narration_sentence_indexes يحتوي على قيمة غير صحيحة (ليست int): {i!r}"
                )
            if i < 0 or i >= num_sentences:
                raise ValueError(
                    f"scene_blueprint[{idx}] يحتوي على index خارج النطاق: {i} "
                    f"(عدد جمل السكربت: {num_sentences})."
                )
            used_sentence_indexes.add(i)

    # coverage: every sentence must be referenced by at least one scene
    missing_indexes = [i for i in range(num_sentences) if i not in used_sentence_indexes]
    if missing_indexes:
        raise ValueError(
            f"يوجد {len(missing_indexes)} جملة من script.full_script_sentences غير مرتبطة بأي Scene. "
            f"الفهارس الناقصة: {missing_indexes}."
        )

    # ----- visual_direction -----
    visual_direction = data.get("visual_direction")
    if not isinstance(visual_direction, dict):
        raise ValueError("'visual_direction' يجب أن يكون كائنًا (object).")
    for key in ["visual_style", "character_direction", "environment_direction"]:
        _require_non_empty_string(visual_direction, key, "'visual_direction'")
    if "continuity_rules" not in visual_direction:
        raise ValueError("'visual_direction' ينقصه الحقل 'continuity_rules'.")
    continuity_rules = visual_direction.get("continuity_rules")
    if not isinstance(continuity_rules, list):
        raise ValueError("'visual_direction.continuity_rules' يجب أن يكون قائمة (list).")
    if not continuity_rules:
        raise ValueError("'visual_direction.continuity_rules' يجب ألا تكون فارغة.")

    # ----- image_prompts -----
    image_prompts = data.get("image_prompts")
    if not isinstance(image_prompts, list) or not image_prompts:
        raise ValueError("'image_prompts' يجب أن يكون قائمة غير فارغة!")

    scenes_with_prompts = set()
    for idx, item in enumerate(image_prompts):
        if not isinstance(item, dict):
            raise ValueError(f"image_prompts[{idx}] ليس كائنًا (object).")
        for key in ["scene_id", "prompt", "visual_purpose"]:
            if key not in item:
                raise ValueError(f"image_prompts[{idx}] ينقصه الحقل '{key}'.")
        item_scene_id = item.get("scene_id")
        if not isinstance(item_scene_id, str) or not item_scene_id.strip():
            raise ValueError(f"image_prompts[{idx}].scene_id غير صالح.")
        if item_scene_id not in valid_scene_ids:
            raise ValueError(
                f"image_prompts[{idx}].scene_id '{item_scene_id}' غير موجود في scene_blueprint."
            )
        if item_scene_id in scenes_with_prompts:
            raise ValueError(
                f"تكرار scene_id داخل image_prompts: '{item_scene_id}' "
                f"(يُفترض وجود Image Prompt واحد لكل Scene)."
            )
        scenes_with_prompts.add(item_scene_id)

        for str_key in ["prompt", "visual_purpose"]:
            val = item.get(str_key)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"image_prompts[{idx}].{str_key} يجب أن يكون string غير فارغ."
                )

    # every scene must have at least one prompt
    missing_prompts = [sid for sid in valid_scene_ids if sid not in scenes_with_prompts]
    if missing_prompts:
        raise ValueError(
            f"بعض الـScenes لا تحتوي على Image Prompt: {sorted(missing_prompts)}."
        )

    # ----- word count log -----
    if 800 <= total_words <= 900:
        logger.info(f"🎯 حجم السكربت مثالي: {total_words} كلمة.")
    else:
        logger.warning(
            f"ℹ️ حجم السكربت: {total_words} كلمة (المستهدف: 800 - 900 كلمة)."
        )

    return data


def generate_stage1_script(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = f"Here is the raw input JSON for the episode:\n{json.dumps(episode_data, ensure_ascii=False, indent=2)}"

    raw_response = call_gemini_with_fallback(
        system_instruction=STAGE_1_SYSTEM_PROMPT,
        user_prompt=user_prompt
    )

    cleaned_json_text = _clean_json_string(raw_response)
    parsed_json = json.loads(cleaned_json_text)
    return validate_script_output(parsed_json)
