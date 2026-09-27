import json
import re
import copy
import logging
from typing import Dict, Any, List, Tuple
from gemini_engine import call_gemini_with_fallback

logger = logging.getLogger("Stage1Generator")


# ===========================================================================
# CONSTANTS
# ===========================================================================

MIN_SENTENCES = 60
MAX_SENTENCES = 80
PREFERRED_MIN_SENTENCES = 60
PREFERRED_MAX_SENTENCES = 70

MAX_FULL_GENERATION_ATTEMPTS = 4
MAX_FIELD_REPAIR_ATTEMPTS = 4
MAX_SPLIT_REPAIR_ATTEMPTS = 4
MAX_PHASE_D_ROUNDS = 2

CHARACTER_DNA = "Orange character, muscular body, smooth head, two big white eyes, no mouth, black shorts."

APPROVED_VISUAL_BEATS = {
    "establishing",
    "symbolic_setup",
    "development",
    "escalation",
    "pattern_interrupt",
    "visual_reveal",
    "metaphor_transformation",
    "quiet_reflection",
    "ending_tail",
}

MIN_ENDING_TAIL_SECONDS = 8
MAX_ENDING_TAIL_SECONDS = 12


# ===========================================================================
# PHASE A — SCRIPT GENERATION (script_text only)
# ===========================================================================

PHASE_A_SYSTEM_PROMPT = """You are an elite AI Episode Director for a high-retention English-language YouTube channel about human behavior, psychology, philosophy, ethics, language, linguistics, literature, civilizations, history of ideas, important books, references, and meaningful human contradictions.

The channel explores why human beings think, feel, speak, choose, suffer, resist change, build identities, create meaning, and contradict themselves.

The channel is NOT primarily a fitness channel. Do NOT default to muscles, exercise, dieting, body transformation, gym settings, workout imagery, or physical-training metaphors unless the episode data explicitly requires them.
You are NOT a generic scriptwriter. You think in terms of narrative architecture, emotional pacing, and viewer retention.

Your ONLY task right now: write the complete spoken script for one YouTube episode, as ONE continuous, coherent piece of English text, based on the input episode data.

INPUT DATA STRUCTURE
You will receive a JSON object with:
"id", "topic", "the_myth", "the_truth", "the_analogy", "key_points", "actionable_solution", "core_takeaway", "comment_question".

INTERNAL DIRECTING WORKFLOW (execute silently before writing):
1. Analyze the episode idea deeply and extract the essential information.
2. Define a UNIQUE, non-generic ANGLE.
3. Define the CORE IDEA in one sharp sentence.
4. Define the CENTRAL CONFLICT (desire vs. obstacle, belief vs. reality).
5. Define an UNEXPECTED INSIGHT that reframes the topic.
6. Choose a NARRATIVE ARCHITECTURE that fits the topic — do NOT always use the same rigid order.
   Use functional beats such as: Hook, Open Loop, Pattern Interrupt, Problem, Mechanism, Story/Analogy,
   Escalation, Solution, Application, Payoff, Ending Callback. Reorder freely to fit the topic.
7. Build an EMOTIONAL PROGRESSION (e.g. curiosity → tension → clarity → motivation → resolve).
8. Define the PAYOFF that connects back to the HOOK.
9. THEN write the script.

SCRIPT REQUIREMENTS:
- The script must feel like ONE connected, cohesive piece — never disconnected bullet points.
- Strong hook in the opening lines.
- A human question or tension early on.
- Build curiosity and escalation.
- Introduce and correctly frame a relevant thinker, scholar, author, historical source, or book, when supplied
  or clearly appropriate — never invent quotations, studies, or unsupported medical/psychological claims.
- AUTHORITATIVE REFERENCE PRESERVATION (CRITICAL): If the episode_data supplies a relevant thinker, author,
  scholar, philosopher, historical figure, researcher, or a specific book/reference/source, the generated script
  MUST explicitly preserve and mention that exact named reference in the script text. Do NOT substitute it with
  a vague gesture like "some thinkers believe" or "a famous book says". Use the actual name that was provided.
  If multiple named references are supplied, preserve and mention each of them. If no named reference is supplied,
  do not invent one — but if one is clearly appropriate and well-established, you may reference it accurately,
  and once you do, it must be the real, correct name.
- Explain the idea clearly.
- Include a meaningful contradiction or insight.
- End with a practical implication and a memorable callback to the hook.
- Natural, conversational spoken English — as if talking directly to the viewer.
- Avoid a rigid Hook → Intro → Myth → Truth → Tips template; the beats may be reordered.
- Avoid generic motivational filler and excessive repetition.
- Avoid extremely long compound sentences, semicolons, or em-dashes; prefer short, speakable sentences.
- Use natural connecting words (And, But, So, That's why, Here's the thing) so it never feels choppy.
- Every sentence must end with ".", "?", or "!".
- HARD REQUIREMENT: the final locally extracted sentence list MUST contain at least 60 and at most 80
  sentences inclusive. This is NOT approximate. Never produce fewer than 60 sentences. Never produce more
  than 80 sentences. A later step splits the script deterministically and will reject any count outside
  the 60-80 range.
- Do NOT artificially restrict yourself to any specific word-count band. Word count is not the controlling
  requirement; narrative coherence and sentence count are.
- Preserve the meaning of the_truth, key_points, and actionable_solution. Do not invent facts beyond
  episode_data. Distinguish opinions, interpretations, and established findings.
- The episode must contain a philosophical paradigm shift.
- The ending must contain a new frame that changes the meaning of the opening problem.
- The ending must leave the viewer with one deep reflective question.
- Practical implications must not dominate the final section.

ENDING AND PARADIGM SHIFT REQUIREMENTS:

- Do not end the episode as a numbered self-help list.
- Do not make the final section feel like generic personal-development advice.
- If practical guidance is needed, embed it inside a broader philosophical reframing.
- The ending must create a paradigm shift from the viewer's initial belief to a more nuanced understanding.
- The final meaning must reframe the central problem instead of merely summarizing the solution.
- End with a memorable reflective question that remains open after the video ends.
- The final question must not be a promotional call to action.
- Do not end with phrases such as "subscribe", "watch the next video", "like", "share", or "comment below".
- The final sentences must feel contemplative, unresolved, and intellectually meaningful.

OUTPUT FORMAT (CRITICAL):
Return a single valid, parsable JSON object ONLY. No markdown, no backticks, no commentary.

{
  "script_text": "the complete continuous spoken script as one single string"
}

Do NOT include scene_plan, visual_bible, retention_plan, quality_report, or full_script_sentences here.
Only script_text is required at this stage.
"""


def _generate_initial_script(episode_data: Dict[str, Any]) -> str:
    user_prompt = (
        "Here is the raw input JSON for the episode:\n"
        f"{json.dumps(episode_data, ensure_ascii=False, indent=2)}\n\n"
        "Write the complete continuous spoken script now. "
        "If the episode data supplies a relevant thinker, author, scholar, or book/reference, "
        "you MUST explicitly name and mention it in the script. "
        "Return ONLY the JSON object described in the system instructions."
    )
    raw = call_gemini_with_fallback(
        system_instruction=PHASE_A_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_json_safe(raw)
    if not isinstance(parsed, dict) or "script_text" not in parsed:
        raise ValueError("Phase A: الاستجابة يجب أن تحتوي على الحقل 'script_text'.")
    script_text = parsed["script_text"]
    if not isinstance(script_text, str) or not script_text.strip():
        raise ValueError("Phase A: 'script_text' فارغ أو غير صالح.")
    return script_text


# ===========================================================================
# PHASE A REPAIR — expand / compress the frozen script (count repair)
# ===========================================================================

PHASE_A_EXPAND_SYSTEM_PROMPT = """You are a script-repair specialist for a YouTube script pipeline.
You will receive a complete existing script that currently produces FEWER than 60 sentences.

Your task: return a revised COMPLETE script that:
- Preserves the original meaning, angle, and narrative arc.
- Preserves the central claim exactly in meaning.
- Preserves the episode angle.
- Preserves the named thinker, scholar, author, or historical source.
- Preserves the referenced book or source.
- Preserves the analogy.
- Preserves ALL key points.
- Preserves the actionable solution.
- Preserves the ending and the callback to the hook.
- Preserves factual meaning. Do NOT invent facts, quotations, studies, or references.
- Adds natural, meaningful sentences only — expanding on ideas, examples, mechanism, or implications
  already present or clearly implied.
- Does NOT introduce a new topic.
- Does NOT delete an essential idea.
- Does NOT add filler, padding, or repetition merely to hit a number.
- The added sentences MUST read as if they were part of the original script from the start: same
  voice, same pacing, same natural transitions (And, But, So, That's why, Here's the thing). Integrate
  each addition into the existing flow of ideas — do NOT append a separate block of extra sentences
  at the end, and do NOT create a list-like or disconnected passage anywhere in the script.
- After the revision, reading the full script start to finish must feel like ONE person telling ONE
  continuous story — a reader must NOT be able to tell which sentences were added.
- MUST reach at least 60 sentences after a deterministic local splitter and MUST NOT exceed 80 sentences.
- Remains one continuous, coherent piece of natural spoken English.
- Every sentence ends with ".", "?", or "!".

Return a single valid JSON object ONLY, no markdown, no commentary:
{
  "script_text": "the complete revised script as one string"
}
"""

PHASE_A_COMPRESS_SYSTEM_PROMPT = """You are a script-repair specialist for a YouTube script pipeline.
You will receive a complete existing script that currently produces MORE than 80 sentences.

Your task: return a revised COMPLETE script that:
- Preserves the meaning, important details, emotional progression, and ending.
- Preserves the central claim exactly in meaning.
- Preserves the episode angle.
- Preserves the named thinker, scholar, author, or historical source.
- Preserves the referenced book or source.
- Preserves the analogy.
- Preserves ALL key points.
- Preserves the actionable solution.
- Preserves the ending and the callback to the hook.
- Preserves factual meaning. Do NOT invent facts, quotations, studies, or references.
- Does NOT introduce a new topic.
- Does NOT delete an essential idea.
- Naturally merges or trims redundant sentences.
- The trimming MUST preserve the natural flow: after merging or cutting, the transitions between
  remaining sentences must still read smoothly (And, But, So, That's why, Here's the thing), with no
  abrupt jumps, gaps in logic, or leftover dangling references to a removed sentence.
- After the revision, reading the full script start to finish must feel like ONE person telling ONE
  continuous story, exactly as coherent as the original — not a shortened list of leftovers.
- MUST produce no more than 80 sentences after a deterministic local splitter (aim for 60-75) and MUST
  NOT fall below 60 sentences.
- Does not create overly long, unnatural, run-on sentences merely to reduce the count.
- Remains one continuous, coherent piece of natural spoken English.
- Every sentence ends with ".", "?", or "!".

Return a single valid JSON object ONLY, no markdown, no commentary:
{
  "script_text": "the complete revised script as one string"
}
"""


def _repair_expand_script(episode_data: Dict[str, Any], script_text: str, current_count: int) -> str:
    deficit = max(MIN_SENTENCES - current_count, 0)
    target_low = MIN_SENTENCES
    target_high = min(MAX_SENTENCES, PREFERRED_MAX_SENTENCES)
    user_prompt = (
        f"The current script produced only {current_count} sentences after deterministic splitting, "
        f"which is below the required minimum of {MIN_SENTENCES}.\n"
        f"You must add AT LEAST {deficit} additional full sentences to the script (not just one or two), "
        f"so the new total lands between {target_low} and {target_high} sentences.\n"
        "Spread the new sentences across SEVERAL different points in the script — for example inside the "
        "mechanism explanation, the examples, the escalation, and the application/solution — instead of "
        "adding them all in one place or tacking them onto the end.\n"
        "Every added sentence must connect naturally to the sentences immediately before and after it, "
        "using the same conversational voice as the rest of the script. The result must read as ONE "
        "seamless, coherent script, not an original script plus extra sentences bolted on.\n\n"
        f"Episode data (context only):\n{json.dumps(episode_data, ensure_ascii=False, indent=2)}\n\n"
        f"Frozen current script:\n{script_text}\n\n"
        "Return ONLY the revised script_text JSON object now."
    )
    raw = call_gemini_with_fallback(
        system_instruction=PHASE_A_EXPAND_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_json_safe(raw)
    if not isinstance(parsed, dict) or "script_text" not in parsed:
        raise ValueError("Phase C (expand): الاستجابة يجب أن تحتوي على 'script_text'.")
    new_text = parsed["script_text"]
    if not isinstance(new_text, str) or not new_text.strip():
        raise ValueError("Phase C (expand): 'script_text' فارغ.")
    return new_text


def _repair_compress_script(episode_data: Dict[str, Any], script_text: str, current_count: int) -> str:
    excess = max(current_count - MAX_SENTENCES, 0)
    target_low = PREFERRED_MIN_SENTENCES
    target_high = MAX_SENTENCES
    user_prompt = (
        f"The current script produced {current_count} sentences after deterministic splitting, "
        f"which exceeds the maximum of {MAX_SENTENCES}.\n"
        f"You must remove or merge AT LEAST {excess} sentences' worth of content, "
        f"so the new total lands between {target_low} and {target_high} sentences (aim for 60-75).\n"
        "Prefer merging redundant or overlapping sentences over deleting ideas outright, and make sure "
        "every remaining transition still reads smoothly — no abrupt jumps, no leftover references to "
        "something that was cut. The result must read as ONE seamless, coherent script, exactly as "
        "connected as the original, not a shortened list of leftover fragments.\n\n"
        f"Episode data (context only):\n{json.dumps(episode_data, ensure_ascii=False, indent=2)}\n\n"
        f"Frozen current script:\n{script_text}\n\n"
        "Return ONLY the revised script_text JSON object now."
    )
    raw = call_gemini_with_fallback(
        system_instruction=PHASE_A_COMPRESS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_json_safe(raw)
    if not isinstance(parsed, dict) or "script_text" not in parsed:
        raise ValueError("Phase C (compress): الاستجابة يجب أن تحتوي على 'script_text'.")
    new_text = parsed["script_text"]
    if not isinstance(new_text, str) or not new_text.strip():
        raise ValueError("Phase C (compress): 'script_text' فارغ.")
    return new_text


# ===========================================================================
# PHASE B — DETERMINISTIC SENTENCE EXTRACTION
# ===========================================================================

_ABBREVIATIONS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "Gen.", "Rep.", "Sen.",
    "Ph.D.", "M.D.", "e.g.", "i.e.", "etc.", "vs.", "U.S.", "U.K.", "U.N.", "U.S.A.",
]

_PLACEHOLDER = "\uE000"

_SENTENCE_END_PATTERN = re.compile(r'[.!?]+["\'”’\)\]\}]*')


def _split_into_sentences(text: str) -> List[str]:
    """Deterministic sentence splitter."""
    protected = text.strip()

    for abbr in _ABBREVIATIONS:
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        protected_abbr = abbr.replace(".", _PLACEHOLDER)

        def _replacer(match, _protected_abbr=protected_abbr):
            return _protected_abbr

        protected = pattern.sub(_replacer, protected)

    protected = re.sub(r"(\d)\.(\d)", r"\1" + _PLACEHOLDER + r"\2", protected)

    sentences: List[str] = []
    start = 0
    length = len(protected)

    for m in _SENTENCE_END_PATTERN.finditer(protected):
        end_pos = m.end()
        if end_pos < length and not protected[end_pos].isspace():
            continue

        chunk = protected[start:end_pos]
        restored = chunk.replace(_PLACEHOLDER, ".").strip()
        if restored:
            sentences.append(restored)

        start = end_pos
        while start < length and protected[start].isspace():
            start += 1

    if start < length:
        remainder = protected[start:].replace(_PLACEHOLDER, ".").strip()
        if remainder:
            sentences.append(remainder)

    return sentences


def _has_terminal_punctuation(text: str) -> bool:
    """Return True if `text` ends with terminal punctuation, ignoring any
    trailing closing quotation marks or brackets."""
    stripped = text.rstrip()
    while stripped and stripped[-1] in "\"'”’)]}":
        stripped = stripped[:-1].rstrip()
    return bool(stripped) and stripped[-1] in ".?!"


def _validate_sentence_list(sentences: List[str]) -> List[str]:
    """Clean sentences: strip whitespace, ensure ending punctuation, drop empties."""
    cleaned: List[str] = []
    for s in sentences:
        if not isinstance(s, str):
            continue
        s_clean = s.strip()
        if not s_clean:
            continue
        if not _has_terminal_punctuation(s_clean):
            s_clean += "."
        cleaned.append(s_clean)
    return cleaned


def _extract_sentences_from_script(script_text: str) -> List[str]:
    if not isinstance(script_text, str) or not script_text.strip():
        raise ValueError("script_text فارغ أو غير صالح.")
    raw_sentences = _split_into_sentences(script_text)
    cleaned = _validate_sentence_list(raw_sentences)
    if not cleaned:
        raise ValueError("لم يتم استخراج أي جملة صالحة من script_text.")
    return cleaned


# ===========================================================================
# PHASE C ORCHESTRATION — bounded repair loop until count is valid
# ===========================================================================

def _generate_valid_sentence_list(
    episode_data: Dict[str, Any], initial_script_text: str
) -> Tuple[List[str], str]:
    """Runs Phase B, then Phase C repair (bounded), until MIN<=count<=MAX or gives up."""
    script_text = initial_script_text
    sentences = _extract_sentences_from_script(script_text)
    count = len(sentences)

    attempt = 0
    while not (MIN_SENTENCES <= count <= MAX_SENTENCES) and attempt < MAX_SPLIT_REPAIR_ATTEMPTS:
        attempt += 1
        logger.info(
            f"Phase C repair attempt {attempt}/{MAX_SPLIT_REPAIR_ATTEMPTS}: "
            f"current sentence count = {count}."
        )
        if count < MIN_SENTENCES:
            script_text = _repair_expand_script(episode_data, script_text, count)
        else:
            script_text = _repair_compress_script(episode_data, script_text, count)

        sentences = _extract_sentences_from_script(script_text)
        count = len(sentences)

    if not (MIN_SENTENCES <= count <= MAX_SENTENCES):
        raise ValueError(
            f"تعذر الوصول إلى عدد جمل صالح بعد {MAX_SPLIT_REPAIR_ATTEMPTS} محاولات إصلاح. "
            f"آخر عدد جمل: {count} (المطلوب بين {MIN_SENTENCES} و {MAX_SENTENCES})."
        )

    return sentences, script_text


# ===========================================================================
# PHASE D — AUXILIARY STRUCTURE GENERATION SYSTEM PROMPT
# ===========================================================================

PHASE_D_SYSTEM_PROMPT = """You are an elite AI Episode Director completing the post-production planning
for a YouTube script that has ALREADY been written and FROZEN.

You will receive:
- The original episode_data.
- The frozen, final list of script sentences, each tagged with its 0-based index in the form [N] sentence.

CRITICAL — AUTHORITATIVE SOURCE:
"full_script_sentences" is the ONLY authoritative source for image generation, audio generation,
subtitles, scene_plan, and every downstream stage. Its contents, order, and indices are FINAL and
OUT OF YOUR CONTROL. You must NOT alter, paraphrase, reorder, merge, split, translate, or shorten any
sentence. Nothing you output may override, replace, or mutate the frozen sentence list.

Your task is to produce ONLY the auxiliary structure around this frozen script:
hook, sections, creative_brief, paradigm_shift, retention_plan, audio_arc, ending_tail,
scene_plan, visual_bible, quality_report.

CHARACTER DNA (MUST NOT CHANGE):
"Orange character, muscular body, smooth head, two big white eyes, no mouth, black shorts."
This character is a VISUAL NARRATOR, not a fitness symbol. It represents curiosity, thought, and
the inner life of ideas. Do NOT stage it in gyms, workouts, or fitness contexts unless the episode
data explicitly requires it. For non-fitness episodes, place it in symbolic environments such as
libraries, archives, theaters, ancient cities, classrooms, streets, quiet rooms, mirrors, maps,
manuscripts, museums, rooftops, or abstract mental spaces.

SCENE PLAN INDEXING RULES (CRITICAL):
- N = the exact number of frozen sentences given to you.
- Valid 0-based indices are 0 through N-1.
- The first scene MUST start at sentence_start = 0.
- The final scene MUST end at sentence_end = N-1. NEVER use N itself — it is out of range.
- Scenes MUST be contiguous: each next scene's sentence_start = previous scene's sentence_end + 1.
- No gaps, no overlaps, no negative indices.
- scene_id starts at 1 and increases sequentially matching list order.
- The number of scenes must adapt naturally to N (do not force a fixed count).
- Every scene must add a real VISUAL IDEA, not literally illustrate the sentence.
- Every scene must include character_action AND visual_concept, staying consistent with CHARACTER DNA.
- Do not make every scene just a different pose against an empty background.
- Do not add text inside images unless absolutely essential.

RETENTION_PLAN STRUCTURE RULES (CRITICAL):
"retention_plan" must have this exact shape:
{
  "hook_strategy": "string",
  "open_loops": ["string", "string"],
  "pattern_interrupts": ["string", "string"],
  "escalation_points": ["string", "string"],
  "main_reveal": "string",
  "payoff": "string",
  "ending_callback": "string"
}
open_loops, pattern_interrupts, and escalation_points MUST always be real JSON arrays of non-empty
strings, even if there is only one item (wrap it in an array). Never return them as a single string.

PARADIGM SHIFT RULES:

- The episode must contain one clear paradigm shift.
- old_belief must describe the viewer's initial common belief.
- hidden_assumption must expose the assumption underneath that belief.
- new_frame must reinterpret the topic in a deeper and more nuanced way.
- practical_implication must describe a consequence of the new frame without becoming a numbered checklist.
- final_question must be philosophical and reflective.
- final_question MUST be an actual interrogative sentence ending with a question mark ("?").
- final_question must not be promotional.
- final_question must not ask the viewer to subscribe, like, share, or comment.

VISUAL METAPHOR ARC RULES:

- The episode must have one central visual metaphor.
- The central visual metaphor must evolve across the episode.
- opening_state, development_state, escalation_state, revelation_state, and ending_state must describe different visual states.
- The ending_state must not simply repeat the opening_state.
- The visual metaphor must be symbolic rather than a literal illustration of the narration.
- The visual metaphor must remain compatible with the existing Orange character DNA.

VISUAL BEAT RULES (CRITICAL):

- Every scene must contain exactly one visual_beat.
- visual_beat must be one of the approved values:
  establishing, symbolic_setup, development, escalation, pattern_interrupt,
  visual_reveal, metaphor_transformation, quiet_reflection, ending_tail.
- Do not assign the same visual_beat to every scene.
- At least one scene must use visual_reveal or metaphor_transformation.
- At least one scene must use quiet_reflection.
- quiet_reflection MUST NOT be the final scene. It must appear before the ending_tail scene.
- Exactly one scene MUST use ending_tail, and it MUST be either the final scene
  or the scene immediately before the final scene.
- visual_beat must describe the dramatic function of the image, not merely its subject.

AUDIO ARC RULES:

- audio_arc is a planning structure only.
- dramatic_silence_sentence_indices must contain valid zero-based sentence indices.
- music_drop_sentence_indices must contain valid zero-based sentence indices.
- At least one valid dramatic silence index must be provided.
- Do not place dramatic silence before every important sentence.
- foley_motifs must relate to the central visual metaphor.
- The audio arc must contain variation between opening, tension, revelation, reflection, and ending.
- Do not describe one continuous music track with identical intensity from beginning to end.

ENDING TAIL RULES:

- ending_tail.duration_seconds must be between 8 and 12 inclusive.
- The spoken narration must end before the ending tail begins.
- The ending tail must not contain spoken promotional CTA.
- visual_action must describe a quiet visual action.
- camera_behavior must describe an almost imperceptible or very slow camera movement.
- music_behavior must describe a gradual fade.
- ambience_behavior must describe subtle ambience that may continue briefly after the music fades.
- end_screen_safe_area must identify a visually calm area for a future end screen.

OUTPUT FORMAT (CRITICAL):
Return a single valid, parsable JSON object ONLY. No markdown, no backticks, no commentary.
Do NOT include "full_script_sentences" in your output — it is supplied separately and frozen.

Required JSON schema:

{
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
  "paradigm_shift": {
    "old_belief": "string",
    "hidden_assumption": "string",
    "new_frame": "string",
    "practical_implication": "string",
    "final_question": "string (must end with '?')"
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
  "audio_arc": {
    "opening_mode": "string",
    "tension_mode": "string",
    "revelation_mode": "string",
    "reflection_mode": "string",
    "ending_mode": "string",
    "dramatic_silence_sentence_indices": [0],
    "music_drop_sentence_indices": [0],
    "foley_motifs": ["string"]
  },
  "ending_tail": {
    "duration_seconds": 10,
    "spoken_audio_ends_before_tail": true,
    "visual_action": "string",
    "camera_behavior": "string",
    "music_behavior": "string",
    "ambience_behavior": "string",
    "end_screen_safe_area": "string"
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
      "continuity_notes": "string",
      "visual_beat": "establishing"
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
    "visual_progression": "string",
    "visual_metaphor_arc": {
      "central_metaphor": "string",
      "opening_state": "string",
      "development_state": "string",
      "escalation_state": "string",
      "revelation_state": "string",
      "ending_state": "string"
    }
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
"""


def _generate_auxiliary_structure(
    episode_data: Dict[str, Any], frozen_sentences: List[str]
) -> Dict[str, Any]:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(frozen_sentences))
    N = len(frozen_sentences)

    user_prompt = (
        f"N = {N} (total frozen sentences). Valid indices are 0 through {N - 1}.\n\n"
        f"Episode data:\n{json.dumps(episode_data, ensure_ascii=False, indent=2)}\n\n"
        f"Frozen script sentences (DO NOT MODIFY, indices are final):\n{numbered}\n\n"
        "Return ONLY the auxiliary JSON object described in the system instructions now."
    )

    raw = call_gemini_with_fallback(
        system_instruction=PHASE_D_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_json_safe(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Phase D: الاستجابة ليست كائن JSON.")
    return parsed


# ===========================================================================
# PARTIAL-REPAIR SYSTEM PROMPTS
# ===========================================================================

SCENE_PLAN_REPAIR_SYSTEM_PROMPT = """You are a scene-plan repair specialist for a YouTube script pipeline.
You will receive a FROZEN list of sentences. You must NOT modify, reorder, paraphrase, merge, or split them.
"full_script_sentences" is the ONLY authoritative source for images, audio, subtitles, scene_plan, and
all downstream stages. Your ONLY job is to output a valid "scene_plan" JSON array for those sentences.

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
  composition, motion_potential, transition, continuity_notes, visual_beat.
- Every scene MUST include a visual_beat field.
- visual_beat MUST be exactly one of:
  establishing, symbolic_setup, development, escalation,
  pattern_interrupt, visual_reveal, metaphor_transformation,
  quiet_reflection, ending_tail.
- At least one scene must use visual_reveal or metaphor_transformation.
- At least one scene must use quiet_reflection.
- quiet_reflection MUST NOT be the final scene; it must precede the ending_tail scene.
- Exactly one scene MUST use ending_tail, and it MUST be either the final scene
  or the scene immediately before the final scene.
- Do not modify, reorder, paraphrase, merge, or split the frozen sentences.
- Every scene must add a real VISUAL IDEA (not just literal illustration).
- Character DNA must be consistent: Orange character, muscular body, smooth head, two big white eyes, no mouth, black shorts.
  This character is a VISUAL NARRATOR, not a fitness symbol. Avoid gym/fitness staging unless the episode
  explicitly requires it. Prefer symbolic environments: libraries, archives, theaters, ancient cities,
  classrooms, streets, rooms, mirrors, maps, manuscripts, museums, or abstract mental spaces.
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
- visual_metaphor_arc (JSON object)

- visual_bible MUST include visual_metaphor_arc.
- visual_metaphor_arc MUST contain:
  central_metaphor,
  opening_state,
  development_state,
  escalation_state,
  revelation_state,
  ending_state.
- All values must be non-empty strings.
- The visual states must evolve and must not all describe the same image.

NOTE: The character is a VISUAL NARRATOR, not a fitness symbol. Environment style should favor
symbolic, idea-driven spaces (libraries, archives, theaters, ancient cities, classrooms, streets,
rooms, mirrors, maps, manuscripts, museums, abstract mental spaces), unless the episode explicitly
requires otherwise.

No markdown. No backticks. No explanation.
Return only the JSON object.
"""


QUALITY_REPORT_REPAIR_SYSTEM_PROMPT = """You are a quality-report repair specialist.
You will receive the full auxiliary structure of an episode: creative_brief, paradigm_shift,
retention_plan, audio_arc, ending_tail, scene_plan, and visual_bible, plus the frozen script.

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

Your scores MUST take into account the paradigm_shift, the audio_arc, and the ending_tail,
not only the script and the scene plan.

No markdown. No backticks. No explanation.
Return only the JSON object.
"""


_REPAIR_PARADIGM_SHIFT_SYSTEM_PROMPT = """You are a paradigm-shift repair specialist for a YouTube script pipeline.
The frozen sentence list is authoritative.
Do not modify the script.
Return only the requested JSON field.
Preserve all existing valid fields.

Return ONLY a valid "paradigm_shift" JSON object with EXACTLY these fields:
- old_belief (non-empty string) — the viewer's initial common belief.
- hidden_assumption (non-empty string) — the assumption underneath that belief.
- new_frame (non-empty string) — a deeper, more nuanced reinterpretation of the topic.
- practical_implication (non-empty string) — a consequence of the new frame, not a numbered checklist.
- final_question (non-empty string) — philosophical, reflective, non-promotional.
  MUST be an actual interrogative sentence that ENDS WITH A QUESTION MARK ("?").
  Must not ask the viewer to subscribe, like, share, or comment.

No markdown. No backticks. No explanation.
Return only the JSON object.
"""


_REPAIR_AUDIO_ARC_SYSTEM_PROMPT = """You are an audio-arc repair specialist for a YouTube script pipeline.
The frozen sentence list is authoritative.
Do not modify the script.
Return only the requested JSON field.
Preserve all existing valid fields.

Return ONLY a valid "audio_arc" JSON object with EXACTLY these fields:
- opening_mode (non-empty string)
- tension_mode (non-empty string)
- revelation_mode (non-empty string)
- reflection_mode (non-empty string)
- ending_mode (non-empty string)
- dramatic_silence_sentence_indices (JSON array of valid zero-based sentence indices, at least one)
- music_drop_sentence_indices (JSON array of valid zero-based sentence indices)
- foley_motifs (JSON array of non-empty strings, at least one)

All indices MUST be valid (0 <= index < N).
Do not place dramatic silence before every important sentence.

No markdown. No backticks. No explanation.
Return only the JSON object.
"""


_REPAIR_ENDING_TAIL_SYSTEM_PROMPT = """You are an ending-tail repair specialist for a YouTube script pipeline.
The frozen sentence list is authoritative.
Do not modify the script.
Return only the requested JSON field.
Preserve all existing valid fields.

Return ONLY a valid "ending_tail" JSON object with EXACTLY these fields:
- duration_seconds (number between 8 and 12 inclusive)
- spoken_audio_ends_before_tail (must be true)
- visual_action (non-empty string, a quiet visual action)
- camera_behavior (non-empty string, almost imperceptible or very slow camera movement)
- music_behavior (non-empty string, gradual fade)
- ambience_behavior (non-empty string, subtle ambience that may continue briefly after the music fades)
- end_screen_safe_area (non-empty string, a visually calm area for a future end screen)

Must not contain a spoken promotional CTA (no subscribe/like/share/comment).

No markdown. No backticks. No explanation.
Return only the JSON object.
"""


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
    "core_idea", "unique_angle", "central_conflict", "unexpected_insight",
    "episode_concept", "episode_format", "target_audience", "viewer_problem",
    "viewer_outcome", "tone", "emotional_arc",
]

_RETENTION_PLAN_STRING_FIELDS = ["hook_strategy", "main_reveal", "payoff", "ending_callback"]
_RETENTION_PLAN_LIST_FIELDS = ["open_loops", "pattern_interrupts", "escalation_points"]

_VISUAL_BIBLE_STRING_FIELDS = [
    "character_dna", "environment_style", "color_logic", "lighting_style",
    "camera_language", "composition_rules", "visual_progression",
]

_QUALITY_REPORT_SCORE_FIELDS = [
    "hook_score", "originality_score", "retention_score",
    "narrative_coherence_score", "visual_storytelling_score",
]
_QUALITY_REPORT_STRING_FIELDS = ["repetition_check", "factual_caution_check"]

_SCENE_REQUIRED_STRING_FIELDS = [
    "script_segment", "narrative_purpose", "emotion", "visual_concept",
    "character_action", "environment", "camera", "composition",
    "motion_potential", "transition", "continuity_notes",
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
                raise ValueError(f"retention_plan.{field}[{i}] يجب أن يكون نصًا غير فارغ.")


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
# NEW field-level validators (paradigm_shift / visual_metaphor_arc /
# visual_beat / audio_arc / ending_tail)
# ===========================================================================

def _validate_paradigm_shift(data: Dict[str, Any]) -> None:
    paradigm_shift = data.get("paradigm_shift")

    if not isinstance(paradigm_shift, dict):
        raise ValueError("paradigm_shift must be an object.")

    required_keys = [
        "old_belief",
        "hidden_assumption",
        "new_frame",
        "practical_implication",
        "final_question",
    ]

    for key in required_keys:
        value = paradigm_shift.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"paradigm_shift.{key} must be a non-empty string."
            )

    final_question_raw = paradigm_shift["final_question"].strip()

    if not final_question_raw.endswith("?"):
        raise ValueError(
            "paradigm_shift.final_question must end with a question mark ('?')."
        )

    final_question_lower = final_question_raw.lower()

    prohibited_cta_terms = [
        "subscribe",
        "like",
        "share",
        "comment below",
        "watch the next video",
        "اشترك",
        "إعجاب",
        "شارك",
        "اكتب تعليق",
        "شاهد الفيديو التالي",
    ]

    if any(term in final_question_lower for term in prohibited_cta_terms):
        raise ValueError(
            "paradigm_shift.final_question must not be promotional."
        )


def _validate_visual_metaphor_arc(data: Dict[str, Any]) -> None:
    visual_bible = data.get("visual_bible")

    if not isinstance(visual_bible, dict):
        raise ValueError("visual_bible must be an object.")

    arc = visual_bible.get("visual_metaphor_arc")

    if not isinstance(arc, dict):
        raise ValueError(
            "visual_bible.visual_metaphor_arc must be an object."
        )

    required_keys = [
        "central_metaphor",
        "opening_state",
        "development_state",
        "escalation_state",
        "revelation_state",
        "ending_state",
    ]

    for key in required_keys:
        value = arc.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"visual_metaphor_arc.{key} must be a non-empty string."
            )

    states = [
        arc["opening_state"].strip(),
        arc["development_state"].strip(),
        arc["escalation_state"].strip(),
        arc["revelation_state"].strip(),
        arc["ending_state"].strip(),
    ]

    if len(set(states)) < 3:
        raise ValueError(
            "visual_metaphor_arc must contain at least three distinct visual states."
        )


def _validate_visual_beats(data: Dict[str, Any]) -> None:
    scene_plan = data.get("scene_plan")

    if not isinstance(scene_plan, list) or not scene_plan:
        raise ValueError("scene_plan must be a non-empty list.")

    beats: List[str] = []

    for index, scene in enumerate(scene_plan):
        if not isinstance(scene, dict):
            raise ValueError(
                f"scene_plan[{index}] must be an object."
            )

        beat = scene.get("visual_beat")

        if beat not in APPROVED_VISUAL_BEATS:
            raise ValueError(
                f"scene_plan[{index}].visual_beat is invalid: {beat!r}"
            )

        beats.append(beat)

    if len(set(beats)) < 2:
        raise ValueError(
            "scene_plan must contain at least two different visual_beat values."
        )

    if not any(
        beat in {"visual_reveal", "metaphor_transformation"}
        for beat in beats
    ):
        raise ValueError(
            "scene_plan must contain visual_reveal or metaphor_transformation."
        )

    last_index = len(beats) - 1

    quiet_reflection_indices = [
        index for index, beat in enumerate(beats)
        if beat == "quiet_reflection"
    ]

    if not quiet_reflection_indices:
        raise ValueError(
            "scene_plan must contain at least one quiet_reflection beat."
        )

    if last_index in quiet_reflection_indices:
        raise ValueError(
            "scene_plan must not place quiet_reflection at the final scene; "
            "quiet_reflection must precede the ending_tail scene."
        )

    ending_tail_indices = [
        index for index, beat in enumerate(beats)
        if beat == "ending_tail"
    ]

    if not ending_tail_indices:
        raise ValueError(
            "scene_plan must contain at least one ending_tail visual beat."
        )

    if any(
        index not in {last_index, last_index - 1}
        for index in ending_tail_indices
    ):
        raise ValueError(
            "ending_tail visual beats are only allowed in the final scene "
            "or the scene immediately before it."
        )

    first_ending_tail_index = min(ending_tail_indices)
    if not any(
        index < first_ending_tail_index
        for index in quiet_reflection_indices
    ):
        raise ValueError(
            "scene_plan must place at least one quiet_reflection beat before "
            "the ending_tail scene."
        )


def _validate_audio_arc(data: Dict[str, Any], sentence_count: int) -> None:
    audio_arc = data.get("audio_arc")

    if not isinstance(audio_arc, dict):
        raise ValueError("audio_arc must be an object.")

    required_string_keys = [
        "opening_mode",
        "tension_mode",
        "revelation_mode",
        "reflection_mode",
        "ending_mode",
    ]

    for key in required_string_keys:
        value = audio_arc.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"audio_arc.{key} must be a non-empty string."
            )

    for key in [
        "dramatic_silence_sentence_indices",
        "music_drop_sentence_indices",
    ]:
        indices = audio_arc.get(key)

        if not isinstance(indices, list):
            raise ValueError(
                f"audio_arc.{key} must be an array."
            )

        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError(
                    f"audio_arc.{key} contains a non-integer index."
                )

            if index < 0 or index >= sentence_count:
                raise ValueError(
                    f"audio_arc.{key} contains an out-of-range index: {index}."
                )

    silence_indices = audio_arc["dramatic_silence_sentence_indices"]

    if not silence_indices:
        raise ValueError(
            "audio_arc must contain at least one dramatic silence index."
        )

    foley_motifs = audio_arc.get("foley_motifs")

    if not isinstance(foley_motifs, list) or not foley_motifs:
        raise ValueError(
            "audio_arc.foley_motifs must be a non-empty array."
        )

    for motif in foley_motifs:
        if not isinstance(motif, str) or not motif.strip():
            raise ValueError(
                "audio_arc.foley_motifs must contain non-empty strings."
            )


def _validate_ending_tail(data: Dict[str, Any]) -> None:
    ending_tail = data.get("ending_tail")

    if not isinstance(ending_tail, dict):
        raise ValueError("ending_tail must be an object.")

    required_keys = [
        "duration_seconds",
        "spoken_audio_ends_before_tail",
        "visual_action",
        "camera_behavior",
        "music_behavior",
        "ambience_behavior",
        "end_screen_safe_area",
    ]

    for key in required_keys:
        if key not in ending_tail:
            raise ValueError(
                f"ending_tail.{key} is missing."
            )

    duration = ending_tail["duration_seconds"]

    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError(
            "ending_tail.duration_seconds must be numeric."
        )

    if not (
        MIN_ENDING_TAIL_SECONDS
        <= float(duration)
        <= MAX_ENDING_TAIL_SECONDS
    ):
        raise ValueError(
            "ending_tail.duration_seconds must be between 8 and 12."
        )

    if ending_tail["spoken_audio_ends_before_tail"] is not True:
        raise ValueError(
            "spoken_audio_ends_before_tail must be true."
        )

    for key in [
        "visual_action",
        "camera_behavior",
        "music_behavior",
        "ambience_behavior",
        "end_screen_safe_area",
    ]:
        value = ending_tail[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"ending_tail.{key} must be a non-empty string."
            )

    combined_text = " ".join(
        str(ending_tail[key]).lower()
        for key in [
            "visual_action",
            "camera_behavior",
            "music_behavior",
            "ambience_behavior",
        ]
    )

    prohibited_cta_terms = [
        "spoken cta",
        "call to action",
        "subscribe",
        "like",
        "share",
        "comment below",
        "اشترك",
        "إعجاب",
        "شارك",
        "اكتب تعليق",
    ]

    if any(term in combined_text for term in prohibited_cta_terms):
        raise ValueError(
            "ending_tail must not contain a spoken promotional CTA."
        )


# ===========================================================================
# Sentence validation (final gate — used on the frozen list)
# ===========================================================================

def _extract_and_validate_full_script_sentences(data: Dict[str, Any]) -> Tuple[List[str], int]:
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
        if not _has_terminal_punctuation(s_clean):
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
            raise ValueError(f"scene_plan[{idx}]: scene_id يجب أن يكون رقمًا صحيحًا موجبًا (وجدنا {scene_id}).")

        for field in _SCENE_REQUIRED_STRING_FIELDS:
            _require_non_empty_string(scene, field, f"scene_plan[{idx}]")

        if s_start < 0 or s_end < 0:
            raise ValueError(f"scene_plan[{idx}]: لا يمكن أن تكون الفهارس سالبة.")
        if s_start > s_end:
            raise ValueError(f"scene_plan[{idx}]: sentence_start أكبر من sentence_end.")
        if s_end >= total_sentences:
            raise ValueError(
                f"scene_plan[{idx}]: sentence_end={s_end} خارج نطاق full_script_sentences "
                f"(الحد الأقصى {total_sentences - 1}). عدد الجمل الفعلي N = {total_sentences}."
            )

        if idx == 0:
            if s_start != 0:
                raise ValueError(f"scene_plan[0]: يجب أن يبدأ أول مشهد عند sentence_start=0 (وجدنا {s_start}).")
        else:
            expected_start = previous_end + 1
            if s_start < expected_start:
                raise ValueError(
                    f"scene_plan[{idx}]: تداخل مع المشهد السابق. يجب أن يبدأ عند {expected_start} لكنه بدأ عند {s_start}."
                )
            if s_start > expected_start:
                raise ValueError(
                    f"scene_plan[{idx}]: توجد فجوة بين المشهد السابق وهذا المشهد. "
                    f"يجب أن يبدأ عند {expected_start} لكنه بدأ عند {s_start}."
                )

        for i in range(s_start, s_end + 1):
            if covered[i]:
                raise ValueError(f"scene_plan[{idx}]: الجملة رقم {i} مُغطاة في أكثر من مشهد (تكرار).")
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
    "id", "topic", "hook", "sections", "full_script_sentences",
    "creative_brief", "paradigm_shift", "retention_plan", "audio_arc",
    "ending_tail", "scene_plan", "visual_bible", "quality_report",
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
        logger.info(
            f"ℹ️ عدد الجمل: {sentence_count} (مقبول ضمن الحدود {MIN_SENTENCES}-{MAX_SENTENCES})."
        )

    logger.info(f"ℹ️ إجمالي عدد الكلمات: {total_words} (لا يوجد حد أقصى/أدنى إلزامي).")

    _validate_creative_brief(data)
    _validate_retention_plan(data)
    _validate_visual_bible(data)
    _validate_quality_report(data)
    _validate_scene_plan(data, sentence_count)

    _validate_paradigm_shift(data)
    _validate_visual_metaphor_arc(data)
    _validate_visual_beats(data)
    _validate_audio_arc(data, sentence_count)
    _validate_ending_tail(data)

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
    full_script_sentences: List[str], episode_data: Dict[str, Any], error_msg: Any
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
        "No gaps. No overlaps. scene_id must start at 1 and increment by 1.\n"
        "Every scene MUST include visual_beat, one of:\n"
        "establishing, symbolic_setup, development, escalation, pattern_interrupt, "
        "visual_reveal, metaphor_transformation, quiet_reflection, ending_tail.\n"
        "MANDATORY BEAT CONSTRAINTS:\n"
        "- Exactly one scene (near the very end) MUST use ending_tail.\n"
        "  It must be the final scene or the scene immediately before the final scene.\n"
        "- At least one scene MUST use quiet_reflection, and quiet_reflection MUST NOT be\n"
        "  the final scene — it must appear before the ending_tail scene.\n\n"
        f"Previous validation failure:\n{error_msg}\n\n"
        f"Episode data (context only):\n{ep_json}\n\n"
        f"Frozen full_script_sentences (DO NOT MODIFY):\n{sentences_json}\n\n"
        "Return ONLY the scene_plan JSON array now."
    )

    raw = call_gemini_with_fallback(system_instruction=SCENE_PLAN_REPAIR_SYSTEM_PROMPT, user_prompt=user_prompt)
    parsed = _parse_json_safe(raw)

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        if isinstance(parsed.get("scene_plan"), list):
            return parsed["scene_plan"]
        raise ValueError("استجابة إصلاح scene_plan يجب أن تكون JSON array أو كائنًا يحتوي على scene_plan كقائمة.")
    raise ValueError("استجابة إصلاح scene_plan ليست قائمة JSON صالحة.")


def _repair_retention_plan(
    full_script_sentences: List[str], current_retention_plan: Any, episode_data: Dict[str, Any], error_msg: Any
) -> Dict[str, Any]:
    sentences_json = json.dumps(full_script_sentences, ensure_ascii=False, indent=2)
    current_json = json.dumps(current_retention_plan, ensure_ascii=False, indent=2) if current_retention_plan is not None else "null"
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

    raw = call_gemini_with_fallback(system_instruction=RETENTION_PLAN_REPAIR_SYSTEM_PROMPT, user_prompt=user_prompt)
    parsed = _parse_json_safe(raw)

    if not isinstance(parsed, dict):
        raise ValueError("استجابة إصلاح retention_plan يجب أن تكون كائن JSON.")
    if isinstance(parsed.get("retention_plan"), dict):
        return parsed["retention_plan"]
    return parsed


def _repair_visual_bible(
    full_script_sentences: List[str], current_visual_bible: Any, episode_data: Dict[str, Any], error_msg: Any
) -> Dict[str, Any]:
    sentences_json = json.dumps(full_script_sentences, ensure_ascii=False, indent=2)
    current_json = json.dumps(current_visual_bible, ensure_ascii=False, indent=2) if current_visual_bible is not None else "null"
    ep_json = json.dumps(episode_data, ensure_ascii=False, indent=2)

    user_prompt = (
        "This is a fresh independent repair request.\n"
        "The provided full_script_sentences is frozen and must not be changed.\n"
        "Return only the requested JSON field (visual_bible).\n"
        "Do not return the full episode object.\n"
        "Do not use markdown.\n\n"
        f"Reminder: character_dna MUST be exactly: \"{CHARACTER_DNA}\"\n"
        "visual_bible MUST also include visual_metaphor_arc with keys: "
        "central_metaphor, opening_state, development_state, escalation_state, "
        "revelation_state, ending_state.\n\n"
        f"Previous validation failure:\n{error_msg}\n\n"
        f"Episode data (context only):\n{ep_json}\n\n"
        f"Frozen full_script_sentences (DO NOT MODIFY):\n{sentences_json}\n\n"
        f"Current (possibly broken) visual_bible:\n{current_json}\n\n"
        "Return ONLY the visual_bible JSON object now."
    )

    raw = call_gemini_with_fallback(system_instruction=VISUAL_BIBLE_REPAIR_SYSTEM_PROMPT, user_prompt=user_prompt)
    parsed = _parse_json_safe(raw)

    if not isinstance(parsed, dict):
        raise ValueError("استجابة إصلاح visual_bible يجب أن تكون كائن JSON.")
    if isinstance(parsed.get("visual_bible"), dict):
        return parsed["visual_bible"]
    return parsed


# ---------------------------------------------------------------------------
# QUALITY REPORT REPAIR
# ---------------------------------------------------------------------------
# IMPORTANT: the original `_repair_quality_report` signature is preserved
# exactly as it was. The extended logic (which also takes paradigm_shift,
# audio_arc, and ending_tail into account) lives in a NEW internal function
# `_repair_quality_report_extended`. The public wrapper delegates to it,
# passing None for the three new fields so existing callers stay compatible.
# ---------------------------------------------------------------------------

def _repair_quality_report(
    full_script_sentences: List[str],
    creative_brief: Any,
    retention_plan: Any,
    scene_plan: Any,
    visual_bible: Any,
    current_quality_report: Any,
    error_msg: Any,
) -> Dict[str, Any]:
    """
    ORIGINAL signature — preserved unchanged.
    Delegates to `_repair_quality_report_extended` with the new optional
    fields (paradigm_shift / audio_arc / ending_tail) set to None.
    """
    return _repair_quality_report_extended(
        full_script_sentences=full_script_sentences,
        creative_brief=creative_brief,
        paradigm_shift=None,
        retention_plan=retention_plan,
        audio_arc=None,
        ending_tail=None,
        scene_plan=scene_plan,
        visual_bible=visual_bible,
        current_quality_report=current_quality_report,
        error_msg=error_msg,
    )


def _repair_quality_report_extended(
    full_script_sentences: List[str],
    creative_brief: Any,
    paradigm_shift: Any,
    retention_plan: Any,
    audio_arc: Any,
    ending_tail: Any,
    scene_plan: Any,
    visual_bible: Any,
    current_quality_report: Any,
    error_msg: Any,
) -> Dict[str, Any]:
    """
    Internal extended version of the quality_report repair. Accepts
    paradigm_shift, audio_arc, and ending_tail so the quality_report scores
    reflect the full auxiliary structure — not only the script and scene plan.
    """
    sentences_json = json.dumps(full_script_sentences, ensure_ascii=False, indent=2)
    cb_json = json.dumps(creative_brief, ensure_ascii=False, indent=2)
    ps_json = json.dumps(paradigm_shift, ensure_ascii=False, indent=2)
    rp_json = json.dumps(retention_plan, ensure_ascii=False, indent=2)
    aa_json = json.dumps(audio_arc, ensure_ascii=False, indent=2)
    et_json = json.dumps(ending_tail, ensure_ascii=False, indent=2)
    sp_json = json.dumps(scene_plan, ensure_ascii=False, indent=2)
    vb_json = json.dumps(visual_bible, ensure_ascii=False, indent=2)
    current_json = json.dumps(current_quality_report, ensure_ascii=False, indent=2) if current_quality_report is not None else "null"

    user_prompt = (
        "This is a fresh independent repair request.\n"
        "Return only the requested JSON field (quality_report).\n"
        "Do not return the full episode object.\n"
        "Do not use markdown.\n\n"
        "Scores must be numbers between 0 and 10. approved must be a real boolean.\n"
        "You MUST evaluate the paradigm_shift, audio_arc, and ending_tail as part of your scoring,\n"
        "not only the script and the scene plan.\n\n"
        f"Previous validation failure:\n{error_msg}\n\n"
        f"Frozen full_script_sentences:\n{sentences_json}\n\n"
        f"creative_brief:\n{cb_json}\n\n"
        f"paradigm_shift:\n{ps_json}\n\n"
        f"retention_plan:\n{rp_json}\n\n"
        f"audio_arc:\n{aa_json}\n\n"
        f"ending_tail:\n{et_json}\n\n"
        f"scene_plan:\n{sp_json}\n\n"
        f"visual_bible:\n{vb_json}\n\n"
        f"Current (possibly broken) quality_report:\n{current_json}\n\n"
        "Return ONLY the quality_report JSON object now."
    )

    raw = call_gemini_with_fallback(system_instruction=QUALITY_REPORT_REPAIR_SYSTEM_PROMPT, user_prompt=user_prompt)
    parsed = _parse_json_safe(raw)

    if not isinstance(parsed, dict):
        raise ValueError("استجابة إصلاح quality_report يجب أن تكون كائن JSON.")
    if isinstance(parsed.get("quality_report"), dict):
        return parsed["quality_report"]
    return parsed


# ---------------------------------------------------------------------------
# NEW individual repair calls: paradigm_shift / audio_arc / ending_tail
# ---------------------------------------------------------------------------

def _repair_paradigm_shift(
    full_script_sentences: List[str], current_paradigm_shift: Any, episode_data: Dict[str, Any], error_msg: Any
) -> Dict[str, Any]:
    sentences_json = json.dumps(full_script_sentences, ensure_ascii=False, indent=2)
    current_json = json.dumps(current_paradigm_shift, ensure_ascii=False, indent=2) if current_paradigm_shift is not None else "null"
    ep_json = json.dumps(episode_data, ensure_ascii=False, indent=2)

    user_prompt = (
        "This is a fresh independent repair request.\n"
        "The provided full_script_sentences is frozen and must not be changed.\n"
        "Return only the requested JSON field (paradigm_shift).\n"
        "Do not return the full episode object.\n"
        "Do not use markdown.\n\n"
        "CRITICAL: final_question MUST be an actual interrogative sentence ending with '?'.\n\n"
        f"Previous validation failure:\n{error_msg}\n\n"
        f"Episode data (context only):\n{ep_json}\n\n"
        f"Frozen full_script_sentences (DO NOT MODIFY):\n{sentences_json}\n\n"
        f"Current (possibly broken) paradigm_shift:\n{current_json}\n\n"
        "Return ONLY the paradigm_shift JSON object now."
    )

    raw = call_gemini_with_fallback(system_instruction=_REPAIR_PARADIGM_SHIFT_SYSTEM_PROMPT, user_prompt=user_prompt)
    parsed = _parse_json_safe(raw)

    if not isinstance(parsed, dict):
        raise ValueError("استجابة إصلاح paradigm_shift يجب أن تكون كائن JSON.")
    if isinstance(parsed.get("paradigm_shift"), dict):
        return parsed["paradigm_shift"]
    return parsed


def _repair_audio_arc(
    full_script_sentences: List[str], current_audio_arc: Any, episode_data: Dict[str, Any], error_msg: Any
) -> Dict[str, Any]:
    N = len(full_script_sentences)
    sentences_json = json.dumps(full_script_sentences, ensure_ascii=False, indent=2)
    current_json = json.dumps(current_audio_arc, ensure_ascii=False, indent=2) if current_audio_arc is not None else "null"
    ep_json = json.dumps(episode_data, ensure_ascii=False, indent=2)

    user_prompt = (
        "This is a fresh independent repair request.\n"
        "The provided full_script_sentences is frozen and must not be changed.\n"
        "Return only the requested JSON field (audio_arc).\n"
        "Do not return the full episode object.\n"
        "Do not use markdown.\n\n"
        f"Number of sentences N = {N}. Valid indices are 0 through {N - 1}.\n"
        "Reminder: dramatic_silence_sentence_indices and music_drop_sentence_indices must be JSON arrays of valid indices.\n"
        "At least one dramatic silence index is required.\n\n"
        f"Previous validation failure:\n{error_msg}\n\n"
        f"Episode data (context only):\n{ep_json}\n\n"
        f"Frozen full_script_sentences (DO NOT MODIFY):\n{sentences_json}\n\n"
        f"Current (possibly broken) audio_arc:\n{current_json}\n\n"
        "Return ONLY the audio_arc JSON object now."
    )

    raw = call_gemini_with_fallback(system_instruction=_REPAIR_AUDIO_ARC_SYSTEM_PROMPT, user_prompt=user_prompt)
    parsed = _parse_json_safe(raw)

    if not isinstance(parsed, dict):
        raise ValueError("استجابة إصلاح audio_arc يجب أن تكون كائن JSON.")
    if isinstance(parsed.get("audio_arc"), dict):
        return parsed["audio_arc"]
    return parsed


def _repair_ending_tail(
    full_script_sentences: List[str], current_ending_tail: Any, episode_data: Dict[str, Any], error_msg: Any
) -> Dict[str, Any]:
    sentences_json = json.dumps(full_script_sentences, ensure_ascii=False, indent=2)
    current_json = json.dumps(current_ending_tail, ensure_ascii=False, indent=2) if current_ending_tail is not None else "null"
    ep_json = json.dumps(episode_data, ensure_ascii=False, indent=2)

    user_prompt = (
        "This is a fresh independent repair request.\n"
        "The provided full_script_sentences is frozen and must not be changed.\n"
        "Return only the requested JSON field (ending_tail).\n"
        "Do not return the full episode object.\n"
        "Do not use markdown.\n\n"
        "Reminder: duration_seconds must be between 8 and 12 inclusive.\n"
        "spoken_audio_ends_before_tail must be true.\n"
        "No spoken promotional CTA allowed.\n\n"
        f"Previous validation failure:\n{error_msg}\n\n"
        f"Episode data (context only):\n{ep_json}\n\n"
        f"Frozen full_script_sentences (DO NOT MODIFY):\n{sentences_json}\n\n"
        f"Current (possibly broken) ending_tail:\n{current_json}\n\n"
        "Return ONLY the ending_tail JSON object now."
    )

    raw = call_gemini_with_fallback(system_instruction=_REPAIR_ENDING_TAIL_SYSTEM_PROMPT, user_prompt=user_prompt)
    parsed = _parse_json_safe(raw)

    if not isinstance(parsed, dict):
        raise ValueError("استجابة إصلاح ending_tail يجب أن تكون كائن JSON.")
    if isinstance(parsed.get("ending_tail"), dict):
        return parsed["ending_tail"]
    return parsed


# ===========================================================================
# Partial repair: per-field retry wrappers
# ===========================================================================

def _repair_scene_plan_with_attempts(
    full_script_sentences: List[str], data: Dict[str, Any], episode_data: Dict[str, Any], first_error: Any,
) -> List[Dict[str, Any]]:
    N = len(full_script_sentences)
    last_err: Any = first_error

    for attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
        try:
            sp = _repair_scene_plan(full_script_sentences, episode_data, last_err)
            test_data = copy.deepcopy(data)
            test_data["scene_plan"] = sp
            _validate_scene_plan(test_data, N)
            _validate_visual_beats(test_data)
            return sp
        except Exception as e:
            last_err = e
            logger.warning(f"scene_plan repair attempt {attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} failed: {e}")

    raise ValueError(f"فشل إصلاح scene_plan بعد {MAX_FIELD_REPAIR_ATTEMPTS} محاولات. آخر خطأ: {last_err}")


def _repair_retention_plan_with_attempts(
    full_script_sentences: List[str], data: Dict[str, Any], episode_data: Dict[str, Any], first_error: Any,
) -> Dict[str, Any]:
    last_err: Any = first_error
    current = data.get("retention_plan")

    for attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
        try:
            rp = _repair_retention_plan(full_script_sentences, current, episode_data, last_err)
            test_data = copy.deepcopy(data)
            test_data["retention_plan"] = rp
            _validate_retention_plan(test_data)
            return rp
        except Exception as e:
            last_err = e
            logger.warning(f"retention_plan repair attempt {attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} failed: {e}")

    raise ValueError(f"فشل إصلاح retention_plan بعد {MAX_FIELD_REPAIR_ATTEMPTS} محاولات. آخر خطأ: {last_err}")


def _repair_visual_bible_with_attempts(
    full_script_sentences: List[str], data: Dict[str, Any], episode_data: Dict[str, Any], first_error: Any,
) -> Dict[str, Any]:
    last_err: Any = first_error
    current = data.get("visual_bible")

    for attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
        try:
            vb = _repair_visual_bible(full_script_sentences, current, episode_data, last_err)
            test_data = copy.deepcopy(data)
            test_data["visual_bible"] = vb
            _validate_visual_bible(test_data)
            _validate_visual_metaphor_arc(test_data)
            return vb
        except Exception as e:
            last_err = e
            logger.warning(f"visual_bible repair attempt {attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} failed: {e}")

    raise ValueError(f"فشل إصلاح visual_bible بعد {MAX_FIELD_REPAIR_ATTEMPTS} محاولات. آخر خطأ: {last_err}")


def _repair_quality_report_with_attempts(
    data: Dict[str, Any], episode_data: Dict[str, Any], first_error: Any,
) -> Dict[str, Any]:
    """
    Calls the EXTENDED quality_report repair internally, so paradigm_shift,
    audio_arc, and ending_tail are properly considered in scoring.
    The public `_repair_quality_report` signature remains unchanged.
    """
    last_err: Any = first_error
    current = data.get("quality_report")

    for attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
        try:
            qr = _repair_quality_report_extended(
                full_script_sentences=data["full_script_sentences"],
                creative_brief=data.get("creative_brief"),
                paradigm_shift=data.get("paradigm_shift"),
                retention_plan=data.get("retention_plan"),
                audio_arc=data.get("audio_arc"),
                ending_tail=data.get("ending_tail"),
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
            logger.warning(f"quality_report repair attempt {attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} failed: {e}")

    raise ValueError(f"فشل إصلاح quality_report بعد {MAX_FIELD_REPAIR_ATTEMPTS} محاولات. آخر خطأ: {last_err}")


# ---------------------------------------------------------------------------
# NEW retry wrappers for paradigm_shift / audio_arc / ending_tail
# ---------------------------------------------------------------------------

def _repair_paradigm_shift_with_attempts(
    full_script_sentences: List[str], data: Dict[str, Any], episode_data: Dict[str, Any], first_error: Any,
) -> Dict[str, Any]:
    last_err: Any = first_error
    current = data.get("paradigm_shift")

    for attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
        try:
            ps = _repair_paradigm_shift(full_script_sentences, current, episode_data, last_err)
            test_data = copy.deepcopy(data)
            test_data["paradigm_shift"] = ps
            _validate_paradigm_shift(test_data)
            return ps
        except Exception as e:
            last_err = e
            logger.warning(f"paradigm_shift repair attempt {attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} failed: {e}")

    raise ValueError(f"فشل إصلاح paradigm_shift بعد {MAX_FIELD_REPAIR_ATTEMPTS} محاولات. آخر خطأ: {last_err}")


def _repair_audio_arc_with_attempts(
    full_script_sentences: List[str], data: Dict[str, Any], episode_data: Dict[str, Any], first_error: Any,
) -> Dict[str, Any]:
    last_err: Any = first_error
    current = data.get("audio_arc")
    N = len(full_script_sentences)

    for attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
        try:
            aa = _repair_audio_arc(full_script_sentences, current, episode_data, last_err)
            test_data = copy.deepcopy(data)
            test_data["audio_arc"] = aa
            _validate_audio_arc(test_data, N)
            return aa
        except Exception as e:
            last_err = e
            logger.warning(f"audio_arc repair attempt {attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} failed: {e}")

    raise ValueError(f"فشل إصلاح audio_arc بعد {MAX_FIELD_REPAIR_ATTEMPTS} محاولات. آخر خطأ: {last_err}")


def _repair_ending_tail_with_attempts(
    full_script_sentences: List[str], data: Dict[str, Any], episode_data: Dict[str, Any], first_error: Any,
) -> Dict[str, Any]:
    last_err: Any = first_error
    current = data.get("ending_tail")

    for attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
        try:
            et = _repair_ending_tail(full_script_sentences, current, episode_data, last_err)
            test_data = copy.deepcopy(data)
            test_data["ending_tail"] = et
            _validate_ending_tail(test_data)
            return et
        except Exception as e:
            last_err = e
            logger.warning(f"ending_tail repair attempt {attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} failed: {e}")

    raise ValueError(f"فشل إصلاح ending_tail بعد {MAX_FIELD_REPAIR_ATTEMPTS} محاولات. آخر خطأ: {last_err}")


# ===========================================================================
# Merge + partial repair orchestrator (operates on the frozen sentence set)
# ===========================================================================

def _merge_and_validate_partial_repairs(
    base_data: Dict[str, Any], episode_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    base_data already has a valid, frozen full_script_sentences.
    Attempts to repair only the auxiliary fields that fail validation:
      paradigm_shift / retention_plan / audio_arc / ending_tail /
      visual_bible (including visual_metaphor_arc) /
      scene_plan (including visual_beat) / quality_report.
    creative_brief has no repair path — its failure bubbles up so the caller
    can trigger a fresh auxiliary-generation attempt (Phase D), never touching
    the frozen sentences.
    """
    data = copy.deepcopy(base_data)

    sentences = data.get("full_script_sentences")
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("_merge_and_validate_partial_repairs: full_script_sentences missing or empty.")
    N = len(sentences)

    if not _field_is_valid(_validate_creative_brief, data):
        _validate_creative_brief(data)  # raises with the real reason

    if not _field_is_valid(_validate_paradigm_shift, data):
        logger.info("Detected invalid paradigm_shift. Attempting partial repair...")
        ps = _repair_paradigm_shift_with_attempts(
            full_script_sentences=sentences, data=data, episode_data=episode_data,
            first_error="paradigm_shift validation failed.",
        )
        data["paradigm_shift"] = ps
        _validate_paradigm_shift(data)

    if not _field_is_valid(_validate_retention_plan, data):
        logger.info("Detected invalid retention_plan. Attempting partial repair...")
        rp = _repair_retention_plan_with_attempts(
            full_script_sentences=sentences, data=data, episode_data=episode_data,
            first_error="retention_plan validation failed.",
        )
        data["retention_plan"] = rp
        _validate_retention_plan(data)

    if not _field_is_valid(_validate_audio_arc, data, N):
        logger.info("Detected invalid audio_arc. Attempting partial repair...")
        aa = _repair_audio_arc_with_attempts(
            full_script_sentences=sentences, data=data, episode_data=episode_data,
            first_error="audio_arc validation failed.",
        )
        data["audio_arc"] = aa
        _validate_audio_arc(data, N)

    if not _field_is_valid(_validate_ending_tail, data):
        logger.info("Detected invalid ending_tail. Attempting partial repair...")
        et = _repair_ending_tail_with_attempts(
            full_script_sentences=sentences, data=data, episode_data=episode_data,
            first_error="ending_tail validation failed.",
        )
        data["ending_tail"] = et
        _validate_ending_tail(data)

    if not _field_is_valid(_validate_visual_bible, data) or not _field_is_valid(_validate_visual_metaphor_arc, data):
        logger.info("Detected invalid visual_bible. Attempting partial repair...")
        first_err: Any = "visual_bible validation failed."
        try:
            _validate_visual_bible(data)
            _validate_visual_metaphor_arc(data)
        except ValueError as e:
            first_err = e
        vb = _repair_visual_bible_with_attempts(
            full_script_sentences=sentences, data=data, episode_data=episode_data,
            first_error=first_err,
        )
        data["visual_bible"] = vb
        _validate_visual_bible(data)
        _validate_visual_metaphor_arc(data)

    if not _field_is_valid(_validate_scene_plan, data, N) or not _field_is_valid(_validate_visual_beats, data):
        logger.info("Detected invalid scene_plan. Attempting partial repair...")
        first_err_sp: Any = "scene_plan validation failed."
        try:
            _validate_scene_plan(data, N)
            _validate_visual_beats(data)
        except ValueError as e:
            first_err_sp = e

        sp = _repair_scene_plan_with_attempts(
            full_script_sentences=sentences, data=data, episode_data=episode_data, first_error=first_err_sp,
        )
        data["scene_plan"] = sp
        _validate_scene_plan(data, N)
        _validate_visual_beats(data)

    if not _field_is_valid(_validate_quality_report, data):
        logger.info("Detected invalid quality_report. Attempting partial repair...")
        qr = _repair_quality_report_with_attempts(
            data=data, episode_data=episode_data, first_error="quality_report validation failed.",
        )
        data["quality_report"] = qr
        _validate_quality_report(data)

    return validate_script_output(data)


# ===========================================================================
# Public generator
# ===========================================================================

def generate_stage1_script(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Staged pipeline:
      Phase A: generate script_text only.
      Phase B: deterministically split into sentences.
      Phase C: bounded repair (expand/compress) until 60<=N<=80 inclusive.
      Phase D: freeze sentences, generate auxiliary structure, repair only
                auxiliary fields as needed. Never mutate the frozen sentences.
                If Phase D fails after its bounded retries, Phase D is retried
                with the SAME frozen sentence list before falling back to a
                full Phase A regeneration. Regenerating the complete script is
                a last resort, not the first response to an auxiliary failure.

    "full_script_sentences" is the ONLY authoritative source for image
    generation, audio generation, subtitles, scene_plan, and all downstream
    stages. It is never mutated by the auxiliary structure.
    """
    last_error: Any = None

    for full_attempt in range(MAX_FULL_GENERATION_ATTEMPTS):
        logger.info(f"Full generation attempt {full_attempt + 1}/{MAX_FULL_GENERATION_ATTEMPTS}")

        # ---- Phase A ----
        try:
            script_text = _generate_initial_script(episode_data)

            # ---- Phase B + C ----
            frozen_sentences, _final_script_text = _generate_valid_sentence_list(episode_data, script_text)
            N = len(frozen_sentences)
            total_words = sum(len(s.split()) for s in frozen_sentences)
            logger.info(f"Sentence count frozen at N={N}, total_words={total_words}.")

        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {full_attempt + 1}: Phase A/B/C failed: {e}")
            continue

        # ---- Phase D (with its own bounded retry loop, then a re-round with the SAME frozen list) ----
        aux_data = None
        aux_last_error: Any = None

        for phase_d_round in range(MAX_PHASE_D_ROUNDS):
            if phase_d_round > 0:
                logger.info(
                    f"Retrying Phase D with the SAME frozen sentence list "
                    f"(round {phase_d_round + 1}/{MAX_PHASE_D_ROUNDS}, N={N})."
                )

            for aux_attempt in range(MAX_FIELD_REPAIR_ATTEMPTS):
                try:
                    aux_candidate = _generate_auxiliary_structure(episode_data, frozen_sentences)

                    # Frozen / authoritative fields — never trust the model for these.
                    aux_candidate["id"] = episode_data.get("id", aux_candidate.get("id", 0))
                    aux_candidate["topic"] = episode_data.get("topic", aux_candidate.get("topic", ""))
                    aux_candidate["full_script_sentences"] = frozen_sentences
                    aux_candidate["total_word_count"] = total_words

                    aux_data = _merge_and_validate_partial_repairs(aux_candidate, episode_data)
                    break
                except Exception as e:
                    aux_last_error = e
                    logger.warning(
                        f"Phase D round {phase_d_round + 1}/{MAX_PHASE_D_ROUNDS}, "
                        f"attempt {aux_attempt + 1}/{MAX_FIELD_REPAIR_ATTEMPTS} failed: {e}"
                    )

            if aux_data is not None:
                break

        if aux_data is None:
            last_error = ValueError(
                f"فشل توليد الهيكل المساعد (Phase D) بعد "
                f"{MAX_PHASE_D_ROUNDS * MAX_FIELD_REPAIR_ATTEMPTS} محاولات "
                f"باستخدام نفس قائمة الجمل المجمّدة. آخر خطأ: {aux_last_error}"
            )
            logger.warning(str(last_error))
            # Fall through to regenerate Phase A on the next outer iteration.
            continue

        # Final guarantee: sentences are exactly the frozen list, untouched.
        aux_data["full_script_sentences"] = frozen_sentences
        aux_data["total_word_count"] = total_words

        return validate_script_output(aux_data)

    raise ValueError(
        f"فشل توليد سكربت المرحلة الأولى بعد {MAX_FULL_GENERATION_ATTEMPTS} محاولات كاملة. آخر خطأ: {last_error}"
    )
