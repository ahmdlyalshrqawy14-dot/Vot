import json
import re
import logging
from typing import Dict, Any
from gemini_engine import call_gemini_with_fallback

logger = logging.getLogger("Stage1Generator")

STAGE_1_SYSTEM_PROMPT = """You are an elite YouTube Scriptwriter and Content Strategist specializing in self-development, fitness psychology, and audience retention.
Your task is to take a raw input JSON containing episode metadata and turn it into a high-retention, 5-to-6-minute YouTube script (target: 800 - 900 words, at an average speaking rate of 150 words per minute).

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

SCRIPTWRITING & PACING RULES
TONE & STYLE: 
- Write like a professional YouTube creator speaking directly to the viewer.
- Conversational, authoritative yet warm and relatable spoken English.
- The entire script must feel like ONE continuous, cohesive piece — not a list of disconnected lines.
- Use natural transitions between ideas so the narration flows smoothly from one sentence to the next.
- Avoid robotic or staccato delivery. The viewer should feel they are listening to a well-written video, not a series of short announcements.

SENTENCE CADENCE (IMPORTANT):
- Keep sentences relatively short and easy to speak in one breath.
- Every sentence must end with a period, question mark, or exclamation mark.
- Prefer clear, direct sentences. Avoid extremely long compound sentences, semicolons, or em-dashes.
- However, allow natural connecting words and soft transitions (And, But, So, That's why, Here's the thing, etc.) so the script does not feel choppy or broken.

STRUCTURE & WORD BUDGET (800 - 900 Words Total):
- THE HOOK (approx. 40 - 50 words): One powerful opening that creates a pattern interrupt and strong curiosity.
- THE INTRO (approx. 100 - 120 words): Connect the topic to real daily struggles and frustration.
- PART 1 - THE MYTH & THE SCIENCE (approx. 200 words): Expand "the_myth" and "the_truth" with vivid, concrete examples.
- PART 2 - THE ANALOGY EXPANDED (approx. 150 words): Turn "the_analogy" into a clear, memorable mental picture.
- PART 3 - KEY LESSONS & ATTENTION RESETS (approx. 150 words): Cover "key_points". Insert occasional rhetorical questions to keep attention.
- PART 4 - ACTIONABLE BLUEPRINT (approx. 150 words): Break down "actionable_solution" into clear, practical steps.
- OUTRO & CTA (approx. 60 words): Deliver the "core_takeaway" with a natural call to action.
- ENGAGEMENT QUESTION (approx. 25 words): End with the "comment_question".

ABSOLUTELY NO SHORTCUTS: Do not summarize or skip key points. Expand every idea into concrete scenes, sensory details, and practical examples.

FLOW REQUIREMENT (CRITICAL):
The final list of sentences in "full_script_sentences" must read as a continuous professional narration when spoken one after another. 
Do not make every sentence start abruptly. Create smooth logical progression from the first sentence to the last.

OUTPUT FORMAT
You must return a valid, parsable JSON object ONLY, with no markdown wrappers, no backticks, and no conversational filler:
{
  "id": 0,
  "topic": "string",
  "total_word_count": 0,
  "hook": "string",
  "sections": [
    {
      "section_name": "hook",
      "sentences": ["sentence 1", "sentence 2"]
    },
    {
      "section_name": "intro",
      "sentences": ["sentence 1", "sentence 2"]
    },
    {
      "section_name": "myth_and_truth",
      "sentences": ["..."]
    },
    {
      "section_name": "analogy",
      "sentences": ["..."]
    },
    {
      "section_name": "key_points",
      "sentences": ["..."]
    },
    {
      "section_name": "actionable_blueprint",
      "sentences": ["..."]
    },
    {
      "section_name": "takeaway_and_cta",
      "sentences": ["..."]
    },
    {
      "section_name": "comment_question",
      "sentences": ["..."]
    }
  ],
  "full_script_sentences": [
    "Chronological list of every single sentence in the script ending with a period."
  ]
}"""


def _clean_json_string(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def validate_script_output(data: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = ["id", "topic", "hook", "sections", "full_script_sentences"]
    for k in required_keys:
        if k not in data:
            raise ValueError(f"الحقل المطلوب '{k}' مفقود من مخرجات JSON!")

    sentences = data.get("full_script_sentences", [])
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("مصفوفة 'full_script_sentences' فارغة!")

    cleaned_sentences = []
    total_words = 0
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        if not (s_clean.endswith(".") or s_clean.endswith("?") or s_clean.endswith("!")):
            s_clean += "."
        cleaned_sentences.append(s_clean)
        total_words += len(s_clean.split())

    data["full_script_sentences"] = cleaned_sentences
    data["total_word_count"] = total_words

    if 800 <= total_words <= 900:
        logger.info(f"🎯 حجم السكربت مثالي: {total_words} كلمة.")
    else:
        logger.warning(f"ℹ️ حجم السكربت: {total_words} كلمة (المستهدف: 800 - 900 كلمة).")

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
