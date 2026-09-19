import os
import json
import sys
import re
from google import genai
from google.genai import types

MODELS_PRIORITY = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash"  # احتياطي في حال عدم توفر أسماء الإصدارات التجريبية
]

# القائمة الرسمية الحصرية للانفعالات وفق دليل الإنتاج
APPROVED_EMOTIONS = (
    "[admiration], [aggression], [amusement], [anger], [anxiety], [apology], [approval], "
    "[awe], [boredom], [calm], [celebration], [concern], [contempt], [contentment], "
    "[curiosity], [determination], [disapproval], [disbelief], [disgust], [embarrassment], "
    "[empathy], [enthusiasm], [excitement], [fear], [frustration], [gasps], [gratitude], "
    "[hope], [humor], [interest], [joy], [laughs], [longing], [love], [nervousness], "
    "[nostalgia], [pride], [relief], [sadness], [sarcasm], [satisfaction], [shock], "
    "[sighs], [suspense], [sympathy], [tenderness], [tiredness], [trust], [uncertainty], "
    "[urgency], [vulnerability], [warning], [whispers], [wonder], [cries]"
)

def get_next_episode(bank_path="episodes_bank.json"):
    if not os.path.exists(bank_path):
        print(f"Error: {bank_path} not found.")
        sys.exit(1)
        
    with open(bank_path, "r", encoding="utf-8") as f:
        episodes = json.load(f)
        
    for ep in episodes:
        if ep.get("status") == "pending":
            return ep, episodes
            
    print("No pending episodes found in queue.")
    sys.exit(0)

def clean_json_response(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text

def expand_episode_with_fallback(episode_data, api_key):
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
You are a Lead YouTube Scriptwriter and Biomechanics Production Specialist.

Transform this episode blueprint into a full, high-retention YouTube production package:
{json.dumps(episode_data, indent=2)}

STRICT PRODUCTION RULES:
1. Language: 100% English across all fields.
2. Script Length & Pacing: Expand into 40 to 55 punchy sentences. Every sentence MUST end with a period (.), express one complete thought, and be suited for a single natural breath.
3. Narrative Structure:
   - In metadata: 3 distinct clickable titles (Curiosity, Pain Point, Outcome).
   - In timeline: Open with the strongest pattern-interrupt hook, translate anatomy/science into vivid relatable scenarios, insert a retention-reset question mid-way, present the core takeaway ("The Zaytouna"), add a smart tailored subscribe CTA, and finish with a high-engagement comment question.
4. Base Character Image Prompts (1-to-1 Mapping for BOTH Timeline AND Thumbnails):
   You MUST use this EXACT template for every single image prompt in timeline AND in thumbnails:
   "Create a 2D muscular character illustration in the exact style of the provided example. The figure should be orange, with a smooth head, large white oval eyes, no mouth, and a defined muscular body wearing black shorts. Maintain the same proportions and facial features as in the reference. [INSERT DYNAMIC POSE HERE]. Background must be plain white. Clean, cel-shaded, and expressive style. Keep the style consistent across all generated images."
   - STRICT CONSTRAINT: Do NOT add text overlays, subtitles, split screens, or colored backgrounds. Only plain white background. Alter ONLY the [INSERT DYNAMIC POSE HERE].
   - Thumbnails must depict extreme, high-visibility expressive character poses representing each angle (Curiosity, Pain Point, Outcome) using the same exact Base Prompt.
5. Google TTS Emotion Tagging (STRICT WHITELIST):
   - You are ONLY permitted to use emotions from this exact list:
     {APPROVED_EMOTIONS}
   - STRICT FORBIDDEN WORDS: NEVER use unapproved emotions like [authority], [focus], [caution], [reflective], [serious], etc. If you want authority, use [trust] or [determination]. If you want caution, use [warning] or [concern]. If you want focus, use [interest] or [calm].
   - Put 1 to 2 emotions in square brackets at the start of each sentence, followed by the spoken sentence with natural emojis placed inside.
   - Do NOT repeat the exact same emotion 3 times consecutively.
6. Description Formatting:
   The description field must strictly follow this structure:
   [Line 1-2: Hook and core benefit summary]
   [Brief 2-3 sentence overview of the video's science-backed solution]
   
   Question of the day: [The exact comment question]
   Subscribe: [Tailored call to action]
7. Tags: A single comma-separated line from broad to niche keywords.

Return ONLY a valid raw JSON object matching this schema:
{{
  "episode_id": "ep_{episode_data['id']:03d}",
  "topic_name": "{episode_data['topic']}",
  "character_color": "orange",
  "metadata": {{
    "titles": {{ "curiosity": "", "pain_point": "", "outcome_gain": "" }},
    "description": "",
    "tags": ""
  }},
  "thumbnails": [
    {{ "angle": "curiosity", "prompt": "Exact base prompt with extreme curiosity pose" }},
    {{ "angle": "pain_point", "prompt": "Exact base prompt with extreme defeat/pain pose" }},
    {{ "angle": "outcome_gain", "prompt": "Exact base prompt with extreme victory/strength pose" }}
  ],
  "timeline": [
    {{
      "id": 1,
      "script_sentence": "Sentence here.",
      "voice_tts": "[emotion] \\"Sentence with emoji.\\"",
      "image_prompt": "Exact base prompt with pose."
    }}
  ]
}}
"""

    last_error = None
    for model_name in MODELS_PRIORITY:
        print(f"Attempting expansion using model: {model_name}...")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            
            cleaned_text = clean_json_response(response.text)
            parsed_json = json.loads(cleaned_text)
            print(f"Success with model: {model_name}")
            return parsed_json
            
        except Exception as e:
            print(f"Warning: Model {model_name} failed. Error: {e}")
            last_error = e
            continue

    print(f"Error: All models failed. Last error: {last_error}")
    sys.exit(1)

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing.")
        sys.exit(1)
        
    print("Loading pending episode from bank...")
    episode, all_episodes = get_next_episode()
    print(f"Selected Episode {episode['id']}: {episode['topic']}")
    
    print("Expanding episode with strict rules...")
    expanded_data = expand_episode_with_fallback(episode, api_key)
    
    output_file = "current_episode.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(expanded_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved production package to {output_file} successfully.")

if __name__ == "__main__":
    main()
