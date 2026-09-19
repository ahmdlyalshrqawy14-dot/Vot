import os
import json
import sys
import re
from google import genai
from google.genai import types

# قائمة النماذج حسب أولويتك المحددة
MODELS_PRIORITY = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash",
    "gemini-3.5-flash-lite"
]

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
    """تنظيف الرد في حال تم إرجاع علامات الماركداون"""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text

def expand_episode_with_fallback(episode_data, api_key):
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
You are a senior YouTube scriptwriter, biomechanics expert, and creative director for an evidence-based fitness channel.

Transform this episode blueprint into a full, high-retention YouTube production package:
Blueprint:
{json.dumps(episode_data, indent=2)}

STRICT PRODUCTION RULES:
1. Language: 100% English across all fields.
2. Script Length & Pacing: Expand into 40 to 55 punchy sentences. Every sentence MUST end with a period (.), express one complete thought, and be suited for a single natural breath.
3. Narrative Structure: Include 3 hooks in metadata. In the timeline, use the strongest hook first, develop relatable analogies, insert a retention-reset question mid-way, state the core takeaway ("The Zaytouna"), add a tailored subscribe CTA, and conclude with the comment question.
4. Image Prompts (1-to-1 Mapping): Provide an image prompt for EVERY sentence. Use this exact base prompt, altering ONLY the pose/action and maintaining the orange color:
   "Create a 2D muscular character illustration in the exact style of the provided example. The figure should be orange, with a smooth head, large white oval eyes, no mouth, and a defined muscular body wearing black shorts. Maintain the same proportions and facial features as in the reference. [INSERT SPECIFIC POSE HERE]. Background must be plain white. Clean, cel-shaded, and expressive style. Keep the style consistent across all generated images."
5. Voice TTS: Wrap 1 to 2 emotions in square brackets (e.g., [shock] [urgency], [calm], [sarcasm]) selected from the approved list, followed by the spoken sentence containing natural emojis. Never repeat the same emotion 3 times consecutively.
6. Packaging: Generate 3 distinct titles (Curiosity, Pain Point, Outcome), 3 thumbnail prompts matching those angles, an engaging description, and comma-separated tags.

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
    {{ "angle": "curiosity", "prompt": "" }},
    {{ "angle": "pain_point", "prompt": "" }},
    {{ "angle": "outcome_gain", "prompt": "" }}
  ],
  "timeline": [
    {{
      "id": 1,
      "script_sentence": "Sentence here.",
      "voice_tts": "[emotion] \\"Sentence with emoji.\\"",
      "image_prompt": "Base character prompt with pose."
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

    print(f"Error: All fallback models failed. Last error: {last_error}")
    sys.exit(1)

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing.")
        sys.exit(1)
        
    print("Loading pending episode from bank...")
    episode, all_episodes = get_next_episode()
    print(f"Selected Episode {episode['id']}: {episode['topic']}")
    
    print("Expanding episode with fallback pipeline...")
    expanded_data = expand_episode_with_fallback(episode, api_key)
    
    output_file = "current_episode.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(expanded_data, f, indent=2, ensure_ascii=False)
        
    print(f"Saved production package to {output_file} successfully.")

if __name__ == "__main__":
    main()
