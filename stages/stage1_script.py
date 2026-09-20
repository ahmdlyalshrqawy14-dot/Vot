import json
from pathlib import Path
from config.models import generate_json_fallback
from config.settings import PROMPTS_DIR, CHARACTER_COLOR

def run_stage_1(episode: dict, episode_dir: Path) -> dict:
    """توليد مصفوفة المشاهد المهيكلة للسكربت الطويل"""
    script_json_path = episode_dir / "script_beats.json"
    
    # Checkpoint: إذا تم توليده مسبقاً نتخطى لإكمال باقي المراحل
    if script_json_path.exists():
        print("--- [Stage 1] Script beats already exist. Loading cached version. ---")
        with open(script_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    prompt_file = PROMPTS_DIR / "01_long_script.txt"
    system_prompt = prompt_file.read_text(encoding="utf-8")
    
    user_prompt = f"""
    Please generate the 5-minute deep-dive YouTube script for this topic:
    Topic Data:
    {json.dumps(episode, indent=2, ensure_ascii=False)}
    
    Remember: Target ~45-50 beats (650-750 words) to ensure a full 5-minute educational experience.
    """

    print(f"\n--- [Stage 1] Generating 5-Minute Structured Beats for Episode #{episode.get('id')} ---")
    data = generate_json_fallback(
        stage_name="script",
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # حقن لون الشخصية في جميع البرومبتات
    for beat in data.get("beats", []):
        beat["visual_prompt"] = beat["visual_prompt"].replace("{COLOR}", CHARACTER_COLOR)

    with open(script_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"--- [Stage 1] Generated {len(data.get('beats', []))} beats successfully! ---")
    return data
