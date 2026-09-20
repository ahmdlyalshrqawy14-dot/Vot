import json
from pathlib import Path
from config.models import generate_with_fallback
from config.settings import PROMPTS_DIR

def run_stage_1(episode: dict, episode_dir: Path) -> str:
    """
    المرحلة الأولى: توليد السكربت الإنجليزي الكامل للحلقة
    """
    prompt_file = PROMPTS_DIR / "01_script.txt"
    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    user_prompt = f"Episode Information:\n{json.dumps(episode, indent=2)}"

    print(f"\n--- [Stage 1] Generating Script for Episode #{episode.get('id')} ---")
    script_text = generate_with_fallback(
        stage_name="script",
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # حفظ السكربت في مجلد الحلقة
    script_file = episode_dir / "script.md"
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_text)

    print(f"--- [Stage 1] Script saved successfully to: {script_file} ---")
    return script_text
