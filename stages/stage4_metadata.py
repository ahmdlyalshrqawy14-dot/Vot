import json
from pathlib import Path
from config.models import generate_with_fallback
from config.settings import PROMPTS_DIR, CHARACTER_COLOR

def run_stage_4(episode: dict, script_text: str, episode_dir: Path) -> str:
    """
    المرحلة الرابعة: توليد العناوين، برومبتات الثامبنيل، الوصف، والكلمات الدلالية
    """
    prompt_file = PROMPTS_DIR / "04_metadata.txt"
    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # استبدال لون الشخصية في برومبتات الصورة المصغرة
    system_prompt = system_prompt.replace("{COLOR}", CHARACTER_COLOR)

    user_prompt = f"""
Episode Data:
{json.dumps(episode, indent=2)}

Full Video Script:
{script_text}
""".strip()

    print(f"\n--- [Stage 4] Generating YouTube Metadata & Thumbnails ---")
    metadata_text = generate_with_fallback(
        stage_name="metadata",
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # حفظ المخرجات في مجلد الحلقة
    metadata_file = episode_dir / "metadata.md"
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write(metadata_text)

    print(f"--- [Stage 4] Metadata saved successfully to: {metadata_file} ---")
    return metadata_text
