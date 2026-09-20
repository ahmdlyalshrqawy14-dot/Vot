import json
from pathlib import Path
from config.models import generate_text_fallback
from config.settings import PROMPTS_DIR

def run_stage_4(episode: dict, script_data: dict, episode_dir: Path) -> str:
    metadata_file = episode_dir / "metadata.md"
    if metadata_file.exists():
        return metadata_file.read_text(encoding="utf-8")

    prompt_file = PROMPTS_DIR / "04_metadata.txt"
    system_prompt = prompt_file.read_text(encoding="utf-8")

    all_spoken = " ".join([b["spoken_text"] for b in script_data.get("beats", [])])
    user_prompt = f"""
    Topic Details:
    {json.dumps(episode, indent=2, ensure_ascii=False)}

    Spoken Script:
    {all_spoken}
    """

    print(f"\n--- [Stage 4] Generating SEO YouTube Metadata & Titles ---")
    meta_text = generate_text_fallback("metadata", system_prompt, user_prompt)
    metadata_file.write_text(meta_text, encoding="utf-8")
    return meta_text
