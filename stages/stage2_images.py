import re
from pathlib import Path
from config.models import generate_with_fallback
from config.settings import PROMPTS_DIR, CHARACTER_COLOR

def run_stage_2(script_text: str, episode_dir: Path) -> str:
    """
    المرحلة الثانية: استخراج وتوليد أوامر الصور التعبيرية لكل جملة
    """
    prompt_file = PROMPTS_DIR / "02_images.txt"
    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    # حقن لون الشخصية المعتمد في تعليمات النظام
    system_prompt = system_prompt.replace("{COLOR}", CHARACTER_COLOR)

    user_prompt = f"English Script:\n{script_text}"

    print(f"\n--- [Stage 2] Generating Image Prompts ---")
    prompts_text = generate_with_fallback(
        stage_name="images",
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # تنظيف المخرجات واستخراج المحتوى من داخل صندوق الكود إن وجد
    clean_prompts = prompts_text.strip()
    code_block_match = re.search(r"```(?:text)?\s*(.*?)\s*```", clean_prompts, re.DOTALL)
    if code_block_match:
        clean_prompts = code_block_match.group(1).strip()

    # حفظ البرومبتات في مجلد مخرجات الحلقة
    prompts_file = episode_dir / "image_prompts.txt"
    with open(prompts_file, "w", encoding="utf-8") as f:
        f.write(clean_prompts)

    print(f"--- [Stage 2] Image prompts saved successfully to: {prompts_file} ---")
    return clean_prompts
