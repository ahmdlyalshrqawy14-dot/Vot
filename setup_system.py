import os
from pathlib import Path

BASE_DIR = Path("C:/Vot")
FILES = {}

# 1. قائمة النماذج المعتمدة
FILES["config/models.py"] = '''import os
import time
import json
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# قائمة النماذج بالترتيب المعتمد
MODEL_FALLBACK_LIST = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

KEY_MAPPING = {
    "script": os.getenv("GEMINI_API_KEY"),
    "metadata": os.getenv("GEMINI_API_KEY_2")
}

def generate_text_fallback(stage_name: str, system_prompt: str, user_prompt: str) -> str:
    api_key = KEY_MAPPING.get(stage_name) or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(f"API Key for stage '{stage_name}' is missing.")

    client = genai.Client(api_key=api_key)
    last_error = None

    for model_name in MODEL_FALLBACK_LIST:
        try:
            print(f"[{stage_name.upper()}] Requesting {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                )
            )
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"[{stage_name.upper()}] Model {model_name} warning: {str(e)[:100]}")
            last_error = e
            time.sleep(1)

    raise RuntimeError(f"All fallback models failed for {stage_name}. Last error: {last_error}")

def generate_json_fallback(stage_name: str, system_prompt: str, user_prompt: str) -> dict:
    raw_text = generate_text_fallback(stage_name, system_prompt, user_prompt)
    json_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw_text, re.DOTALL)
    clean_str = json_match.group(1).strip() if json_match else raw_text.strip()
    try:
        return json.loads(clean_str)
    except json.JSONDecodeError:
        start = clean_str.find("{")
        end = clean_str.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(clean_str[start:end])
        raise ValueError(f"Failed to parse JSON response: {clean_str[:200]}")
'''

# 2. فاحص الصور والرقعة الذكية
FILES["stages/vision_sorter.py"] = '''import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFilter
from google import genai
from dotenv import load_dotenv
from config.models import MODEL_FALLBACK_LIST as MODELS

load_dotenv()

API_KEYS = [
    k for k in [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GEMINI_API_KEY_2"),
        os.getenv("GEMINI_API_KEY_3"),
        os.getenv("GEMINI_API_KEY_4"),
    ] if k
]

def remove_corner_badge(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    patch_w = int(w * 0.14)
    patch_h = int(h * 0.09)
    x0 = w - patch_w
    y0 = h - patch_h

    sample_box = (max(0, x0 - 20), max(0, y0 - 30), min(w, x0 + 10), min(h, y0 - 10))
    sample_crop = img.crop(sample_box)
    stat = sample_crop.resize((1, 1)).getpixel((0, 0))
    bg_color = (stat[0], stat[1], stat[2])

    mask = Image.new("L", (patch_w, patch_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([4, 4, patch_w, patch_h], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=5))

    patch = Image.new("RGB", (patch_w, patch_h), bg_color)
    img.paste(patch, (x0, y0), mask)
    return img

def read_badge_number(image_path: Path, key_idx: int = 0) -> int:
    if not API_KEYS:
        return 0

    api_key = API_KEYS[key_idx % len(API_KEYS)]
    client = genai.Client(api_key=api_key)

    prompt = (
        "Focus on the bottom-right and bottom-left corners of this image. "
        "Find the small badge or label indicating the frame/scene number, formatted like '#01', '#12', or '#45'. "
        "Respond with ONLY the integer number between 1 and 45. "
        "If no number is visible, return 0."
    )

    for m in MODELS:
        try:
            with Image.open(image_path) as img:
                res = client.models.generate_content(
                    model=m,
                    contents=[img, prompt]
                )
                text = res.text.strip() if res.text else ""
                match = re.search(r"\b([1-9]|[1-3][0-9]|4[0-5])\b", text)
                if match:
                    return int(match.group(1))
                return 0
        except Exception:
            continue
    return 0

def sort_and_save_images_parallel(image_files: list, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    detected_results = {}

    def worker(idx_and_path):
        idx, p = idx_and_path
        num = read_badge_number(p, idx)
        return p, num

    workers_count = min(8, len(image_files) or 1)
    with ThreadPoolExecutor(max_workers=workers_count) as executor:
        results = executor.map(worker, enumerate(image_files))
        for p, num in results:
            detected_results[p] = num

    assigned = {}
    unassigned = []

    for p in image_files:
        n = detected_results.get(p, 0)
        if 1 <= n <= 45 and n not in assigned:
            assigned[n] = p
        else:
            unassigned.append(p)

    missing_slots = [i for i in range(1, 46) if i not in assigned]
    for slot, p in zip(missing_slots, unassigned):
        assigned[slot] = p

    saved_count = 0
    for beat_id, p in assigned.items():
        dest = target_dir / f"beat_{beat_id:03d}.png"
        try:
            with Image.open(p) as img:
                clean_img = remove_corner_badge(img)
                clean_img.save(dest, "PNG", quality=98)
            saved_count += 1
        except Exception as e:
            print(f"[SAVE ERROR] {dest}: {e}")

    return saved_count
'''

# 3. ملفات الـ TXT لـ Google Flow
FILES["stages/stage2_telegram_prompts.py"] = '''import json
from pathlib import Path
from config.settings import CHARACTER_COLOR

def generate_google_flow_files(script_data: dict, metadata_text: str, episode_dir: Path) -> list:
    beats = script_data.get("beats", [])
    files_dir = episode_dir / "prompts_txt"
    files_dir.mkdir(parents=True, exist_ok=True)

    base_template = (
        "Create a 2D muscular character illustration in the exact style of the provided example. "
        f"The figure should be {CHARACTER_COLOR}, with a smooth head, large white oval eyes, no mouth, "
        "and a defined muscular body wearing black shorts. Maintain the same proportions and facial features "
        "as in the reference. {action} "
        "In the bottom-right corner of the image, render a small, clean, unobtrusive badge text '#{b_id:02d}'. "
        "Background must be plain white. Clean, cel-shaded, and expressive style. "
        "Keep the style consistent across all generated images."
    )

    output_packages = []

    # دفعة 1 (01 إلى 20)
    b1_lines = [base_template.format(action=b.get("visual_prompt", "athletic pose").replace("\\n", " ").strip(), b_id=b["id"]) for b in beats[0:20]]
    file_b1 = files_dir / "01_Batch_1_Beats_01_to_20.txt"
    file_b1.write_text("\\n".join(b1_lines), encoding="utf-8")
    output_packages.append({"path": file_b1, "caption": "📄 <b>الدفعة الأولى: المشاهد (01 إلى 20)</b>\\n20 سطراً لـ Google Flow."})

    # دفعة 2 (21 إلى 40)
    b2_lines = [base_template.format(action=b.get("visual_prompt", "athletic pose").replace("\\n", " ").strip(), b_id=b["id"]) for b in beats[20:40]]
    file_b2 = files_dir / "02_Batch_2_Beats_21_to_40.txt"
    file_b2.write_text("\\n".join(b2_lines), encoding="utf-8")
    output_packages.append({"path": file_b2, "caption": "📄 <b>الدفعة الثانية: المشاهد (21 إلى 40)</b>\\n20 سطراً لـ Google Flow."})

    # دفعة 3 (41 إلى 45)
    b3_lines = [base_template.format(action=b.get("visual_prompt", "athletic pose").replace("\\n", " ").strip(), b_id=b["id"]) for b in beats[40:45]]
    file_b3 = files_dir / "03_Batch_3_Beats_41_to_45.txt"
    file_b3.write_text("\\n".join(b3_lines), encoding="utf-8")
    output_packages.append({"path": file_b3, "caption": "📄 <b>الدفعة الثالثة: المشاهد (41 إلى 45)</b>\\n5 أسطر لـ Google Flow."})

    # الأغلفة
    thumbs = [
        ("Shocking Curiosity", f"Extremely shocked and horrified, hands on head, wide eyes, {CHARACTER_COLOR} muscular character leaning towards camera."),
        ("Pain-Point Angle", f"Intense agony, tightly clutching knee joint, dynamic strain, {CHARACTER_COLOR} muscular character in black shorts."),
        ("Ultimate Result", f"Triumphant victory pose flexing biceps, muscular dominance, {CHARACTER_COLOR} muscular character standing tall.")
    ]
    t_lines = [f"[{title}]\\nYouTube Clickable Thumbnail, 16:9, 2D style, {CHARACTER_COLOR} muscular character in black shorts. {act} Plain white background." for title, act in thumbs]
    file_t = files_dir / "04_Thumbnails.txt"
    file_t.write_text("\\n\\n".join(t_lines), encoding="utf-8")
    output_packages.append({"path": file_t, "caption": "🖼️ <b>برومبتات الأغلفة الثلاثة (Thumbnails)</b>"})

    return output_packages
'''

# 4. ملف حماية البيانات السرية (.gitignore)
FILES[".gitignore"] = '''
.env
output/
__pycache__/
*.pyc
*.zip
*.wav
*.mp4
*.ts
'''

for rel_path, content in FILES.items():
    p = BASE_DIR / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\\n")
    print(f"Updated: {rel_path}")

print("\\n[SUCCESS] ALL SYSTEM FILES CLEANED & UPDATED SUCCESSFULLY!")