import os
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
                match = re.search(r"([1-9]|[1-3][0-9]|4[0-5])", text)
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

    return saved_count\n