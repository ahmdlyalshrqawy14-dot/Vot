import time
import urllib.parse
from pathlib import Path
import requests
from PIL import Image
from openai import AzureOpenAI
from config.settings import (
    AZURE_IMAGE_KEY,
    AZURE_IMAGE_ENDPOINT,
    AZURE_IMAGE_DEPLOYMENT,
    IMAGE_SIZE
)

def verify_image(file_path: Path) -> bool:
    if not file_path.exists() or file_path.stat().st_size < 35 * 1024:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def generate_via_azure(prompt: str) -> bytes:
    client = AzureOpenAI(
        api_version="2024-02-01",
        azure_endpoint=AZURE_IMAGE_ENDPOINT,
        api_key=AZURE_IMAGE_KEY
    )
    response = client.images.generate(
        model=AZURE_IMAGE_DEPLOYMENT,
        prompt=prompt,
        n=1,
        size=IMAGE_SIZE,  # 1792x1024 شاشة عريضة أصلية
        quality="standard"
    )
    img_resp = requests.get(response.data[0].url, timeout=50)
    img_resp.raise_for_status()
    return img_resp.content

def generate_via_pollinations(prompt: str) -> bytes:
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1792&height=1024&nologo=true"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    return response.content

def run_stage_2(script_data: dict, episode_dir: Path):
    images_dir = episode_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    beats = script_data.get("beats", [])
    total = len(beats)

    print(f"\n--- [Stage 2] Rendering {total} Wide Visual Beats (16:9) ---")

    for beat in beats:
        b_id = beat["id"]
        image_path = images_dir / f"beat_{b_id:03d}.png"

        if verify_image(image_path):
            continue

        prompt = beat["visual_prompt"]
        print(f"[Image {b_id:03d}/{total:03d}] Generating...")

        # محاولة التوليد عبر Azure مع Exponential Backoff ذكي
        success = False
        for attempt in range(3):
            try:
                img_bytes = generate_via_azure(prompt)
                with open(image_path, "wb") as f:
                    f.write(img_bytes)
                if verify_image(image_path):
                    success = True
                    break
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg:
                    wait_time = (attempt + 1) * 20
                    print(f"[Image {b_id:03d}] Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[Image {b_id:03d}] Azure attempt {attempt+1} failed: {err_msg[:90]}")
                    time.sleep(3)

        # البديل التلقائي إذا استمر الفشل
        if not success:
            print(f"[Image {b_id:03d}] Falling back to high-res secondary engine...")
            try:
                img_bytes = generate_via_pollinations(prompt)
                with open(image_path, "wb") as f:
                    f.write(img_bytes)
            except Exception as e:
                print(f"[Image {b_id:03d}] Critical: Image failed completely: {e}")

        # مهلة آمنة لاستهلاك الكوتا السحابية
        time.sleep(2)

    print(f"--- [Stage 2] Completed Visual Asset Generation ---")
