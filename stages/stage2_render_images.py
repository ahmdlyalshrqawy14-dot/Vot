import os
import time
import urllib.parse
from pathlib import Path
import requests
from PIL import Image
from openai import AzureOpenAI
from config.settings import (
    AZURE_IMAGE_KEY,
    AZURE_IMAGE_ENDPOINT,
    AZURE_IMAGE_DEPLOYMENT
)

def verify_image(file_path: Path) -> bool:
    """
    التحقق من أن الملف تم تحميله بالكامل وأنه صورة صالحة وغير تالفة
    """
    if not file_path.exists():
        return False
    # التحقق من أن حجم الملف أكبر من 30 كيلوبايت
    if file_path.stat().st_size < 30 * 1024:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def generate_via_azure(prompt: str) -> bytes:
    """
    التوليد عبر مايكروسوفت Azure DALL-E 3
    """
    if not AZURE_IMAGE_KEY or not AZURE_IMAGE_ENDPOINT:
        raise ValueError("Azure Image credentials missing.")

    client = AzureOpenAI(
        api_version="2024-02-01",
        azure_endpoint=AZURE_IMAGE_ENDPOINT,
        api_key=AZURE_IMAGE_KEY
    )

    response = client.images.generate(
        model=AZURE_IMAGE_DEPLOYMENT or "dall-e-3",
        prompt=prompt,
        n=1,
        size="1024x1024",
        quality="standard"
    )

    image_url = response.data[0].url
    img_resp = requests.get(image_url, timeout=40)
    img_resp.raise_for_status()
    return img_resp.content

def generate_via_pollinations(prompt: str) -> bytes:
    """
    التوليد عبر البديل المجاني المباشر Pollinations.ai
    """
    encoded_prompt = urllib.parse.quote(prompt)
    # رابط Pollinations مع أبعاد 1024x1024 وإلغاء اللوجو وتحديد موديل سريع ونظيف
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    return response.content

def render_all_images(prompts_text: str, episode_dir: Path):
    """
    قراءة البرومبتات وتوليد صورة لكل جملة تسلسلياً مع الفحص ونظام التبديل
    """
    images_dir = episode_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # تقسيم البرومبتات على أساس الأسطر الفارغة
    raw_prompts = [p.strip() for p in prompts_text.strip().split("\n\n") if p.strip()]
    
    # في حال لم تكن مفصولة بأسطر فارغة، نأخذ كل سطر مستقل
    if len(raw_prompts) <= 1 and "\n" in prompts_text:
        raw_prompts = [p.strip() for p in prompts_text.strip().split("\n") if p.strip()]

    total = len(raw_prompts)
    print(f"\n--- [Stage 2.5] Rendering {total} Images Sequentially ---")

    for idx, prompt in enumerate(raw_prompts, start=1):
        image_path = images_dir / f"{idx:03d}.png"

        # التحقق إذا كانت الصورة موجودة وسليمة مسبقاً (لتوفير الموارد)
        if verify_image(image_path):
            print(f"[Image {idx:03d}/{total:03d}] Already exists and verified. Skipping.")
            continue

        success = False
        image_bytes = None

        # 1. المحاولة الأولى: مايكروسوفت Azure DALL-E 3
        print(f"[Image {idx:03d}/{total:03d}] Generating via Azure DALL-E 3...")
        try:
            image_bytes = generate_via_azure(prompt)
            # حفظ مبدئي للفحص
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            if verify_image(image_path):
                print(f"[Image {idx:03d}/{total:03d}] Success via Azure.")
                success = True
            else:
                print(f"[Image {idx:03d}/{total:03d}] Azure image failed verification.")
        except Exception as e:
            print(f"[Image {idx:03d}/{total:03d}] Azure failed: {str(e)[:120]}")

        # 2. المحاولة الثانية: البديل Pollinations في حال فشل مايكروسوفت
        if not success:
            print(f"[Image {idx:03d}/{total:03d}] Falling back to Pollinations.ai...")
            try:
                image_bytes = generate_via_pollinations(prompt)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                if verify_image(image_path):
                    print(f"[Image {idx:03d}/{total:03d}] Success via Pollinations fallback.")
                    success = True
                else:
                    print(f"[Image {idx:03d}/{total:03d}] Pollinations image failed verification.")
            except Exception as e:
                print(f"[Image {idx:03d}/{total:03d}] Pollinations failed: {str(e)[:120]}")

        if not success:
            print(f"[ERROR] Could not generate valid image for index {idx:03d} from any source.")

        # مهلة ثانيتين بين كل صورة لتفادي إجهاد السيرفرات أو تجاوز المعدل
        time.sleep(2)

    print(f"--- [Stage 2.5] Finished rendering images. Stored in: {images_dir} ---")
