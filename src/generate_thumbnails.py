import os
import sys
import time
import requests
from io import BytesIO
from PIL import Image
from openai import AzureOpenAI

# البرومبت الأساسي الحرفي من الدليل (المتغيران فقط: اللون والحركة)
BASE_TEMPLATE = (
    "Create a 2D muscular character illustration in the exact style of the provided example. "
    "The figure should be {color}, with a smooth head, large white oval eyes, no mouth, and a defined muscular body wearing black shorts. "
    "Maintain the same proportions and facial features as in the reference. "
    "{pose} "
    "Background must be plain white. Clean, cel-shaded, and expressive style. Keep the style consistent across all generated images."
)

COLOR = "orange"

# الحركات الثلاث وفق الزوايا التسويقية المحددة في الدليل
THUMBNAIL_POSES = {
    # 1. زاوية الفضول (Curiosity)
    "thumb_curiosity": (
        "Leaning forward aggressively toward the viewer, one hand scratching the smooth bald head in utter shock and confusion, "
        "eyes wide open with extreme curiosity, other hand pointing downward."
    ),
    # 2. زاوية الوجع (Pain Point)
    "thumb_pain_point": (
        "Collapsed down on one knee in intense physical agony, clutching both hands tightly around the knee joint, "
        "body hunched over expressing severe distress and pain."
    ),
    # 3. زاوية المكسب والنتيجة (Outcome / Gain)
    "thumb_outcome_gain": (
        "Standing tall and proud in an explosive, powerful double-bicep victory flex pose, "
        "chest puffed out, radiating complete dominance, peak physical health, and celebration."
    )
}

def init_azure():
    endpoint = os.getenv("AZURE_IMAGE_ENDPOINT")
    api_key = os.getenv("AZURE_IMAGE_KEY")
    deployment = os.getenv("AZURE_IMAGE_DEPLOYMENT", "gpt-image-2.5-flare")

    if not endpoint or not api_key:
        print("[ERROR] Missing Azure credentials in environment.")
        sys.exit(1)

    return AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version="2024-02-01"), deployment

def generate_thumbnail(client, deployment, prompt, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            res = client.images.generate(
                model=deployment,
                prompt=prompt,
                size="1024x1024",
                quality="hd",
                n=1
            )
            url = res.data[0].url
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content)).convert("RGB")
                return img
        except Exception as e:
            print(f"[WARN] Attempt {attempt} failed: {e}")
            time.sleep(attempt * 5)
    return None

def main():
    os.makedirs("images/thumbnails", exist_ok=True)
    client, deployment = init_azure()

    print("--- [START] Generating 3 YouTube Thumbnails (PDF Guide Compliant) ---")

    for key, pose in THUMBNAIL_POSES.items():
        output_path = f"images/thumbnails/{key}.png"

        # التحقق من وجود الصورة وصحتها مسبقاً
        if os.path.exists(output_path) and os.path.getsize(output_path) > 10240:
            print(f"[SKIP] {key} already exists.")
            continue

        full_prompt = BASE_TEMPLATE.format(color=COLOR, pose=pose)
        print(f"\n[REQUEST] Generating {key}...")

        img = generate_thumbnail(client, deployment, full_prompt)
        if img:
            # تحويل الأبعاد إلى 1280x720 القياسية ليوتيوب بنقاء عالي
            img_resized = img.resize((1280, 720), Image.Resampling.LANCZOS)
            img_resized.save(output_path, "PNG", quality=95)
            print(f"[SUCCESS] Saved: {output_path}")
        else:
            print(f"[ERROR] Failed to generate {key} after retries.")

        # راحة أمان لتجنب تجاوز الـ Rate Limit
        time.sleep(5)

    print("\n--- [DONE] Thumbnails pipeline finished. ---")

if __name__ == "__main__":
    main()
