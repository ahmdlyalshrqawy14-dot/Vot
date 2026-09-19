import os
import json
import sys
import time
import urllib.parse
import requests
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. إعداد عملاء جوجل للمفاتيح المتوفرة
# ---------------------------------------------------------------------------
def init_google_clients():
    keys = []
    # جلب المفاتيح الأربعة من البيئة
    env_names = ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]
    for name in env_names:
        key = os.getenv(name)
        if key and key.strip():
            keys.append((name, key.strip()))
            
    if not keys:
        print("Warning: No Google API keys found in environment.")
        return []
        
    print(f"Loaded {len(keys)} Google API keys into active pool.")
    return keys

# ---------------------------------------------------------------------------
# 2. محركات التوليد (Google -> Pollinations -> Hugging Face)
# ---------------------------------------------------------------------------
def try_google_generate(api_key: str, prompt: str) -> bytes:
    client = genai.Client(api_key=api_key)
    models = ["imagen-3.0-generate-002", "imagen-3.0-fast-generate-001"]
    
    last_err = None
    for model_name in models:
        try:
            result = client.models.generate_images(
                model=model_name,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="16:9",
                    output_mime_type="image/png"
                )
            )
            if result.generated_images:
                return result.generated_images[0].image.image_bytes
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Google generation failed: {last_err}")

def try_pollinations_generate(prompt: str) -> bytes:
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&model=flux&nologo=true&seed={int(time.time() * 1000) % 100000}"
    response = requests.get(url, timeout=45)
    if response.status_code == 200:
        return response.content
    raise RuntimeError(f"Pollinations error {response.status_code}")

def try_huggingface_generate(hf_token: str, prompt: str) -> bytes:
    api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {"num_inference_steps": 4, "aspect_ratio": "16:9"}
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if response.status_code == 200:
        return response.content
    raise RuntimeError(f"Hugging Face error {response.status_code}: {response.text}")

# ---------------------------------------------------------------------------
# 3. حفظ وضبط أبعاد الصورة (16:9)
# ---------------------------------------------------------------------------
def save_processed_image(image_bytes: bytes, output_path: str):
    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    image = image.resize((1280, 720), Image.Resampling.LANCZOS)
    image.save(output_path, "PNG", quality=95)

# ---------------------------------------------------------------------------
# 4. إدارة شبكة الإنقاذ الذاتية لكل صورة
# ---------------------------------------------------------------------------
def process_single_image(task, google_keys, hf_token):
    item_id = task["id"]
    prompt = task["prompt"]
    out_path = task["output_path"]
    
    if os.path.exists(out_path):
        print(f"[{item_id}] Already exists. Skipping.")
        return True

    # 1. محاولة التوليد عبر تجمع جوجل (Google Pool)
    if google_keys:
        # تحديد المفتاح الأساسي بالدور
        start_idx = (item_id if isinstance(item_id, int) else 0) % len(google_keys)
        # ترتيب المحاولات بدءاً من المفتاح المخصص ثم باقي المفاتيح تلقائياً
        ordered_keys = google_keys[start_idx:] + google_keys[:start_idx]
        
        for key_name, key_val in ordered_keys:
            try:
                img_bytes = try_google_generate(key_val, prompt)
                save_processed_image(img_bytes, out_path)
                print(f"[{item_id}] SUCCESS -> Generated via Google ({key_name})")
                return True
            except Exception as e:
                print(f"[{item_id}] Google ({key_name}) busy/failed. Transferring to next Google key...")
                time.sleep(0.5)

    # 2. خط الدفاع الثاني: Pollinations.ai
    try:
        print(f"[{item_id}] All Google keys exhausted. Falling back to Pollinations (FLUX)...")
        img_bytes = try_pollinations_generate(prompt)
        save_processed_image(img_bytes, out_path)
        print(f"[{item_id}] SUCCESS -> Generated via Pollinations")
        return True
    except Exception as e:
        print(f"[{item_id}] Pollinations failed: {e}")

    # 3. خط الدفاع الثالث والأخير: Hugging Face
    if hf_token:
        try:
            print(f"[{item_id}] Falling back to Hugging Face (FLUX Schnell)...")
            img_bytes = try_huggingface_generate(hf_token, prompt)
            save_processed_image(img_bytes, out_path)
            print(f"[{item_id}] SUCCESS -> Generated via Hugging Face")
            return True
        except Exception as e:
            print(f"[{item_id}] Hugging Face failed: {e}")

    print(f"[{item_id}] FATAL: All engines failed for this image.")
    return False

# ---------------------------------------------------------------------------
# 5. الدالة الرئيسية وإدارة التوازي المتزامن
# ---------------------------------------------------------------------------
def main():
    if not os.path.exists("current_episode.json"):
        print("Error: current_episode.json not found.")
        sys.exit(1)
        
    with open("current_episode.json", "r", encoding="utf-8") as f:
        episode = json.load(f)
        
    google_keys = init_google_clients()
    hf_token = os.getenv("HF_TOKEN")
    
    os.makedirs("images/timeline", exist_ok=True)
    os.makedirs("images/thumbnails", exist_ok=True)
    
    tasks = []
    
    # مهام المشاهد (Timeline)
    for item in episode.get("timeline", []):
        tasks.append({
            "id": item["id"],
            "prompt": item["image_prompt"],
            "output_path": f"images/timeline/{item['id']:03d}.png"
        })
        
    # مهام الثامبنيل (Thumbnails)
    for idx, thumb in enumerate(episode.get("thumbnails", [])):
        angle = thumb.get("angle", f"angle_{idx+1}")
        tasks.append({
            "id": f"thumb_{angle}",
            "prompt": thumb.get("prompt", ""),
            "output_path": f"images/thumbnails/thumb_{angle}.png"
        })

    print(f"\n=======================================================")
    print(f"Starting Multi-Engine Parallel Production for {len(tasks)} images")
    print(f"Concurrency: {max(len(google_keys), 4)} Workers in parallel")
    print(f"=======================================================\n")
    
    start_time = time.time()
    max_workers = max(len(google_keys), 4)
    failed_count = 0
    
    # تشغيل المسارات المتوازية
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, task, google_keys, hf_token): task for task in tasks}
        for future in as_completed(futures):
            success = future.result()
            if not success:
                failed_count += 1
                
    elapsed = round(time.time() - start_time, 2)
    print(f"\n=======================================================")
    print(f"Image generation finished in {elapsed}s.")
    print(f"Total: {len(tasks)} | Successful: {len(tasks) - failed_count} | Failed: {failed_count}")
    print(f"=======================================================")
    
    if failed_count > 0:
        print(f"Error: {failed_count} images failed to generate.")
        sys.exit(1)

if __name__ == "__main__":
    main()
