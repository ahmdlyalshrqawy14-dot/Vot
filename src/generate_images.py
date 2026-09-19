import os
import json
import sys
import time
import base64
import urllib.parse
import requests
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from huggingface_hub import InferenceClient

# النماذج المعتمدة من جوجل بالترتيب المطلوب
GOOGLE_MODELS_PRIORITY = [
    "gemini-3.1-flash-lite-image",
    "gemini-3.1-flash-image"
]

# ---------------------------------------------------------------------------
# 1. إعداد مفاتيح جوجل
# ---------------------------------------------------------------------------
def init_google_clients():
    keys = []
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
    last_err = None
    
    for model_name in GOOGLE_MODELS_PRIORITY:
        try:
            # استدعاء generate_content بالصيغة المعتمدة لإنتاج الصور
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="16:9"
                    )
                )
            )
            
            # استخراج بايتات الصورة من استجابة الموديل
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.inline_data and part.inline_data.data:
                                data = part.inline_data.data
                                if isinstance(data, str):
                                    return base64.b64decode(data)
                                return data
                                
            if hasattr(response, "parts") and response.parts:
                for part in response.parts:
                    if part.inline_data and part.inline_data.data:
                        data = part.inline_data.data
                        if isinstance(data, str):
                            return base64.b64decode(data)
                        return data
                        
        except Exception as e:
            last_err = e
            continue
            
    raise RuntimeError(f"Google error: {last_err}")

def try_pollinations_generate(prompt: str) -> bytes:
    encoded = urllib.parse.quote(prompt)
    seed = int(time.time() * 1000) % 999999
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&model=flux&nologo=true&seed={seed}"
    
    # محاولات متعددة مع تأخير لتفادي الـ Rate Limit 429
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=45)
            if response.status_code == 200:
                return response.content
            if response.status_code == 429:
                time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2)
            
    raise RuntimeError("Pollinations rate limit or server error.")

def try_huggingface_generate(hf_token: str, prompt: str) -> bytes:
    """استدعاء مكتبة Hugging Face الرسمية لحل مشاكل الـ DNS"""
    client = InferenceClient(api_key=hf_token)
    image = client.text_to_image(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell"
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

# ---------------------------------------------------------------------------
# 3. حفظ وضبط أبعاد الصورة (16:9)
# ---------------------------------------------------------------------------
def save_processed_image(image_bytes: bytes, output_path: str):
    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    image = image.resize((1280, 720), Image.Resampling.LANCZOS)
    image.save(output_path, "PNG", quality=95)

# ---------------------------------------------------------------------------
# 4. شبكة الإنقاذ والتوزيع الذكي
# ---------------------------------------------------------------------------
def process_single_image(task, google_keys, hf_token):
    item_id = task["id"]
    prompt = task["prompt"]
    out_path = task["output_path"]
    
    if os.path.exists(out_path):
        return True

    # 1. محاولة جوجل عبر المفاتيح الأربعة والنماذج بالترتيب
    if google_keys:
        start_idx = (item_id if isinstance(item_id, int) else 0) % len(google_keys)
        ordered_keys = google_keys[start_idx:] + google_keys[:start_idx]
        
        for key_name, key_val in ordered_keys:
            try:
                img_bytes = try_google_generate(key_val, prompt)
                save_processed_image(img_bytes, out_path)
                print(f"[{item_id}] SUCCESS -> Generated via Google ({key_name})")
                return True
            except Exception as e:
                print(f"[{item_id}] Google ({key_name}) failed: {e}")

    # 2. خط الدفاع الثاني: Pollinations.ai
    try:
        print(f"[{item_id}] Trying Pollinations (FLUX)...")
        img_bytes = try_pollinations_generate(prompt)
        save_processed_image(img_bytes, out_path)
        print(f"[{item_id}] SUCCESS -> Generated via Pollinations")
        return True
    except Exception as e:
        print(f"[{item_id}] Pollinations failed: {e}")

    # 3. خط الدفاع الثالث: Hugging Face
    if hf_token:
        try:
            print(f"[{item_id}] Trying Hugging Face (FLUX Schnell)...")
            img_bytes = try_huggingface_generate(hf_token, prompt)
            save_processed_image(img_bytes, out_path)
            print(f"[{item_id}] SUCCESS -> Generated via Hugging Face")
            return True
        except Exception as e:
            print(f"[{item_id}] Hugging Face failed: {e}")

    print(f"[{item_id}] FATAL: All engines failed.")
    return False

# ---------------------------------------------------------------------------
# 5. الدالة الرئيسية
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
    for item in episode.get("timeline", []):
        tasks.append({
            "id": item["id"],
            "prompt": item["image_prompt"],
            "output_path": f"images/timeline/{item['id']:03d}.png"
        })
        
    for idx, thumb in enumerate(episode.get("thumbnails", [])):
        angle = thumb.get("angle", f"angle_{idx+1}")
        tasks.append({
            "id": f"thumb_{angle}",
            "prompt": thumb.get("prompt", ""),
            "output_path": f"images/thumbnails/thumb_{angle}.png"
        })

    print(f"\n=======================================================")
    print(f"Starting Multi-Engine Parallel Production for {len(tasks)} images")
    print(f"Primary Engine: Google ({', '.join(GOOGLE_MODELS_PRIORITY)})")
    print(f"Active Google Keys: {len(google_keys)}")
    print(f"=======================================================\n")
    
    start_time = time.time()
    failed_count = 0
    
    # 4 مسارات متزامنة للعمل المتوازي
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_single_image, task, google_keys, hf_token): task for task in tasks}
        for future in as_completed(futures):
            if not future.result():
                failed_count += 1
                
    elapsed = round(time.time() - start_time, 2)
    print(f"\n=======================================================")
    print(f"Completed in {elapsed}s | Success: {len(tasks) - failed_count} | Failed: {failed_count}")
    print(f"=======================================================")
    
    if failed_count > 0:
        print(f"Error: {failed_count} images failed to generate.")
        sys.exit(1)

if __name__ == "__main__":
    main()
