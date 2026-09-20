import os
import time
from google import genai
from google.genai import types

# قائمة أولويات النماذج بالترتيب الصارم المحدد
MODEL_FALLBACK_LIST = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

# خريطة توزيع المفاتيح الأربعة:
# المفتاح الثاني هو العادي (Standard) وخُصص للمرحلة الرابعة (Metadata)
KEY_MAPPING = {
    "script": os.getenv("GEMINI_API_KEY"),         # Stage 1 (Pro)
    "images": os.getenv("GEMINI_API_KEY_3"),       # Stage 2 (Pro)
    "audio": os.getenv("GEMINI_API_KEY_4"),        # Stage 3 (Pro)
    "metadata": os.getenv("GEMINI_API_KEY_2")      # Stage 4 (Standard)
}

def generate_with_fallback(stage_name: str, system_prompt: str, user_prompt: str) -> str:
    """
    استدعاء النموذج مع التبديل التلقائي عبر قائمة النماذج الستة عند حدوث أي خطأ
    """
    api_key = KEY_MAPPING.get(stage_name)
    if not api_key:
        raise ValueError(f"API Key for stage '{stage_name}' is missing in environment variables.")

    client = genai.Client(api_key=api_key)

    last_error = None
    for model_name in MODEL_FALLBACK_LIST:
        try:
            print(f"[{stage_name.upper()}] Attempting with model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                )
            )
            if response.text:
                print(f"[{stage_name.upper()}] Successfully generated using: {model_name}")
                return response.text.strip()
        except Exception as e:
            print(f"[{stage_name.upper()}] Model {model_name} failed: {str(e)}")
            last_error = e
            time.sleep(2)  # فاصل زمني ثانيتين قبل محاولة النموذج التالي في القائمة

    raise RuntimeError(f"All 6 fallback models failed for stage '{stage_name}'. Last error: {last_error}")
