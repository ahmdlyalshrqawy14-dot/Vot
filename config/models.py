import os
import time
from google import genai
from google.genai import types

# قائمة أولويات النماذج مع الانتقال للبديل تلقائياً
MODEL_FALLBACK_LIST = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b"
]

# خريطة المفاتيح حسب الدور والمرحلة
# المفتاح 2 هو العادي (Standard) مخصص للمرحلة الرابعة
KEY_MAPPING = {
    "script": os.getenv("GEMINI_API_KEY"),         # Stage 1 (Pro)
    "images": os.getenv("GEMINI_API_KEY_3"),       # Stage 2 (Pro)
    "audio": os.getenv("GEMINI_API_KEY_4"),        # Stage 3 (Pro)
    "metadata": os.getenv("GEMINI_API_KEY_2")      # Stage 4 (Standard)
}

def generate_with_fallback(stage_name: str, system_prompt: str, user_prompt: str) -> str:
    """
    استدعاء Gemini مع التبديل التلقائي بين النماذج وتوزيع المفاتيح
    """
    api_key = KEY_MAPPING.get(stage_name)
    if not api_key:
        raise ValueError(f"API Key for stage '{stage_name}' is not set in environment variables.")

    client = genai.Client(api_key=api_key)

    last_error = None
    for model_name in MODEL_FALLBACK_LIST:
        try:
            print(f"[{stage_name.upper()}] Trying model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                )
            )
            if response.text:
                print(f"[{stage_name.upper()}] Success with: {model_name}")
                return response.text.strip()
        except Exception as e:
            print(f"[{stage_name.upper()}] Failed with {model_name}: {str(e)}")
            last_error = e
            time.sleep(2)  # انتظار قصير قبل تجربة النموذج التالي

    raise RuntimeError(f"All models failed for stage '{stage_name}'. Last error: {last_error}")
