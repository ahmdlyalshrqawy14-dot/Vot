import os
import time
import json
import re
from google import genai
from google.genai import types

MODEL_FALLBACK_LIST = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

KEY_MAPPING = {
    "script": os.getenv("GEMINI_API_KEY"),
    "metadata": os.getenv("GEMINI_API_KEY_2")
}

def generate_text_fallback(stage_name: str, system_prompt: str, user_prompt: str) -> str:
    """استدعاء الموديل للنصوص العادية مع التبديل التلقائي"""
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
            time.sleep(2)

    raise RuntimeError(f"All fallback models failed for {stage_name}. Last error: {last_error}")

def generate_json_fallback(stage_name: str, system_prompt: str, user_prompt: str) -> dict:
    """استدعاء الموديل وإجبار المخرجات على هيئة كائن JSON نظيف تماماً"""
    raw_text = generate_text_fallback(stage_name, system_prompt, user_prompt)
    
    # استخراج كود JSON الصافي حتى لو وضع الموديل ماركداون حوله
    json_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw_text, re.DOTALL)
    clean_str = json_match.group(1).strip() if json_match else raw_text.strip()
    
    try:
        return json.loads(clean_str)
    except json.JSONDecodeError:
        # محاولة أخيرة للتنظيف
        start = clean_str.find("{")
        end = clean_str.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(clean_str[start:end])
        raise ValueError(f"Failed to parse JSON response: {clean_str[:200]}")
