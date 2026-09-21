import time
import json
import base64
import logging
import threading
import requests
from typing import Dict, Any, List, Optional
from config import GEMINI_KEYS, GEMINI_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("GeminiEngine")

FAST_TIMEOUT = 25  # ثانية

_lock = threading.Lock()
_request_counter = 0
_model_cooldown: Dict[str, float] = {}
COOLDOWN_DURATION = 300
_fastest_model = None


def _format_model_name(model_name: str) -> str:
    clean = model_name.strip().lower().replace(" ", "-")
    if not clean.startswith("gemini-"):
        clean = f"gemini-{clean}"
    return clean


def _get_next_key_index() -> int:
    global _request_counter
    with _lock:
        idx = _request_counter % len(GEMINI_KEYS)
        _request_counter += 1
        return idx


def _get_active_models() -> List[str]:
    now = time.time()
    with _lock:
        models = [m for m in GEMINI_MODELS if now > _model_cooldown.get(m, 0)]
        if not models:
            models = list(GEMINI_MODELS)
        if _fastest_model and _fastest_model in models:
            models.remove(_fastest_model)
            models.insert(0, _fastest_model)
        return models


def call_gemini_with_fallback(
    system_instruction: str, 
    user_prompt: str, 
    response_mime_type: str = "application/json"
) -> str:
    global _fastest_model

    if not GEMINI_KEYS:
        raise ValueError("خطأ: لم يتم العثور على مفاتيح GEMINI في ملف .env!")

    generation_config = {"temperature": 0.7}
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type

    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": generation_config
    }

    starting_key_idx = _get_next_key_index()
    total_keys = len(GEMINI_KEYS)

    for attempt in range(total_keys):
        current_key_idx = (starting_key_idx + attempt) % total_keys
        api_key = GEMINI_KEYS[current_key_idx]
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
        models_pool = _get_active_models()

        for raw_model in models_pool:
            model_id = _format_model_name(raw_model)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=FAST_TIMEOUT
                )

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            with _lock:
                                _fastest_model = raw_model
                            return parts[0]["text"]

                if response.status_code == 503:
                    with _lock:
                        _model_cooldown[raw_model] = time.time() + COOLDOWN_DURATION

            except requests.exceptions.RequestException:
                pass

    raise RuntimeError("❌ فشلت كافة المفاتيح والنماذج في تلبية الطلب!")


def call_gemini_vision_with_fallback(image_bytes: bytes, mime_type: str, user_prompt: str) -> str:
    """
    استدعاء Gemini Vision لتحليل وفحص الصور بدقة عالية.
    """
    global _fastest_model

    if not GEMINI_KEYS:
        raise ValueError("خطأ: لم يتم العثور على مفاتيح GEMINI في ملف .env!")

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": b64_image
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1  # درجة حرارة منخفضة جداً للدقة في قراءة الأرقام
        }
    }

    starting_key_idx = _get_next_key_index()
    total_keys = len(GEMINI_KEYS)

    for attempt in range(total_keys):
        current_key_idx = (starting_key_idx + attempt) % total_keys
        api_key = GEMINI_KEYS[current_key_idx]
        models_pool = _get_active_models()

        for raw_model in models_pool:
            model_id = _format_model_name(raw_model)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=FAST_TIMEOUT
                )

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            with _lock:
                                _fastest_model = raw_model
                            return parts[0]["text"]

                if response.status_code == 503:
                    with _lock:
                        _model_cooldown[raw_model] = time.time() + COOLDOWN_DURATION

            except requests.exceptions.RequestException:
                pass

    raise RuntimeError("❌ تعذر قراءة الصورة عبر Gemini Vision!")
