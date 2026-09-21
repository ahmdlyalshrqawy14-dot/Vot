import time
import json
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

# قفل لمنع التضارب بين العمليات المتوازية (Thread-Safety)
_lock = threading.Lock()
_request_counter = 0

# ذاكرة استبعاد النماذج المضغوطة 503
_model_cooldown: Dict[str, float] = {}
COOLDOWN_DURATION = 300  # 5 دقائق

# ذاكرة أفضل نموذج شغال حالياً
_fastest_model = None


def _format_model_name(model_name: str) -> str:
    clean = model_name.strip().lower().replace(" ", "-")
    if not clean.startswith("gemini-"):
        clean = f"gemini-{clean}"
    return clean


def _get_next_key_index() -> int:
    """توزيع الحمل بالتساوي: إعطاء كل طلب جديد مفتاحاً مختلفاً (Round-Robin)"""
    global _request_counter
    with _lock:
        idx = _request_counter % len(GEMINI_KEYS)
        _request_counter += 1
        return idx


def _get_active_models() -> List[str]:
    """جلب النماذج مع استبعاد المعزولة مؤقتاً وتقديم النموذج السريع"""
    now = time.time()
    with _lock:
        models = [m for m in GEMINI_MODELS if now > _model_cooldown.get(m, 0)]
        if not models:
            models = list(GEMINI_MODELS)
        
        # وضع النموذج الشغال أولاً دائماً
        if _fastest_model and _fastest_model in models:
            models.remove(_fastest_model)
            models.insert(0, _fastest_model)
        return models


def call_gemini_with_fallback(
    system_instruction: str, 
    user_prompt: str, 
    response_mime_type: str = "application/json"
) -> str:
    """
    استدعاء واجهة Gemini بموزع أحمال تناوبي (Load Balancer)
    مع صمام أمان Fallback متكامل.
    """
    global _fastest_model

    if not GEMINI_KEYS:
        raise ValueError("خطأ: لم يتم العثور على أي مفتاح GEMINI في ملف .env!")

    generation_config = {"temperature": 0.7}
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type

    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": generation_config
    }

    # تحديد المفتاح المخصص لهذا الطلب بتناوب عادل
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

            logger.info(f"⚖️ [موزع الأحمال] مفتاح [{current_key_idx + 1}] ({masked_key}) ➔ نموذج '{raw_model}'")

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
                            logger.info(f"✅ نجاح الطلب عبر مفتاح [{current_key_idx + 1}] والنموذج '{raw_model}'!")
                            return parts[0]["text"]

                # في حال ضغط الخوادم (503)، عزل النموذج فوراً
                if response.status_code == 503:
                    logger.warning(f"⚠️ النموذج '{raw_model}' عليه ضغط (503). عزله لمدة 5 دقائق.")
                    with _lock:
                        _model_cooldown[raw_model] = time.time() + COOLDOWN_DURATION
                else:
                    logger.warning(f"⚠️ كود {response.status_code} على مفتاح [{current_key_idx + 1}]: {response.text[:120]}")

            except requests.exceptions.RequestException as exc:
                logger.warning(f"⚠️ تعثر اتصال بالنموذج '{raw_model}': {exc}")

        logger.warning(f"🚨 تحويل الطلب تلقائياً لمفتاح بديل بعد تعثر مفتاح [{current_key_idx + 1}]...")

    raise RuntimeError("❌ فشلت كافة المفاتيح والنماذج في تلبية الطلب!")
