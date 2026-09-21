import json
import logging
import requests
from typing import Dict, Any
from config import GEMINI_KEYS, GEMINI_MODELS, REQUEST_TIMEOUT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("GeminiEngine")


def _format_model_name(model_name: str) -> str:
    clean = model_name.strip().lower().replace(" ", "-")
    if not clean.startswith("gemini-"):
        clean = f"gemini-{clean}"
    return clean


def call_gemini_with_fallback(system_instruction: str, user_prompt: str) -> str:
    """
    استدعاء Gemini عبر المفاتيح والنماذج المتعاقبة.
    """
    if not GEMINI_KEYS:
        raise ValueError("خطأ: لم يتم العثور على أي مفتاح GEMINI في متغيرات البيئة!")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.7
        }
    }

    # حلقة المفاتيح (صمامات الأمان)
    for key_index, api_key in enumerate(GEMINI_KEYS, start=1):
        masked_key = f"{api_key[:6]}...{api_key[-4:]}" if len(api_key) > 10 else "***"
        logger.info(f"🔑 تجربة المفتاح رقم [{key_index}] ({masked_key})")

        # حلقة النماذج الستة بالترتيب
        for model_index, raw_model in enumerate(GEMINI_MODELS, start=1):
            model_id = _format_model_name(raw_model)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

            logger.info(f"   ⏳ تجربة النموذج ({model_index}/{len(GEMINI_MODELS)}): '{raw_model}' [{model_id}]")

            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )

                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            logger.info(f"   ✅ تم النجاح بالمفتاح [{key_index}] والنموذج '{raw_model}'!")
                            return parts[0]["text"]

                logger.warning(f"   ⚠️ فشل الطلب (رمز {response.status_code}): {response.text[:160]}")

            except requests.exceptions.RequestException as exc:
                logger.warning(f"   ⚠️ استثناء شبكة أثناء الاتصال بالنموذج '{raw_model}': {exc}")

        logger.warning(f"🚨 استُنفدت جميع النماذج على المفتاح [{key_index}]. الانتقال التلقائي للمفتاح التالي...")

    raise RuntimeError("❌ خطأ فادح: فشلت كافة المحاولات عبر جميع المفاتيح والنماذج المتاحة!")
