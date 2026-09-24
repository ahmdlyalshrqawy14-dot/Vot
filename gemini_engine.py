import base64
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from config import GEMINI_KEYS, GEMINI_MODELS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("GeminiEngine")


FAST_TIMEOUT = 25
COOLDOWN_DURATION = 300

_lock = threading.Lock()
_request_counter = 0
_model_cooldown: Dict[str, float] = {}
_fastest_model: Optional[str] = None


# ---------------------------------------------------------------------------
# Configuration and state helpers
# ---------------------------------------------------------------------------


def _get_valid_keys() -> List[str]:
    """Return non-empty, stripped API keys without changing their order."""
    result: List[str] = []
    for key in GEMINI_KEYS or []:
        if isinstance(key, str):
            key = key.strip()
            if key:
                result.append(key)
    return result


def _get_configured_models() -> List[str]:
    """Return non-empty configured model names without inventing models."""
    result: List[str] = []
    for model in GEMINI_MODELS or []:
        if isinstance(model, str) and model.strip():
            result.append(model.strip())
    return result


def _format_model_name(model_name: str) -> str:
    clean = model_name.strip().lower().replace(" ", "-")
    if not clean.startswith("gemini-"):
        clean = f"gemini-{clean}"
    return clean


def _mask_key(api_key: str) -> str:
    """Return a safe log representation; never log the complete key."""
    if not api_key or len(api_key) <= 10:
        return "***"
    return f"{api_key[:6]}...{api_key[-4:]}"


def _get_next_key_index(total_keys: int) -> int:
    """Return a fair, thread-safe starting index for the next request."""
    global _request_counter
    if total_keys <= 0:
        return 0
    with _lock:
        index = _request_counter % total_keys
        _request_counter += 1
        return index


def _get_active_models() -> List[str]:
    """Return models in configured order, preferring the last successful model."""
    global _fastest_model
    configured = _get_configured_models()
    now = time.time()

    with _lock:
        active = [
            model
            for model in configured
            if now >= _model_cooldown.get(model, 0)
        ]

        # If every configured model is cooling down, do not deadlock the engine.
        if not active:
            active = list(configured)

        if _fastest_model and _fastest_model in active:
            active.remove(_fastest_model)
            active.insert(0, _fastest_model)

        return active


def _apply_model_cooldown(raw_model: str) -> None:
    with _lock:
        _model_cooldown[raw_model] = time.time() + COOLDOWN_DURATION


def _mark_fastest_model(raw_model: str) -> None:
    global _fastest_model
    with _lock:
        _fastest_model = raw_model


# ---------------------------------------------------------------------------
# Response and logging helpers
# ---------------------------------------------------------------------------


def _extract_text_from_response(data: Any) -> Optional[str]:
    """Extract non-empty Gemini text without altering the returned content."""
    if not isinstance(data, dict):
        return None

    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text

    return None


def _parse_json_response(response: requests.Response) -> Optional[Any]:
    try:
        return response.json()
    except (ValueError, TypeError):
        return None


def _log_attempt(
    key_index: int,
    masked_key: str,
    model_id: str,
    status: Optional[int],
    category: str,
    detail: str = "",
) -> None:
    message = (
        f"[Gemini] key#{key_index} ({masked_key}) "
        f"model={model_id} status={status} category={category}"
    )
    if detail:
        message += f" detail={detail}"
    logger.warning(message)


# ---------------------------------------------------------------------------
# Shared fallback executor
# ---------------------------------------------------------------------------


def _execute_fallback(payload: Dict[str, Any], operation: str) -> str:
    """Try each key/model combination once and return the first usable text."""
    keys = _get_valid_keys()
    if not keys:
        raise ValueError("خطأ: لم يتم العثور على مفاتيح GEMINI صالحة في ملف .env!")

    models = _get_configured_models()
    if not models:
        raise ValueError("خطأ: لم يتم العثور على نماذج GEMINI صالحة في config.py!")

    start_index = _get_next_key_index(len(keys))
    failures: List[str] = []

    for key_offset in range(len(keys)):
        key_index = (start_index + key_offset) % len(keys)
        api_key = keys[key_index]
        masked_key = _mask_key(api_key)
        models_pool = _get_active_models()

        for raw_model in models_pool:
            model_id = _format_model_name(raw_model)
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_id}:generateContent?key={api_key}"
            )

            try:
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=FAST_TIMEOUT,
                )
            except requests.exceptions.Timeout:
                _log_attempt(key_index, masked_key, model_id, None, "timeout")
                failures.append("timeout")
                continue
            except requests.exceptions.RequestException as exc:
                _log_attempt(
                    key_index,
                    masked_key,
                    model_id,
                    None,
                    "network_error",
                    type(exc).__name__,
                )
                failures.append("network_error")
                continue

            status = getattr(response, "status_code", None)

            if status == 200:
                data = _parse_json_response(response)
                if data is None:
                    _log_attempt(key_index, masked_key, model_id, status, "invalid_json")
                    failures.append("invalid_json")
                    continue

                text = _extract_text_from_response(data)
                if text is None:
                    _log_attempt(key_index, masked_key, model_id, status, "no_usable_text")
                    failures.append("no_usable_text")
                    continue

                _mark_fastest_model(raw_model)
                logger.info(
                    f"[Gemini] {operation} success via key#{key_index} "
                    f"model={model_id}"
                )
                return text

            if status in (401, 403):
                category = "unauthorized" if status == 401 else "forbidden"
                _log_attempt(key_index, masked_key, model_id, status, category)
                failures.append(category)
                # An unauthorized key is not worth trying with other models.
                break

            if status == 404:
                _log_attempt(key_index, masked_key, model_id, status, "model_not_found")
                failures.append("model_not_found")
                continue

            if status == 429:
                _log_attempt(key_index, masked_key, model_id, status, "rate_limited")
                failures.append("rate_limited")
                continue

            if status == 400:
                _log_attempt(key_index, masked_key, model_id, status, "bad_request")
                failures.append("bad_request")
                continue

            if status == 503:
                _log_attempt(key_index, masked_key, model_id, status, "unavailable")
                failures.append("unavailable")
                _apply_model_cooldown(raw_model)
                continue

            if status in (500, 502, 504):
                _log_attempt(key_index, masked_key, model_id, status, "server_error")
                failures.append("server_error")
                continue

            _log_attempt(key_index, masked_key, model_id, status, "unknown_status")
            failures.append("unknown_status")

    categories = sorted(set(failures)) if failures else ["unknown"]
    raise RuntimeError(
        f"❌ فشلت كافة المفاتيح والنماذج في تلبية طلب {operation}. "
        f"categories={categories}"
    )


# ---------------------------------------------------------------------------
# Public API — signatures intentionally unchanged
# ---------------------------------------------------------------------------


def call_gemini_with_fallback(
    system_instruction: str,
    user_prompt: str,
    response_mime_type: str = "application/json",
) -> str:
    generation_config: Dict[str, Any] = {"temperature": 0.7}
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type

    payload: Dict[str, Any] = {
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]}
        ],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": generation_config,
    }

    return _execute_fallback(payload, operation="text")


def call_gemini_vision_with_fallback(
    image_bytes: bytes,
    mime_type: str,
    user_prompt: str,
) -> str:
    """Analyze an image through the same safe key/model fallback engine."""
    if not image_bytes:
        raise ValueError("لا توجد بيانات صورة لتحليلها.")
    if not isinstance(mime_type, str) or not mime_type.strip():
        raise ValueError("نوع MIME للصورة غير صالح أو فارغ.")

    try:
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"فشل ترميز الصورة: {type(exc).__name__}") from exc

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type.strip(),
                            "data": encoded_image,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {"temperature": 0.1},
    }

    return _execute_fallback(payload, operation="vision")
