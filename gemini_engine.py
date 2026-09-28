import base64
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter

from config import GEMINI_KEYS, GEMINI_MODELS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("GeminiEngine")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

CONNECT_TIMEOUT = 5            # ثواني للاتصال
READ_TIMEOUT = 60              # ثواني لانتظار الرد
TOTAL_DEADLINE = 90            # سقف زمني كلي للطلب الواحد
MAX_ATTEMPTS = 8               # أقصى عدد محاولات للطلب الواحد
MIN_ATTEMPT_TIME = 3           # لو الوقت المتبقي أقل من كده مفيش محاولة جديدة
MAX_WAIT_FOR_RECOVERY = 3.0    # أقصى انتظار لزوج/مفتاح قرب يفوق
MAX_WAITS_PER_REQUEST = 3

BREAKER_THRESHOLD = 2          # عدد الفشل المتتالي قبل فتح الصمام
BREAKER_BASE_COOLDOWN = 20     # أول مدة راحة (تتضاعف مع كل فشل)
BREAKER_MAX_COOLDOWN = 300
PROBE_WINDOW = 30              # نافذة المحاولة الوحيدة (half-open)

DEFAULT_RETRY_DELAY = 30       # لو Gemini مرجعش retryDelay
DAILY_QUOTA_COOLDOWN = 3600    # لو الحصة اليومية خلصت
KEY_429_LONG_COOLDOWN = 60     # لو المفتاح اتضرب 429 تلات مرات ورا بعض
KEY_DEAD_INVALID = 3600        # مفتاح غلط / منتهي / 401
KEY_DEAD_FORBIDDEN = 1800      # 403 أو FAILED_PRECONDITION

EWMA_ALPHA = 0.3
BAD_REQUEST_FAIL_FAST = 2      # 400 على موديلين مختلفين = الـ payload نفسه غلط

_BLOCK_FINISH_REASONS = {
    "SAFETY",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "SPII",
    "IMAGE_SAFETY",
}


# ---------------------------------------------------------------------------
# Exceptions (كلها وارثة من RuntimeError فمفيش حاجة هتتكسر عندك)
# ---------------------------------------------------------------------------


class GeminiEngineError(RuntimeError):
    """Base error for the Gemini engine."""


class GeminiBadRequestError(GeminiEngineError):
    """The payload itself is invalid; no key can fix it."""


class GeminiSafetyBlockError(GeminiEngineError):
    """Gemini blocked the prompt/response for safety reasons."""


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


class _KeyState:
    __slots__ = (
        "cooldown_until",
        "dead_until",
        "consec_429",
        "last_used",
        "inflight",
        "last_error",
    )

    def __init__(self) -> None:
        self.cooldown_until = 0.0
        self.dead_until = 0.0
        self.consec_429 = 0
        self.last_used = 0.0
        self.inflight = 0
        self.last_error = ""


class _PairState:
    __slots__ = ("failures", "trips", "open_until")

    def __init__(self) -> None:
        self.failures = 0
        self.trips = 0
        self.open_until = 0.0


_lock = threading.Lock()
_key_states: Dict[str, _KeyState] = {}
_pair_states: Dict[Tuple[str, str], _PairState] = {}
_model_latency: Dict[str, float] = {}
_dead_models: Set[str] = set()

_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=4, pool_maxsize=16)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def _get_key_locked(api_key: str) -> _KeyState:
    state = _key_states.get(api_key)
    if state is None:
        state = _KeyState()
        _key_states[api_key] = state
    return state


def _get_pair_locked(api_key: str, model: str) -> _PairState:
    pair = (api_key, model)
    state = _pair_states.get(pair)
    if state is None:
        state = _PairState()
        _pair_states[pair] = state
    return state


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _get_valid_keys() -> List[str]:
    result: List[str] = []
    for key in GEMINI_KEYS or []:
        if isinstance(key, str):
            key = key.strip()
            if key and key not in result:
                result.append(key)
    return result


def _get_configured_models() -> List[str]:
    result: List[str] = []
    for model in GEMINI_MODELS or []:
        if isinstance(model, str):
            model = model.strip()
            if model and model not in result:
                result.append(model)
    return result


def _format_model_name(model_name: str) -> str:
    clean = model_name.strip().lower().replace(" ", "-")
    if clean.startswith("models/"):
        clean = clean[len("models/"):]
    if not clean.startswith(("gemini-", "gemma-")):
        clean = f"gemini-{clean}"
    return clean


def _mask_key(api_key: str) -> str:
    if not api_key or len(api_key) <= 10:
        return "***"
    return f"{api_key[:6]}...{api_key[-4:]}"


# ---------------------------------------------------------------------------
# Valves: key level
# ---------------------------------------------------------------------------


def _release_key(api_key: str) -> None:
    with _lock:
        state = _get_key_locked(api_key)
        state.inflight = max(0, state.inflight - 1)


def _block_key(api_key: str, seconds: float, dead: bool, reason: str) -> None:
    with _lock:
        state = _get_key_locked(api_key)
        until = time.time() + seconds
        if dead:
            state.dead_until = max(state.dead_until, until)
        else:
            state.cooldown_until = max(state.cooldown_until, until)
        state.last_error = reason


def _register_key_429(api_key: str, delay: float, daily: bool) -> None:
    """A 429 rests the pair; repeated 429s rest the whole key."""
    with _lock:
        state = _get_key_locked(api_key)
        state.consec_429 += 1
        state.last_error = "rate_limited"
        now = time.time()
        if daily:
            return
        if state.consec_429 >= 3:
            state.cooldown_until = max(
                state.cooldown_until, now + max(delay, KEY_429_LONG_COOLDOWN)
            )
        elif state.consec_429 == 2:
            state.cooldown_until = max(state.cooldown_until, now + delay)


# ---------------------------------------------------------------------------
# Valves: pair level (circuit breaker) + latency
# ---------------------------------------------------------------------------


def _observe_latency_locked(model: str, sample: float) -> None:
    previous = _model_latency.get(model)
    if previous is None:
        _model_latency[model] = sample
    else:
        _model_latency[model] = previous + EWMA_ALPHA * (sample - previous)


def _pair_success(api_key: str, model: str, latency: float) -> None:
    with _lock:
        pair = _get_pair_locked(api_key, model)
        pair.failures = 0
        pair.trips = 0
        pair.open_until = 0.0
        _observe_latency_locked(model, latency)
        key_state = _get_key_locked(api_key)
        key_state.consec_429 = 0
        key_state.last_error = ""


def _pair_failure(
    api_key: str,
    model: str,
    category: str,
    cooldown: Optional[float] = None,
    latency_sample: Optional[float] = None,
) -> None:
    with _lock:
        now = time.time()
        pair = _get_pair_locked(api_key, model)
        pair.failures += 1

        if cooldown is not None:
            pair.trips = max(pair.trips, 1)
            pair.open_until = now + cooldown
        elif pair.trips > 0 or pair.failures >= BREAKER_THRESHOLD:
            pair.trips += 1
            duration = min(
                BREAKER_BASE_COOLDOWN * (2 ** (pair.trips - 1)),
                BREAKER_MAX_COOLDOWN,
            )
            pair.open_until = now + duration

        if latency_sample is not None:
            _observe_latency_locked(model, latency_sample)

        _get_key_locked(api_key).last_error = category


def _mark_dead_model(raw_model: str) -> None:
    with _lock:
        _dead_models.add(raw_model)


# ---------------------------------------------------------------------------
# Pair selection: least-loaded healthy key, then best (fastest) model
# ---------------------------------------------------------------------------


def _pick_pair(
    keys: List[str],
    models: List[str],
    tried: Set[Tuple[str, str]],
) -> Tuple[Optional[Tuple[int, str, str]], Optional[float]]:
    now = time.time()
    earliest: Optional[float] = None

    def note(ts: float) -> None:
        nonlocal earliest
        earliest = ts if earliest is None else min(earliest, ts)

    with _lock:
        usable = [m for m in models if m not in _dead_models] or list(models)

        ranked: List[Tuple[int, float, int, str]] = []
        for index, key in enumerate(keys):
            state = _get_key_locked(key)
            ready = max(state.cooldown_until, state.dead_until)
            if now < ready:
                if any((key, m) not in tried for m in usable):
                    note(ready)
                continue
            ranked.append((state.inflight, state.last_used, index, key))
        ranked.sort()

        for _, _, index, key in ranked:
            best: Optional[Tuple[Tuple[float, int], str, _PairState]] = None
            for order, model in enumerate(usable):
                if (key, model) in tried:
                    continue
                pair = _get_pair_locked(key, model)
                if now < pair.open_until:
                    note(pair.open_until)
                    continue
                score = (_model_latency.get(model, 0.0), order)
                if best is None or score < best[0]:
                    best = (score, model, pair)

            if best is not None:
                _, model, pair = best
                if pair.trips > 0:
                    # half-open: محاولة تجريبية واحدة بس
                    pair.open_until = now + PROBE_WINDOW
                key_state = _get_key_locked(key)
                key_state.inflight += 1
                key_state.last_used = now
                return (index, key, model), None

        return None, earliest


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _extract_text_from_response(data: Any) -> Optional[str]:
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

        chunks: List[str] = []
        for part in parts:
            if not isinstance(part, dict) or part.get("thought") is True:
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)

        joined = "".join(chunks)
        if joined.strip():
            return joined

    return None


def _raise_if_blocked(data: Any) -> None:
    """Safety blocks won't change with another key/model: fail fast."""
    if not isinstance(data, dict):
        return

    feedback = data.get("promptFeedback")
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        raise GeminiSafetyBlockError(
            f"تم حجب الطلب من Gemini لأسباب أمنية: {feedback.get('blockReason')}"
        )

    candidates = data.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            reason = candidate.get("finishReason")
            if reason in _BLOCK_FINISH_REASONS:
                raise GeminiSafetyBlockError(
                    f"تم حجب الرد من Gemini لأسباب أمنية: {reason}"
                )


def _parse_json_response(response: requests.Response) -> Optional[Any]:
    try:
        return response.json()
    except (ValueError, TypeError):
        return None


def _parse_duration(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.match(r"^\s*([\d.]+)\s*s\s*$", value)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _parse_error(response: requests.Response) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "status": "",
        "message": "",
        "reason": "",
        "retry_delay": None,
        "daily": False,
    }

    data = _parse_json_response(response)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    error = data.get("error") if isinstance(data, dict) else None

    if isinstance(error, dict):
        info["status"] = str(error.get("status") or "")
        info["message"] = str(error.get("message") or "")[:300]
        details = error.get("details")
        if isinstance(details, list):
            for item in details:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get("@type", ""))
                if kind.endswith("RetryInfo"):
                    info["retry_delay"] = _parse_duration(item.get("retryDelay"))
                elif kind.endswith("ErrorInfo"):
                    info["reason"] = str(item.get("reason") or "")

    try:
        raw = (response.text or "")[:3000].lower()
    except Exception:
        raw = ""
    info["daily"] = "perday" in raw or "per day" in raw

    return info


def _is_invalid_key_error(info: Dict[str, Any]) -> bool:
    message = info["message"].lower()
    return (
        info["reason"] in ("API_KEY_INVALID", "API_KEY_EXPIRED")
        or "api key not valid" in message
        or "api key expired" in message
        or "api_key_invalid" in message
    )


def _log_attempt(
    key_index: int,
    model_id: str,
    status: Optional[int],
    category: str,
    detail: str = "",
) -> None:
    message = f"[Gemini] key#{key_index} model={model_id} status={status} category={category}"
    if detail:
        message += f" detail={detail}"
    logger.warning(message)


def _describe_keys(keys: List[str]) -> str:
    now = time.time()
    lines: List[str] = []
    with _lock:
        for index, key in enumerate(keys):
            state = _get_key_locked(key)
            if now < state.dead_until:
                status = f"معطّل {int(state.dead_until - now)}ث"
            elif now < state.cooldown_until:
                status = f"مضغوط {int(state.cooldown_until - now)}ث"
            else:
                status = "سليم"
            lines.append(f"key#{index}: {status} (آخر خطأ={state.last_error or '-'})")
    return " | ".join(lines)


# ---------------------------------------------------------------------------
# Shared fallback executor
# ---------------------------------------------------------------------------


def _execute_fallback(payload: Dict[str, Any], operation: str) -> str:
    keys = _get_valid_keys()
    if not keys:
        raise ValueError("خطأ: لم يتم العثور على مفاتيح GEMINI صالحة في ملف .env!")

    models = _get_configured_models()
    if not models:
        raise ValueError("خطأ: لم يتم العثور على نماذج GEMINI صالحة في config.py!")

    started = time.time()
    deadline = started + TOTAL_DEADLINE
    tried: Set[Tuple[str, str]] = set()
    failures: List[Tuple[int, str, str]] = []
    attempts = 0
    waits = 0
    bad_requests = 0
    last_bad_message = ""
    stop_reason = "exhausted"

    while attempts < MAX_ATTEMPTS:
        remaining = deadline - time.time()
        if remaining <= MIN_ATTEMPT_TIME:
            stop_reason = "deadline"
            break

        choice, earliest = _pick_pair(keys, models, tried)

        if choice is None:
            if earliest is not None and waits < MAX_WAITS_PER_REQUEST:
                wait = earliest - time.time()
                if 0 < wait <= MAX_WAIT_FOR_RECOVERY and wait < remaining - MIN_ATTEMPT_TIME:
                    waits += 1
                    time.sleep(wait + 0.05)
                    continue
            stop_reason = "no_available_pair"
            break

        key_index, api_key, raw_model = choice
        attempts += 1
        tried.add((api_key, raw_model))
        model_id = _format_model_name(raw_model)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_id}:generateContent"
        )
        read_timeout = min(READ_TIMEOUT, max(remaining - 1, MIN_ATTEMPT_TIME))

        response: Optional[requests.Response] = None
        error_category = ""
        error_detail = ""
        started_call = time.time()
        try:
            response = _session.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                json=payload,
                timeout=(CONNECT_TIMEOUT, read_timeout),
            )
        except requests.exceptions.Timeout:
            error_category = "timeout"
        except requests.exceptions.RequestException as exc:
            error_category = "network_error"
            error_detail = type(exc).__name__
        finally:
            _release_key(api_key)

        latency = time.time() - started_call

        # ---- network level failures ----
        if response is None:
            _pair_failure(
                api_key,
                raw_model,
                error_category,
                latency_sample=READ_TIMEOUT if error_category == "timeout" else None,
            )
            _log_attempt(key_index, model_id, None, error_category, error_detail)
            failures.append((key_index, model_id, error_category))
            continue

        status = getattr(response, "status_code", None)

        # ---- success ----
        if status == 200:
            data = _parse_json_response(response)
            if data is None:
                _pair_failure(api_key, raw_model, "invalid_json")
                _log_attempt(key_index, model_id, status, "invalid_json")
                failures.append((key_index, model_id, "invalid_json"))
                continue

            text = _extract_text_from_response(data)
            if text is None:
                _raise_if_blocked(data)
                _pair_failure(api_key, raw_model, "no_usable_text")
                _log_attempt(key_index, model_id, status, "no_usable_text")
                failures.append((key_index, model_id, "no_usable_text"))
                continue

            _pair_success(api_key, raw_model, latency)
            logger.info(
                f"[Gemini] {operation} success via key#{key_index} "
                f"model={model_id} latency={latency:.2f}s attempts={attempts}"
            )
            return text

        # ---- HTTP errors ----
        info = _parse_error(response)

        if status == 401 or (status == 400 and _is_invalid_key_error(info)):
            _block_key(api_key, KEY_DEAD_INVALID, True, "invalid_key")
            _log_attempt(key_index, model_id, status, "invalid_key")
            failures.append((key_index, model_id, "invalid_key"))
            continue

        if status == 403:
            _block_key(api_key, KEY_DEAD_FORBIDDEN, True, "forbidden")
            _log_attempt(key_index, model_id, status, "forbidden")
            failures.append((key_index, model_id, "forbidden"))
            continue

        if status == 404:
            _mark_dead_model(raw_model)
            _log_attempt(key_index, model_id, status, "model_not_found")
            failures.append((key_index, model_id, "model_not_found"))
            continue

        if status == 429:
            delay = info["retry_delay"] or DEFAULT_RETRY_DELAY
            delay = min(max(delay, 1.0), DAILY_QUOTA_COOLDOWN)
            if info["daily"]:
                delay = DAILY_QUOTA_COOLDOWN
            _pair_failure(api_key, raw_model, "rate_limited", cooldown=delay)
            _register_key_429(api_key, delay, info["daily"])
            _log_attempt(
                key_index, model_id, status, "rate_limited",
                f"retry_in={int(delay)}s daily={info['daily']}",
            )
            failures.append((key_index, model_id, "rate_limited"))
            continue

        if status == 400:
            if info["status"] == "FAILED_PRECONDITION":
                _block_key(api_key, KEY_DEAD_FORBIDDEN, True, "failed_precondition")
                _log_attempt(key_index, model_id, status, "failed_precondition", info["message"])
                failures.append((key_index, model_id, "failed_precondition"))
                continue

            bad_requests += 1
            last_bad_message = info["message"] or "Bad request"
            _pair_failure(api_key, raw_model, "bad_request")
            _log_attempt(key_index, model_id, status, "bad_request", info["message"])
            failures.append((key_index, model_id, "bad_request"))
            if bad_requests >= BAD_REQUEST_FAIL_FAST:
                raise GeminiBadRequestError(
                    f"الطلب غير صالح (تكرر على أكثر من نموذج): {last_bad_message}"
                )
            continue

        if status == 503:
            _pair_failure(api_key, raw_model, "unavailable", cooldown=BREAKER_BASE_COOLDOWN)
            _log_attempt(key_index, model_id, status, "unavailable")
            failures.append((key_index, model_id, "unavailable"))
            continue

        if status in (500, 502, 504):
            _pair_failure(api_key, raw_model, "server_error")
            _log_attempt(key_index, model_id, status, "server_error")
            failures.append((key_index, model_id, "server_error"))
            continue

        _pair_failure(api_key, raw_model, "unknown_status")
        _log_attempt(key_index, model_id, status, "unknown_status", info["message"])
        failures.append((key_index, model_id, "unknown_status"))

    # ---- everything failed ----
    categories = sorted({category for _, _, category in failures})

    if categories == ["bad_request"]:
        raise GeminiBadRequestError(
            f"الطلب غير صالح: {last_bad_message or 'Bad request'}"
        )

    elapsed = time.time() - started
    raise GeminiEngineError(
        f"❌ فشلت كافة المفاتيح والنماذج في تلبية طلب {operation} "
        f"(السبب={stop_reason}, محاولات={attempts}, زمن={elapsed:.1f}ث, "
        f"categories={categories or ['none']}) | {_describe_keys(keys)}"
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
