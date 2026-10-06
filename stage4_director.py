import json
import logging
import hashlib
import threading
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import math
try:
    from gemini_engine import call_gemini_with_fallback
except ImportError:
    call_gemini_with_fallback = None

logger = logging.getLogger("Stage4Director")

_AI_PROMPT_VERSION = "v3"

class AIEditorialDirector:
    _instance: Optional["AIEditorialDirector"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._available = call_gemini_with_fallback is not None
        self._plan_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._MAX_PLAN_CACHE = 32

    @classmethod
    def get(cls) -> "AIEditorialDirector":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def is_available(self) -> bool:
        return self._available

    def plan_montage(
        self,
        renderable_timeline: List[Dict[str, Any]],
        durations: List[float],
        transitions: List[str],
    ) -> Dict[str, Any]:
        if not self.is_available():
            return {}

        cache_key = self._timeline_signature(
            renderable_timeline, durations, transitions
        )

        with self._cache_lock:
            cached = self._plan_cache.get(cache_key)
            if cached is not None:
                self._plan_cache.move_to_end(cache_key)
                return cached

        try:
            prompt = self._build_prompt(
                renderable_timeline, durations, transitions
            )
            raw = call_gemini_with_fallback(
                payload={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}},
                operation="plan_montage"
            )

            # extract text
            if isinstance(raw, dict) and "candidates" in raw and raw["candidates"]:
                content = raw["candidates"][0].get("content", {})
                if "parts" in content and content["parts"]:
                    raw = content["parts"][0].get("text", "")
            elif isinstance(raw, str):
                pass
            else:
                raw = str(raw)

            plan = self._parse_plan(raw, len(renderable_timeline))
            if plan:
                with self._cache_lock:
                    self._plan_cache[cache_key] = plan
                    self._plan_cache.move_to_end(cache_key)
                    while len(self._plan_cache) > self._MAX_PLAN_CACHE:
                        self._plan_cache.popitem(last=False)
                logger.info(
                    f"🎬 Gemini: خطة مونتاج لـ {len(plan.get('shots', []))} لقطة."
                )
            return plan or {}
        except Exception as e:
            logger.warning(f"فشل تخطيط AI: {e} — العودة للقواعد المحلية.")
            return {}

    def _build_prompt(
        self,
        timeline: List[Dict[str, Any]],
        durations: List[float],
        transitions: List[str],
    ) -> str:
        items = []
        tr_list = list(transitions) + [""]
        for i, (it, d, tr) in enumerate(zip(timeline, durations, tr_list)):
            role = str(it.get("narrative_role") or "").lower().strip()
            text = str(it.get("text") or "")[:180]
            items.append({
                "idx": i,
                "role": role,
                "duration": round(float(d), 2),
                "text": text,
                "local_transition_after": tr,
            })
        tl_json = json.dumps(items, ensure_ascii=False)

        system = (
            "أنت مخرج مونتاج وثائقي عالمي. "
            "مهمتك: خطة إبداعية ذكية ومتقدمة لسلسلة لقطات. "
            "أجب بـ JSON فقط بالصيغة التالية:\n"
            "{\n"
            '  "pacing": {"peaks": [int], "calm": [int], "breathing_after": [int]},\n'
            '  "shots": [\n'
            '    {"idx": int, "motion": str, "intensity": float, '
            '"transition_after": str, "whoosh_strength": float, '
            '"ambience_boost": float, "keyword": str|null}\n'
            "  ],\n"
            '  "visual_overlays": [\n'
            '    {"timestamp_start": float, "duration": float, "keyword": str, "screen_position": "top_right" | "center_pop" | "lower_third_corner"}\n'
            "  ]\n"
            "}\n\n"
            "قواعد الحركة (motion):\n"
            "- punch_in: زووم سريع وقوي جداً عند كلمات التأكيد القصوى.\n"
            "- slow_pan: حركة بطيئة أفقية (يسار/يمين) أو عمودية (أعلى/أسفل).\n"
            "- drift: حركة بطيئة وعائمة ببطء شديد للأجواء التأملية.\n"
            "- legacy fallbacks (zoom_in, zoom_out, pan_left, etc... are allowed).\n"
            "- transition_after ∈ {cut, crossfade_soft, crossfade_medium, crossfade_deep}.\n"
            "- intensity ∈ [0.3, 1.25] (استخدم أعلى شدة مع punch_in).\n"
            "- whoosh_strength ∈ [0.0, 1.0].\n"
        )

        user = f"اللقطات:\n{tl_json}\n\nأعد JSON فقط دون أي شرح."
        return system + "\n" + user

    def _parse_plan(self, raw: str, n_shots: int) -> Dict[str, Any]:
        if not raw:
            return {}
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("`")
            if s.lower().startswith("json"):
                s = s[4:]
            s = s.strip()
        first_brace = s.find("{")
        last_brace = s.rfind("}")
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            return {}
        s = s[first_brace:last_brace + 1]

        try:
            data = json.loads(s)
        except json.JSONDecodeError as e:
            logger.debug(f"تعذّر تحليل JSON من AI: {e}")
            return {}
        if not isinstance(data, dict):
            return {}

        shots_raw = data.get("shots")
        if not isinstance(shots_raw, list):
            return {}

        cleaned_shots: List[Dict[str, Any]] = []
        seen_idx = set()
        for entry in shots_raw:
            if not isinstance(entry, dict):
                continue

            raw_idx = entry.get("idx")
            if isinstance(raw_idx, bool):
                continue
            if isinstance(raw_idx, float):
                if not math.isfinite(raw_idx) or not raw_idx.is_integer():
                    continue
                idx = int(raw_idx)
            elif isinstance(raw_idx, int):
                idx = raw_idx
            else:
                try:
                    idx = int(raw_idx)
                except (TypeError, ValueError, OverflowError):
                    continue

            if idx < 0 or idx >= n_shots or idx in seen_idx:
                continue
            seen_idx.add(idx)

            cleaned_shots.append({
                "idx": idx,
                "motion": str(entry.get("motion", "")).lower().strip(),
                "intensity": self._safe_float(entry.get("intensity"), 0.55),
                "transition_after": str(entry.get("transition_after", "")).lower().strip(),
                "whoosh_strength": self._safe_float(entry.get("whoosh_strength"), 0.5),
                "ambience_boost": self._safe_float(entry.get("ambience_boost"), 0.0),
                "keyword": entry.get("keyword")
            })

        data["shots"] = cleaned_shots

        # Fix pacing
        pacing_raw = data.get("pacing", {})
        if not isinstance(pacing_raw, dict):
            pacing_raw = {}

        def _safe_int_list(value: Any) -> List[int]:
            if not isinstance(value, (list, tuple)):
                return []
            out: List[int] = []
            for x in value:
                if isinstance(x, bool):
                    continue
                if isinstance(x, int):
                    out.append(x)
                elif isinstance(x, float):
                    if not math.isfinite(x) or not x.is_integer():
                        continue
                    out.append(int(x))
            return out

        data["pacing"] = {
            "peaks": _safe_int_list(pacing_raw.get("peaks")),
            "calm": _safe_int_list(pacing_raw.get("calm")),
            "breathing_after": _safe_int_list(pacing_raw.get("breathing_after")),
        }

        # Fix visual_overlays
        visual_overlays_raw = data.get("visual_overlays", [])
        if not isinstance(visual_overlays_raw, list):
            visual_overlays_raw = []

        cleaned_overlays = []
        for overlay in visual_overlays_raw:
            if not isinstance(overlay, dict):
                continue

            ts = self._safe_float(overlay.get("timestamp_start"), -1.0)
            dur = self._safe_float(overlay.get("duration"), -1.0)
            kw = str(overlay.get("keyword", "")).strip()
            pos = str(overlay.get("screen_position", "")).strip()

            if ts >= 0 and dur > 0 and kw and pos in ("top_right", "center_pop", "lower_third_corner"):
                cleaned_overlays.append({
                    "timestamp_start": ts,
                    "duration": dur,
                    "keyword": kw,
                    "screen_position": pos
                })

        data["visual_overlays"] = cleaned_overlays

        return data

    def _safe_float(self, val, default=0.0):
        try:
            f = float(val)
            return f if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _timeline_signature(
        timeline: List[Dict[str, Any]],
        durations: List[float],
        transitions: List[str],
    ) -> str:
        h = hashlib.sha256()
        h.update(_AI_PROMPT_VERSION.encode("ascii"))
        h.update(b"\n")
        for i, (it, d) in enumerate(zip(timeline, durations)):
            role = str(it.get("narrative_role") or "")
            text = str(it.get("text") or "")[:200]
            h.update(role.encode("utf-8", "ignore"))
            h.update(b"|")
            h.update(text.encode("utf-8", "ignore"))
            h.update(b"|")
            h.update(f"{round(float(d), 2)}".encode("ascii"))
            h.update(b"|")
            tr = transitions[i] if i < len(transitions) else ""
            h.update(str(tr).encode("utf-8", "ignore"))
            h.update(b"\n")
        return h.hexdigest()[:24]
