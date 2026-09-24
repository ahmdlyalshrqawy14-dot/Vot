import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _clean_string_list(values):
    """Return non-empty stripped unique strings while preserving order."""
    cleaned = []
    seen = set()

    for value in values:
        if value is None:
            continue

        text = str(value).strip()
        if not text:
            continue

        if text not in seen:
            seen.add(text)
            cleaned.append(text)

    return cleaned


# Gemini API keys loaded safely from up to four environment variables.
GEMINI_KEYS = _clean_string_list([
    os.getenv("GEMINI_API_KEY", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
])

# Official configured model order is preserved.
GEMINI_MODELS = _clean_string_list([
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
])

# Microsoft Azure Speech settings.
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "uaenorth").strip() or "uaenorth"

# Approved male-only voices for stage 3.
GOOGLE_MALE_VOICES = _clean_string_list([
    "en-US-Journey-D",
    "en-US-Studio-Q",
    "en-US-Neural2-D"
])

AZURE_MALE_VOICES = _clean_string_list([
    "en-US-GuyNeural",
    "en-US-DavisNeural",
    "en-US-TonyNeural",
    "en-US-JasonNeural",
    "en-US-BrianMultilingualNeural"
])

REQUEST_TIMEOUT = 90
