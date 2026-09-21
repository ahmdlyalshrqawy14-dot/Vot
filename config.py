import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# مفاتيح Gemini الأربعة (صمامات الأمان)
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY", "").strip(),
    os.getenv("GEMINI_API_KEY_2", "").strip(),
    os.getenv("GEMINI_API_KEY_3", "").strip(),
    os.getenv("GEMINI_API_KEY_4", "").strip(),
]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

# النماذج الرسمية بالترتيب
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

# إعدادات Microsoft Azure Speech
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", "").strip()
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "uaenorth").strip()

# الأصوات الرجالية المعتمدة حصراً للمرحلة الثالثة (Male Only)
GOOGLE_MALE_VOICES = [
    "en-US-Journey-D",
    "en-US-Studio-Q",
    "en-US-Neural2-D"
]

AZURE_MALE_VOICES = [
    "en-US-GuyNeural",
    "en-US-DavisNeural",
    "en-US-TonyNeural",
    "en-US-JasonNeural"
]

REQUEST_TIMEOUT = 90
