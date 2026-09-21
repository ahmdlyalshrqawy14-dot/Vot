import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# تحميل ملف .env في حال التشغيل المحلي على السيرفر
load_dotenv(BASE_DIR / ".env")

# جلب المفاتيح الأربعة كصمامات أمان متعاقبة
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY", "").strip(),
    os.getenv("GEMINI_API_KEY_2", "").strip(),
    os.getenv("GEMINI_API_KEY_3", "").strip(),
    os.getenv("GEMINI_API_KEY_4", "").strip(),
]
GEMINI_KEYS = [k for k in GEMINI_KEYS if k]

# مصفوفة النماذج المحددة بالترتيب
GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

REQUEST_TIMEOUT = 90  # مهلة الاتصال بالثواني
