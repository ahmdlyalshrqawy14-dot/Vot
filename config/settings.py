import os
from pathlib import Path

# المسارات الأساسية للمشروع
BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "output"

# ملف الحلقات في المجلد الرئيسي
DATA_FILE = BASE_DIR / "episodes.json"

# إعدادات مظهر شخصية الفيديو
CHARACTER_COLOR = "orange"  # يمكنك تغييره مستقبلاً إلى أي لون معتمد (مثل grey أو blue)

# إعدادات مايكروسوفت للصوت (Azure Speech - صوت الكوتش الرياضي)
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_VOICE_NAME = "en-US-TonyNeural"  # صوت الكوتش الحازم والمحفز
AZURE_VOICE_RATE = "+5%"               # سرعة إلقاء ديناميكية تناسب إيقاع يوتيوب

# إعدادات مايكروسوفت لتوليد الصور (Azure OpenAI / DALL-E)
AZURE_IMAGE_KEY = os.getenv("AZURE_IMAGE_KEY")
AZURE_IMAGE_ENDPOINT = os.getenv("AZURE_IMAGE_ENDPOINT")
AZURE_IMAGE_DEPLOYMENT = os.getenv("AZURE_IMAGE_DEPLOYMENT", "dall-e-3")
