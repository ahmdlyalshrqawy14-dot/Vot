import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"
DATA_FILE = BASE_DIR / "episodes.json"

# مظهر الشخصية وهوية القناة
CHARACTER_COLOR = "orange"
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FPS = 30

# أبعاد الصور العريضة لتفادي الحواف الميتة
IMAGE_SIZE = "1792x1024"

# Azure Speech Settings
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_VOICE_NAME = "en-US-TonyNeural"
AZURE_VOICE_RATE = "+0%"  # سرعة طبيعية تناسب الشرح الرياضي العميق

# Azure OpenAI / DALL-E Settings
AZURE_IMAGE_KEY = os.getenv("AZURE_IMAGE_KEY")
AZURE_IMAGE_ENDPOINT = os.getenv("AZURE_IMAGE_ENDPOINT")
AZURE_IMAGE_DEPLOYMENT = os.getenv("AZURE_IMAGE_DEPLOYMENT", "dall-e-3")

# Telegram Notifier Settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
