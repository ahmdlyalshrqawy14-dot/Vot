from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "output"
DATA_FILE = BASE_DIR / "episodes.json"

CHARACTER_COLOR = "orange"
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FPS = 30

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
AZURE_VOICE_NAME = "en-US-TonyNeural"
AZURE_VOICE_RATE = "+0%"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
