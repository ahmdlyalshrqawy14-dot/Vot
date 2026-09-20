import os
from pathlib import Path
import requests

# بيانات البوت ومحادثتك
BOT_TOKEN = "8908972160:AAFdZRxPnry9VRnFX028VyANSdoWsWafsY8"
CHAT_ID = "5381280076"

def send_telegram_results(episode: dict, episode_dir: Path):
    ep_id = episode.get("id")
    topic = episode.get("topic")
    video_path = episode_dir / "final_video.mp4"
    metadata_path = episode_dir / "metadata.md"

    # تجهيز رسالة الملخص
    caption = f"🎬 *Episode #{ep_id}: {topic}*\n\n"
    caption += "✅ تم اكتمال إنتاج ورندرة الفيديو بنجاح (1080p)!\n\n"

    if metadata_path.exists():
        try:
            lines = [l.strip() for l in metadata_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            caption += "📌 *العناوين المقترحة:*\n" + "\n".join(lines[:3])
        except Exception:
            pass

    if len(caption) > 1000:
        caption = caption[:950] + "..."

    print(f"\n--- [Telegram] Sending Video to Chat ID: {CHAT_ID} ---")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVideo"

    if video_path.exists():
        try:
            with open(video_path, "rb") as vf:
                data = {
                    "chat_id": CHAT_ID,
                    "caption": caption,
                    "parse_mode": "Markdown",
                    "supports_streaming": "true"
                }
                files = {"video": vf}
                resp = requests.post(url, data=data, files=files, timeout=240)
                if resp.status_code == 200:
                    print("[Telegram] Video delivered successfully!")
                    return
                else:
                    print(f"[Telegram] Failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"[Telegram] Upload error: {e}")

    # رسالة تنبيه لو تعذر رفع الفيديو كملف
    msg_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(msg_url, json={
            "chat_id": CHAT_ID,
            "text": f"⚠️ *Episode #{ep_id}: {topic}*\nاكتملت الحلقة ولكن حدث خطأ في رفع الفيديو إلى تيليجرام.",
            "parse_mode": "Markdown"
        }, timeout=30)
    except Exception:
        pass
