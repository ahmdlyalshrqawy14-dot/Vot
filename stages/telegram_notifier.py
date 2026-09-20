import html
from pathlib import Path
import requests
from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_telegram_results(episode: dict, episode_dir: Path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] Bot Token or Chat ID not configured. Skipping.")
        return

    ep_id = episode.get("id")
    topic = episode.get("topic")
    video_path = episode_dir / "final_video.mp4"
    metadata_path = episode_dir / "metadata.md"

    caption = f"🎬 <b>Episode #{ep_id}: {html.escape(str(topic))}</b>\n\n"
    caption += "✅ <b>اكتمل إنتاج الفيديو الطويل (5 دقائق - 1080p بنجاح!)</b>\n\n"

    if metadata_path.exists():
        try:
            lines = [l.strip() for l in metadata_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            caption += "📌 <b>العناوين المقترحة:</b>\n" + html.escape("\n".join(lines[:3]))
        except Exception:
            pass

    if len(caption) > 950:
        caption = caption[:920] + "..."

    print(f"[Telegram] Uploading Video to Chat ID: {TELEGRAM_CHAT_ID}...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"

    if video_path.exists():
        try:
            with open(video_path, "rb") as vf:
                data = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption,
                    "parse_mode": "HTML",
                    "supports_streaming": "true"
                }
                resp = requests.post(url, data=data, files={"video": vf}, timeout=300)
                if resp.status_code == 200:
                    print("[Telegram] Video sent successfully!")
                    return
        except Exception as e:
            print(f"[Telegram] Failed to upload video: {e}")

    # إرسال رسالة نصية في حال تجاوز حجم الملف ليميت التيليجرام العادي
    msg_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(msg_url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"✅ اكتملت الحلقة #{ep_id} وتم حفظ الفيديو في الـ Artifacts السحابية.",
        "parse_mode": "HTML"
    }, timeout=20)
