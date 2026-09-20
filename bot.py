import os
import gc
import re
import json
import time
import zipfile
from pathlib import Path
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from PIL import Image

from config.settings import DATA_FILE, OUTPUT_DIR, TELEGRAM_BOT_TOKEN
from stages.stage1_script import run_stage_1
from stages.stage3_audio import run_stage_3
from stages.stage4_metadata import run_stage_4
from stages.stage2_telegram_prompts import generate_google_flow_files
from stages.vision_sorter import sort_and_save_images_parallel, read_badge_number
from stages.stage5_video import run_stage_5

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is missing in environment variables.")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="HTML")

SESSION = {
    "current_ep_id": None
}

def clean_memory():
    gc.collect()

def load_episodes():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_episodes(episodes):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)

def get_collected_count(episode_dir: Path) -> int:
    images_dir = episode_dir / "images"
    if not images_dir.exists():
        return 0
    return len(list(images_dir.glob("beat_*.png")))

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

@bot.message_handler(commands=['start', 'reset'])
def send_welcome(message):
    SESSION["current_ep_id"] = None
    clean_memory()

    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(
        InlineKeyboardButton("▶️ إنتاج الحلقة التالية تلقائياً", callback_data="prod_auto"),
        InlineKeyboardButton("🔢 اختيار رقم حلقة معين", callback_data="prod_choose")
    )

    bot.send_message(
        message.chat.id,
        "🔄 <b>تم تصفير الجلسة وتفعيل نظام الفحص البصري (Gemini Vision 2K)!</b>\n\n"
        "اختر ماذا تريد أن نفعل الآن:",
        reply_markup=markup
    )

@bot.callback_query_handler(func=lambda call: True)
def handle_callbacks(call):
    chat_id = call.message.chat.id
    if call.data == "prod_auto":
        episodes = load_episodes()
        pending = [ep for ep in episodes if ep.get("status") == "pending"]
        if not pending:
            bot.send_message(chat_id, "✅ جميع الحلقات مكتملة بالفعل!")
            return
        start_episode_pipeline(chat_id, pending[0])
    elif call.data == "prod_choose":
        bot.send_message(chat_id, "أرسل رقم الحلقة المطلوب إنتاجها (مثال: <code>1</code>):")
    elif call.data == "render_now":
        trigger_render(chat_id)

@bot.message_handler(func=lambda m: m.text and m.text.isdigit())
def handle_episode_id_input(message):
    ep_id = int(message.text)
    episodes = load_episodes()
    target = next((ep for ep in episodes if ep.get("id") == ep_id), None)
    if not target:
        bot.send_message(message.chat.id, f"❌ الحلقة رقم {ep_id} غير موجودة.")
        return
    start_episode_pipeline(message.chat.id, target)

def start_episode_pipeline(chat_id, episode: dict):
    ep_id = episode.get("id")
    SESSION["current_ep_id"] = ep_id
    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    images_dir = episode_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    bot.send_message(
        chat_id,
        f"⏳ <b>جاري إعداد الحلقة #{ep_id}: {episode.get('topic')}</b>\n"
        f"توليد السكربت، صوت أزور البشري، وإنشاء ملفات البرومبتات لـ Google Flow..."
    )

    try:
        script_data = run_stage_1(episode, episode_dir)
        run_stage_3(script_data, episode_dir)
        meta_text = run_stage_4(episode, script_data, episode_dir)
        file_packages = generate_google_flow_files(script_data, meta_text, episode_dir)

        bot.send_message(chat_id, "✅ <b>تم تجهيز ملفات الـ TXT بنجاح! سأرسلها لك الآن:</b>")

        for item in file_packages:
            with open(item["path"], "rb") as doc:
                bot.send_document(chat_id, doc, caption=item["caption"])
            time.sleep(0.4)

        bot.send_message(
            chat_id,
            "📥 <b>إرسال الصور:</b>\n\n"
            "• يمكنك إرسال الـ 45 صورة دفعة واحدة داخل <b>ملف ZIP</b> واحد.\n"
            "• سيقوم الذكاء الاصطناعي (Gemini Vision) بفحص بادج الركن السفلي لكل صورة لتحديد ترتيبها الدقيق تلقائياً."
        )
        clean_memory()
    except Exception as e:
        bot.send_message(chat_id, f"⚠️ خطأ: <code>{str(e)}</code>\nاكتب /start للمحاولة مجدداً.")

# -------------------------------------------------------------
# استقبال الصور وفحصها بالذكاء الاصطناعي
# -------------------------------------------------------------
@bot.message_handler(content_types=['photo', 'document'])
def handle_incoming_files(message):
    ep_id = SESSION.get("current_ep_id")
    if not ep_id:
        bot.send_message(message.chat.id, "⚠️ لا توجد حلقة نشطة حالياً. اكتب /start.")
        return

    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    images_dir = episode_dir / "images"
    temp_dir = episode_dir / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. ملف ZIP (فحص جماعي متوازي عبر المفاتيح الأربعة)
    if message.content_type == 'document' and message.document.file_name.lower().endswith('.zip'):
        status_msg = bot.send_message(
            message.chat.id,
            "📦 <b>استلمت ملف الـ ZIP!</b>\n"
            "🔍 جاري فحص الكادرات بالتوازي وقراءة أرقام المشاهد بنموذج Gemini Vision..."
        )
        file_info = bot.get_file(message.document.file_id)
        zip_bytes = bot.download_file(file_info.file_path)
        zip_path = temp_dir / f"batch_{int(time.time())}.zip"
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        unpacked_count = 0
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extract_folder = temp_dir / f"unzipped_{int(time.time())}"
            extract_folder.mkdir(parents=True, exist_ok=True)
            zip_ref.extractall(extract_folder)

            valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
            extracted_files = [
                p for p in extract_folder.rglob("*")
                if p.is_file() and p.suffix.lower() in valid_exts and "__MACOSX" not in str(p) and not p.name.startswith(".")
            ]
            extracted_files.sort(key=natural_sort_key)

            if extracted_files:
                unpacked_count = sort_and_save_images_parallel(extracted_files, images_dir)

        clean_memory()
        report_progress(message.chat.id, f"✅ تم التعرف على بادجات الكادرات وتسكين <b>{unpacked_count}</b> صورة بنجاح.")
        return

    # 2. صورة فردية
    if message.content_type == 'photo':
        file_id = message.photo[-1].file_id
        orig_name = ""
    else:
        file_id = message.document.file_id
        orig_name = message.document.file_name or ""

    file_info = bot.get_file(file_id)
    downloaded = bot.download_file(file_info.file_path)

    temp_img = temp_dir / f"single_{int(time.time()*1000)}.png"
    with open(temp_img, "wb") as f:
        f.write(downloaded)

    caption = (message.caption or "").strip()
    target_id = None
    num_match = re.search(r"(\d+)", caption) or re.search(r"(\d+)", orig_name)
    if num_match:
        val = int(num_match.group(1))
        if 1 <= val <= 45:
            target_id = val

    # لو لم يكتب الرقم في الاسم، يفحصه Gemini Vision في ثانية واحدة
    if not target_id:
        detected = read_badge_number(temp_img, 0)
        if 1 <= detected <= 45:
            target_id = detected

    # في حال تعذر القراءة، يسكن في أول مشهد شاغر
    if not target_id:
        existing = {int(m.group(1)) for f in images_dir.glob("beat_*.png") if (m := re.search(r"beat_(\d+)", f.stem))}
        for i in range(1, 46):
            if i not in existing:
                target_id = i
                break

    if not target_id:
        target_id = 45

    dest = images_dir / f"beat_{target_id:03d}.png"
    with Image.open(temp_img) as img:
        img.save(dest, "PNG")

    clean_memory()
    report_progress(message.chat.id, f"✅ تم فحص وحفظ المشهد <b>#{target_id:02d}</b>.")

def report_progress(chat_id, prefix: str):
    ep_id = SESSION.get("current_ep_id")
    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    count = get_collected_count(episode_dir)

    text = f"{prefix}\n\n📊 <b>إجمالي الكادرات المستلمة:</b> {count} / 45 كادراً."
    markup = None
    if count >= 40:
        markup = InlineKeyboardMarkup()
        markup.add(InlineKeyboardButton("🚀 ابدأ رندرة ومونتاج الفيديو الآن", callback_data="render_now"))

    bot.send_message(chat_id, text, reply_markup=markup)

def trigger_render(chat_id):
    ep_id = SESSION.get("current_ep_id")
    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    script_json = episode_dir / "script_beats.json"

    with open(script_json, "r", encoding="utf-8") as f:
        script_data = json.load(f)

    bot.send_message(chat_id, "🎬 <b>بدأ المونتاج التلقائي فائق الدقة (1080p + صوت نقي)...</b> ⏳")

    try:
        video_path = run_stage_5(script_data, episode_dir)
        episodes = load_episodes()
        target = next((ep for ep in episodes if ep.get("id") == ep_id), None)
        if target:
            target["status"] = "completed"
            save_episodes(episodes)

        bot.send_message(chat_id, "🎉 <b>اكتمل المونتاج بنجاح! جاري رفع الفيديو إليك...</b>")
        with open(video_path, "rb") as vf:
            bot.send_video(
                chat_id,
                video=vf,
                caption=f"🎬 <b>Episode #{ep_id} Finished!</b>",
                supports_streaming=True,
                timeout=300
            )
        clean_memory()
    except Exception as e:
        bot.send_message(chat_id, f"❌ خطأ أثناء المونتاج: <code>{str(e)}</code>")

if __name__ == "__main__":
    print("==================================================")
    print("   GEMINI VISION MULTI-KEY PIPELINE RUNNING       ")
    print("==================================================")
    try:
        bot.delete_webhook(drop_pending_updates=True)
        bot.infinity_polling(skip_pending=True, timeout=25, long_polling_timeout=25)
    except Exception as err:
        print(f"[ERROR] {err}")
