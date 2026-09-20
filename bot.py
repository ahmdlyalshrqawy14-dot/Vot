import os
import json
import time
from pathlib import Path
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

from config.settings import DATA_FILE, OUTPUT_DIR, TELEGRAM_BOT_TOKEN
from stages.stage1_script import run_stage_1
from stages.stage3_audio import run_stage_3
from stages.stage4_metadata import run_stage_4
from stages.stage2_telegram_prompts import generate_13_telegram_messages
from stages.image_slicer import slice_8_grid, save_single_frame
from stages.stage5_video import run_stage_5

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is missing in environment variables.")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="HTML")

# حالة الجلسة النشطة
SESSION = {
    "current_ep_id": None,
    "pending_file_path": None  # في حال أرسل صورة بدون كابشن
}

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

# -------------------------------------------------------------
# الأوامر الرئيسية
# -------------------------------------------------------------
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    markup = InlineKeyboardMarkup(row_width=1)
    markup.add(
        InlineKeyboardButton("▶️ إنتاج الحلقة التالية تلقائياً", callback_data="prod_auto"),
        InlineKeyboardButton("🔢 اختيار رقم حلقة معين", callback_data="prod_choose")
    )
    bot.send_message(
        message.chat.id,
        "🎬 <b>مرحباً بك في استوديو إنتاج الفيديوهات (5 دقائق)!</b>\n\n"
        "أنا جاهز لإدارة إنتاج الحلقات، كتابة السكربت، توليد الصوت البشري، "
        "وإرسال برومبتات الشيتات لتوليدها بدقة 2K ثم رندرة الفيديو بالكامل.",
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
        bot.send_message(chat_id, "أرسل الآن رقم الحلقة المطلوب إنتاجها (مثال: <code>1</code>):")

    elif call.data.startswith("set_sheet_"):
        sheet_id = int(call.data.split("_")[2])
        process_sheet_assignment(chat_id, sheet_id)

    elif call.data.startswith("set_single_"):
        beat_id = int(call.data.split("_")[2])
        process_single_assignment(chat_id, beat_id)

    elif call.data == "render_now":
        trigger_render(chat_id)

@bot.message_handler(func=lambda m: m.text and m.text.isdigit())
def handle_episode_id_input(message):
    ep_id = int(message.text)
    episodes = load_episodes()
    target = next((ep for ep in episodes if ep.get("id") == ep_id), None)
    if not target:
        bot.send_message(message.chat.id, f"❌ الحلقة رقم {ep_id} غير موجودة في ملف episodes.json.")
        return
    start_episode_pipeline(message.chat.id, target)

# -------------------------------------------------------------
# بدء خط الإنتاج وتوليد الـ 13 رسالة
# -------------------------------------------------------------
def start_episode_pipeline(chat_id, episode: dict):
    ep_id = episode.get("id")
    SESSION["current_ep_id"] = ep_id
    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    bot.send_message(chat_id, f"⏳ <b>جاري إعداد الحلقة #{ep_id}: {episode.get('topic')}</b>\n"
                              f"يتم الآن توليد السكربت الإنجليزي وهندسة المشاهد والصوت...")

    # 1. السكربت والمشاهد
    script_data = run_stage_1(episode, episode_dir)

    # 2. الصوت البشري لكل مشهد
    run_stage_3(script_data, episode_dir)

    # 3. الميتاداتا
    meta_text = run_stage_4(episode, script_data, episode_dir)

    # 4. تجهيز الـ 13 رسالة
    messages = generate_13_telegram_messages(script_data, meta_text, episode_dir)

    bot.send_message(chat_id, f"✅ <b>تم الانتهاء من السكربت والصوت بنجاح!</b>\n"
                              f"سأرسل لك الآن <b>13 رسالة</b> بالبرومبتات.\n"
                              f"• 5 شيتات (كل شيت 8 مربعات مرقمة من الداخل)\n"
                              f"• 5 صور فردية (للمشاهد 41 إلى 45)\n"
                              f"• 3 برومبتات للغلاف (Thumbnail)")

    # إرسال الرسائل بتتابع هادئ
    for m in messages:
        bot.send_message(chat_id, m["text"])
        time.sleep(0.6)

    bot.send_message(
        chat_id,
        "📥 <b>أنا في انتظار استلام الصور الآن:</b>\n\n"
        "1. يمكنك إرسال صورة الشيت كملف (File/Document) للحفاظ على جودة الـ 2K.\n"
        "2. اكتب في تعليق الصورة (Caption) رقم الشيت (مثال: <code>شيت 1</code>) أو رقم الصورة الفردية (مثال: <code>41</code>).\n"
        "3. لو أرسلتها بدون كتابة سأسألك بأزرار تفاعلية لتختار رقمها."
    )

# -------------------------------------------------------------
# استقبال الصور والشيتات وتوزيعها
# -------------------------------------------------------------
@bot.message_handler(content_types=['photo', 'document'])
def handle_incoming_media(message):
    ep_id = SESSION.get("current_ep_id")
    if not ep_id:
        bot.send_message(message.chat.id, "⚠️ لم يتم تحديد حلقة نشطة حالياً. اكتب /start لبدء حلقة.")
        return

    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    temp_dir = episode_dir / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # استخراج معرف الملف
    if message.content_type == 'photo':
        file_id = message.photo[-1].file_id
        ext = "png"
    else:
        file_id = message.document.file_id
        ext = message.document.file_name.split(".")[-1]

    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    temp_path = temp_dir / f"upload_{int(time.time())}.{ext}"
    with open(temp_path, "wb") as f:
        f.write(downloaded_file)

    caption = (message.caption or "").strip().lower()

    # محاولة قراءة الرقم من الكابشن تلقائياً
    # فحص إذا كان شيت (1 إلى 5)
    matched_sheet = None
    for s_id in range(1, 6):
        if f"شيت {s_id}" in caption or f"sheet {s_id}" in caption or caption == str(s_id):
            matched_sheet = s_id
            break

    if matched_sheet:
        start_idx = (matched_sheet - 1) * 8 + 1
        saved = slice_8_grid(temp_path, start_idx, episode_dir / "images")
        report_progress(message.chat.id, f"✅ تم تقطيع <b>الشيت {matched_sheet}</b> وحفظ المشاهد ({start_idx} إلى {start_idx+7})")
        return

    # فحص إذا كانت صورة فردية (41 إلى 45)
    matched_single = None
    for b_id in range(41, 46):
        if str(b_id) in caption:
            matched_single = b_id
            break

    if matched_single:
        save_single_frame(temp_path, matched_single, episode_dir / "images")
        report_progress(message.chat.id, f"✅ تم حفظ <b>المشهد الفردي #{matched_single}</b> بنجاح.")
        return

    # في حال لم يكتب المستخدم شيئاً، نظهر أزرار الاختيار
    SESSION["pending_file_path"] = temp_path
    markup = InlineKeyboardMarkup(row_width=3)
    sheet_btns = [InlineKeyboardButton(f"شيت {i}", callback_data=f"set_sheet_{i}") for i in range(1, 6)]
    markup.add(*sheet_btns)
    single_btns = [InlineKeyboardButton(f"مشهد {i}", callback_data=f"set_single_{i}") for i in range(41, 46)]
    markup.add(*single_btns)

    bot.send_message(
        message.chat.id,
        "❓ <b>استلمت هذه الصورة بدون تحديد. اختر تصنيفها المناسب:</b>",
        reply_markup=markup
    )

def process_sheet_assignment(chat_id, sheet_id: int):
    temp_path = SESSION.get("pending_file_path")
    ep_id = SESSION.get("current_ep_id")
    if not temp_path or not ep_id:
        return

    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    start_idx = (sheet_id - 1) * 8 + 1
    slice_8_grid(temp_path, start_idx, episode_dir / "images")
    SESSION["pending_file_path"] = None
    report_progress(chat_id, f"✅ تم تقطيع <b>الشيت {sheet_id}</b> وتسكين المشاهد ({start_idx} إلى {start_idx+7})")

def process_single_assignment(chat_id, beat_id: int):
    temp_path = SESSION.get("pending_file_path")
    ep_id = SESSION.get("current_ep_id")
    if not temp_path or not ep_id:
        return

    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    save_single_frame(temp_path, beat_id, episode_dir / "images")
    SESSION["pending_file_path"] = None
    report_progress(chat_id, f"✅ تم حفظ <b>المشهد الفردي #{beat_id}</b> بنجاح.")

def report_progress(chat_id, status_prefix: str):
    ep_id = SESSION.get("current_ep_id")
    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    count = get_collected_count(episode_dir)

    text = f"{status_prefix}\n\n📊 <b>تقدم استلام الكادرات:</b> {count} / 45 كادراً."

    markup = None
    if count >= 40:
        markup = InlineKeyboardMarkup()
        markup.add(InlineKeyboardButton("🚀 ابدأ رندرة ومونتاج الفيديو الآن", callback_data="render_now"))

    bot.send_message(chat_id, text, reply_markup=markup)

# -------------------------------------------------------------
# الرندرة وإرسال الفيديو النهائي
# -------------------------------------------------------------
def trigger_render(chat_id):
    ep_id = SESSION.get("current_ep_id")
    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    script_json = episode_dir / "script_beats.json"

    with open(script_json, "r", encoding="utf-8") as f:
        script_data = json.load(f)

    bot.send_message(chat_id, "🎬 <b>بدأت الآن عملية المونتاج والرندرة الكاملة (1080p)...</b>\n"
                              "جاري دمج الكادرات، تطبيق زووم Ken Burns، حرق الترجمة، وضبط الموسيقى... ⏳")

    try:
        video_path = run_stage_5(script_data, episode_dir)

        # تحديث حالة الحلقة في JSON
        episodes = load_episodes()
        target = next((ep for ep in episodes if ep.get("id") == ep_id), None)
        if target:
            target["status"] = "completed"
            save_episodes(episodes)

        # إرسال الفيديو للمستخدم
        bot.send_message(chat_id, "🎉 <b>اكتمل المونتاج بنجاح! جاري رفع الفيديو إليك...</b>")
        with open(video_path, "rb") as vf:
            bot.send_video(
                chat_id,
                video=vf,
                caption=f"🎬 <b>Episode #{ep_id}: Completed Full HD (5-Min)</b>",
                supports_streaming=True,
                timeout=300
            )

    except Exception as e:
        bot.send_message(chat_id, f"❌ حدث خطأ أثناء المونتاج: <code>{str(e)}</code>")

if __name__ == "__main__":
    print("==================================================")
    print("   TELEGRAM PRODUCTION BOT RUNNING (24/7 MODE)    ")
    print("==================================================")
    bot.infinity_polling(skip_pending=True)
