import os
import json
import logging
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import (
    AZURE_MALE_VOICES,
    GOOGLE_MALE_VOICES,
    BASE_DIR
)
from stage1_generator import generate_stage1_script
from stage2_generator import generate_stage2_prompts_batches
from stage3_audio import generate_stage3_audio
from stage4_vision import process_and_verify_images, MissingAssetsError
from stage4_subtitles import align_audio_and_generate_ass
from stage4_composer import render_final_video
from stage5_metadata import generate_stage5_metadata

# إعداد الـ Logging
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("VotTelegramBot")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
EPISODES_FILE = BASE_DIR / "episodes.json"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# جلسات العمل النشطة للمستخدمين {chat_id: {"episode_id": "...", ...}}
user_sessions = {}


def get_episode_by_id_or_pending(target_id=None):
    if not EPISODES_FILE.exists():
        return None
    with open(EPISODES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data if isinstance(data, list) else data.get("episodes", [])
    if not episodes:
        return None
    if target_id:
        for ep in episodes:
            if str(ep.get("id")) == str(target_id):
                return ep
    for ep in episodes:
        if ep.get("status") == "pending":
            return ep
    return episodes[0]


# --- أوامر البوت الأساسية ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 أهلاً بك في نظام إنتاج ومونتاج الفيديوهات الآلي (Vot Engine)!\n\n"
        "الأوامر المتاحة:\n"
        "🔹 /start_episode [id] - لبدء إنتاج حلقة جديدة فوراً.\n"
        "🔹 /check_images - لفحص الصور المرفوعة والتأكد من عدم وجود أي نقص.\n"
        "🔹 /render - لبدء المونتاج والرندرة وإصدار الفيديو النهائي ورسائل النشر.\n"
        "🔹 /status - لمعرفة حالة الحلقة الحالية."
    )
    await update.message.reply_text(msg)


async def start_episode_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    target_id = context.args[0] if context.args else None

    episode = get_episode_by_id_or_pending(target_id)
    if not episode:
        await update.message.reply_text("❌ لم يتم العثور على أي حلقة صالحة في episodes.json!")
        return

    ep_id = str(episode.get("id", "201"))
    topic = episode.get("topic", "")

    user_sessions[chat_id] = {
        "episode_id": ep_id,
        "episode_data": episode,
        "stage1_done": False,
        "stage2_done": False,
        "stage3_done": False,
    }

    await update.message.reply_text(f"🚀 تم بدء الحلقة رقم [{ep_id}]:\n📌 العنوان: {topic}\n\n⏳ جاري تنفيذ المرحلة الأولى (توليد السكربت)...")

    # 1. المرحلة الأولى: توليد السكربت
    try:
        stage1_res = generate_stage1_script(episode)
        s1_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"
        with open(s1_file, "w", encoding="utf-8") as f:
            json.dump(stage1_res, f, ensure_ascii=False, indent=2)

        sentences = stage1_res.get("full_script_sentences", [])
        words_cnt = stage1_res.get("total_word_count", 0)
        user_sessions[chat_id]["sentences"] = sentences
        user_sessions[chat_id]["stage1_done"] = True

        await update.message.reply_text(
            f"✅ اكتملت المرحلة الأولى بنجاح!\n"
            f"📝 إجمالي الجمل: {len(sentences)}\n"
            f"📊 إجمالي الكلمات: {words_cnt} كلمة (المستهدف: 800-900)\n\n"
            f"⏳ جاري تنفيذ المرحلة الثانية (أوامر الصور وتجزئتها)..."
        )
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ في المرحلة الأولى: {str(e)}")
        return

    # 2. المرحلة الثانية: توليد وتجزئة أوامر الصور (بحد أقصى 24 لكل ملف)
    try:
        prompts_dir = OUTPUTS_DIR / f"episode_{ep_id}_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        batches = generate_stage2_prompts_batches(sentences)
        for idx, batch in enumerate(batches, start=1):
            p_file = prompts_dir / f"prompts_part_{idx:02d}.txt"
            with open(p_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(batch))
            
            # إرسال الملفات للمستخدم عبر تليجرام
            await update.message.reply_document(
                document=open(p_file, "rb"),
                caption=f"📦 ملف الأوامر رقم {idx:02d} ({len(batch)} برومبت)"
            )

        user_sessions[chat_id]["stage2_done"] = True
        await update.message.reply_text(f"✅ تم تصدير كافة ملفات أوامر الصور بنجاح ({len(batches)} ملفات).")
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ في المرحلة الثانية: {str(e)}")
        return

    # 3. عرض خيارات محرك الصوت (المرحلة الثالثة)
    keyboard = [
        [
            InlineKeyboardButton("🌐 Google Cloud TTS", callback_data="engine_google"),
            InlineKeyboardButton("⚡ Microsoft Azure Speech", callback_data="engine_azure")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🎙️ اختر محرك التعليق الصوتي للمتابعة:",
        reply_markup=reply_markup
    )


# --- معالجة أزرار المحرك الصوتي والأصوات الرجالية ---

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = update.effective_chat.id
    session = user_sessions.get(chat_id)

    if not session:
        await query.edit_message_text("⚠️ انتهت صلاحية الجلسة، ابدأ مجدداً عبر /start_episode")
        return

    # اختيار المحرك
    if data == "engine_google":
        session["engine"] = "google"
        buttons = [[InlineKeyboardButton(v, callback_data=f"voice_{v}")] for v in GOOGLE_MALE_VOICES]
        await query.edit_message_text("🗣️ اختر الصوت الرجالي المعتمد لـ Google TTS:", reply_markup=InlineKeyboardMarkup(buttons))
        return

    elif data == "engine_azure":
        session["engine"] = "azure"
        buttons = [[InlineKeyboardButton(v, callback_data=f"voice_{v}")] for v in AZURE_MALE_VOICES]
        await query.edit_message_text("🗣️ اختر الصوت الرجالي المعتمد لـ Microsoft Azure:", reply_markup=InlineKeyboardMarkup(buttons))
        return

    # اختيار الصوت وبدء المرحلة الثالثة
    if data.startswith("voice_"):
        selected_voice = data.replace("voice_", "")
        session["voice"] = selected_voice
        engine = session.get("engine", "azure")
        ep_id = session.get("episode_id")
        sentences = session.get("sentences", [])

        await query.edit_message_text(f"⏳ جاري هندسة الصوت التعبيري عبر {engine.upper()} بصوت: {selected_voice}...")

        try:
            audio_path = generate_stage3_audio(
                episode_id=ep_id,
                sentences=sentences,
                engine=engine,
                voice=selected_voice,
                output_dir=OUTPUTS_DIR
            )
            session["audio_path"] = audio_path
            session["stage3_done"] = True

            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"✅ تم توليد ملف الصوت الموحد بنجاح!\n\n"
                    f"📸 المرحلة التالية (الصور):\n"
                    f"قم بتوليد الصور من ملفات الـ TXT المرسلة لك، ثم ارفع الصور هنا في الشات (كصور أو ملفات).\n"
                    f"وبعد الانتهاء اضغط على: /render"
                )
            )
        except Exception as e:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ خطأ أثناء توليد الصوت: {str(e)}")


# --- استقبال الصور وحفظها في مجلد الحلقة ---

async def handle_incoming_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    session = user_sessions.get(chat_id)
    if not session:
        return

    ep_id = session.get("episode_id", "201")
    raw_images_dir = OUTPUTS_DIR / f"episode_{ep_id}_raw_images"
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    file_obj = None
    file_name = None

    if update.message.document:
        doc = update.message.document
        if doc.mime_type and doc.mime_type.startswith("image/"):
            file_obj = await doc.get_file()
            file_name = doc.file_name
    elif update.message.photo:
        photo = update.message.photo[-1]
        file_obj = await photo.get_file()
        file_name = f"photo_{photo.file_unique_id}.png"

    if file_obj:
        save_path = raw_images_dir / file_name
        await file_obj.download_to_drive(save_path)
        logger.info(f"تم استلام صورة وحفظها: {save_path.name}")


# --- فحص النواقص والرندرة النهائية (المرحلة 4 و 5) ---

async def render_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    session = user_sessions.get(chat_id)

    if not session or not session.get("stage3_done"):
        await update.message.reply_text("⚠️ يرجى إكمال المراحل السابقة وتوليد الصوت أولاً عبر /start_episode")
        return

    ep_id = session.get("episode_id")
    sentences = session.get("sentences", [])
    total_expected = len(sentences)
    raw_images_dir = OUTPUTS_DIR / f"episode_{ep_id}_raw_images"
    clean_frames_dir = OUTPUTS_DIR / f"episode_{ep_id}_clean_frames"

    await update.message.reply_text("🔍 جاري فحص الركن السفلي الأيمن للصور بالـ OCR وكشف النواقص...")

    # 1. فحص كاشف النواقص
    try:
        frames = process_and_verify_images(
            uploaded_images_dir=raw_images_dir,
            output_frames_dir=clean_frames_dir,
            expected_total=total_expected
        )
    except MissingAssetsError as m_err:
        await update.message.reply_text(str(m_err))
        return
    except Exception as e:
        await update.message.reply_text(f"❌ خطأ أثناء فحص الصور: {str(e)}")
        return

    await update.message.reply_text(
        f"✅ تم التحقق من كافة الصور ({len(frames)}/{total_expected}) واختفاء الأرقام بالرقعة الذكية!\n\n"
        f"🎬 بدء المزامنة وتوليد الترجمة الحركية والرندرة الآلية عبر FFmpeg (1080p 60fps)..."
    )

    # 2. المزامنة والترجمة الحركية الصفراء والرندرة
    try:
        audio_file = session["audio_path"]
        subtitles_ass = OUTPUTS_DIR / f"episode_{ep_id}_subtitles.ass"
        final_video = OUTPUTS_DIR / f"episode_{ep_id}_final_1080p.mp4"

        timeline = align_audio_and_generate_ass(audio_file, sentences, subtitles_ass)
        render_final_video(frames, timeline, audio_file, subtitles_ass, final_video)

        # إرسال الفيديو إذا كان حجمه مناسباً للتليجرام
        file_size_mb = final_video.stat().st_size / (1024 * 1024)
        if file_size_mb < 49:
            await update.message.reply_video(
                video=open(final_video, "rb"),
                caption=f"🏆 تم إنتاج الفيديو بنجاح بدقة Full HD 1080p (حجم: {file_size_mb:.1f} MB)"
            )
        else:
            await update.message.reply_text(
                f"🏆 تم تصدير الفيديو بنجاح على السيرفر (حجم: {file_size_mb:.1f} MB)!\n"
                f"📁 المسار المحلي: {final_video}"
            )
    except Exception as e:
        await update.message.reply_text(f"❌ حدث خطأ أثناء المونتاج والرندرة: {str(e)}")
        return

    # 3. المرحلة الخامسة: إرسال ميتاداتا النشر كـ 4 رسائل نصية منفصلة
    await update.message.reply_text("📦 جاري صياغة حزمة النشر والتغليف الرقمي (العناوين، الغلاف، الوصف، والتاجز)...")
    try:
        metadata = generate_stage5_metadata(session["episode_data"], sentences)

        # الرسالة 1: العناوين
        await update.message.reply_text(metadata.get("titles_message"))
        # الرسالة 2: أوامر الغلاف
        await update.message.reply_text(metadata.get("thumbnails_message"))
        # الرسالة 3: الوصف
        await update.message.reply_text(metadata.get("description_message"))
        # الرسالة 4: التاجز
        await update.message.reply_text(metadata.get("tags_message"))

        await update.message.reply_text("🎉 اكتملت دورة إنتاج الحلقة بالكامل 100% وجاهزة للنشر على يوتيوب!")
    except Exception as e:
        await update.message.reply_text(f"❌ تعذر استخراج ميتاداتا المرحلة الخامسة: {str(e)}")


def main():
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN غير موجود في متغيرات البيئة!")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("start_episode", start_episode_command))
    app.add_handler(CommandHandler("render", render_command))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_incoming_media))

    print("🤖 بوت التليجرام جاهز ويعمل الآن بنجاح على السيرفر...")
    app.run_polling()


if __name__ == "__main__":
    main()
