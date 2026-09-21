import os
import json
import logging
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
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

# إعداد السجلات
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("VotStudioBot")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
EPISODES_FILE = BASE_DIR / "episodes.json"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# إدارة الجلسات {chat_id: {...}}
user_sessions = {}


def load_all_episodes():
    if not EPISODES_FILE.exists():
        return []
    with open(EPISODES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("episodes", [])


def get_episode(target_id=None):
    episodes = load_all_episodes()
    if not episodes:
        return None
    if target_id is not None:
        for ep in episodes:
            if str(ep.get("id")) == str(target_id):
                return ep
        return None
    # التلقائي: أول حلقة pending
    for ep in episodes:
        if ep.get("status") == "pending":
            return ep
    return episodes[0]


def get_session(chat_id):
    if chat_id not in user_sessions:
        user_sessions[chat_id] = {
            "state": "IDLE",
            "episode_id": None,
            "episode_data": None,
            "sentences": [],
            "audio_path": None,
            "uploaded_count": 0,
        }
    return user_sessions[chat_id]


# -------------------------------------------------------------
# 1. شاشة البداية والترحيب التفاعلية
# -------------------------------------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    session = get_session(update.effective_chat.id)
    session["state"] = "IDLE"

    welcome_text = (
        "<b>🎬 مرحباً بك في Vot Studio | المحرك الآلي لصناعة المحتوى</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "النظام السحابي المتكامل لتحويل أفكار الحلقات إلى فيديوهات يوتيوب احترافية "
        "بجودة <b>1080p 60fps</b> مع هندسة التعليق الصوتي والترجمة الحركية.\n\n"
        "<b>👇 كيف تود أن نبدأ اليوم؟</b>"
    )

    keyboard = [
        [
            InlineKeyboardButton("▶️ بدء الحلقة التالية المجدولة", callback_data="btn_start_next")
        ],
        [
            InlineKeyboardButton("🔢 إدخال رقم حلقة معينة (ID)", callback_data="btn_choose_id"),
            InlineKeyboardButton("📋 استعراض الحلقات المتاحة", callback_data="btn_list_episodes")
        ]
    ]

    await update.message.reply_text(
        welcome_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.HTML
    )


# -------------------------------------------------------------
# 2. معالجة اختيارات القائمة الرئيسية
# -------------------------------------------------------------

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = update.effective_chat.id
    session = get_session(chat_id)

    # 1. خيار بدء الحلقة المجدولة التالية
    if data == "btn_start_next":
        ep = get_episode(target_id=None)
        if not ep:
            await query.edit_message_text("⚠️ لا توجد حلقات متاحة في ملف episodes.json!")
            return
        await show_episode_confirmation(query, ep)

    # 2. خيار إدخال رقم ID مخصص
    elif data == "btn_choose_id":
        session["state"] = "WAITING_EPISODE_ID"
        await query.edit_message_text(
            "🔢 <b>يرجى كتابة رقم الحلقة (ID) الآن في الشات:</b>\n"
            "<i>(مثال: أرسل الرقم 201 أو 101)</i>",
            parse_mode=ParseMode.HTML
        )

    # 3. استعراض قائمة الحلقات
    elif data == "btn_list_episodes":
        episodes = load_all_episodes()
        if not episodes:
            await query.edit_message_text("⚠️ لا توجد حلقات مسجلة.")
            return

        msg = "<b>📋 قائمة الحلقات المسجلة في السيرفر:</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for ep in episodes[:10]:
            st = "🟡 انتظار" if ep.get("status") == "pending" else "🟢 مكتملة"
            msg += f"• <b>ID [{ep.get('id')}]:</b> {ep.get('topic')}\n   └ الحالة: {st}\n"

        buttons = [
            [InlineKeyboardButton("▶️ بدء الحلقة التالية المجدولة", callback_data="btn_start_next")],
            [InlineKeyboardButton("🔙 رجوع للقائمة الرئيسية", callback_data="btn_back_main")]
        ]
        await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(buttons), parse_mode=ParseMode.HTML)

    elif data == "btn_back_main":
        await start_command(query, context)

    # 4. تأكيد بدء إنتاج حلقة معينة
    elif data.startswith("confirm_ep_"):
        target_id = data.replace("confirm_ep_", "")
        ep = get_episode(target_id)
        if ep:
            await run_stage1_and_2(query, context, ep)

    # 5. اختيار محرك الصوت
    elif data == "engine_google":
        session["engine"] = "google"
        buttons = []
        descriptions = {
            "en-US-Journey-D": "أداء حواري ديناميكي معبّر 🔥",
            "en-US-Studio-Q": "صوت استوديو عميق ورخيم 🎙️",
            "en-US-Neural2-D": "إلقاء إخباري ورسمي واضح 📢"
        }
        for v in GOOGLE_MALE_VOICES:
            label = descriptions.get(v, v)
            buttons.append([InlineKeyboardButton(f"🗣️ {label}", callback_data=f"voice_{v}")])
        buttons.append([InlineKeyboardButton("🔙 رجوع لمحركات الصوت", callback_data="btn_reselect_engine")])

        await query.edit_message_text(
            "🌐 <b>محرك Google Cloud TTS | اختر الصوت الرجالي:</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            reply_markup=InlineKeyboardMarkup(buttons),
            parse_mode=ParseMode.HTML
        )

    elif data == "engine_azure":
        session["engine"] = "azure"
        buttons = []
        descriptions = {
            "en-US-GuyNeural": "تلوين انفعالي كامل وشامل 🌟",
            "en-US-DavisNeural": "سرد ناضج وهادئ وقوي 📖",
            "en-US-TonyNeural": "صوت حماسي وواثق وعالي الطاقة ⚡",
            "en-US-JasonNeural": "نبرة شبابية وسريعة وخفيفة 🚀"
        }
        for v in AZURE_MALE_VOICES:
            label = descriptions.get(v, v)
            buttons.append([InlineKeyboardButton(f"🗣️ {label}", callback_data=f"voice_{v}")])
        buttons.append([InlineKeyboardButton("🔙 رجوع لمحركات الصوت", callback_data="btn_reselect_engine")])

        await query.edit_message_text(
            "⚡ <b>محرك Microsoft Azure Speech | اختر الصوت الرجالي:</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            reply_markup=InlineKeyboardMarkup(buttons),
            parse_mode=ParseMode.HTML
        )

    elif data == "btn_reselect_engine":
        await present_audio_engine_choice(query)

    # 6. اختيار الصوت وبدء إنتاج الملف الصوتي (المرحلة الثالثة)
    elif data.startswith("voice_"):
        selected_voice = data.replace("voice_", "")
        session["voice"] = selected_voice
        await run_stage3(query, context)

    # 7. الضغط على زر بدء الرندرة بعد رفع الصور
    elif data == "btn_start_render":
        await run_stage4_and_5(query.message, context)


async def show_episode_confirmation(query, ep):
    ep_id = ep.get("id", "201")
    topic = ep.get("topic", "بدون عنوان")
    myth = ep.get("the_myth", "غير محدد")
    status = ep.get("status", "pending")
    status_tag = "🟡 قيد الانتظار (Ready to Produce)" if status == "pending" else "🟢 تم إنتاجها مسبقاً"

    card_text = (
        f"<b>🎯 بطاقة بيانات الحلقة المحددة</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🆔 <b>رقم الحلقة:</b> <code>{ep_id}</code>\n"
        f"📌 <b>الموضوع:</b> <b>{topic}</b>\n"
        f"💡 <b>الخرافة المستهدفة:</b> <i>{myth}</i>\n"
        f"📊 <b>الحالة الحالية:</b> {status_tag}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"هل تود بدء دورة الإنتاج الآلي لهذه الحلقة؟"
    )

    keyboard = [
        [InlineKeyboardButton("🚀 تأكيد وبدء الإنتاج الآن", callback_data=f"confirm_ep_{ep_id}")],
        [InlineKeyboardButton("❌ إلغاء والعودة للقائمة", callback_data="btn_back_main")]
    ]
    await query.edit_message_text(card_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)


# -------------------------------------------------------------
# 3. معالجة الرسائل النصية (إدخال الـ ID يدوياً)
# -------------------------------------------------------------

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    session = get_session(chat_id)

    if session.get("state") == "WAITING_EPISODE_ID":
        entered_text = update.message.text.strip()
        ep = get_episode(target_id=entered_text)
        if ep:
            session["state"] = "IDLE"
            # إرسال بطاقة التأكيد كرسالة جديدة
            ep_id = ep.get("id")
            card_text = (
                f"<b>🎯 تم العثور على الحلقة بنجاح!</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🆔 <b>رقم الحلقة:</b> <code>{ep_id}</code>\n"
                f"📌 <b>الموضوع:</b> <b>{ep.get('topic')}</b>\n"
                f"💡 <b>الخرافة:</b> <i>{ep.get('the_myth')}</i>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"جاهز للبدء؟"
            )
            keyboard = [
                [InlineKeyboardButton("🚀 تأكيد وبدء الإنتاج الآن", callback_data=f"confirm_ep_{ep_id}")],
                [InlineKeyboardButton("❌ إلغاء والعودة", callback_data="btn_back_main")]
            ]
            await update.message.reply_text(card_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
        else:
            await update.message.reply_text(
                f"❌ لم يتم العثور على حلقة برقم ID <code>{entered_text}</code>.\n"
                f"تأكد من الرقم وحاول مجدداً، أو اضغط /start للعودة للقائمة.",
                parse_mode=ParseMode.HTML
            )


# -------------------------------------------------------------
# 4. تنفيذ المراحل 1 و 2 (توليد السكربت وأوامر الصور)
# -------------------------------------------------------------

async def run_stage1_and_2(query, context, episode):
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    ep_id = str(episode.get("id", "201"))

    session["episode_id"] = ep_id
    session["episode_data"] = episode
    session["uploaded_count"] = 0

    status_card = (
        f"<b>⚙️ جاري معالجة الحلقة #{ep_id}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏳ <b>[ 1/5 ]</b> توليد السكربت الإنجليزي وهندسة الجمل القصيرة...\n"
        f"⚪ <b>[ 2/5 ]</b> توليد وتجزئة أوامر الصور (24 لكل ملف)\n"
        f"⚪ <b>[ 3/5 ]</b> التعليق الصوتي التعبيري\n"
        f"⚪ <b>[ 4/5 ]</b> المونتاج والدمج الآلي (FFmpeg)\n"
        f"⚪ <b>[ 5/5 ]</b> التغليف والنشر الرقمي\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    status_msg = await query.edit_message_text(status_card, parse_mode=ParseMode.HTML)

    # 1. المرحلة الأولى
    try:
        stage1_res = generate_stage1_script(episode)
        s1_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"
        with open(s1_file, "w", encoding="utf-8") as f:
            json.dump(stage1_res, f, ensure_ascii=False, indent=2)

        sentences = stage1_res.get("full_script_sentences", [])
        words_cnt = stage1_res.get("total_word_count", 0)
        session["sentences"] = sentences

        status_card = (
            f"<b>⚙️ جاري معالجة الحلقة #{ep_id}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ <b>[ 1/5 ]</b> تم إنتاج السكربت ({words_cnt} كلمة | {len(sentences)} جملة)\n"
            f"⏳ <b>[ 2/5 ]</b> جاري صياغة أوامر الصور وتجزئتها بدقة (1:1)...\n"
            f"⚪ <b>[ 3/5 ]</b> التعليق الصوتي التعبيري\n"
            f"⚪ <b>[ 4/5 ]</b> المونتاج والدمج الآلي (FFmpeg)\n"
            f"⚪ <b>[ 5/5 ]</b> التغليف والنشر الرقمي\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        await status_msg.edit_text(status_card, parse_mode=ParseMode.HTML)

    except Exception as e:
        await status_msg.edit_text(f"❌ <b>خطأ أثناء توليد السكربت:</b>\n<code>{str(e)}</code>", parse_mode=ParseMode.HTML)
        return

    # 2. المرحلة الثانية: تجزئة ملفات البرومبتات
    try:
        prompts_dir = OUTPUTS_DIR / f"episode_{ep_id}_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        batches = generate_stage2_prompts_batches(sentences)
        for idx, batch in enumerate(batches, start=1):
            p_file = prompts_dir / f"prompts_part_{idx:02d}.txt"
            with open(p_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(batch))

            await context.bot.send_document(
                chat_id=chat_id,
                document=open(p_file, "rb"),
                caption=f"📦 <b>حزمة أوامر الصور: الجزء [{idx:02d}]</b>\n└ يحتوي على <b>{len(batch)}</b> برومبت جاهز للنسخ المباشر.",
                parse_mode=ParseMode.HTML
            )

        status_card = (
            f"<b>✅ اكتملت المرحلتان (1 و 2) بنجاح!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📝 <b>إجمالي الجمل المولدة:</b> <code>{len(sentences)}</code> جملة.\n"
            f"📦 <b>الملفات النصية:</b> تم إرسال <code>{len(batches)}</code> ملفات (.txt).\n"
            f"🎯 <b>الخطوة التالية:</b> تحديد محرك وهندسة الصوت التعبيري."
        )
        await context.bot.send_message(chat_id=chat_id, text=status_card, parse_mode=ParseMode.HTML)
        await present_audio_engine_choice(context.bot, chat_id=chat_id)

    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ <b>خطأ في المرحلة الثانية:</b>\n<code>{str(e)}</code>", parse_mode=ParseMode.HTML)


async def present_audio_engine_choice(bot_or_query, chat_id=None):
    text = (
        "🎙️ <b>المرحلة 3: اختيار محرك وهندسة الصوت التعبيري</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "يرجى تحديد المحرك الصوتي المعتمد لهذه الحلقة:\n\n"
        "• <b>Microsoft Azure:</b> يدعم معيار SSML وتلوين نبرة الصوت بين الغضب، الحماس، والهمس.\n"
        "• <b>Google Cloud TTS:</b> يدعم وسوم الانفعالات الديناميكية والإيموجي السياقي داخل الجمل."
    )
    keyboard = [
        [
            InlineKeyboardButton("⚡ Microsoft Azure Speech (SSML)", callback_data="engine_azure"),
        ],
        [
            InlineKeyboardButton("🌐 Google Cloud TTS (Expressive)", callback_data="engine_google")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if chat_id:
        await bot_or_query.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    else:
        await bot_or_query.edit_message_text(text=text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


# -------------------------------------------------------------
# 5. تنفيذ المرحلة 3 (توليد الصوت الموحد)
# -------------------------------------------------------------

async def run_stage3(query, context):
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    ep_id = session.get("episode_id")
    sentences = session.get("sentences", [])
    engine = session.get("engine", "azure")
    voice = session.get("voice", "en-US-GuyNeural")

    wait_msg = await query.edit_message_text(
        f"⏳ <b>جاري توليد ملف الصوت الموحد عبر {engine.upper()}...</b>\n"
        f"🗣️ الصوت المختار: <code>{voice}</code>\n"
        f"<i>يتم الآن فحص وتطبيق القواعد الصوتية والانفعالات الصارمة...</i>",
        parse_mode=ParseMode.HTML
    )

    try:
        audio_path = generate_stage3_audio(
            episode_id=ep_id,
            sentences=sentences,
            engine=engine,
            voice=voice,
            output_dir=OUTPUTS_DIR
        )
        session["audio_path"] = audio_path

        ready_card = (
            f"<b>🎉 تم توليد التعليق الصوتي الموحد بنجاح!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎙️ <b>الصوت:</b> <code>{voice}</code> ({engine.upper()})\n"
            f"📁 <b>الملف:</b> <code>{audio_path.name}</code>\n\n"
            f"📸 <b>المرحلة 4: استقبال وتجهيز صور الفيديو</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"1️⃣ قم بتوليد الصور على منصة الصور عبر ملفات الـ TXT المرسلة لك.\n"
            f"2️⃣ أرسل الصور هنا في الشات (كصور أو كملفات دفعة واحدة).\n"
            f"3️⃣ سيتعرف السيرفر تلقائياً على ترتيب كل صورة بالـ OCR.\n\n"
            f"👇 <b>عند الانتهاء من رفع كافة الصور ({len(sentences)} صورة)، اضغط الزر أدناه:</b>"
        )
        keyboard = [[InlineKeyboardButton("🎬 فحص الصور وبدء الرندرة الآلية", callback_data="btn_start_render")]]
        await wait_msg.edit_text(ready_card, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)

    except Exception as e:
        await wait_msg.edit_text(f"❌ <b>خطأ أثناء توليد الصوت:</b>\n<code>{str(e)}</code>", parse_mode=ParseMode.HTML)


# -------------------------------------------------------------
# 6. استقبال وتجميع الصور مع عداد فوري
# -------------------------------------------------------------

async def handle_media_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    session = get_session(chat_id)
    ep_id = session.get("episode_id", "201")
    total_expected = len(session.get("sentences", []))

    raw_dir = OUTPUTS_DIR / f"episode_{ep_id}_raw_images"
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_obj = None
    file_name = None

    if update.message.document and update.message.document.mime_type.startswith("image/"):
        file_obj = await update.message.document.get_file()
        file_name = update.message.document.file_name
    elif update.message.photo:
        photo = update.message.photo[-1]
        file_obj = await photo.get_file()
        file_name = f"photo_{photo.file_unique_id}.png"

    if file_obj:
        save_path = raw_dir / file_name
        await file_obj.download_to_drive(save_path)
        session["uploaded_count"] += 1

        # إشعار سريع للمستخدم بالتقدم كل 5 صور أو عند الاكتمال
        current = session["uploaded_count"]
        if current % 5 == 0 or (total_expected > 0 and current == total_expected):
            pct = int((current / total_expected * 100)) if total_expected > 0 else 0
            keyboard = [[InlineKeyboardButton("🎬 فحص الصور وبدء الرندرة الآلية", callback_data="btn_start_render")]]
            await update.message.reply_text(
                f"📥 <b>تم استلام وحفظ:</b> <code>{current} / {total_expected}</code> صورة ({pct}%)\n"
                f"إذا انتهيت من رفع الحزمة كاملة، اضغط على الزر أدناه لبدء المونتاج.",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.HTML
            )


# -------------------------------------------------------------
# 7. تنفيذ المرحلتين 4 و 5 (المونتاج الآلي ورسائل النشر الأربعة)
# -------------------------------------------------------------

async def run_stage4_and_5(msg_obj, context):
    chat_id = msg_obj.chat_id
    session = get_session(chat_id)
    ep_id = session.get("episode_id", "201")
    sentences = session.get("sentences", [])
    total_expected = len(sentences)

    raw_dir = OUTPUTS_DIR / f"episode_{ep_id}_raw_images"
    clean_dir = OUTPUTS_DIR / f"episode_{ep_id}_clean_frames"

    progress_msg = await context.bot.send_message(
        chat_id=chat_id,
        text="🔍 <b>جاري فحص الركن السفلي الأيمن للصور بالـ OCR ومطابقة الترتيب...</b>",
        parse_mode=ParseMode.HTML
    )

    # 1. كشف النواقص وتطبيق الرقعة الذكية
    try:
        frames = process_and_verify_images(
            uploaded_images_dir=raw_dir,
            output_frames_dir=clean_dir,
            expected_total=total_expected
        )
    except MissingAssetsError as m_err:
        keyboard = [[InlineKeyboardButton("🔄 إعادة الفحص بعد رفع النواقص", callback_data="btn_start_render")]]
        await progress_msg.edit_text(
            f"🚨 <b>تنبيه: أصول مفقودة (Missing Images)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"تم التحقق بنجاح من <b>{m_err.found_count}</b> صورة من أصل <b>{m_err.total_expected}</b>.\n\n"
            f"⚠️ <b>الصور المفقودة المطلوب رفعها:</b>\n"
            f"<code>{m_err.missing_indices}</code>\n\n"
            f"قم بتوليد هذه الأرقام ورفعها هنا، ثم اضغط على زر إعادة الفحص.",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.HTML
        )
        return
    except Exception as e:
        await progress_msg.edit_text(f"❌ <b>خطأ أثناء معالجة الصور:</b>\n<code>{str(e)}</code>", parse_mode=ParseMode.HTML)
        return

    # 2. المزامنة والرندرة عبر FFmpeg
    await progress_msg.edit_text(
        f"<b>🎬 بدء المونتاج والرندرة الآلية عبر FFmpeg (1080p 60fps)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ تم مطابقة {len(frames)} صورة وتطبيق الرقعة الذكية واختفاء الأرقام 100%.\n"
        f"⏳ جاري المزامنة الدقيقة بالمللي ثانية وتوليد الترجمة الحركية الصفراء (#FDE047)...\n"
        f"⏳ جاري تطبيق دورة حركات Ken Burns والانتقالات الهوائية (-18dB)...\n\n"
        f"<i>قد تستغرق الرندرة من دقيقتين إلى 4 دقائق حسب سرعة المعالج...</i>",
        parse_mode=ParseMode.HTML
    )

    try:
        audio_file = session["audio_path"]
        subtitles_ass = OUTPUTS_DIR / f"episode_{ep_id}_subtitles.ass"
        final_video = OUTPUTS_DIR / f"episode_{ep_id}_final_1080p.mp4"

        timeline = align_audio_and_generate_ass(audio_file, sentences, subtitles_ass)
        render_final_video(frames, timeline, audio_file, subtitles_ass, final_video)

        file_size_mb = final_video.stat().st_size / (1024 * 1024)
        if file_size_mb < 49:
            await context.bot.send_video(
                chat_id=chat_id,
                video=open(final_video, "rb"),
                caption=f"🏆 <b>فيديو الحلقة #{ep_id} جاهز للنشر!</b>\nالدقة: 1080p Full HD @ 60fps | الحجم: {file_size_mb:.1f} MB",
                parse_mode=ParseMode.HTML
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🏆 <b>تم تصدير الفيديو النهائي بنجاح على السيرفر!</b>\n"
                     f"📊 الحجم: <code>{file_size_mb:.1f} MB</code> (أكبر من حد تيليجرام 50MB)\n"
                     f"📁 المسار المباشر على السيرفر:\n<code>{final_video}</code>",
                parse_mode=ParseMode.HTML
            )

    except Exception as e:
        await progress_msg.edit_text(f"❌ <b>حدث خطأ أثناء الرندرة:</b>\n<code>{str(e)}</code>", parse_mode=ParseMode.HTML)
        return

    # 3. إرسال حزمة النشر الرقمي (المرحلة الخامسة: 4 رسائل منفصلة)
    await context.bot.send_message(
        chat_id=chat_id,
        text="📦 <b>جاري إرسال حزمة النشر الرقمي (العناوين، الغلاف، الوصف، والتاجز)...</b>",
        parse_mode=ParseMode.HTML
    )

    try:
        meta = generate_stage5_metadata(session["episode_data"], sentences)

        # الرسالة الأولى: العناوين
        await context.bot.send_message(chat_id=chat_id, text=meta.get("titles_message"))
        # الرسالة الثانية: أوامر الغلاف
        await context.bot.send_message(chat_id=chat_id, text=meta.get("thumbnails_message"))
        # الرسالة الثالثة: الوصف
        await context.bot.send_message(chat_id=chat_id, text=meta.get("description_message"))
        # الرسالة الرابعة: التاجز
        await context.bot.send_message(chat_id=chat_id, text=meta.get("tags_message"))

        await context.bot.send_message(
            chat_id=chat_id,
            text="🎉 <b>ألف مبروك! اكتملت دورة إنتاج الحلقة بنسبة 100% وأصبحت جاهزة لليوتيوب فوراً.</b>",
            parse_mode=ParseMode.HTML
        )

    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ تعذر استخراج ميتاداتا النشر: {str(e)}")


# -------------------------------------------------------------
# الدالة الأساسية لتشغيل البوت
# -------------------------------------------------------------

def main():
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN غير موجود في ملف .env!")

    app = ApplicationBuilder().token(TOKEN).build()

    # الأوامر والأزرار
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_media_upload))

    print("=" * 60)
    print("🚀 محرك Vot Studio Pro يعمل الآن بنجاح على سيرفر الويندوز...")
    print("=" * 60)
    app.run_polling()


if __name__ == "__main__":
    main()
