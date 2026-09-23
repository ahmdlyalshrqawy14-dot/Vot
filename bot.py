import os
import json
import shutil
import asyncio
import logging
import html
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
    BASE_DIR,
)
from stage1_generator import generate_stage1_script
from stage2_generator import generate_stage2_prompts_batches
from stage3_audio import generate_stage3_audio, generate_voice_preview
from stage4_vision import process_and_verify_images, MissingAssetsError
from stage4_subtitles import align_audio_and_generate_ass
from stage4_composer import render_final_video
from stage5_metadata import generate_stage5_metadata

# -----------------------------------------------------------------
# إعداد السجلات
# -----------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("VotStudioBot")

for _noisy_logger in (
    "httpx",
    "httpcore",
    "telegram",
    "telegram.ext",
    "telegram.request",
    "telegram.ext.Updater",
    "telegram.ext.Application",
):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
EPISODES_FILE = BASE_DIR / "episodes.json"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------
# إدارة الجلسات والمهام
# -----------------------------------------------------------------
user_sessions = {}
user_tasks = {}

VOICE_PREVIEW_TEXT = "Hello, this is a sample of my voice. How do I sound to you?"


def _fresh_session():
    """ينشئ جلسة نظيفة تماماً برمز تعريف فريد (token)."""
    return {
        "state": "IDLE",
        "episode_id": None,
        "episode_data": None,
        "sentences": [],
        "audio_path": None,
        "uploaded_count": 0,
        "engine": None,
        "voice": None,
        "cancelled": False,
        "token": object(),
        "stage1_result": None,
        "stage1_approved": False,
        "stage1_plan_message_id": None,
        "manual_map": {},                 # {int(slot): Path}
        "manual_queue": [],               # [Path, ...]
        "manual_missing": [],             # [int, ...]
        "manual_current": None,           # Path الحالية
        "manual_unindexed_files": [],     # [Path, ...]
        "manual_missing_indices": [],     # [int, ...]
    }


def get_session(chat_id):
    if chat_id not in user_sessions:
        user_sessions[chat_id] = _fresh_session()
    return user_sessions[chat_id]


def _is_stale(chat_id, session):
    current = user_sessions.get(chat_id)
    return current is not session or session.get("cancelled", False)


def _register_task(chat_id):
    try:
        task = asyncio.current_task()
        if task:
            user_tasks[chat_id] = task
    except RuntimeError:
        pass


def _cancel_user_task(chat_id):
    task = user_tasks.get(chat_id)
    if task and not task.done():
        try:
            task.cancel()
            logger.info(f"🛑 تم إرسال Cancel للـ Task الجاري في {chat_id}")
            return True
        except Exception as e:
            logger.warning(f"فشل إلغاء الـ Task: {e}")
    return False


def _esc(val, fallback: str = "—") -> str:
    """يهرّب أي قيمة قادمة من النموذج قبل وضعها في رسائل HTML."""
    if val is None:
        return fallback
    try:
        s = str(val)
    except Exception:
        return fallback
    if not s.strip():
        return fallback
    return html.escape(s)


def _as_list(val):
    return val if isinstance(val, list) else []


def _as_dict(val):
    return val if isinstance(val, dict) else {}


def _build_stage1_summary(ep_id, episode, stage1_res) -> str:
    """يبني رسالة ملخص المرحلة الأولى (HTML Safe)."""
    if not isinstance(stage1_res, dict):
        stage1_res = {}

    cb = _as_dict(stage1_res.get("creative_brief"))
    rp = _as_dict(stage1_res.get("retention_plan"))
    qr = _as_dict(stage1_res.get("quality_report"))

    scene_plan = _as_list(stage1_res.get("scene_plan"))
    open_loops = _as_list(rp.get("open_loops"))
    pattern_interrupts = _as_list(rp.get("pattern_interrupts"))
    sentences = _as_list(stage1_res.get("full_script_sentences"))

    words_cnt = stage1_res.get("total_word_count", 0)
    try:
        words_cnt_int = int(words_cnt)
    except Exception:
        words_cnt_int = 0

    approved = bool(qr.get("approved"))
    approved_tag = "✅ نعم" if approved else "⚠️ لا"

    topic = _esc(episode.get("topic", "—")) if isinstance(episode, dict) else "—"

    lines = [
        f"<b>📋 ملخص المرحلة الأولى — الحلقة #{_esc(ep_id)}</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📌 <b>الموضوع:</b> {topic}",
        f"📝 <b>عدد الكلمات:</b> <code>{words_cnt_int}</code>",
        f"🧩 <b>عدد الجمل:</b> <code>{len(sentences)}</code>",
        f"🎬 <b>عدد المشاهد في الخطة:</b> <code>{len(scene_plan)}</code>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "<b>🧠 الفكرة الإبداعية (Creative Brief)</b>",
        f"├ 💡 <b>الفكرة الأساسية:</b> {_esc(cb.get('core_idea'))}",
        f"├ 🎯 <b>الزاوية الفريدة:</b> {_esc(cb.get('unique_angle'))}",
        f"├ 🎞️ <b>صيغة الحلقة:</b> {_esc(cb.get('episode_format'))}",
        f"└ 🎁 <b>مخرجات المشاهد:</b> {_esc(cb.get('viewer_outcome'))}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "<b>🪝 الخطاف (Hook)</b>",
        f"{_esc(stage1_res.get('hook'))}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "<b>📈 خطة الاحتفاظ (Retention Plan)</b>",
        f"├ 🎣 <b>استراتيجية الخطاف:</b> {_esc(rp.get('hook_strategy'))}",
        f"├ 🔁 <b>عدد الحلقات المفتوحة:</b> <code>{len(open_loops)}</code>",
        f"├ ⚡ <b>عدد مقاطعات النمط:</b> <code>{len(pattern_interrupts)}</code>",
        f"└ 🎁 <b>المكافأة النهائية:</b> {_esc(rp.get('payoff'))}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"<b>✅ تقرير الجودة:</b> معتمد = {approved_tag}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "هل تود اعتماد هذه الخطة والانتقال إلى <b>المرحلة الثانية</b>؟",
    ]
    return "\n".join(lines)


def _cleanup_episode_temp_files(ep_id):
    """يحذف كل المجلدات المؤقتة للحلقة لتفادي استهلاك المساحة."""
    if not ep_id:
        return
    targets = [
        OUTPUTS_DIR / f"episode_{ep_id}_prompts",
        OUTPUTS_DIR / f"episode_{ep_id}_raw_images",
        OUTPUTS_DIR / f"episode_{ep_id}_clean_frames",
        OUTPUTS_DIR / f"episode_{ep_id}_clean_frames_renamed",
        OUTPUTS_DIR / f"episode_{ep_id}_temp_segments",
        OUTPUTS_DIR / f"episode_{ep_id}_temp_segments_audio",
        OUTPUTS_DIR / "temp_segments",
        *OUTPUTS_DIR.glob("episode_*_temp_segments_ep*"),
    ]
    for t in targets:
        try:
            if t.exists() and t.is_dir():
                shutil.rmtree(t, ignore_errors=True)
                logger.info(f"🧹 تم حذف المجلد المؤقت: {t.name}")
        except Exception as e:
            logger.warning(f"تعذّر حذف {t}: {e}")


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
    for ep in episodes:
        if ep.get("status") == "pending":
            return ep
    return episodes[0]


def _format_missing_indices(missing, limit=15):
    try:
        items = list(missing)
    except TypeError:
        return str(missing)
    if len(items) <= limit:
        return ", ".join(str(x) for x in items)
    shown = ", ".join(str(x) for x in items[:limit])
    return f"{shown} ... (+{len(items) - limit} أخرى)"


# =================================================================
# 1. /start → إعادة تشغيل كاملة (Hard Reset)
# =================================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    cancelled = _cancel_user_task(chat_id)

    old_session = user_sessions.get(chat_id)
    old_ep_id = old_session.get("episode_id") if old_session else None

    if old_session:
        old_session["cancelled"] = True

    user_sessions[chat_id] = _fresh_session()
    _cleanup_episode_temp_files(old_ep_id)

    logger.info(
        f"♻️ Hard Reset للمستخدم {chat_id} | cancelled_task={cancelled} | cleaned_ep={old_ep_id}"
    )

    reset_badge = (
        "♻️ <b>تم تنفيذ إعادة التشغيل الكاملة بنجاح</b>\n"
        "├ تم إيقاف أي عملية جارية فوراً\n"
        "├ تم مسح الذاكرة المؤقتة (Session Purge)\n"
        "└ تم حذف الملفات المؤقتة من السيرفر\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

    welcome_text = (
        "<b>🎬 مرحباً بك في Vot Studio | المحرك الآلي لصناعة المحتوى</b>\n"
        f"{reset_badge}"
        "النظام السحابي المتكامل لتحويل أفكار الحلقات إلى فيديوهات يوتيوب احترافية "
        "بجودة <b>1080p 60fps</b> مع هندسة التعليق الصوتي والترجمة الحركية.\n\n"
        "<b>👇 كيف تود أن نبدأ اليوم؟</b>"
    )

    keyboard = [
        [InlineKeyboardButton("▶️ بدء الحلقة التالية المجدولة", callback_data="btn_start_next")],
        [
            InlineKeyboardButton("🔢 إدخال رقم حلقة معينة (ID)", callback_data="btn_choose_id"),
            InlineKeyboardButton("📋 استعراض الحلقات المتاحة", callback_data="btn_list_episodes"),
        ],
    ]

    await update.message.reply_text(
        welcome_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.HTML,
    )


# =================================================================
# 2. معالجة اختيارات القائمة الرئيسية
# =================================================================

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = update.effective_chat.id
    session = get_session(chat_id)

    if session.get("cancelled"):
        return

    if data == "btn_start_next":
        ep = get_episode(target_id=None)
        if not ep:
            await query.edit_message_text("⚠️ لا توجد حلقات متاحة في ملف episodes.json!")
            return
        await show_episode_confirmation(query, ep)

    elif data == "btn_choose_id":
        session["state"] = "WAITING_EPISODE_ID"
        await query.edit_message_text(
            "🔢 <b>يرجى كتابة رقم الحلقة (ID) الآن في الشات:</b>\n"
            "<i>(مثال: أرسل الرقم 201 أو 101)</i>",
            parse_mode=ParseMode.HTML,
        )

    elif data == "btn_list_episodes":
        episodes = load_all_episodes()
        if not episodes:
            await query.edit_message_text("⚠️ لا توجد حلقات مسجلة.")
            return

        msg = "<b>📋 قائمة الحلقات المسجلة في السيرفر:</b>\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        for ep in episodes[:10]:
            st = "🟡 انتظار" if ep.get("status") == "pending" else "🟢 مكتملة"
            msg += f"• <b>ID [{ep.get('id')}]:</b> {_esc(ep.get('topic'))}\n   └ الحالة: {st}\n"

        buttons = [
            [InlineKeyboardButton("▶️ بدء الحلقة التالية المجدولة", callback_data="btn_start_next")],
            [InlineKeyboardButton("🔙 رجوع للقائمة الرئيسية", callback_data="btn_back_main")],
        ]
        await query.edit_message_text(
            msg, reply_markup=InlineKeyboardMarkup(buttons), parse_mode=ParseMode.HTML
        )

    elif data == "btn_back_main":
        await start_command(query, context)

    elif data.startswith("confirm_ep_"):
        target_id = data.replace("confirm_ep_", "")
        ep = get_episode(target_id)
        if ep:
            _cancel_user_task(chat_id)
            user_tasks[chat_id] = asyncio.create_task(
                run_stage1_only(query, context, ep)
            )

    elif data.startswith("approve_stage1_"):
        target_id = data.replace("approve_stage1_", "")
        active_id = session.get("episode_id")

        if not active_id or str(active_id) != str(target_id):
            logger.warning(
                f"⚠️ approve_stage1 قديم/غير مطابق | target={target_id} | active={active_id}"
            )
            try:
                await query.answer("⚠️ هذا الزر لا يخص الحلقة النشطة الحالية.", show_alert=True)
            except Exception:
                pass
            try:
                await query.edit_message_text(
                    "⚠️ <b>زر الاعتماد قديم أو لا يخص الحلقة النشطة حالياً.</b>\n"
                    "تم تجاهل الطلب دون أي تغيير في الجلسة.\n"
                    "استخدم /start لبدء حلقة جديدة.",
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                pass
            return

        if session.get("stage1_approved") is True:
            logger.info(f"⏩ تم اعتماد المرحلة مسبقاً للحلقة {target_id}")
            try:
                await query.answer("ℹ️ تم اعتماد هذه المرحلة مسبقاً بالفعل.", show_alert=True)
            except Exception:
                pass
            return

        stage1_res = session.get("stage1_result")
        has_stage1_in_session = isinstance(stage1_res, dict)
        s1_file = OUTPUTS_DIR / f"stage1_episode_{target_id}.json"
        has_stage1_on_disk = s1_file.exists()

        if not has_stage1_in_session and not has_stage1_on_disk:
            logger.warning(
                f"⚠️ لا توجد بيانات مرحلة أولى للحلقة {target_id} "
                f"(session={has_stage1_in_session}, disk={has_stage1_on_disk})"
            )
            try:
                await query.answer("⚠️ لا توجد بيانات مرحلة أولى معتمدة لهذه الحلقة.", show_alert=True)
            except Exception:
                pass
            try:
                await query.edit_message_text(
                    "❌ <b>لا توجد بيانات مرحلة أولى صالحة لهذه الحلقة.</b>\n"
                    "الرجاء إعادة توليد المرحلة الأولى عبر /start.",
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                pass
            return

        _cancel_user_task(chat_id)
        user_tasks[chat_id] = asyncio.create_task(
            run_stage2_after_approval(query, context)
        )

    elif data.startswith("regenerate_stage1_"):
        target_id = data.replace("regenerate_stage1_", "")
        ep = session.get("episode_data") or get_episode(target_id)
        if ep:
            _cancel_user_task(chat_id)
            user_tasks[chat_id] = asyncio.create_task(
                run_stage1_only(query, context, ep)
            )

    elif data.startswith("preview_voice_"):
        preview_voice = data.replace("preview_voice_", "")
        try:
            preview_path = await asyncio.to_thread(
                generate_voice_preview,
                voice=preview_voice,
                text=VOICE_PREVIEW_TEXT,
                output_dir=OUTPUTS_DIR,
            )
            with open(preview_path, "rb") as audio_file:
                await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=audio_file,
                    title=f"Voice Preview — {preview_voice}",
                    caption=(
                        f"🎧 <b>عينة صوت Azure:</b> <code>{preview_voice}</code>\n"
                        f"<i>{VOICE_PREVIEW_TEXT}</i>"
                    ),
                    parse_mode=ParseMode.HTML,
                )
        except Exception as e:
            logger.error(f"فشل توليد معاينة الصوت {preview_voice}: {e}")
            try:
                await query.answer(f"❌ فشل توليد العينة: {e}", show_alert=True)
            except Exception:
                pass
        return

    elif data.startswith("voice_"):
        selected_voice = data.replace("voice_", "")
        session["voice"] = selected_voice
        session["engine"] = "azure"
        user_tasks[chat_id] = asyncio.create_task(
            run_stage3(query, context)
        )

    elif data == "btn_start_render":
        user_tasks[chat_id] = asyncio.create_task(
            run_stage4_and_5(query.message, context, allow_partial=False)
        )

    elif data == "btn_force_render":
        user_tasks[chat_id] = asyncio.create_task(
            run_stage4_and_5(query.message, context, allow_partial=True)
        )

    elif data == "manual_cancel":
        session["state"] = "IDLE"
        session["manual_map"] = {}
        session["manual_queue"] = []
        session["manual_missing"] = []
        session["manual_current"] = None
        try:
            await query.edit_message_caption(
                caption=(
                    "❌ <b>تم إلغاء الوضع اليدوي.</b>\n"
                    "يمكنك رفع الصور الناقصة أو المتابعة بالصور المتوفرة."
                ),
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            try:
                await query.message.delete()
            except Exception:
                pass
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ <b>تم إلغاء الوضع اليدوي.</b>",
                parse_mode=ParseMode.HTML,
            )
        return

    elif data == "btn_manual_assign":
        unindexed = session.get("manual_unindexed_files", []) or []
        missing_idx = session.get("manual_missing_indices", []) or []
        if not unindexed:
            await query.answer("لا توجد صور بحاجة لتعيين يدوي.", show_alert=True)
            return

        session["manual_map"] = {}
        session["manual_queue"] = list(unindexed)
        session["manual_missing"] = list(missing_idx)
        session["manual_current"] = None
        session["state"] = "MANUAL_ASSIGN"

        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass

        await show_next_manual_image(context, chat_id)

    elif data.startswith("manual_assign_"):
        slot = data.replace("manual_assign_", "")
        await handle_manual_assign(query, context, slot)

    elif data == "manual_skip":
        session["manual_current"] = None
        try:
            await query.answer("⏭️ تم تخطي هذه الصورة")
        except Exception:
            pass
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        await show_next_manual_image(context, chat_id)


async def show_episode_confirmation(query, ep):
    ep_id = ep.get("id", "201")
    topic = ep.get("topic", "بدون عنوان")
    myth = ep.get("the_myth", "غير محدد")
    status = ep.get("status", "pending")
    status_tag = "🟡 قيد الانتظار (Ready to Produce)" if status == "pending" else "🟢 تم إنتاجها مسبقاً"

    card_text = (
        f"<b>🎯 بطاقة بيانات الحلقة المحددة</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🆔 <b>رقم الحلقة:</b> <code>{_esc(ep_id)}</code>\n"
        f"📌 <b>الموضوع:</b> <b>{_esc(topic)}</b>\n"
        f"💡 <b>الخرافة المستهدفة:</b> <i>{_esc(myth)}</i>\n"
        f"📊 <b>الحالة الحالية:</b> {status_tag}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"هل تود بدء <b>المرحلة الأولى</b> (السكربت + الخطة الإبداعية) لهذه الحلقة؟"
    )

    keyboard = [
        [InlineKeyboardButton("🚀 تأكيد وبدء المرحلة الأولى", callback_data=f"confirm_ep_{ep_id}")],
        [InlineKeyboardButton("❌ إلغاء والعودة للقائمة", callback_data="btn_back_main")],
    ]
    await query.edit_message_text(
        card_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.HTML,
    )


# =================================================================
# 3. الرسائل النصية
# =================================================================

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    session = get_session(chat_id)

    if session.get("cancelled"):
        return

    if session.get("state") == "WAITING_EPISODE_ID":
        entered_text = update.message.text.strip()
        ep = get_episode(target_id=entered_text)
        if ep:
            session["state"] = "IDLE"
            ep_id = ep.get("id")
            card_text = (
                f"<b>🎯 تم العثور على الحلقة بنجاح!</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🆔 <b>رقم الحلقة:</b> <code>{_esc(ep_id)}</code>\n"
                f"📌 <b>الموضوع:</b> <b>{_esc(ep.get('topic'))}</b>\n"
                f"💡 <b>الخرافة:</b> <i>{_esc(ep.get('the_myth'))}</i>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"جاهز للبدء؟"
            )
            keyboard = [
                [InlineKeyboardButton("🚀 تأكيد وبدء المرحلة الأولى", callback_data=f"confirm_ep_{ep_id}")],
                [InlineKeyboardButton("❌ إلغاء والعودة", callback_data="btn_back_main")],
            ]
            await update.message.reply_text(
                card_text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.HTML,
            )
        else:
            await update.message.reply_text(
                f"❌ لم يتم العثور على حلقة برقم ID <code>{_esc(entered_text)}</code>.\n"
                f"تأكد من الرقم وحاول مجدداً، أو اضغط /start للعودة للقائمة.",
                parse_mode=ParseMode.HTML,
            )


# =================================================================
# 4. المرحلة الأولى فقط + عرض الخطة واعتماد المستخدم
# =================================================================

async def run_stage1_only(query, context, episode):
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    ep_id = str(episode.get("id", "201"))

    session["episode_id"] = ep_id
    session["episode_data"] = episode
    session["uploaded_count"] = 0
    session["stage1_approved"] = False
    session["stage1_result"] = None

    if _is_stale(chat_id, session):
        return

    status_card = (
        f"<b>⚙️ [ 1/2 ] جاري تشغيل المرحلة الأولى للحلقة #{_esc(ep_id)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏳ توليد السكربت الإنجليزي + هندسة الجمل القصيرة + الخطة الإبداعية...\n"
        f"<i>لن تبدأ المرحلة الثانية حتى تعتمد الخطة بنفسك.</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    try:
        status_msg = await query.edit_message_text(status_card, parse_mode=ParseMode.HTML)
    except Exception:
        return

    session["stage1_plan_message_id"] = status_msg.message_id

    try:
        stage1_res = await asyncio.to_thread(generate_stage1_script, episode)

        if _is_stale(chat_id, session):
            logger.info(f"⛔ تم إيقاف المرحلة 1 للحلقة {ep_id} بسبب /start")
            return

        s1_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"
        with open(s1_file, "w", encoding="utf-8") as f:
            json.dump(stage1_res, f, ensure_ascii=False, indent=2)

        sentences = _as_list(stage1_res.get("full_script_sentences"))
        session["stage1_result"] = stage1_res
        session["sentences"] = sentences

        summary_text = _build_stage1_summary(ep_id, episode, stage1_res)

        keyboard = [
            [InlineKeyboardButton(
                "✅ اعتماد الخطة والانتقال للمرحلة الثانية",
                callback_data=f"approve_stage1_{ep_id}",
            )],
            [InlineKeyboardButton(
                "🔄 إعادة توليد المرحلة الأولى",
                callback_data=f"regenerate_stage1_{ep_id}",
            )],
            [InlineKeyboardButton("❌ إلغاء والعودة للقائمة", callback_data="btn_back_main")],
        ]

        await status_msg.edit_text(
            summary_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.HTML,
        )

    except asyncio.CancelledError:
        logger.info(f"🛑 تم إلغاء المرحلة 1 للحلقة {ep_id}")
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        logger.exception(f"خطأ في المرحلة الأولى للحلقة {ep_id}")
        try:
            await status_msg.edit_text(
                f"❌ <b>خطأ أثناء توليد المرحلة الأولى:</b>\n"
                f"<code>{_esc(str(e))}</code>\n\n"
                f"<i>يمكنك إعادة المحاولة عبر /start.</i>",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass


# =================================================================
# 4.b المرحلة الثانية — تُشغّل فقط بعد اعتماد المستخدم
# =================================================================

async def run_stage2_after_approval(query, context):
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    ep_id = session.get("episode_id")

    if _is_stale(chat_id, session):
        return

    if not ep_id:
        try:
            await query.edit_message_text("⚠️ لا توجد حلقة نشطة. اضغط /start للبدء.")
        except Exception:
            pass
        return

    stage1_res = session.get("stage1_result")
    if not isinstance(stage1_res, dict):
        s1_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"
        if s1_file.exists():
            try:
                with open(s1_file, "r", encoding="utf-8") as f:
                    stage1_res = json.load(f)
                session["stage1_result"] = stage1_res
            except Exception as e:
                logger.error(f"فشل قراءة stage1_episode_{ep_id}.json: {e}")
                stage1_res = None

    if not isinstance(stage1_res, dict):
        try:
            await query.edit_message_text(
                "❌ <b>تعذّر العثور على بيانات المرحلة الأولى.</b>\n"
                "الرجاء إعادة توليد المرحلة الأولى من جديد.",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass
        return

    sentences = _as_list(stage1_res.get("full_script_sentences"))
    if not sentences:
        sentences = session.get("sentences") or []

    if not sentences:
        try:
            await query.edit_message_text(
                "❌ <b>لا توجد جمل صالحة لتشغيل المرحلة الثانية.</b>\n"
                "الرجاء إعادة توليد المرحلة الأولى.",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass
        return

    session["sentences"] = sentences
    session["stage1_approved"] = True

    status_card = (
        f"<b>⚙️ [ 2/2 ] جاري تشغيل المرحلة الثانية للحلقة #{_esc(ep_id)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ تم اعتماد خطة المرحلة الأولى.\n"
        f"⏳ جاري صياغة أوامر الصور وتجزئتها بدقة (1:1)...\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    try:
        status_msg = await query.edit_message_text(status_card, parse_mode=ParseMode.HTML)
    except Exception:
        return

    try:
        prompts_dir = OUTPUTS_DIR / f"episode_{ep_id}_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        batches = await asyncio.to_thread(
            generate_stage2_prompts_batches,
            sentences,
            stage1_result=stage1_res,
        )

        if _is_stale(chat_id, session):
            logger.info(f"⛔ تم إيقاف المرحلة 2 للحلقة {ep_id} بسبب /start")
            return

        for idx, batch in enumerate(batches, start=1):
            if _is_stale(chat_id, session):
                return

            p_file = prompts_dir / f"prompts_part_{idx:02d}.txt"
            with open(p_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(batch))

            with open(p_file, "rb") as fp:
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=fp,
                    caption=(
                        f"📦 <b>حزمة أوامر الصور: الجزء [{idx:02d}]</b>\n"
                        f"└ يحتوي على <b>{len(batch)}</b> برومبت جاهز للنسخ المباشر."
                    ),
                    parse_mode=ParseMode.HTML,
                )

        if _is_stale(chat_id, session):
            return

        done_card = (
            f"<b>✅ اكتملت المرحلة الثانية بنجاح!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📝 <b>إجمالي الجمل:</b> <code>{len(sentences)}</code> جملة.\n"
            f"📦 <b>الملفات النصية:</b> تم إرسال <code>{len(batches)}</code> ملفات (.txt).\n"
            f"🎯 <b>الخطوة التالية:</b> تحديد الصوت التعبيري."
        )
        try:
            await status_msg.edit_text(done_card, parse_mode=ParseMode.HTML)
        except Exception:
            await context.bot.send_message(chat_id=chat_id, text=done_card, parse_mode=ParseMode.HTML)

        await present_audio_engine_choice(context.bot, chat_id=chat_id)

    except asyncio.CancelledError:
        logger.info(f"🛑 تم إلغاء المرحلة 2 للحلقة {ep_id}")
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        logger.exception(f"خطأ في المرحلة الثانية للحلقة {ep_id}")
        try:
            await status_msg.edit_text(
                f"❌ <b>خطأ في المرحلة الثانية:</b>\n<code>{_esc(str(e))}</code>",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass


async def present_audio_engine_choice(bot_or_query, chat_id=None):
    text = (
        "🎙️ <b>المرحلة 3: اختيار الصوت التعبيري (Microsoft Azure)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "يمكنك الاستماع لعينة فورية لأي صوت قبل الاختيار عبر زر <b>🎧 استمع للعينة</b>،\n"
        "ثم اضغط على اسم الصوت لاعتماده وبدء توليد التعليق الصوتي.\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    descriptions = {
        "en-US-GuyNeural": "تلوين انفعالي كامل وشامل 🌟",
        "en-US-DavisNeural": "سرد ناضج وهادئ وقوي 📖",
        "en-US-TonyNeural": "صوت حماسي وواثق وعالي الطاقة ⚡",
        "en-US-JasonNeural": "نبرة شبابية وسريعة وخفيفة 🚀",
    }

    buttons = []
    for v in AZURE_MALE_VOICES:
        label = descriptions.get(v, v)
        buttons.append([
            InlineKeyboardButton(f"🗣️ {label}", callback_data=f"voice_{v}"),
            InlineKeyboardButton("🎧 استمع للعينة", callback_data=f"preview_voice_{v}"),
        ])

    reply_markup = InlineKeyboardMarkup(buttons)

    if chat_id:
        await bot_or_query.send_message(
            chat_id=chat_id, text=text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )
    else:
        await bot_or_query.edit_message_text(
            text=text, reply_markup=reply_markup, parse_mode=ParseMode.HTML
        )


# =================================================================
# 5. المرحلة 3 — تمرير stage1_result لـ Voice Director (مع Disk Fallback)
# =================================================================

async def run_stage3(query, context):
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    ep_id = session.get("episode_id")
    sentences = session.get("sentences", [])
    engine = "azure"
    voice = session.get("voice", "en-US-GuyNeural")

    # ✅ استرجاع مخرجات المرحلة الأولى الكاملة لدعم Voice Director
    episode_context = session.get("stage1_result")
    if not isinstance(episode_context, dict):
        s1_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"
        if s1_file.exists():
            try:
                with open(s1_file, "r", encoding="utf-8") as f:
                    episode_context = json.load(f)
                session["stage1_result"] = episode_context
            except Exception as e:
                logger.error(f"فشل قراءة سياق المرحلة الأولى من القرص: {e}")
                episode_context = session.get("episode_data")
        else:
            episode_context = session.get("episode_data")

    if _is_stale(chat_id, session):
        return

    try:
        wait_msg = await query.edit_message_text(
            f"⏳ <b>جاري توليد ملف الصوت الموحد عبر {engine.upper()}...</b>\n"
            f"🗣️ الصوت المختار: <code>{voice}</code>\n"
            f"<i>يتم الآن فحص وتطبيق القواعد الصوتية والانفعالات الصارمة...</i>",
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        return

    try:
        audio_path = await asyncio.to_thread(
            generate_stage3_audio,
            episode_id=ep_id,
            sentences=sentences,
            engine=engine,
            voice=voice,
            output_dir=OUTPUTS_DIR,
            episode_context=episode_context,
        )

        if _is_stale(chat_id, session):
            logger.info(f"⛔ تم إلغاء المرحلة 3 للحلقة {ep_id} بسبب /start")
            return

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
        keyboard = [
            [InlineKeyboardButton("🎬 فحص الصور وبدء الرندرة الآلية", callback_data="btn_start_render")]
        ]
        await wait_msg.edit_text(
            ready_card, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML
        )

    except asyncio.CancelledError:
        logger.info(f"🛑 تم إلغاء المرحلة 3 للحلقة {ep_id}")
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        await wait_msg.edit_text(
            f"❌ <b>خطأ أثناء توليد الصوت:</b>\n<code>{_esc(str(e))}</code>",
            parse_mode=ParseMode.HTML,
        )


# =================================================================
# 6. استقبال الصور
# =================================================================

async def handle_media_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    session = get_session(chat_id)

    if _is_stale(chat_id, session) or not session.get("episode_id"):
        return

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

        current = session["uploaded_count"]
        if current % 5 == 0 or (total_expected > 0 and current == total_expected):
            pct = int((current / total_expected * 100)) if total_expected > 0 else 0
            keyboard = [
                [InlineKeyboardButton("🎬 فحص الصور وبدء الرندرة الآلية", callback_data="btn_start_render")]
            ]
            await update.message.reply_text(
                f"📥 <b>تم استلام وحفظ:</b> <code>{current} / {total_expected}</code> صورة ({pct}%)\n"
                f"إذا انتهيت من رفع الحزمة كاملة، اضغط على الزر أدناه لبدء المونتاج.",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.HTML,
            )


# =================================================================
# دالة المعالجة اليدوية + حماية Anti-Spam + توحيد نوع المفتاح
# =================================================================

async def handle_manual_assign(query, context, slot):
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    current = session.get("manual_current")
    if not current:
        return

    try:
        slot_int = int(slot)
    except (ValueError, TypeError):
        logger.warning(f"⚠️ slot غير صالح: {slot}")
        return

    session["manual_map"][slot_int] = current
    session["manual_current"] = None

    try:
        await query.answer(f"✅ تم تعيين الصورة للرقم {slot_int}")
    except Exception:
        pass

    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass

    logger.info(f"✅ تعيين يدوي: {Path(current).name} → slot {slot_int}")
    await show_next_manual_image(context, chat_id)


async def show_next_manual_image(context, chat_id):
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    queue = session.get("manual_queue", []) or []

    if not queue:
        session["manual_current"] = None
        session["state"] = "IDLE"
        keyboard = [
            [InlineKeyboardButton(
                "🎬 بدء الرندرة بعد التعيين اليدوي",
                callback_data="btn_start_render"
            )],
        ]
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "✅ <b>اكتمل التعيين اليدوي بنجاح!</b>\n"
                f"عدد الصور المعيَّنة: <code>{len(session.get('manual_map', {}))}</code>\n"
                "اضغط الزر أدناه لبدء الرندرة."
            ),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.HTML,
        )
        return

    current_path = queue.pop(0)
    session["manual_queue"] = queue
    session["manual_current"] = current_path

    assigned_keys = set(session.get("manual_map", {}).keys())
    remaining_slots = sorted([
        s for s in session.get("manual_missing", [])
        if int(s) not in assigned_keys
    ])

    buttons = []
    row = []
    for s in remaining_slots:
        row.append(InlineKeyboardButton(
            f"#{s}", callback_data=f"manual_assign_{s}"
        ))
        if len(row) == 5:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    buttons.append([
        InlineKeyboardButton("❌ صورة غلط / تجاهل", callback_data="manual_skip"),
        InlineKeyboardButton("❌ إلغاء الوضع اليدوي", callback_data="manual_cancel"),
    ])

    try:
        with open(current_path, "rb") as img:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=img,
                caption=(
                    f"🖼️ <b>تعيين يدوي — متبقٍ {len(queue)}</b>\n"
                    f"الملف: <code>{_esc(Path(current_path).name)}</code>\n\n"
                    f"👇 اختر رقم الكادر (من النواقص) الذي تنتمي إليه هذه الصورة:\n"
                    f"أو اضغط <b>❌ صورة غلط</b> لو الصورة مش تبع الشغل أصلاً."
                ),
                reply_markup=InlineKeyboardMarkup(buttons),
                parse_mode=ParseMode.HTML,
            )
    except Exception as e:
        logger.error(f"فشل إرسال الصورة للتعيين اليدوي: {e}")
        await show_next_manual_image(context, chat_id)


# =================================================================
# 7. المرحلتان 4 و 5 — مع دعم Partial/Force Render + Manual Assignment
# =================================================================

async def run_stage4_and_5(msg_obj, context, allow_partial: bool = False):
    chat_id = msg_obj.chat_id
    session = get_session(chat_id)

    if _is_stale(chat_id, session):
        return

    ep_id = session.get("episode_id", "201")
    sentences = session.get("sentences", [])
    total_expected = len(sentences)

    raw_dir = OUTPUTS_DIR / f"episode_{ep_id}_raw_images"
    clean_dir = OUTPUTS_DIR / f"episode_{ep_id}_clean_frames"

    mode_tag = "⚡ رندرة قسرية بالصور المتوفرة" if allow_partial else "🔍 فحص كامل"
    try:
        progress_msg = await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"{mode_tag}\n"
                f"<i>جاري فحص الركن السفلي الأيمن للصور بالـ OCR ومطابقة الترتيب...</i>"
            ),
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        return

    try:
        frames = await asyncio.to_thread(
            process_and_verify_images,
            uploaded_images_dir=raw_dir,
            output_frames_dir=clean_dir,
            expected_total=total_expected,
            allow_partial=allow_partial,
            manual_assignments=session.get("manual_map", {}),
        )
        if _is_stale(chat_id, session):
            logger.info(f"⛔ تم إلغاء المرحلة 4 (فحص الصور) للحلقة {ep_id}")
            return
    except MissingAssetsError as m_err:
        if _is_stale(chat_id, session):
            return

        missing_preview = _format_missing_indices(m_err.missing_indices)
        unindexed_files = list(getattr(m_err, "unindexed_files", []) or [])
        session["manual_unindexed_files"] = unindexed_files
        session["manual_missing_indices"] = list(m_err.missing_indices)

        keyboard = []
        if unindexed_files:
            keyboard.append([
                InlineKeyboardButton(
                    f"🧩 تعيين يدوي لـ {len(unindexed_files)} صورة غير مُفهرسة",
                    callback_data="btn_manual_assign",
                )
            ])
        keyboard.append([
            InlineKeyboardButton(
                f"⏩ متابعة ورندرة بالصور المتوفرة ({m_err.found_count} صورة)",
                callback_data="btn_force_render",
            )
        ])
        keyboard.append([
            InlineKeyboardButton(
                "🔄 إعادة الفحص بعد رفع النواقص",
                callback_data="btn_start_render",
            )
        ])

        extra_note = (
            f"\n🧩 <b>صور غير مُفهرسة (فشل قراءة الـ OCR):</b> <code>{len(unindexed_files)}</code>"
            if unindexed_files else ""
        )

        await progress_msg.edit_text(
            f"🚨 <b>تنبيه: أصول مفقودة (Missing Images)</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"تم التحقق بنجاح من <b>{m_err.found_count}</b> صورة من أصل <b>{m_err.total_expected}</b>.{extra_note}\n\n"
            f"⚠️ <b>الصور المفقودة المطلوب رفعها:</b>\n"
            f"<code>{_esc(missing_preview)}</code>\n\n"
            f"👇 <b>الخيارات المتاحة:</b>\n"
            f"• <b>تعيين يدوي</b>: اربط الصور التي فشل قراءة رقمها بكادراتها يدوياً.\n"
            f"• <b>متابعة ورندرة</b>: فيديو أقصر بالصور المتوفرة فقط.\n"
            f"• <b>إعادة الفحص</b>: بعد رفع النواقص للحصول على الفيديو الكامل.",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.HTML,
        )
        return
    except asyncio.CancelledError:
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        await progress_msg.edit_text(
            f"❌ <b>خطأ أثناء معالجة الصور:</b>\n<code>{_esc(str(e))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    partial_note = (
        "⚡ <b>الوضع القسري:</b> سيتم إنتاج فيديو بالصور المتوفرة فقط وتخطي الجمل المفقودة.\n"
        if allow_partial else ""
    )
    await progress_msg.edit_text(
        f"<b>🎬 بدء المونتاج والرندرة الآلية عبر FFmpeg (1080p 60fps)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{partial_note}"
        f"✅ تم مطابقة {len(frames)} صورة وتطبيق الرقعة الذكية واختفاء الأرقام 100%.\n"
        f"⏳ جاري المزامنة الدقيقة بالمللي ثانية وتوليد الترجمة الحركية الصفراء (#FDE047)...\n"
        f"⏳ جاري تطبيق دورة حركات Ken Burns والانتقالات الهوائية (-18dB)...\n\n"
        f"<i>قد تستغرق الرندرة من دقيقتين إلى 4 دقائق حسب سرعة المعالج...</i>",
        parse_mode=ParseMode.HTML,
    )

    try:
        audio_file = session["audio_path"]
        subtitles_ass = OUTPUTS_DIR / f"episode_{ep_id}_subtitles.ass"
        final_video = OUTPUTS_DIR / f"episode_{ep_id}_final_1080p.mp4"

        timeline = await asyncio.to_thread(
            align_audio_and_generate_ass, audio_file, sentences, subtitles_ass
        )
        if _is_stale(chat_id, session):
            logger.info(f"⛔ تم إلغاء المزامنة للحلقة {ep_id}")
            return

        await asyncio.to_thread(
            render_final_video, frames, timeline, audio_file, subtitles_ass, final_video
        )
        if _is_stale(chat_id, session):
            logger.info(f"⛔ تم إلغاء الرندرة للحلقة {ep_id} قبل النشر")
            return

        file_size_mb = final_video.stat().st_size / (1024 * 1024)
        render_badge = "⚡ (رندرة قسرية)" if allow_partial else "🏆"
        if file_size_mb < 49:
            with open(final_video, "rb") as fv:
                await context.bot.send_video(
                    chat_id=chat_id,
                    video=fv,
                    caption=(
                        f"{render_badge} <b>فيديو الحلقة #{ep_id} جاهز للنشر!</b>\n"
                        f"الدقة: 1080p Full HD @ 60fps | الحجم: {file_size_mb:.1f} MB"
                    ),
                    parse_mode=ParseMode.HTML,
                )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"{render_badge} <b>تم تصدير الفيديو النهائي بنجاح على السيرفر!</b>\n"
                    f"📊 الحجم: <code>{file_size_mb:.1f} MB</code> (أكبر من حد تيليجرام 50MB)\n"
                    f"📁 المسار المباشر على السيرفر:\n<code>{final_video}</code>"
                ),
                parse_mode=ParseMode.HTML,
            )

    except asyncio.CancelledError:
        logger.info(f"🛑 تم إلغاء الرندرة للحلقة {ep_id}")
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        await progress_msg.edit_text(
            f"❌ <b>حدث خطأ أثناء الرندرة:</b>\n<code>{_esc(str(e))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    await context.bot.send_message(
        chat_id=chat_id,
        text="📦 <b>جاري إرسال حزمة النشر الرقمي (العناوين، الغلاف، الوصف، والتاجز)...</b>",
        parse_mode=ParseMode.HTML,
    )

    try:
        meta = await asyncio.to_thread(
            generate_stage5_metadata, session["episode_data"], sentences
        )
        if _is_stale(chat_id, session):
            return

        await context.bot.send_message(chat_id=chat_id, text=meta.get("titles_message"))
        await context.bot.send_message(chat_id=chat_id, text=meta.get("thumbnails_message"))
        await context.bot.send_message(chat_id=chat_id, text=meta.get("description_message"))
        await context.bot.send_message(chat_id=chat_id, text=meta.get("tags_message"))

        await context.bot.send_message(
            chat_id=chat_id,
            text="🎉 <b>ألف مبروك! اكتملت دورة إنتاج الحلقة بنسبة 100% وأصبحت جاهزة لليوتيوب فوراً.</b>",
            parse_mode=ParseMode.HTML,
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        await context.bot.send_message(
            chat_id=chat_id, text=f"⚠️ تعذر استخراج ميتاداتا النشر: {_esc(str(e))}"
        )


# =================================================================
# 8. نقطة التشغيل
# =================================================================

def main():
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN غير موجود في ملف .env!")

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_media_upload))

    print("=" * 60)
    print("🚀 محرك Vot Studio Pro يعمل الآن — Stage1 Approval Flow + Manual Assignment")
    print("=" * 60)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
