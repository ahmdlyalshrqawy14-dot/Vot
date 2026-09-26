import os
import re
import json
import time
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
# ثوابت
# -----------------------------------------------------------------
SENTENCE_MIN = 60
SENTENCE_MAX = 80
DEFAULT_VOICE = "en-US-BrianMultilingualNeural"

# نمط مجلدات المونتاج المؤقتة التي ينشئها stage4_composer.py
_COMPOSER_TEMP_DIR_RE = re.compile(r"^temp_segments_[0-9a-fA-F\-]+$")
_COMPOSER_TEMP_MAX_AGE_SECONDS = 1800  # 30 دقيقة = "قديم بشكل واضح"

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
        "awaiting_script_edit": False,
        "manual_map": {},                 # {int(slot): Path}  نهائي حتى يقوم المستخدم بإعادة التعيين
        "manual_queue": [],               # [Path, ...]
        "manual_missing": [],             # [int, ...]
        "manual_current": None,           # Path الحالية
        "manual_unindexed_files": [],     # [Path, ...]
        "manual_missing_indices": [],     # [int, ...]
        "ocr_indexed_map": {},            # {int(slot): str(path)} من آخر فحص OCR
        # =============================================================
        # نظام مراجعة وترتيب الصور المستقل (قبل المونتاج)
        # =============================================================
        "image_review_mode": None,           # None | "quick" | "full"
        "image_review_items": [],            # [int(slot), ...]
        "image_review_index": 0,             # int
        "image_review_map": {},              # {int(slot): str(path)} خريطة العمل
        "image_review_original": {},         # نسخة من الخريطة الأصلية قبل التعديلات
        "image_review_confirmed": {},        # {int(slot): True}
        "image_review_confirmed_flag": False,
        "image_review_summary": None,        # dict ملخص جاهز للعرض
        "image_review_message_id": None,
        "image_review_missing": [],          # [int(slot), ...]
        "image_review_duplicates": [],       # [int(slot), ...]
        "image_review_excluded": [],         # [int(slot), ...] الصور المستبعدة
        "image_review_manual_edits": [],     # [{"old": int, "new": int}, ...]
        "image_review_pending_slot": None,   # int | None
        "final_image_map": {},               # {int(slot): str(path)} معتمدة نهائياً
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


# =================================================================
# FORMAT-ONLY NORMALIZER (للسكريبت اليدوي)
# =================================================================

def normalize_script_sentences(raw_text: str) -> list[str]:
    """
    تنظيف تنسيقي فقط للسكريبت — لا إعادة صياغة ولا ترجمة ولا تغيير معنى.
    """
    if raw_text is None or not isinstance(raw_text, str):
        raise ValueError("النص المُرسل فارغ أو غير صالح.")

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    parts = re.split(r'(?<=[.!?])(?=\s)|\n+', text)

    sentences: list[str] = []
    for part in parts:
        s = part.strip()
        if not s:
            continue
        s = re.sub(r'\s+', ' ', s)
        if s[-1] not in '.!?':
            s = s + '.'
        sentences.append(s)

    if not sentences:
        raise ValueError("لم يتم العثور على أي جمل صالحة في النص المُرسل.")

    return sentences


def _save_normalized_sentences(ep_id, session, normalized: list[str]) -> None:
    """
    يحفظ الجمل المُنسّقة داخل stage1_result + session + stage1_episode_{id}.json
    """
    stage1_res = session.get("stage1_result")
    if not isinstance(stage1_res, dict):
        stage1_res = {}

    stage1_res["full_script_sentences"] = list(normalized)

    try:
        stage1_res["total_word_count"] = int(
            sum(len(s.split()) for s in normalized)
        )
    except Exception:
        pass

    session["stage1_result"] = stage1_res
    session["sentences"] = list(normalized)

    if not ep_id:
        return
    s1_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"
    try:
        with open(s1_file, "w", encoding="utf-8") as f:
            json.dump(stage1_res, f, ensure_ascii=False, indent=2)
        logger.info(
            f"💾 تم حفظ السكريبت المُنسّق للحلقة {ep_id} ({len(normalized)} جملة)"
        )
    except Exception as e:
        logger.error(f"فشل حفظ stage1_episode_{ep_id}.json بعد التعديل اليدوي: {e}")


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
    """يحذف مجلدات الحلقة المؤقتة القديمة (لا يمس ملفات المصدر أو الفيديو النهائي)."""
    if not ep_id:
        return
    targets = [
        OUTPUTS_DIR / f"episode_{ep_id}_prompts",
        OUTPUTS_DIR / f"episode_{ep_id}_raw_images",
        OUTPUTS_DIR / f"episode_{ep_id}_clean_frames",
        OUTPUTS_DIR / f"episode_{ep_id}_clean_frames_renamed",
        OUTPUTS_DIR / f"episode_{ep_id}_temp_segments",
        OUTPUTS_DIR / f"episode_{ep_id}_temp_segments_audio",
        *OUTPUTS_DIR.glob("episode_*_temp_segments_ep*"),
    ]
    for t in targets:
        try:
            if t.exists() and t.is_dir():
                shutil.rmtree(t, ignore_errors=True)
                logger.info(f"🧹 تم حذف المجلد المؤقت: {t.name}")
        except Exception as e:
            logger.warning(f"تعذّر حذف {t}: {e}")


def _cleanup_stale_composer_temp_dirs(max_age_seconds: int = _COMPOSER_TEMP_MAX_AGE_SECONDS):
    """
    يحذف مجلدات stage4_composer المؤقتة (temp_segments_<uuid>) القديمة فقط.

    - يطابق النمط temp_segments_<uuid> فقط.
    - لا يحذف أي شيء أحدث من max_age_seconds (حماية الجلسات النشطة الأخرى).
    - لا يحذف صوراً أو صوتاً أو ترجمات أو فيديو نهائياً.
    """
    try:
        entries = list(OUTPUTS_DIR.iterdir())
    except Exception as e:
        logger.warning(f"failed to iterate OUTPUTS_DIR for cleanup: {e}")
        return
    now = time.time()
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
            if not _COMPOSER_TEMP_DIR_RE.match(entry.name):
                continue
            try:
                mtime = entry.stat().st_mtime
            except Exception:
                continue
            if now - mtime < max_age_seconds:
                continue
            shutil.rmtree(entry, ignore_errors=True)
            logger.info(f"🧹 removed stale composer temp dir: {entry.name}")
        except Exception as e:
            logger.warning(f"failed to remove stale temp dir {entry}: {e}")


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
    

def mark_episode_completed(episode_id):
    """
    يحدّث حالة الحلقة في episodes.json من pending إلى completed.
    يرجع True لو تم التحديث بنجاح، False لو فشل.
    """
    if not EPISODES_FILE.exists():
        logger.warning("episodes.json غير موجود، تعذّر تحديث الحالة")
        return False

    try:
        with open(EPISODES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        episodes = data if isinstance(data, list) else data.get("episodes", [])
        updated = False

        for ep in episodes:
            if str(ep.get("id")) == str(episode_id):
                if ep.get("status") != "completed":
                    ep["status"] = "completed"
                    updated = True
                break

        if not updated:
            logger.warning(f"لم يتم العثور على الحلقة {episode_id} لتحديث حالتها")
            return False

        with open(EPISODES_FILE, "w", encoding="utf-8") as f:
            json.dump(episodes, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ تم تحديث حالة الحلقة {episode_id} إلى completed")
        return True

    except Exception as e:
        logger.error(f"فشل تحديث حالة الحلقة {episode_id}: {e}")
        return False


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

async def start_command(update_or_query, context: ContextTypes.DEFAULT_TYPE):
    if isinstance(update_or_query, Update):
        chat_id = update_or_query.effective_chat.id
        reply_message = update_or_query.message
    else:
        chat_id = update_or_query.message.chat_id
        reply_message = update_or_query.message

    cancelled = _cancel_user_task(chat_id)

    old_session = user_sessions.get(chat_id)
    old_ep_id = old_session.get("episode_id") if old_session else None

    if old_session:
        old_session["cancelled"] = True

    user_sessions[chat_id] = _fresh_session()
    _cleanup_episode_temp_files(old_ep_id)
    _cleanup_stale_composer_temp_dirs()

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

    await reply_message.reply_text(
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

    # -------------------------------------------------------------
    # القوائم الرئيسية
    # -------------------------------------------------------------
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

    elif data.startswith("edit_stage1_"):
        target_id = data.replace("edit_stage1_", "")
        active_id = session.get("episode_id")

        if not active_id or str(active_id) != str(target_id):
            logger.warning(
                f"⚠️ edit_stage1 قديم/غير مطابق | target={target_id} | active={active_id}"
            )
            try:
                await query.answer("⚠️ هذا الزر لا يخص الحلقة النشطة الحالية.", show_alert=True)
            except Exception:
                pass
            return

        if session.get("stage1_approved") is True:
            try:
                await query.answer("ℹ️ تم اعتماد هذه المرحلة بالفعل.", show_alert=True)
            except Exception:
                pass
            return

        session["awaiting_script_edit"] = True
        session["state"] = "WAITING_SCRIPT_EDIT"

        await query.edit_message_text(
            "✏️ <b>وضع تعديل السكريبت اليدوي</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "الصق الآن <b>السكريبت الكامل</b> كنص عادي:\n"
            "• <b>جملة واحدة لكل سطر</b> (مُفضّل) — أو نص متصل.\n"
            "• سيقوم البوت <b>فقط</b> بتصحيح علامات الترقيم والتنسيق — "
            "<b>بدون</b> إعادة صياغة أو ترجمة أو تغيير للمعنى.\n"
            f"• <b>عدد الجمل يجب أن يكون بين {SENTENCE_MIN} و {SENTENCE_MAX} جملة.</b>\n"
            "• بعد التأكيد ستظهر لك معاينة، وسيتبقى عليك الضغط على "
            "<b>✅ اعتماد</b> للانتقال إلى المرحلة الثانية.\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            parse_mode=ParseMode.HTML,
        )

    elif data.startswith("regenerate_stage1_"):
        target_id = data.replace("regenerate_stage1_", "")
        ep = session.get("episode_data") or get_episode(target_id)
        if ep:
            _cancel_user_task(chat_id)
            user_tasks[chat_id] = asyncio.create_task(
                run_stage1_only(query, context, ep)
            )

    # -------------------------------------------------------------
    # معاينة الصوت — التحقق من قائمة AZURE_MALE_VOICES
    # -------------------------------------------------------------
    elif data.startswith("preview_voice_"):
        preview_voice = data[len("preview_voice_"):]
        if preview_voice not in AZURE_MALE_VOICES:
            logger.warning(f"⚠️ preview_voice غير مصرح به: {preview_voice}")
            try:
                await query.answer("⚠️ صوت غير مُعرَّف في الإعدادات.", show_alert=True)
            except Exception:
                pass
            return
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

    # -------------------------------------------------------------
    # اختيار الصوت — التحقق من قائمة AZURE_MALE_VOICES
    # -------------------------------------------------------------
    elif data.startswith("voice_"):
        selected_voice = data[len("voice_"):]
        if selected_voice not in AZURE_MALE_VOICES:
            logger.warning(f"⚠️ voice غير مصرح به: {selected_voice}")
            try:
                await query.answer("⚠️ صوت غير مُعرَّف في الإعدادات.", show_alert=True)
            except Exception:
                pass
            return
        session["voice"] = selected_voice
        session["engine"] = "azure"
        user_tasks[chat_id] = asyncio.create_task(
            run_stage3(query, context)
        )

    # -------------------------------------------------------------
    # بدء فحص الصور
    # -------------------------------------------------------------
    elif data == "btn_start_render":
        user_tasks[chat_id] = asyncio.create_task(
            run_image_verification(query.message, context)
        )

    # -------------------------------------------------------------
    # الوضع اليدوي
    # -------------------------------------------------------------
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
                    "يمكنك رفع الصور الناقصة ثم إعادة الفحص."
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

        if not isinstance(session.get("manual_map"), dict):
            session["manual_map"] = {}

        already = set()
        for p in session["manual_map"].values():
            try:
                already.add(Path(p).resolve())
            except Exception:
                continue

        session["manual_queue"] = [
            p for p in unindexed
            if Path(p).resolve() not in already
        ]
        session["manual_missing"] = list(missing_idx)
        session["manual_current"] = None
        session["state"] = "MANUAL_ASSIGN"

        if not session["manual_queue"]:
            await query.answer("كل الصور تم تعيينها يدويًا.", show_alert=True)
            await _finalize_after_manual(context, chat_id)
            return

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

    # =============================================================
    # نظام مراجعة الصور — اختيار الوضع
    # =============================================================
    elif data == "review_mode_quick":
        await start_image_review(chat_id, context, mode="quick")

    elif data == "review_mode_full":
        await start_image_review(chat_id, context, mode="full")

    # =============================================================
    # نظام مراجعة الصور — أزرار البطاقة
    # =============================================================
    elif data == "review_confirm":
        await handle_review_confirm(chat_id, context)

    elif data == "review_change_num":
        await handle_review_change_num(chat_id, context)

    elif data == "review_prev":
        await handle_review_navigate(chat_id, context, delta=-1)

    elif data == "review_next":
        await handle_review_navigate(chat_id, context, delta=+1)

    elif data == "review_skip":
        await handle_review_skip(chat_id, context)

    elif data == "review_finish":
        await show_review_summary(chat_id, context)

    elif data == "review_noop":
        return

    # =============================================================
    # نظام مراجعة الصور — شاشة الملخص
    # =============================================================
    elif data == "review_summary_approve":
        user_tasks[chat_id] = asyncio.create_task(
            approve_and_render(query.message, context)
        )

    elif data == "review_summary_edit":
        await start_image_review(chat_id, context, mode="full")

    elif data == "review_summary_cancel":
        session["state"] = "IDLE"
        session["image_review_mode"] = None
        session["image_review_items"] = []
        session["image_review_index"] = 0
        await query.edit_message_text(
            "❌ <b>تم إلغاء مراجعة الصور.</b>\nيمكنك العودة للقائمة الرئيسية عبر /start.",
            parse_mode=ParseMode.HTML,
        )


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

    state = session.get("state")

    if state == "WAITING_EPISODE_ID":
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
        return

    # =============================================================
    # استقبال السكريبت المُعدَّل يدوياً (WAITING_SCRIPT_EDIT)
    # =============================================================
    if state == "WAITING_SCRIPT_EDIT":
        if _is_stale(chat_id, session):
            return

        raw = update.message.text or ""
        try:
            normalized = normalize_script_sentences(raw)
        except ValueError as ve:
            await update.message.reply_text(
                f"⚠️ <b>لم أتمكن من قراءة أي جمل صالحة من النص.</b>\n"
                f"<i>{_esc(str(ve))}</i>\n\n"
                f"أرسل النص مرة أخرى (يُفضّل جملة واحدة لكل سطر).",
                parse_mode=ParseMode.HTML,
            )
            return
        except Exception as e:
            logger.exception("خطأ غير متوقع أثناء تنسيق السكريبت اليدوي")
            await update.message.reply_text(
                f"❌ <b>خطأ أثناء التنسيق:</b>\n<code>{_esc(str(e))}</code>",
                parse_mode=ParseMode.HTML,
            )
            return

        # ✅ فرض قاعدة عدد الجمل 60..80 قبل الحفظ
        sentence_count = len(normalized)
        if sentence_count < SENTENCE_MIN or sentence_count > SENTENCE_MAX:
            logger.info(
                f"⛔ رفض سكريبت يدوي للحلقة {session.get('episode_id')} "
                f"بعدد جمل {sentence_count} (المسموح {SENTENCE_MIN}-{SENTENCE_MAX})"
            )
            await update.message.reply_text(
                f"❌ <b>عدد الجمل غير مقبول.</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>العدد المُرسل:</b> <code>{sentence_count}</code>\n"
                f"✅ <b>النطاق المطلوب:</b> <code>{SENTENCE_MIN}</code> – <code>{SENTENCE_MAX}</code> جملة.\n\n"
                f"لم يتم حفظ السكريبت، ولم يتم الاعتماد.\n"
                f"عدّل النص وأرسله مرة أخرى (أنت لا تزال في وضع تعديل السكريبت).",
                parse_mode=ParseMode.HTML,
            )
            return

        ep_id = session.get("episode_id")
        _save_normalized_sentences(ep_id, session, normalized)

        # لا يُعتبر معتمداً حتى يضغط زر الاعتماد
        session["awaiting_script_edit"] = False
        session["state"] = "IDLE"
        session["stage1_approved"] = False

        n = sentence_count
        first_three = normalized[:3]
        last_two = normalized[-2:] if n > 3 else []

        def _fmt(lst):
            if not lst:
                return "  —"
            return "\n".join(f"  • {_esc(s)}" for s in lst)

        preview_lines = [
            "<b>✅ تم تنسيق السكريبت بنجاح</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"🧩 <b>عدد الجمل:</b> <code>{n}</code>",
            "",
            "<b>أول 3 جمل:</b>",
            _fmt(first_three),
        ]
        if last_two:
            preview_lines.append("")
            preview_lines.append("<b>آخر جملتين:</b>")
            preview_lines.append(_fmt(last_two))
        preview_lines.extend([
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "<i>⚠️ لم يتم الاعتماد بعد — اضغط الزر أدناه للانتقال للمرحلة الثانية.</i>",
        ])

        keyboard = [
            [InlineKeyboardButton(
                "✅ اعتماد السكريبت المعدّل والانتقال للمرحلة الثانية",
                callback_data=f"approve_stage1_{ep_id}",
            )],
            [InlineKeyboardButton(
                "✏️ تعديل مرة أخرى",
                callback_data=f"edit_stage1_{ep_id}",
            )],
            [InlineKeyboardButton(
                "🔄 إعادة توليد من الصفر",
                callback_data=f"regenerate_stage1_{ep_id}",
            )],
            [InlineKeyboardButton("❌ إلغاء", callback_data="btn_back_main")],
        ]

        await update.message.reply_text(
            "\n".join(preview_lines),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.HTML,
        )
        return

    if state == "WAITING_IMAGE_CORRECTION":
        await handle_image_correction_text(update, context)
        return


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
    session["awaiting_script_edit"] = False

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
                "✏️ تعديل السكريبت يدويًا",
                callback_data=f"edit_stage1_{ep_id}",
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
    """
    يعرض كل الأصوات المُعرَّفة في AZURE_MALE_VOICES ديناميكياً،
    مع زر اختيار + زر معاينة لكل صوت.
    """
    text = (
        "🎙️ <b>المرحلة 3: اختيار الصوت التعبيري (Microsoft Azure)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "يمكنك الاستماع لعينة فورية لأي صوت قبل الاختيار عبر زر <b>🎧 استمع للعينة</b>،\n"
        "ثم اضغط على اسم الصوت لاعتماده وبدء توليد التعليق الصوتي.\n"
        f"<i>الصوت الافتراضي عند عدم الاختيار: <code>{DEFAULT_VOICE}</code></i>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    descriptions = {
        "en-US-BrianMultilingualNeural": "الصوت المعتمد لقناة VOT — سرد هادئ وإنساني ⭐",
        "en-US-GuyNeural": "تلوين انفعالي كامل وشامل 🌟",
        "en-US-DavisNeural": "سرد ناضج وهادئ وقوي 📖",
        "en-US-TonyNeural": "صوت حماسي وواثق وعالي الطاقة ⚡",
        "en-US-JasonNeural": "نبرة شبابية وسريعة وخفيفة 🚀",
    }

    buttons = []
    # Iterate ديناميكياً على القائمة المُعدَّة في config
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
# 5. المرحلة 3 — تمرير stage1_result لـ Voice Director
# =================================================================

async def run_stage3(query, context):
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    ep_id = session.get("episode_id")
    sentences = session.get("sentences", [])
    engine = "azure"
    voice = session.get("voice") or DEFAULT_VOICE

    if _is_stale(chat_id, session):
        return

    # ✅ تحقق من أن الصوت من القائمة المُعدَّة
    if voice not in AZURE_MALE_VOICES:
        try:
            await query.edit_message_text(
                f"❌ <b>الصوت المختار غير مُعرَّف في الإعدادات:</b>\n"
                f"<code>{_esc(voice)}</code>\n"
                f"الرجاء اختيار صوت من القائمة المُعتمدة.",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass
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
        episode_context = session.get("stage1_result")

        if not isinstance(episode_context, dict):
            s1_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"

            if not s1_file.exists():
                raise RuntimeError(
                    f"لم يتم العثور على نتيجة المرحلة الأولى للحلقة {ep_id}. "
                    "لا يمكن توليد الصوت بدون الخطة الإبداعية المعتمدة."
                )

            try:
                with open(s1_file, "r", encoding="utf-8") as f:
                    episode_context = json.load(f)
            except Exception as e:
                raise RuntimeError(
                    f"تعذر قراءة نتيجة المرحلة الأولى للحلقة {ep_id}: {e}"
                )

            if not isinstance(episode_context, dict):
                raise RuntimeError(
                    f"نتيجة المرحلة الأولى المخزّنة للحلقة {ep_id} تالفة أو ليست كائن JSON صالحًا."
                )

            session["stage1_result"] = episode_context

        if not isinstance(episode_context, dict):
            raise RuntimeError(
                f"نتيجة المرحلة الأولى للحلقة {ep_id} ليست كائنًا صالحًا."
            )

        context_sentences = episode_context.get("full_script_sentences")

        if not isinstance(context_sentences, list) or not context_sentences:
            raise RuntimeError(
                f"نتيجة المرحلة الأولى للحلقة {ep_id} "
                "لا تحتوي على full_script_sentences صالحة."
            )

        if not isinstance(sentences, list) or not sentences:
            raise RuntimeError(
                f"لا توجد جمل صالحة في جلسة الحلقة {ep_id}."
            )

        if sentences != context_sentences:
            raise RuntimeError(
                f"جمل الجلسة لا تطابق full_script_sentences "
                f"في نتيجة المرحلة الأولى للحلقة {ep_id}. "
                "تم إيقاف المرحلة الثالثة لمنع إنتاج صوت بسياق غير متطابق."
            )

        if _is_stale(chat_id, session):
            return

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
            [InlineKeyboardButton("🎬 فحص الصور وبدء المراجعة", callback_data="btn_start_render")]
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
        logger.exception(f"خطأ في المرحلة الثالثة للحلقة {ep_id}")
        try:
            await wait_msg.edit_text(
                f"❌ <b>خطأ أثناء توليد الصوت:</b>\n<code>{_esc(str(e))}</code>",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass


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

        # أي رفع جديد يُبطل أي اعتماد سابق (سيُعاد الفحص من الصفر)
        session["image_review_confirmed_flag"] = False
        session["final_image_map"] = {}

        current = session["uploaded_count"]
        if current % 5 == 0 or (total_expected > 0 and current == total_expected):
            pct = int((current / total_expected * 100)) if total_expected > 0 else 0
            keyboard = [
                [InlineKeyboardButton("🎬 فحص الصور وبدء المراجعة", callback_data="btn_start_render")]
            ]
            await update.message.reply_text(
                f"📥 <b>تم استلام وحفظ:</b> <code>{current} / {total_expected}</code> صورة ({pct}%)\n"
                f"إذا انتهيت من رفع الحزمة كاملة، اضغط على الزر أدناه لبدء الفحص والمراجعة.",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.HTML,
            )


# =================================================================
# المعالجة اليدوية
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


async def _build_map_from_ocr_and_manual(session) -> dict:
    """يدمج نتيجة OCR + التعيين اليدوي. اختيار المستخدم يطغى."""
    merged = {}
    for k, v in (session.get("ocr_indexed_map") or {}).items():
        try:
            merged[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    for k, v in (session.get("manual_map") or {}).items():
        try:
            merged[int(k)] = str(v)
        except (TypeError, ValueError):
            continue
    return merged


def _compute_duplicates_from_map(review_map):
    """يُرجع قائمة أرقام مكررة (نفس المسار لأكثر من slot)."""
    seen = {}
    duplicates = set()
    for slot, p in review_map.items():
        try:
            key = str(Path(p).resolve())
        except Exception:
            key = str(p)
        if key in seen:
            duplicates.add(int(seen[key]))
            duplicates.add(int(slot))
        else:
            seen[key] = int(slot)
    return sorted(duplicates)


def _reset_image_review_state(session, review_map, total_expected):
    """
    يعيد بناء كل حقول مراجعة الصور من الخريطة الحالية.
    لا يحتفظ بحالة قديمة.
    """
    review_map = {int(k): str(v) for k, v in (review_map or {}).items() if v}
    session["image_review_map"] = dict(review_map)
    session["image_review_original"] = dict(review_map)
    session["image_review_missing"] = [
        s for s in range(1, total_expected + 1) if s not in review_map
    ]
    session["image_review_duplicates"] = _compute_duplicates_from_map(review_map)
    session["image_review_excluded"] = []
    session["image_review_manual_edits"] = []
    session["image_review_confirmed"] = {}
    session["image_review_confirmed_flag"] = False
    session["image_review_summary"] = None
    session["image_review_mode"] = None
    session["image_review_index"] = 0
    session["image_review_items"] = []
    session["final_image_map"] = {}


async def _finalize_after_manual(context, chat_id):
    """
    بعد انتهاء المستخدم من التعيين اليدوي:
    - نبني الخريطة من OCR + Manual ونُعيد ضبط كل حالة المراجعة.
    - لا نعرض أبداً خيار رندرة جزئية.
    """
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    total_expected = len(session.get("sentences", []) or [])
    review_map = await _build_map_from_ocr_and_manual(session)

    _reset_image_review_state(session, review_map, total_expected)
    session["state"] = "IDLE"

    found = len(session["image_review_map"])
    missing = session["image_review_missing"]
    duplicates = session["image_review_duplicates"]
    manual_count = len(session.get("manual_map") or {})

    fully_complete = (found == total_expected) and (not missing) and (not duplicates)

    if fully_complete:
        keyboard = [
            [InlineKeyboardButton("📋 مراجعة الصور واعتماد الترتيب", callback_data="review_mode_full")],
            [InlineKeyboardButton("📊 عرض الملخص النهائي مباشرة", callback_data="review_finish")],
            [InlineKeyboardButton("❌ إلغاء", callback_data="btn_back_main")],
        ]
        text = (
            f"✅ <b>تم تجميع كل الصور ({found}/{total_expected})</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🧩 <b>تعيين يدوي:</b> <code>{manual_count}</code>\n"
            f"👇 يمكنك مراجعة الترتيب ثم اعتماده لبدء المونتاج."
        )
    else:
        missing_txt = _format_missing_indices(missing) if missing else "—"
        keyboard = [
            [InlineKeyboardButton("🔄 إعادة الفحص بعد رفع النواقص", callback_data="btn_start_render")],
            [InlineKeyboardButton("❌ إلغاء والعودة", callback_data="btn_back_main")],
        ]
        text = (
            f"⚠️ <b>لسه فيه نواقص — لا يمكن بدء المونتاج</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ <b>المعتمد:</b> <code>{found}</code> / <code>{total_expected}</code>\n"
            f"⚠️ <b>الناقص:</b> <code>{len(missing)}</code>\n"
            f"    └ {_esc(missing_txt)}\n"
            f"🔁 <b>التكرارات:</b> <code>{len(duplicates)}</code>\n\n"
            f"<i>ارفع الصور الناقصة، ثم اضغط «إعادة الفحص».</i>\n"
            f"<i>لا يوجد خيار للرندرة الجزئية.</i>"
        )

    await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.HTML,
    )


async def show_next_manual_image(context, chat_id):
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    queue = session.get("manual_queue", []) or []

    if not queue:
        session["manual_current"] = None
        await _finalize_after_manual(context, chat_id)
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
# 7. فحص الصور — لا يوجد allow_partial من الـ UI
# =================================================================

async def run_image_verification(msg_obj, context):
    """
    يشغّل process_and_verify_images مع allow_partial=False دائماً.
    بعد الفحص الناجح: يُعيد بناء كل حالة المراجعة من الخريطة الجديدة
    (لا توجد بيانات ناقصة قديمة).
    عند MissingAssetsError: يعرض فقط: تعيين يدوي / إعادة فحص / إلغاء.
    """
    chat_id = msg_obj.chat_id
    session = get_session(chat_id)

    if _is_stale(chat_id, session):
        return

    ep_id = session.get("episode_id", "201")
    sentences = session.get("sentences", [])
    total_expected = len(sentences)

    if total_expected == 0:
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ لا توجد جمل في الجلسة. اضغط /start من جديد.",
            parse_mode=ParseMode.HTML,
        )
        return

    raw_dir = OUTPUTS_DIR / f"episode_{ep_id}_raw_images"
    clean_dir = OUTPUTS_DIR / f"episode_{ep_id}_clean_frames"

    try:
        progress_msg = await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "🔍 <b>فحص كامل</b>\n"
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
            allow_partial=False,
            manual_assignments=session.get("manual_map", {}),
        )
        if _is_stale(chat_id, session):
            logger.info(f"⛔ تم إلغاء فحص الصور للحلقة {ep_id}")
            return
    except MissingAssetsError as m_err:
        if _is_stale(chat_id, session):
            return

        missing_preview = _format_missing_indices(m_err.missing_indices)
        unindexed_files = list(getattr(m_err, "unindexed_files", []) or [])
        missing_indices = [int(x) for x in (m_err.missing_indices or [])]

        # ✅ إعادة بناء الحالة من نتيجة الفحص الجديدة فقط
        ocr_map = getattr(m_err, "indexed_images", None) or {}
        ocr_map_int = {}
        for k, v in ocr_map.items():
            try:
                ocr_map_int[int(k)] = str(v)
            except (TypeError, ValueError):
                continue

        session["ocr_indexed_map"] = ocr_map_int
        session["manual_unindexed_files"] = unindexed_files
        session["manual_missing_indices"] = missing_indices

        # لا تمسح manual_map (اختيارات المستخدم النهائية)
        # لكن لا تسمح ببقاء نتائج OCR قديمة تتعارض مع الفحص الجديد
        _reset_image_review_state(session, ocr_map_int, total_expected)
        # بعد reset، خريطة المراجعة هي OCR فقط + أي manual_valid مدمج لاحقاً.
        # لكن دعنا نطبّق manual_map فوقها إذا كان الملف موجوداً
        merged = dict(session["image_review_map"])
        for slot, p in (session.get("manual_map") or {}).items():
            try:
                sp = int(slot)
            except (TypeError, ValueError):
                continue
            if p and Path(p).exists():
                merged[sp] = str(p)
        session["image_review_map"] = merged
        session["image_review_original"] = dict(merged)
        session["image_review_missing"] = [
            s for s in range(1, total_expected + 1) if s not in merged
        ]
        session["image_review_duplicates"] = _compute_duplicates_from_map(merged)

        keyboard = []
        if unindexed_files:
            keyboard.append([
                InlineKeyboardButton(
                    f"🧩 تعيين يدوي لـ {len(unindexed_files)} صورة",
                    callback_data="btn_manual_assign",
                )
            ])
        keyboard.append([
            InlineKeyboardButton(
                "🔄 إعادة الفحص بعد رفع النواقص",
                callback_data="btn_start_render",
            )
        ])
        keyboard.append([
            InlineKeyboardButton("❌ إلغاء والعودة للقائمة", callback_data="btn_back_main")
        ])

        extra_note = (
            f"\n🧩 <b>صور فشل قراءة رقمها (تحتاج تعيين يدوي):</b> "
            f"<code>{len(unindexed_files)}</code>"
            if unindexed_files else ""
        )

        await progress_msg.edit_text(
            f"📊 <b>جرد الصور</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ تم التحقق من <b>{m_err.found_count}</b> صورة من أصل "
            f"<b>{m_err.total_expected}</b>.{extra_note}\n\n"
            f"⚠️ <b>الأرقام الناقصة ({len(missing_indices)}):</b>\n"
            f"<code>{_esc(missing_preview)}</code>\n\n"
            f"<i>⚠️ لا يمكن بدء المونتاج قبل اكتمال كل الأرقام.</i>\n\n"
            f"👇 <b>اختار:</b>\n"
            f"• <b>تعيين يدوي</b>: لو فيه صور زيادة أو فشل الـ OCR — هتشوف الصورة وتختار رقمها.\n"
            f"• <b>إعادة الفحص</b>: بعد ما ترفع الصور الناقصة.",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.HTML,
        )
        return
    except asyncio.CancelledError:
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        logger.exception(f"خطأ أثناء فحص الصور للحلقة {ep_id}")
        await progress_msg.edit_text(
            f"❌ <b>خطأ أثناء معالجة الصور:</b>\n<code>{_esc(str(e))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # -------------------------------------------------------------
    # فحص ناجح كامل → إعادة بناء كل الحالة من الصفر
    # -------------------------------------------------------------
    review_map = {}
    for idx, path in enumerate(frames):
        review_map[idx + 1] = str(path)

    # manual_map له الأولوية على نتيجة OCR في نفس الخانة
    for slot, path in (session.get("manual_map") or {}).items():
        try:
            sp = int(slot)
        except (ValueError, TypeError):
            continue
        if path and Path(path).exists():
            review_map[sp] = str(path)

    # ✅ إعادة بناء كاملة لكل حقول المراجعة
    _reset_image_review_state(session, review_map, total_expected)
    session["ocr_indexed_map"] = {int(k): str(v) for k, v in review_map.items()}

    found = len(session["image_review_map"])
    missing_count = len(session["image_review_missing"])
    duplicates_count = len(session["image_review_duplicates"])

    keyboard = [
        [InlineKeyboardButton("⚡ مراجعة الصور التي تحتاج تدقيقًا فقط", callback_data="review_mode_quick")],
        [InlineKeyboardButton("📋 مراجعة جميع الصور بالترتيب", callback_data="review_mode_full")],
        [InlineKeyboardButton("📊 عرض الملخص النهائي مباشرة", callback_data="review_finish")],
    ]

    await progress_msg.edit_text(
        f"<b>✅ تم فحص الصور بنجاح — جاهز للمراجعة</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📸 <b>الصور المكتشفة:</b> <code>{found}</code> من أصل <code>{total_expected}</code>\n"
        f"⚠️ <b>الأرقام المفقودة:</b> <code>{missing_count}</code>\n"
        f"🔁 <b>التكرارات:</b> <code>{duplicates_count}</code>\n\n"
        f"<i>⚠️ لن تبدأ الرندرة إلا بعد اعتمادك الصريح من شاشة الملخص، وبشرط اكتمال كل الأرقام.</i>\n\n"
        f"👇 <b>اختر وضع المراجعة:</b>",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.HTML,
    )


# =================================================================
# 8. نظام مراجعة الصور — دوال التشغيل
# =================================================================

def _get_quick_review_items(session):
    items = set()
    for s in session.get("image_review_missing", []) or []:
        try:
            items.add(int(s))
        except (ValueError, TypeError):
            continue
    for s in session.get("image_review_duplicates", []) or []:
        try:
            items.add(int(s))
        except (ValueError, TypeError):
            continue
    for e in session.get("image_review_manual_edits", []) or []:
        try:
            items.add(int(e.get("new")))
        except (ValueError, TypeError):
            continue
    return sorted(items)


async def start_image_review(chat_id, context, mode: str):
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    review_map = session.get("image_review_map", {}) or {}
    if not review_map:
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ لا توجد صور لمراجعتها. ارفع الصور ثم اضغط زر الفحص.",
            parse_mode=ParseMode.HTML,
        )
        return

    if mode == "quick":
        items = _get_quick_review_items(session)
        if not items:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "✅ <b>لا توجد صور تحتاج تدقيقاً.</b>\n"
                    "يمكنك عرض الملخص النهائي مباشرة واعتماد الترتيب."
                ),
                parse_mode=ParseMode.HTML,
            )
            await show_review_summary(chat_id, context)
            return
    else:
        total_expected = len(session.get("sentences", []))
        items = list(range(1, total_expected + 1))

    session["image_review_mode"] = mode
    session["image_review_items"] = items
    session["image_review_index"] = 0
    session["state"] = "IMAGE_REVIEW"

    await show_review_image(chat_id, context)


def _build_review_caption(session, slot):
    items = session.get("image_review_items", [])
    idx = session.get("image_review_index", 0)
    review_map = session.get("image_review_map", {}) or {}
    path = review_map.get(slot)
    is_confirmed = slot in (session.get("image_review_confirmed") or {})
    was_edited = any(int(e.get("new", 0)) == int(slot) for e in session.get("image_review_manual_edits", []) or [])
    is_missing = slot in (session.get("image_review_missing") or [])
    is_duplicate = slot in (session.get("image_review_duplicates") or [])
    is_excluded = slot in (session.get("image_review_excluded") or [])

    if is_missing:
        status = "❌ مفقودة — لم يتم رفعها"
    elif is_excluded:
        status = "🚫 مستبعدة"
    elif is_duplicate:
        status = "⚠️ رقم مكرر"
    elif is_confirmed:
        status = "✅ مُعتمد"
    else:
        status = "⏳ بانتظار المراجعة"

    if was_edited:
        status += " ✏️ (مُعدَّل يدوياً)"

    file_display = Path(path).name if path else "—"

    lines = [
        f"<b>🖼️ مراجعة الصور — {idx + 1} / {len(items)}</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🔢 <b>الرقم الحالي (Slot):</b> <code>#{slot}</code>",
        f"📊 <b>الترتيب في المراجعة:</b> <code>{idx + 1}</code> من <code>{len(items)}</code>",
        f"📁 <b>الملف:</b> <code>{_esc(file_display)}</code>",
        f"📌 <b>الحالة:</b> {status}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "<i>💡 يمكنك تكبير الصورة من تيليجرام بالضغط عليها.</i>",
    ]
    return "\n".join(lines)


def _build_review_keyboard(session, slot):
    items = session.get("image_review_items", [])
    idx = session.get("image_review_index", 0)
    total = len(items)

    review_map = session.get("image_review_map", {}) or {}
    path_exists = slot in review_map
    is_excluded = slot in (session.get("image_review_excluded") or [])

    row1 = []
    if path_exists and not is_excluded:
        row1.append(InlineKeyboardButton("✅ تأكيد الرقم الحالي", callback_data="review_confirm"))
    row1.append(InlineKeyboardButton("✏️ تغيير الرقم", callback_data="review_change_num"))

    row2 = [
        InlineKeyboardButton("❌ استبعاد الصورة" if not is_excluded else "♻️ استرجاع الصورة", callback_data="review_skip"),
    ]

    row3 = []
    if idx > 0:
        row3.append(InlineKeyboardButton("⬅️ السابقة", callback_data="review_prev"))
    row3.append(InlineKeyboardButton(f"({idx + 1}/{total})", callback_data="review_noop"))
    if idx < total - 1:
        row3.append(InlineKeyboardButton("➡️ التالية", callback_data="review_next"))

    row4 = [InlineKeyboardButton("🏁 إنهاء المراجعة وعرض الملخص", callback_data="review_finish")]

    return InlineKeyboardMarkup([row1, row2, row3, row4])


async def show_review_image(chat_id, context):
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    items = session.get("image_review_items", []) or []
    idx = session.get("image_review_index", 0)

    if not items:
        await context.bot.send_message(
            chat_id=chat_id,
            text="ℹ️ لا توجد صور للمراجعة.",
            parse_mode=ParseMode.HTML,
        )
        return

    if idx < 0:
        idx = 0
    if idx >= len(items):
        idx = len(items) - 1

    session["image_review_index"] = idx
    slot = items[idx]

    review_map = session.get("image_review_map", {}) or {}
    path = review_map.get(slot)

    caption = _build_review_caption(session, slot)
    keyboard = _build_review_keyboard(session, slot)

    prev_msg_id = session.get("image_review_message_id")
    if prev_msg_id:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=prev_msg_id)
        except Exception:
            pass
        session["image_review_message_id"] = None

    if path and Path(path).exists():
        try:
            with open(path, "rb") as img:
                sent = await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=img,
                    caption=caption,
                    reply_markup=keyboard,
                    parse_mode=ParseMode.HTML,
                )
            session["image_review_message_id"] = sent.message_id
            return
        except Exception as e:
            logger.error(f"فشل إرسال صورة المراجعة: {e}")

    sent = await context.bot.send_message(
        chat_id=chat_id,
        text=caption,
        reply_markup=keyboard,
        parse_mode=ParseMode.HTML,
    )
    session["image_review_message_id"] = sent.message_id


async def handle_review_confirm(chat_id, context):
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    items = session.get("image_review_items", []) or []
    idx = session.get("image_review_index", 0)
    if not items:
        return

    slot = items[idx]
    review_map = session.get("image_review_map", {}) or {}
    if slot not in review_map:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ لا يمكن تأكيد الرقم <code>#{slot}</code> — لا توجد صورة مرتبطة به.",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass
        return

    confirmed = session.get("image_review_confirmed") or {}
    confirmed[int(slot)] = True
    session["image_review_confirmed"] = confirmed

    if idx < len(items) - 1:
        session["image_review_index"] = idx + 1
        await show_review_image(chat_id, context)
    else:
        await show_review_summary(chat_id, context)


async def handle_review_navigate(chat_id, context, delta: int):
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    items = session.get("image_review_items", []) or []
    if not items:
        return

    idx = session.get("image_review_index", 0)
    new_idx = max(0, min(len(items) - 1, idx + delta))
    session["image_review_index"] = new_idx
    await show_review_image(chat_id, context)


async def handle_review_skip(chat_id, context):
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    items = session.get("image_review_items", []) or []
    idx = session.get("image_review_index", 0)
    if not items:
        return

    slot = items[idx]
    excluded = set(session.get("image_review_excluded") or [])
    if slot in excluded:
        excluded.discard(slot)
    else:
        excluded.add(slot)

    session["image_review_excluded"] = sorted(excluded)
    await show_review_image(chat_id, context)


async def handle_review_change_num(chat_id, context):
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    items = session.get("image_review_items", []) or []
    idx = session.get("image_review_index", 0)
    if not items:
        return

    slot = items[idx]
    session["state"] = "WAITING_IMAGE_CORRECTION"
    session["image_review_pending_slot"] = int(slot)

    total_expected = len(session.get("sentences", []))
    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"✏️ <b>تعديل رقم الصورة</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🔢 الرقم الحالي: <code>#{slot}</code>\n"
                f"📥 أرسل الآن <b>الرقم الجديد</b> كرسالة نصية (من <code>1</code> إلى <code>{total_expected}</code>).\n"
                f"<i>لإلغاء التعديل اضغط /start أو انتظر ثم اكتب الرقم نفسه.</i>"
            ),
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        pass


async def handle_image_correction_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    session = get_session(chat_id)

    if _is_stale(chat_id, session):
        return

    text = (update.message.text or "").strip()
    old_slot = session.get("image_review_pending_slot")
    if old_slot is None:
        session["state"] = "IDLE"
        return

    try:
        new_slot = int(text)
    except (ValueError, TypeError):
        await update.message.reply_text(
            "⚠️ الرجاء إرسال رقم صحيح فقط.",
            parse_mode=ParseMode.HTML,
        )
        return

    total_expected = len(session.get("sentences", []))
    if new_slot < 1 or new_slot > total_expected:
        await update.message.reply_text(
            f"⚠️ الرقم خارج النطاق المسموح (1 إلى {total_expected}).",
            parse_mode=ParseMode.HTML,
        )
        return

    review_map = session.get("image_review_map", {}) or {}
    current_path = review_map.get(int(old_slot))

    if not current_path:
        await update.message.reply_text(
            f"⚠️ لا توجد صورة مرتبطة بالرقم الحالي <code>#{old_slot}</code>.",
            parse_mode=ParseMode.HTML,
        )
        session["state"] = "IMAGE_REVIEW"
        session["image_review_pending_slot"] = None
        return

    if int(new_slot) != int(old_slot):
        other_path = review_map.get(int(new_slot))
        review_map[int(old_slot)] = other_path if other_path else current_path
        review_map[int(new_slot)] = current_path

        edits = session.get("image_review_manual_edits") or []
        edits.append({"old": int(old_slot), "new": int(new_slot)})
        if other_path:
            edits.append({"old": int(new_slot), "new": int(old_slot), "swapped": True})
        session["image_review_manual_edits"] = edits

        confirmed = session.get("image_review_confirmed") or {}
        confirmed.pop(int(old_slot), None)
        confirmed.pop(int(new_slot), None)
        session["image_review_confirmed"] = confirmed

    session["image_review_map"] = review_map

    session["image_review_duplicates"] = _compute_duplicates_from_map(review_map)
    session["image_review_missing"] = [
        s for s in range(1, total_expected + 1) if s not in review_map
    ]

    session["state"] = "IMAGE_REVIEW"
    session["image_review_pending_slot"] = None

    await update.message.reply_text(
        f"✅ <b>تم تحديث الرقم:</b> <code>#{old_slot}</code> → <code>#{new_slot}</code>",
        parse_mode=ParseMode.HTML,
    )

    await show_review_image(chat_id, context)


# =================================================================
# 9. شاشة الملخص النهائي
# =================================================================

def _compute_review_summary(session):
    total_expected = len(session.get("sentences", []))
    review_map = session.get("image_review_map", {}) or {}
    excluded = set(session.get("image_review_excluded") or [])

    ordered_slots = sorted([int(s) for s in review_map.keys() if int(s) not in excluded])
    missing = [s for s in range(1, total_expected + 1) if s not in ordered_slots]

    path_to_slots = {}
    for s in ordered_slots:
        p = review_map[s]
        try:
            key = str(Path(p).resolve())
        except Exception:
            key = str(p)
        path_to_slots.setdefault(key, []).append(s)

    duplicates = []
    for p, slots in path_to_slots.items():
        if len(slots) > 1:
            duplicates.append(sorted(slots))

    manual_edits = list(session.get("image_review_manual_edits") or [])

    return {
        "total_expected": total_expected,
        "ordered_count": len(ordered_slots),
        "ordered_slots": ordered_slots,
        "missing": missing,
        "duplicates": duplicates,
        "excluded": sorted(excluded),
        "manual_edits": manual_edits,
    }


async def show_review_summary(chat_id, context):
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    review_map = session.get("image_review_map", {}) or {}
    if not review_map:
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ لا توجد صور لاعتمادها. ارفع الصور أولاً.",
            parse_mode=ParseMode.HTML,
        )
        return

    summary = _compute_review_summary(session)
    session["image_review_summary"] = summary
    session["state"] = "IMAGE_REVIEW_SUMMARY"

    total_expected = summary["total_expected"]
    ordered_count = summary["ordered_count"]
    missing = summary["missing"]
    duplicates = summary["duplicates"]
    excluded = summary["excluded"]
    manual_edits = summary["manual_edits"]

    duplicates_txt = "—" if not duplicates else " | ".join(
        "، ".join(f"#{s}" for s in grp) for grp in duplicates
    )
    excluded_txt = "—" if not excluded else ", ".join(f"#{s}" for s in excluded)
    edits_txt = "—" if not manual_edits else ", ".join(
        f"#{e.get('old')}→#{e.get('new')}" for e in manual_edits
    )
    missing_txt = "—" if not missing else _format_missing_indices(missing)

    fully_complete = (ordered_count == total_expected) and (not missing) and (not duplicates)

    if fully_complete:
        header = "✅ <b>جاهز للاعتماد</b>"
    else:
        header = "⚠️ <b>الترتيب غير مكتمل — لن تبدأ الرندرة</b>"

    text = (
        f"<b>📊 ملخص مراجعة الصور النهائي</b>\n"
        f"{header}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>العدد المتوقع:</b> <code>{total_expected}</code>\n"
        f"✅ <b>عدد الصور المرتبة:</b> <code>{ordered_count}</code>\n"
        f"⚠️ <b>الأرقام المفقودة:</b> <code>{len(missing)}</code>\n"
        f"    └ {_esc(missing_txt)}\n"
        f"🔁 <b>التكرارات:</b> <code>{len(duplicates)}</code>\n"
        f"    └ {_esc(duplicates_txt)}\n"
        f"🚫 <b>الصور المستبعدة:</b> <code>{len(excluded)}</code>\n"
        f"    └ {_esc(excluded_txt)}\n"
        f"✏️ <b>تعديلات يدوية:</b> <code>{len(manual_edits)}</code>\n"
        f"    └ {_esc(edits_txt)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>⚠️ لن يبدأ المونتاج إلا بعد ضغطك على زر الاعتماد، وبشرط اكتمال كل الأرقام.</i>"
    )

    row1 = []
    if fully_complete:
        row1.append(InlineKeyboardButton(
            "✅ اعتماد الترتيب وبدء المونتاج",
            callback_data="review_summary_approve",
        ))
    else:
        row1.append(InlineKeyboardButton(
            "🔄 إعادة الفحص بعد رفع النواقص",
            callback_data="btn_start_render",
        ))

    row2 = [
        InlineKeyboardButton("🔄 العودة للمراجعة الكاملة", callback_data="review_summary_edit"),
        InlineKeyboardButton("❌ إلغاء", callback_data="review_summary_cancel"),
    ]

    await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=InlineKeyboardMarkup([row1, row2]),
        parse_mode=ParseMode.HTML,
    )


# =================================================================
# 10. اعتماد الترتيب + بدء الرندرة الكاملة فقط
# =================================================================

def _validate_final_map(final_map, total_expected):
    """
    يتحقق من أن الخريطة النهائية كاملة وصحيحة رقمياً وفيزيائياً.
    يُرجع (ok: bool, errors: list[str]).
    """
    errors = []
    slots = sorted(final_map.keys())

    # 1. يجب أن تكون المفاتيح هي 1..total_expected بالضبط
    expected_set = set(range(1, total_expected + 1))
    actual_set = set(slots)
    if actual_set != expected_set:
        missing_slots = sorted(expected_set - actual_set)
        extra_slots = sorted(actual_set - expected_set)
        if missing_slots:
            errors.append(
                f"أرقام مفقودة: {_format_missing_indices(missing_slots)}"
            )
        if extra_slots:
            errors.append(
                f"أرقام خارج النطاق: {_format_missing_indices(extra_slots)}"
            )

    # 2. كل مسار موجود وملف فعلي غير فارغ
    resolved_seen = {}
    for slot in slots:
        p = final_map[slot]
        try:
            pp = Path(p)
        except Exception:
            errors.append(f"مسار غير صالح للرقم #{slot}")
            continue
        if not pp.exists():
            errors.append(f"ملف غير موجود للرقم #{slot}: {pp.name}")
            continue
        if not pp.is_file():
            errors.append(f"ليس ملفًا للرقم #{slot}: {pp.name}")
            continue
        try:
            if pp.stat().st_size == 0:
                errors.append(f"ملف فارغ للرقم #{slot}: {pp.name}")
                continue
        except Exception:
            errors.append(f"تعذّر قراءة حالة الملف للرقم #{slot}")
            continue
        try:
            key = str(pp.resolve())
        except Exception:
            key = str(pp)
        if key in resolved_seen:
            errors.append(
                f"مسار مكرر بين #{resolved_seen[key]} و #{slot}"
            )
        else:
            resolved_seen[key] = slot

    return (not errors), errors


async def approve_and_render(msg_obj, context):
    chat_id = msg_obj.chat_id
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    total_expected = len(session.get("sentences", []))
    review_map = session.get("image_review_map", {}) or {}
    excluded = set(session.get("image_review_excluded") or [])

    final_map = {}
    for slot, p in review_map.items():
        try:
            s = int(slot)
        except (ValueError, TypeError):
            continue
        if s in excluded:
            continue
        if p:
            final_map[s] = str(p)

    ok, errors = _validate_final_map(final_map, total_expected)

    if not ok:
        errors_txt = "\n".join(f"• {_esc(e)}" for e in errors[:8])
        keyboard = [
            [InlineKeyboardButton("🔄 إعادة الفحص", callback_data="btn_start_render")],
            [InlineKeyboardButton("📋 العودة للمراجعة", callback_data="review_mode_full")],
            [InlineKeyboardButton("❌ إلغاء", callback_data="review_summary_cancel")],
        ]
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"❌ <b>لا يمكن بدء المونتاج — خريطة الصور غير مكتملة أو غير صالحة.</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{errors_txt}\n\n"
                f"<i>لا يوجد خيار للرندرة الجزئية. ارفع الصور الناقصة، صحّح الأرقام، "
                f"ثم أعد الفحص.</i>"
            ),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.HTML,
        )
        return

    session["final_image_map"] = final_map
    session["image_review_confirmed_flag"] = True

    await run_final_render(msg_obj, context)


async def run_final_render(msg_obj, context):
    """
    ينفّذ المونتاج الكامل فقط. لا يوجد allow_partial.
    """
    chat_id = msg_obj.chat_id
    session = get_session(chat_id)

    if _is_stale(chat_id, session):
        return

    ep_id = session.get("episode_id", "201")
    sentences = session.get("sentences", [])
    total_expected = len(sentences)

    final_map = session.get("final_image_map") or {}
    if not final_map:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ لا توجد صور معتمدة لبدء المونتاج.",
            parse_mode=ParseMode.HTML,
        )
        return

    ok, errors = _validate_final_map(final_map, total_expected)
    if not ok:
        errors_txt = "\n".join(f"• {_esc(e)}" for e in errors[:8])
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"❌ <b>تم رفض بدء المونتاج — خريطة الصور غير مكتملة.</b>\n"
                f"{errors_txt}\n\n"
                f"<i>ارفع الصور الناقصة أو صحّح الأرقام ثم أعد الفحص.</i>"
            ),
            parse_mode=ParseMode.HTML,
        )
        return

    # ترتيب صريح حسب الرقم (لا نعتمد على ترتيب قائمة)
    ordered_slots = sorted(final_map.keys())
    frames = [str(final_map[s]) for s in ordered_slots]

    try:
        progress_msg = await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"<b>🎬 رندرة كاملة — جاري بدء المونتاج عبر FFmpeg (1080p 60fps)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ تم اعتماد ترتيب <b>{len(frames)}</b> صورة.\n"
                f"⏳ جاري المزامنة الدقيقة بالمللي ثانية وتوليد الترجمة الحركية...\n"
                f"⏳ جاري تطبيق دورة حركات Ken Burns والانتقالات الهوائية (-18dB)...\n\n"
                f"<i>قد تستغرق الرندرة من دقيقتين إلى 4 دقائق حسب سرعة المعالج...</i>"
            ),
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        return

    try:
        audio_file = session.get("audio_path")
        if not audio_file:
            raise RuntimeError("ملف الصوت غير موجود في الجلسة.")
        try:
            ap = Path(audio_file)
        except Exception:
            ap = None
        if not ap or not ap.exists() or not ap.is_file():
            raise RuntimeError("ملف الصوت غير موجود على القرص.")

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
        if file_size_mb < 49:
            with open(final_video, "rb") as fv:
                await context.bot.send_video(
                    chat_id=chat_id,
                    video=fv,
                    caption=(
                        f"🏆 <b>فيديو الحلقة #{ep_id} جاهز للنشر!</b>\n"
                        f"الدقة: 1080p Full HD @ 60fps | الحجم: {file_size_mb:.1f} MB"
                    ),
                    parse_mode=ParseMode.HTML,
                )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🏆 <b>تم تصدير الفيديو النهائي بنجاح على السيرفر!</b>\n"
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


# تحديث حالة الحلقة إلى completed
        mark_episode_completed(ep_id)

# =================================================================
# 11. نقطة التشغيل
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
    print("🚀 محرك Vot Studio Pro يعمل الآن — Complete-Render-Only + 60-80 Sentence Edit + Full Voice List + Image Review")
    print("=" * 60)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
