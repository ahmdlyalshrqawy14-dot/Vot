# bot.py — Vot Studio Pro
# Human Script Review + Stage Manager (Automatic / Manual / Override)
# بدون أي تعديل على ملفات المشروع الأخرى.

import os
import re
import json
import time
import shutil
import asyncio
import logging
import html
import subprocess
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

from config import AZURE_MALE_VOICES, BASE_DIR
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
    "httpx", "httpcore", "telegram", "telegram.ext",
    "telegram.request", "telegram.ext.Updater", "telegram.ext.Application",
):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
EPISODES_FILE = BASE_DIR / "episodes.json"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

_EPISODES_LOCK = asyncio.Lock()

# -----------------------------------------------------------------
# ثوابت
# -----------------------------------------------------------------
SENTENCE_MIN = 60
SENTENCE_MAX = 80
DEFAULT_VOICE = "en-US-BrianMultilingualNeural"
SCRIPT_SEND_CHUNK = 3500

_COMPOSER_TEMP_DIR_RE = re.compile(r"^temp_segments_[0-9a-fA-F\-]+$")
_COMPOSER_TEMP_MAX_AGE_SECONDS = 1800

ALLOWED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".opus", ".flac", ".aac"}

user_sessions = {}
user_tasks = {}

VOICE_PREVIEW_TEXT = "Hello, this is a sample of my voice. How do I sound to you?"


# =================================================================
# STAGE MANAGER — الحالة الموحّدة لكل مرحلة
# =================================================================
STAGE_KEYS = ("1", "2", "3", "4", "5")


def _stage_default():
    return {
        "input_mode": None,       # automatic | manual | manual_override
        "status": "not_started",  # not_started | running | succeeded | failed | awaiting_manual_input | awaiting_approval | approved | skipped
        "source": None,           # ai | user_text | user_file | uploaded_audio | uploaded_images
        "manual_override": False,
        "artifact": None,         # مسار الملف/المجلد الناتج
        "approved": False,
    }


def _ensure_stages(session):
    stages = session.get("stages")
    if not isinstance(stages, dict):
        stages = {}
        session["stages"] = stages
    for k in STAGE_KEYS:
        if k not in stages or not isinstance(stages[k], dict):
            stages[k] = _stage_default()
        else:
            for f, v in _stage_default().items():
                stages[k].setdefault(f, v)
    return stages


def _stage_get(session, n):
    return _ensure_stages(session)[str(n)]


def _stage_set(session, n, **kwargs):
    _ensure_stages(session)[str(n)].update(kwargs)


def _stage_is_approved(session, n):
    return bool(_stage_get(session, n).get("approved"))


def _stage_mark_running(session, n, mode="automatic"):
    _stage_set(session, n, status="running", input_mode=mode)


def _stage_mark_failed(session, n, source=None, artifact=None):
    _stage_set(session, n, status="failed", source=source, artifact=artifact)


def _stage_mark_approved(session, n, source, artifact=None, manual_override=False):
    _stage_set(
        session, n,
        status="approved",
        approved=True,
        source=source,
        artifact=str(artifact) if artifact else None,
        manual_override=bool(manual_override),
    )


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

        "episode_status": None,
        "script_version": 0,
        "script_versions": [],
        "approved_script": None,
        "awaiting_script_replace": False,

        "script_input_buffer": [],
        "script_input_messages": 0,
        "script_input_files": 0,

        "manual_map": {},
        "manual_queue": [],
        "manual_missing": [],
        "manual_current": None,
        "manual_unindexed_files": [],
        "manual_missing_indices": [],
        "ocr_indexed_map": {},

        "image_review_mode": None,
        "image_review_items": [],
        "image_review_index": 0,
        "image_review_map": {},
        "image_review_original": {},
        "image_review_confirmed": {},
        "image_review_confirmed_flag": False,
        "image_review_summary": None,
        "image_review_message_id": None,
        "image_review_missing": [],
        "image_review_duplicates": [],
        "image_review_excluded": [],
        "image_review_manual_edits": [],
        "image_review_pending_slot": None,
        "final_image_map": {},

        # STAGE MANAGER
        "stages": {k: _stage_default() for k in STAGE_KEYS},

        # Stage 2 manual prompts buffer
        "stage2_prompt_buffer": [],
        "stage2_prompt_files": 0,
        "stage2_prompts_dir": None,

        # Stage 5 manual metadata buffer
        "stage5_metadata_buffer": None,
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
# فحص الصوت عبر ffprobe (للتحقق من صلاحية الملف اليدوي)
# =================================================================

def _probe_audio_file(path) -> float:
    """يتحقق من صلاحية ملف الصوت عبر ffprobe ويعيد مدته بالثواني.
    يرفع RuntimeError عند أي مشكلة (ملف تالف، غير صوتي، إلخ)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise RuntimeError("ملف الصوت غير موجود على القرص.")
    if p.stat().st_size == 0:
        raise RuntimeError("ملف الصوت فارغ.")

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type,duration",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(p),
            ],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        raise RuntimeError("ffprobe غير مثبّت على السيرفر — تعذّر التحقق من الملف.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe استغرق وقتًا طويلًا — الملف قد يكون تالفًا.")

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe رفض الملف: {result.stderr.strip() or 'خطأ غير معروف'}")

    durations = []
    for line in (result.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            durations.append(float(line))
        except ValueError:
            continue
    if not durations:
        raise RuntimeError("لم يتم العثور على تدفق صوتي صالح في الملف.")
    duration = max(durations)
    if duration <= 0.1:
        raise RuntimeError(f"مدة الملف غير منطقية: {duration:.3f}s")
    return duration


# =================================================================
# FORMAT-ONLY NORMALIZER
# =================================================================

def normalize_script_sentences(raw_text: str) -> list[str]:
    """تنظيف تنسيقي فقط — لا إعادة صياغة ولا ترجمة."""
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
    stage1_res = session.get("stage1_result")
    if not isinstance(stage1_res, dict):
        stage1_res = {}
    stage1_res["full_script_sentences"] = list(normalized)
    try:
        stage1_res["total_word_count"] = int(sum(len(s.split()) for s in normalized))
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
        logger.info(f"💾 حفظ السكريبت المُنسّق للحلقة {ep_id} ({len(normalized)} جملة)")
    except Exception as e:
        logger.error(f"فشل حفظ stage1_episode_{ep_id}.json: {e}")


def _save_script_version(ep_id, version: int, sentences: list[str]) -> Path:
    p = OUTPUTS_DIR / f"script_v{version}_episode_{ep_id}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"version": version, "sentences": list(sentences)},
                  f, ensure_ascii=False, indent=2)
    return p


def _chunk_script_lines(sentences: list[str]) -> list[str]:
    chunks, buf, buf_len = [], [], 0
    for s in sentences:
        line = s
        if buf and buf_len + len(line) + 1 > SCRIPT_SEND_CHUNK:
            chunks.append("\n".join(buf))
            buf, buf_len = [line], len(line)
        else:
            buf.append(line)
            buf_len += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks


async def _send_full_script_for_review(context, chat_id, ep_id, episode,
                                        sentences, version: int,
                                        manual_override: bool = False):
    total = len(sentences)

    override_note = ""
    if manual_override:
        override_note = (
            "\n⚠️ <b>هذا العدد خارج النطاق الموصى به</b> "
            f"(<code>{total}</code> جملة)."
        )

    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            f"📜 <b>السكريبت الكامل — الإصدار v{version}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 <b>الموضوع:</b> {_esc(episode.get('topic')) if isinstance(episode, dict) else '—'}\n"
            f"🧩 <b>عدد الجمل:</b> <code>{total}</code>"
            f"{override_note}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>راجع النص بالكامل، ثم اختر من الأزرار في نهاية الرسائل.</i>"
        ),
        parse_mode=ParseMode.HTML,
    )

    # ملف .txt
    txt_path = OUTPUTS_DIR / f"script_draft_v{version}_episode_{ep_id}.txt"
    txt_path.write_text("\n".join(sentences), encoding="utf-8")
    with open(txt_path, "rb") as fp:
        await context.bot.send_document(
            chat_id=chat_id,
            document=fp,
            filename=f"script_draft_v{version}.txt",
            caption=(
                f"📄 <b>النسخة الكاملة القابلة للتحرير — v{version}</b>\n"
                f"<i>كل جملة في سطر مستقل. عدّل ثم أعد إرسال النص للاستبدال.</i>"
            ),
            parse_mode=ParseMode.HTML,
        )

    # نسخة نصية داخل الشات
    chunks = _chunk_script_lines(sentences)
    n_chunks = len(chunks)
    if n_chunks == 1:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"<pre>{html.escape(chunks[0])}</pre>",
            parse_mode=ParseMode.HTML,
        )
    else:
        for i, ch in enumerate(chunks, 1):
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"<b>[الجزء {i} من {n_chunks}]</b>\n<pre>{html.escape(ch)}</pre>",
                parse_mode=ParseMode.HTML,
            )

    # ---- الأزرار تعتمد على ما إذا كان العدد داخل النطاق ----
    if manual_override:
        # العدد خارج النطاق: زر الاعتماد اليدوي الصريح فقط
        keyboard = [
            [InlineKeyboardButton(
                "✅ اعتماد يدوي وتجاوز شرط عدد الجمل",
                callback_data=f"s1_force_approve_{ep_id}",
            )],
            [InlineKeyboardButton(
                "📥 إرسال نسخة جديدة كاملة (استبدال)",
                callback_data=f"script_review_replace_{ep_id}",
            )],
            [InlineKeyboardButton(
                "🔄 إعادة توليد من الصفر",
                callback_data=f"regenerate_stage1_{ep_id}",
            )],
            [InlineKeyboardButton("❌ إلغاء والعودة", callback_data="btn_back_main")],
        ]
        prompt_text = (
            "👇 <b>ماذا تريد أن تفعل؟</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ <b>العدد خارج النطاق الموصى به.</b>\n"
            "• <b>اعتماد يدوي وتجاوز الشرط</b>: يبدأ تقسيم الجمل ← الصور ← الصوت.\n"
            "• <b>نسخة جديدة</b>: أرسل السكريبت كاملاً بعد التعديل.\n"
            "• <b>إعادة توليد</b>: يبدأ من الصفر (يفقد النسخة الحالية).\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<i>⚠️ لن يُسجَّل التجاوز إلا بعد ضغطك على زر الاعتماد اليدوي صراحةً.</i>"
        )
    else:
        keyboard = [
            [InlineKeyboardButton(
                "✅ اعتماد السكريبت والانتقال للمرحلة الثانية",
                callback_data=f"approve_stage1_{ep_id}",
            )],
            [InlineKeyboardButton(
                "📥 إرسال نسخة جديدة كاملة (استبدال)",
                callback_data=f"script_review_replace_{ep_id}",
            )],
            [InlineKeyboardButton(
                "🔄 إعادة توليد من الصفر",
                callback_data=f"regenerate_stage1_{ep_id}",
            )],
            [InlineKeyboardButton("❌ إلغاء والعودة", callback_data="btn_back_main")],
        ]
        prompt_text = (
            "👇 <b>ماذا تريد أن تفعل؟</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "• <b>اعتماد</b>: يبدأ تقسيم الجمل ← الصور ← الصوت.\n"
            f"• <b>نسخة جديدة</b>: أرسل السكريبت كاملاً بعد التعديل.\n"
            "• <b>إعادة توليد</b>: يبدأ من الصفر (يفقد النسخة الحالية).\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<i>⚠️ لن يبدأ أي شيء متعلق بالصوت قبل اعتمادك الصريح.</i>"
        )

    await context.bot.send_message(
        chat_id=chat_id,
        text=prompt_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.HTML,
    )


# =================================================================
# دوال عرض تقدّم رفع السكريبت
# =================================================================

def _count_approx_sentences(text: str) -> int:
    if not text:
        return 0
    parts = re.split(r'(?<=[.!?])(?=\s)|\n+', text)
    return len([p for p in parts if p.strip()])


async def show_script_input_progress(chat_id: int, session: dict,
                                     context: ContextTypes.DEFAULT_TYPE):
    buf = session.get("script_input_buffer") or []
    msgs = int(session.get("script_input_messages", 0))
    files = int(session.get("script_input_files", 0))
    ep_id = session.get("episode_id")

    combined = "\n".join(buf)
    approx = _count_approx_sentences(combined)

    is_replace = session.get("state") in ("WAITING_SCRIPT_REPLACE", "WAITING_SCRIPT_EDIT")
    title = "📥 <b>وضع استبدال السكريبت</b>" if is_replace else "✏️ <b>وضع تعديل السكريبت</b>"
    next_v = int(session.get("script_version", 0)) + 1

    text = (
        f"{title}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>المستلَم حتى الآن:</b>\n"
        f"   • رسائل نصية: <code>{msgs}</code>\n"
        f"   • ملفات .txt: <code>{files}</code>\n"
        f"   • جمل مقدّرة: <code>~{approx}</code>\n\n"
        f"📌 <b>يمكنك:</b>\n"
        f"• إرسال باقي الجمل كرسائل منفصلة بالترتيب.\n"
        f"• أو إرسال ملف <code>.txt</code> إضافي.\n"
        f"• عند الانتهاء اضغط <b>✅ انتهيت من الرفع</b>.\n\n"
        f"🎯 <b>الإصدار القادم:</b> <code>v{next_v}</code>   |   "
        f"<b>الموصى:</b> <code>{SENTENCE_MIN}</code>–<code>{SENTENCE_MAX}</code> جملة.\n"
        f"<i>⚠️ لن تتم المعالجة إلا بعد الضغط على زر «انتهيت».</i>"
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton(
            "✅ انتهيت من الرفع — عالج السكريبت",
            callback_data=f"script_input_done_{ep_id}",
        )],
        [InlineKeyboardButton(
            "🗑️ مسح ما تم رفعه والبدء من جديد",
            callback_data=f"script_input_clear_{ep_id}",
        )],
        [InlineKeyboardButton("❌ إلغاء والعودة", callback_data="btn_back_main")],
    ])

    try:
        await context.bot.send_message(
            chat_id=chat_id, text=text,
            reply_markup=keyboard, parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.error(f"فشل إرسال شاشة تقدّم السكريبت: {e}")


def _build_stage1_summary(ep_id, episode, stage1_res) -> str:
    if not isinstance(stage1_res, dict):
        stage1_res = {}
    cb = _as_dict(stage1_res.get("creative_brief"))
    rp = _as_dict(stage1_res.get("retention_plan"))
    qr = _as_dict(stage1_res.get("quality_report"))
    scene_plan = _as_list(stage1_res.get("scene_plan"))
    open_loops = _as_list(rp.get("open_loops"))
    pattern_interrupts = _as_list(rp.get("pattern_interrupts"))
    sentences = _as_list(stage1_res.get("full_script_sentences"))

    try:
        words_cnt_int = int(stage1_res.get("total_word_count", 0))
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
        "<b>🧠 الفكرة الإبداعية</b>",
        f"├ 💡 <b>الفكرة الأساسية:</b> {_esc(cb.get('core_idea'))}",
        f"├ 🎯 <b>الزاوية الفريدة:</b> {_esc(cb.get('unique_angle'))}",
        f"├ 🎞️ <b>صيغة الحلقة:</b> {_esc(cb.get('episode_format'))}",
        f"└ 🎁 <b>مخرجات المشاهد:</b> {_esc(cb.get('viewer_outcome'))}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "<b>🪝 الخطاف (Hook)</b>",
        f"{_esc(stage1_res.get('hook'))}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "<b>📈 خطة الاحتفاظ</b>",
        f"├ 🎣 <b>استراتيجية الخطاف:</b> {_esc(rp.get('hook_strategy'))}",
        f"├ 🔁 <b>عدد الحلقات المفتوحة:</b> <code>{len(open_loops)}</code>",
        f"├ ⚡ <b>عدد مقاطعات النمط:</b> <code>{len(pattern_interrupts)}</code>",
        f"└ 🎁 <b>المكافأة النهائية:</b> {_esc(rp.get('payoff'))}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"<b>✅ تقرير الجودة:</b> معتمد = {approved_tag}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "👇 <i>سيتم إرسال السكريبت الكامل للمراجعة في الرسائل التالية.</i>",
    ]
    return "\n".join(lines)


def _cleanup_episode_temp_files(ep_id):
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
    return None


async def mark_episode_completed(episode_id):
    async with _EPISODES_LOCK:
        if not EPISODES_FILE.exists():
            logger.warning("episodes.json غير موجود، تعذّر تحديث الحالة")
            return False
        try:
            def _do_update():
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
                    return False
                with open(EPISODES_FILE, "w", encoding="utf-8") as f:
                    json.dump(episodes, f, ensure_ascii=False, indent=2)
                return True

            result = await asyncio.to_thread(_do_update)
            if result:
                logger.info(f"✅ تم تحديث حالة الحلقة {episode_id} إلى completed")
            return result
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
# 1. /start → إعادة تشغيل كاملة
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

    logger.info(f"♻️ Hard Reset للمستخدم {chat_id} | cancelled_task={cancelled} | cleaned_ep={old_ep_id}")

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


async def soft_cancel(chat_id, context: ContextTypes.DEFAULT_TYPE, session, msg=None):
    """إلغاء ناعم: يُلغي المهمة الجارية ويعيد الحالة لـ IDLE دون مسح المراحل المعتمدة."""
    _cancel_user_task(chat_id)
    session["state"] = "IDLE"
    session["awaiting_script_replace"] = False
    session["awaiting_script_edit"] = False
    session["script_input_buffer"] = []
    session["script_input_messages"] = 0
    session["script_input_files"] = 0
    session["image_review_pending_slot"] = None
    session["stage2_prompt_buffer"] = []
    session["stage2_prompt_files"] = 0

    target = msg or context.bot
    try:
        if hasattr(target, "edit_message_text"):
            await target.edit_message_text(
                "❌ <b>تم الإلغاء الناعم.</b>\n"
                "<i>المراحل المعتمدة سابقاً لم تُمس. اضغط /start للعودة للقائمة الرئيسية.</i>",
                parse_mode=ParseMode.HTML,
            )
            return
    except Exception:
        pass
    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ <b>تم الإلغاء الناعم.</b>",
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        pass


# =================================================================
# 2. Callback Router
# =================================================================

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = update.effective_chat.id
    session = get_session(chat_id)

    if session.get("cancelled"):
        return

    # ---------------- القوائم الرئيسية ----------------
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
        await query.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(buttons), parse_mode=ParseMode.HTML)

    elif data == "btn_back_main":
        await start_command(query, context)

    elif data == "soft_cancel":
        await soft_cancel(chat_id, context, session, msg=query.message)

    # ---------------- تأكيد الحلقة → عرض خيار الوضع ----------------
    elif data.startswith("confirm_ep_"):
        target_id = data.replace("confirm_ep_", "")
        ep = get_episode(target_id)
        if ep:
            _cancel_user_task(chat_id)
            await show_stage1_mode_choice(query, session, ep)

    # ---------------- Stage 1: اختيار الوضع ----------------
    elif data.startswith("s1_auto_"):
        target_id = data.replace("s1_auto_", "")
        ep = get_episode(target_id) or session.get("episode_data")
        if not ep:
            await query.edit_message_text("⚠️ تعذّر العثور على الحلقة.")
            return
        _cancel_user_task(chat_id)
        user_tasks[chat_id] = asyncio.create_task(
            run_stage1_only(query, context, ep)
        )

    elif data.startswith("s1_man_"):
        target_id = data.replace("s1_man_", "")
        ep = get_episode(target_id) or session.get("episode_data")
        if not ep:
            await query.edit_message_text("⚠️ تعذّر العثور على الحلقة.")
            return
        await enter_stage1_manual_mode(query, session, ep, reason="mode_choice")

    # ---------------- Stage 1: إعادة توليد ----------------
    elif data.startswith("regenerate_stage1_"):
        target_id = data.replace("regenerate_stage1_", "")
        ep = session.get("episode_data") or get_episode(target_id)
        if ep:
            session["script_input_buffer"] = []
            session["script_input_messages"] = 0
            session["script_input_files"] = 0
            _cancel_user_task(chat_id)
            user_tasks[chat_id] = asyncio.create_task(run_stage1_only(query, context, ep))

    # ---------------- Stage 1: اعتماد عادي ----------------
    elif data.startswith("approve_stage1_"):
        target_id = data.replace("approve_stage1_", "")
        active_id = session.get("episode_id")

        if not active_id or str(active_id) != str(target_id):
            try:
                await query.answer("⚠️ هذا الزر لا يخص الحلقة النشطة الحالية.", show_alert=True)
            except Exception:
                pass
            return

        if session.get("stage1_approved") is True:
            try:
                await query.answer("ℹ️ تم اعتماد هذه المرحلة مسبقاً بالفعل.", show_alert=True)
            except Exception:
                pass
            return

        stage1_res = session.get("stage1_result")
        s1_file = OUTPUTS_DIR / f"stage1_episode_{target_id}.json"
        if not isinstance(stage1_res, dict) and not s1_file.exists():
            try:
                await query.answer("⚠️ لا توجد بيانات مرحلة أولى معتمدة.", show_alert=True)
            except Exception:
                pass
            return

        approved_sentences = list(session.get("sentences") or [])
        if not approved_sentences:
            try:
                await query.answer("⚠️ لا توجد جمل معتمدة.", show_alert=True)
            except Exception:
                pass
            return

        session["approved_script"] = approved_sentences
        session["episode_status"] = "script_approved"
        session["stage1_approved"] = True

        s1_res = session.get("stage1_result")
        if isinstance(s1_res, dict):
            s1_res["full_script_sentences"] = list(approved_sentences)
            session["stage1_result"] = s1_res

        # STAGE MANAGER
        stage = _stage_get(session, 1)
        manual_override = bool(stage.get("manual_override"))
        source = stage.get("source") or "ai"
        _stage_mark_approved(
            session, 1, source=source,
            artifact=stage.get("artifact"),
            manual_override=manual_override,
        )

        _cancel_user_task(chat_id)
        # ✅ بوابة: عرض خيار وضع المرحلة الثانية بدلاً من تشغيلها فوراً
        await show_stage2_mode_choice(query, session, context)

    # ---------------- Stage 1: اعتماد يدوي صريح (تجاوز شرط العدد) ----------------
    elif data.startswith("s1_force_approve_"):
        target_id = data.replace("s1_force_approve_", "")
        active_id = session.get("episode_id")

        if not active_id or str(active_id) != str(target_id):
            try:
                await query.answer("⚠️ هذا الزر لا يخص الحلقة النشطة الحالية.", show_alert=True)
            except Exception:
                pass
            return

        if session.get("stage1_approved") is True:
            try:
                await query.answer("ℹ️ تم اعتماد هذه المرحلة مسبقاً بالفعل.", show_alert=True)
            except Exception:
                pass
            return

        approved_sentences = list(session.get("sentences") or [])
        if not approved_sentences:
            try:
                await query.answer("⚠️ لا توجد جمل معتمدة.", show_alert=True)
            except Exception:
                pass
            return

        # ✅ التجاوز اليدوي لا يُثبَّت إلا الآن، بعد ضغط المستخدم الصريح
        _stage_set(session, 1, manual_override=True)

        session["approved_script"] = approved_sentences
        session["episode_status"] = "script_approved"
        session["stage1_approved"] = True

        s1_res = session.get("stage1_result")
        if isinstance(s1_res, dict):
            s1_res["full_script_sentences"] = list(approved_sentences)
            session["stage1_result"] = s1_res

        stage = _stage_get(session, 1)
        source = stage.get("source") or "ai"
        _stage_mark_approved(
            session, 1, source=source,
            artifact=stage.get("artifact"),
            manual_override=True,
        )

        _cancel_user_task(chat_id)
        await show_stage2_mode_choice(query, session, context)

    # ---------------- Stage 1: استبدال ----------------
    elif data.startswith("script_review_replace_"):
        target_id = data.replace("script_review_replace_", "")
        if str(session.get("episode_id")) != str(target_id):
            try:
                await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            except Exception:
                pass
            return
        if session.get("episode_status") == "script_approved":
            try:
                await query.answer("ℹ️ تم اعتماد السكريبت بالفعل.", show_alert=True)
            except Exception:
                pass
            return
        await enter_stage1_manual_mode(query, session, session.get("episode_data"), reason="replace")

    elif data.startswith("edit_stage1_"):
        target_id = data.replace("edit_stage1_", "")
        active_id = session.get("episode_id")
        if not active_id or str(active_id) != str(target_id):
            try:
                await query.answer("⚠️ هذا الزر لا يخص الحلقة النشطة الحالية.", show_alert=True)
            except Exception:
                pass
            return
        if session.get("episode_status") == "script_approved":
            try:
                await query.answer("ℹ️ تم اعتماد السكريبت بالفعل.", show_alert=True)
            except Exception:
                pass
            return
        await enter_stage1_manual_mode(query, session, session.get("episode_data"), reason="edit")

    # ---------------- Stage 1: تجاوز يدوي عند الفشل ----------------
    elif data.startswith("s1_override_"):
        target_id = data.replace("s1_override_", "")
        if str(session.get("episode_id")) != str(target_id):
            try:
                await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            except Exception:
                pass
            return
        await enter_stage1_manual_mode(query, session, session.get("episode_data"), reason="override")

    # ---------------- Stage 1: تجميع السكريبت ----------------
    elif data.startswith("script_input_done_"):
        target_id = data.replace("script_input_done_", "")
        if str(session.get("episode_id")) != str(target_id):
            try:
                await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            except Exception:
                pass
            return
        if session.get("episode_status") == "script_approved":
            try:
                await query.answer("ℹ️ السكريبت معتمد بالفعل.", show_alert=True)
            except Exception:
                pass
            return

        buf = session.get("script_input_buffer") or []
        if not buf:
            try:
                await query.answer("⚠️ لم ترفع أي نص أو ملف بعد.", show_alert=True)
            except Exception:
                pass
            return

        combined = "\n".join(buf)
        try:
            normalized = normalize_script_sentences(combined)
        except ValueError as ve:
            await query.message.reply_text(
                f"⚠️ <b>لم أتمكن من قراءة جمل صالحة.</b>\n<i>{_esc(str(ve))}</i>\n\n"
                f"عدّل النص وأرسله من جديد.",
                parse_mode=ParseMode.HTML,
            )
            return
        except Exception as e:
            logger.exception("خطأ أثناء تنسيق السكريبت المُجمَّع")
            await query.message.reply_text(
                f"❌ <b>خطأ أثناء التنسيق:</b>\n<code>{_esc(str(e))}</code>",
                parse_mode=ParseMode.HTML,
            )
            return

        n = len(normalized)
        in_range = (SENTENCE_MIN <= n <= SENTENCE_MAX)

        # حفظ الإصدار + المتابعة
        ep_id = session.get("episode_id")
        new_version = int(session.get("script_version", 0)) + 1
        v_path = _save_script_version(ep_id, new_version, normalized)
        versions = list(session.get("script_versions") or [])
        versions.append({"version": new_version,
                         "sentences": list(normalized),
                         "path": str(v_path)})
        session["script_versions"] = versions
        session["script_version"] = new_version
        session["sentences"] = list(normalized)

        s1_res = session.get("stage1_result")
        if not isinstance(s1_res, dict):
            s1_res = {}
        s1_res["full_script_sentences"] = list(normalized)
        try:
            s1_res["total_word_count"] = int(sum(len(s.split()) for s in normalized))
        except Exception:
            pass
        session["stage1_result"] = s1_res

        if ep_id:
            try:
                s1_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"
                with open(s1_file, "w", encoding="utf-8") as f:
                    json.dump(s1_res, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"فشل حفظ stage1_episode_{ep_id}.json: {e}")

        session["script_input_buffer"] = []
        session["script_input_messages"] = 0
        session["script_input_files"] = 0
        session["awaiting_script_replace"] = False
        session["awaiting_script_edit"] = False
        session["state"] = "IDLE"

        # STAGE MANAGER
        # ✅ manual_override لا يُثبّت هنا — يبقى False حتى يضغط المستخدم الزر الصريح
        _stage_set(session, 1,
                   input_mode="manual",
                   source="user_text",
                   artifact=str(v_path),
                   status="awaiting_approval",
                   manual_override=False,
                   )

        if not in_range:
            # عرض تحذير مع خيارات (زر التجاوز الصريح)
            keyboard = [
                [InlineKeyboardButton(
                    "✅ اعتماد يدوي وتجاوز شرط العدد",
                    callback_data=f"s1_force_approve_{ep_id}",
                )],
                [InlineKeyboardButton(
                    "✏️ إرسال نسخة معدلة",
                    callback_data=f"script_review_replace_{ep_id}",
                )],
                [InlineKeyboardButton("❌ إلغاء والعودة", callback_data="soft_cancel")],
            ]
            await query.message.reply_text(
                f"⚠️ <b>النسخة اليدوية خارج النطاق الموصى به</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 <b>عدد الجمل:</b> <code>{n}</code>\n"
                f"✅ <b>الموصى به:</b> <code>{SENTENCE_MIN}</code>–<code>{SENTENCE_MAX}</code> جملة.\n"
                f"<i>هذا العدد خارج النطاق. يمكنك اعتماده يدويًا وتجاوز الشرط، "
                f"أو إرسال نسخة معدلة.</i>",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode=ParseMode.HTML,
            )

        await query.message.reply_text(
            f"✅ <b>تم استلام النسخة v{new_version}</b>\n"
            f"🧩 <b>عدد الجمل:</b> <code>{n}</code>\n"
            f"<i>سيتم عرضها الآن للمراجعة. لم يُعتمد شيء بعد.</i>",
            parse_mode=ParseMode.HTML,
        )

        ep = session.get("episode_data") or {}
        await _send_full_script_for_review(
            context, chat_id, ep_id, ep, normalized,
            version=new_version,
            manual_override=(not in_range),
        )

    elif data.startswith("script_input_clear_"):
        target_id = data.replace("script_input_clear_", "")
        if str(session.get("episode_id")) != str(target_id):
            try:
                await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            except Exception:
                pass
            return
        session["script_input_buffer"] = []
        session["script_input_messages"] = 0
        session["script_input_files"] = 0
        try:
            await query.answer("🗑️ تم المسح")
        except Exception:
            pass
        await query.message.reply_text(
            "🗑️ <b>تم مسح كل ما تم رفعه.</b>\n"
            "يمكنك الآن البدء من جديد (نص / ملف .txt / عدة رسائل).",
            parse_mode=ParseMode.HTML,
        )

    # ---------------- Stage 2: اختيار الوضع ----------------
    elif data.startswith("s2_auto_"):
        target_id = data.replace("s2_auto_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        _cancel_user_task(chat_id)
        user_tasks[chat_id] = asyncio.create_task(run_stage2_auto(query, context))

    elif data.startswith("s2_man_"):
        target_id = data.replace("s2_man_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        await enter_stage2_manual_mode(query, session, reason="mode_choice")

    elif data.startswith("s2_approve_"):
        target_id = data.replace("s2_approve_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        await approve_stage2(query, session, context)

    elif data.startswith("s2_retry_"):
        target_id = data.replace("s2_retry_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        _cancel_user_task(chat_id)
        user_tasks[chat_id] = asyncio.create_task(run_stage2_auto(query, context))

    elif data.startswith("s2_clear_"):
        target_id = data.replace("s2_clear_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        session["stage2_prompt_buffer"] = []
        session["stage2_prompt_files"] = 0
        try:
            await query.answer("🗑️ تم المسح")
        except Exception:
            pass
        await query.message.reply_text("🗑️ <b>تم مسح حزمة الأوامر المُجمّعة.</b>", parse_mode=ParseMode.HTML)

    elif data.startswith("s2_done_"):
        target_id = data.replace("s2_done_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        await finalize_stage2_manual(query, session, context)

    # ---------------- Stage 3: اختيار الوضع ----------------
    elif data.startswith("s3_auto_"):
        target_id = data.replace("s3_auto_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        await present_audio_engine_choice(query)

    elif data.startswith("s3_man_"):
        target_id = data.replace("s3_man_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        await enter_stage3_manual_mode(query, session)

    elif data.startswith("s3_replace_"):
        target_id = data.replace("s3_replace_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        await enter_stage3_manual_mode(query, session)

    elif data.startswith("s3_retry_"):
        target_id = data.replace("s3_retry_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        _cancel_user_task(chat_id)
        user_tasks[chat_id] = asyncio.create_task(run_stage3(query, context))

    elif data.startswith("s3_continue_"):
        target_id = data.replace("s3_continue_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        await prompt_stage4_upload(query, session)

    # ---------------- الصوت: معاينة / اختيار ----------------
    elif data.startswith("preview_voice_"):
        preview_voice = data[len("preview_voice_"):]
        if preview_voice not in AZURE_MALE_VOICES:
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

    elif data.startswith("voice_"):
        selected_voice = data[len("voice_"):]
        if selected_voice not in AZURE_MALE_VOICES:
            try:
                await query.answer("⚠️ صوت غير مُعرَّف في الإعدادات.", show_alert=True)
            except Exception:
                pass
            return
        session["voice"] = selected_voice
        session["engine"] = "azure"
        user_tasks[chat_id] = asyncio.create_task(run_stage3(query, context))

    # ---------------- Stage 4: اختيار الوضع ----------------
    elif data.startswith("s4_auto_"):
        target_id = data.replace("s4_auto_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        _stage_set(session, 4, input_mode="automatic", status="running")
        user_tasks[chat_id] = asyncio.create_task(
            run_image_verification(query.message, context)
        )

    elif data.startswith("s4_man_"):
        target_id = data.replace("s4_man_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        _stage_set(session, 4, input_mode="manual")
        # تشغيل الترتيب الأول لبناء index ثم فتح التعيين اليدوي
        user_tasks[chat_id] = asyncio.create_task(
            run_image_verification(query.message, context, force_manual_after=True)
        )

    elif data == "btn_start_render":
        # احتفظ بالزر القديم للتوافق الرجعي
        _stage_set(session, 4, input_mode="automatic", status="running")
        user_tasks[chat_id] = asyncio.create_task(
            run_image_verification(query.message, context)
        )

    # ---------------- Manual assign ----------------
    elif data == "manual_cancel":
        session["state"] = "IDLE"
        session["manual_map"] = {}
        session["manual_queue"] = []
        session["manual_missing"] = []
        session["manual_current"] = None
        try:
            await query.edit_message_caption(
                caption="❌ <b>تم إلغاء الوضع اليدوي.</b>\nيمكنك رفع الصور الناقصة ثم إعادة الترتيب.",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            try:
                await query.message.delete()
            except Exception:
                pass
            await context.bot.send_message(chat_id=chat_id, text="❌ <b>تم إلغاء الوضع اليدوي.</b>", parse_mode=ParseMode.HTML)
        return

    elif data == "btn_manual_assign":
        await start_manual_assign_flow(query, context, session)

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

    elif data == "btn_show_ignored":
        unindexed = session.get("manual_unindexed_files", []) or []
        if not unindexed:
            try:
                await query.answer("لا توجد صور متجاهلة حالياً.", show_alert=True)
            except Exception:
                pass
            return
        names = [Path(p).name for p in unindexed[:20]]
        extra = len(unindexed) - 20 if len(unindexed) > 20 else 0
        text = (
            f"🗑️ <b>الصور المتجاهلة</b> ({len(unindexed)})\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"هذه الصور تم تجاهلها لأنها مكررة أو اسمها غير صالح:\n\n"
            + "\n".join(f"• <code>{_esc(n)}</code>" for n in names)
        )
        if extra > 0:
            text += f"\n\n... و<code>{extra}</code> صورة إضافية."
        try:
            await query.answer()
        except Exception:
            pass
        try:
            await query.message.reply_text(text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"فشل إرسال قائمة الصور المتجاهلة: {e}")

    # ---------------- Image review ----------------
    elif data == "review_mode_quick":
        await start_image_review(chat_id, context, mode="quick")

    elif data == "review_mode_full":
        await start_image_review(chat_id, context, mode="full")

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

    # ---------------- Stage 5: metadata ----------------
    elif data.startswith("s5_approve_auto_"):
        # ✅ اعتماد النتيجة التلقائية (لا يعتمد على stage5_metadata_buffer بالصيغة اليدوية)
        target_id = data.replace("s5_approve_auto_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        await approve_stage5_auto(query, session, context)

    elif data.startswith("s5_retry_"):
        target_id = data.replace("s5_retry_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        user_tasks[chat_id] = asyncio.create_task(run_stage5(query.message, context))

    elif data.startswith("s5_man_"):
        target_id = data.replace("s5_man_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        await enter_stage5_manual_mode(query, session)

    elif data.startswith("s5_skip_"):
        target_id = data.replace("s5_skip_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        _stage_set(session, 5, status="skipped", approved=True, source="user_skip")
        try:
            await query.edit_message_text(
                "⏭️ <b>تم تخطي بيانات النشر.</b>\n"
                "<i>الفيديو النهائي محفوظ على السيرفر. يمكنك توليد الميتاداتا لاحقًا ببدء حلقة جديدة.</i>",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass
        # ✅ إغلاق دورة الإنتاج بعد التخطي
        await _finalize_episode(context, chat_id, session, target_id)

    elif data.startswith("s5_done_"):
        target_id = data.replace("s5_done_", "")
        if str(session.get("episode_id")) != str(target_id):
            await query.answer("⚠️ لا يخص الحلقة النشطة.", show_alert=True)
            return
        await finalize_stage5_manual(query, session, context)

    # ---------------- الرندرة: خيارات ما بعد الفشل ----------------
    elif data == "render_retry":
        user_tasks[chat_id] = asyncio.create_task(run_final_render(query.message, context))

    elif data == "render_replace_audio":
        await enter_stage3_manual_mode(query, session)

    elif data == "render_edit_images":
        await start_image_review(chat_id, context, mode="full")


async def show_episode_confirmation(query, ep):
    ep_id = ep.get("id", "201")
    topic = ep.get("topic", "بدون عنوان")
    myth = ep.get("the_myth", "غير محدد")
    status = ep.get("status", "pending")
    status_tag = "🟡 قيد الانتظار" if status == "pending" else "🟢 تم إنتاجها مسبقاً"

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
# 3. اختيار الوضع لكل مرحلة
# =================================================================

async def show_stage1_mode_choice(query, session, ep):
    ep_id = ep.get("id", "201")
    session["episode_id"] = str(ep_id)
    session["episode_data"] = ep

    text = (
        f"<b>📝 المرحلة 1 — توليد السكريبت للحلقة #{_esc(ep_id)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"اختر طريقة إدخال السكريبت:\n\n"
        f"<b>🤖 تلقائي</b> — يكتبه الذكاء الاصطناعي من بيانات الحلقة.\n"
        f"<b>✍️ يدوي</b> — أنت ترسل السكريبت (نص/ملف .txt).\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>في الحالتين، لن يبدأ شيء قبل اعتمادك الصريح.</i>"
    )
    keyboard = [
        [InlineKeyboardButton("🤖 تشغيل المرحلة الأولى تلقائيًا", callback_data=f"s1_auto_{ep_id}")],
        [InlineKeyboardButton("✍️ إدخال السكريبت يدويًا", callback_data=f"s1_man_{ep_id}")],
        [InlineKeyboardButton("❌ إلغاء والعودة", callback_data="btn_back_main")],
    ]
    try:
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
    except Exception:
        await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)


async def enter_stage1_manual_mode(query, session, ep, reason: str):
    ep_id = (ep or {}).get("id") or session.get("episode_id")
    session["episode_id"] = str(ep_id) if ep_id else None
    session["episode_data"] = ep or session.get("episode_data")
    session["state"] = "WAITING_SCRIPT_REPLACE"
    session["awaiting_script_replace"] = True
    session["awaiting_script_edit"] = False
    session["script_input_buffer"] = []
    session["script_input_messages"] = 0
    session["script_input_files"] = 0

    next_v = int(session.get("script_version", 0)) + 1
    text = (
        f"📥 <b>وضع الإدخال اليدوي للسكريبت</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"يمكنك إرسال السكريبت بإحدى الطرق التالية:\n\n"
        f"<b>1️⃣ رسالة نصية واحدة</b>\n"
        f"<b>2️⃣ ملف نصي (.txt)</b>\n"
        f"<b>3️⃣ رسائل متعددة بالترتيب</b>\n\n"
        f"📊 <b>الموصى به:</b> <code>{SENTENCE_MIN}</code>–<code>{SENTENCE_MAX}</code> جملة.\n"
        f"<i>(يمكنك تجاوز العدد — سيُطلب منك تأكيد صريح بعدين)</i>\n"
        f"📌 <i>تنسيق فقط — بدون إعادة صياغة أو ترجمة.</i>\n"
        f"🎯 سيُسجَّل كإصدار جديد: <b>v{next_v}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    keyboard = [
        [InlineKeyboardButton("🗑️ مسح ما تم رفعه والبدء من جديد",
                              callback_data=f"script_input_clear_{ep_id}")],
        [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
    ]
    try:
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
    except Exception:
        try:
            await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
        except Exception:
            pass


async def show_stage2_mode_choice(query, session, context):
    ep_id = session.get("episode_id")
    text = (
        f"<b>🎨 المرحلة 2 — أوامر الصور (Prompts)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"اختر طريقة إنشاء ملفات الأوامر:\n\n"
        f"<b>🤖 تلقائي</b> — يُولّدها البوت ويُرسلها كملفات .txt.\n"
        f"<b>✍️ يدوي</b> — أنت ترفع ملفات .txt (أو نصًا مباشرًا).\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>⚠️ لن تنتقل للمرحلة الثالثة إلا بعد اعتماد حزمة الأوامر صراحةً.</i>"
    )
    keyboard = [
        [InlineKeyboardButton("🤖 تشغيل تلقائي", callback_data=f"s2_auto_{ep_id}")],
        [InlineKeyboardButton("✍️ رفع أوامر يدوياً", callback_data=f"s2_man_{ep_id}")],
        [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
    ]
    try:
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
    except Exception:
        await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)


async def enter_stage2_manual_mode(query, session, reason="replace"):
    ep_id = session.get("episode_id")
    session["state"] = "WAITING_STAGE2_PROMPTS"
    session["stage2_prompt_buffer"] = []
    session["stage2_prompt_files"] = 0
    session["stage2_prompts_dir"] = str(OUTPUTS_DIR / f"episode_{ep_id}_prompts_manual")

    total = len(session.get("sentences", []))
    text = (
        f"📥 <b>وضع رفع أوامر الصور يدويًا</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"أرسل ملف <code>.txt</code> واحدًا أو أكثر (كل ملف يمثل جزءًا)،\n"
        f"أو الصق الأوامر نصًا في رسالة واحدة أو أكثر.\n\n"
        f"📊 <b>عدد الجمل الكلي:</b> <code>{total}</code> — يُفضَّل تغطية كل جملة بأمر واحد.\n"
        f"💡 يمكنك المزج: نصوص + ملفات. سيتم دمج الكل بالترتيب.\n"
        f"<i>⚠️ لن تُعتمد الأوامر إلا بعد الضغط على «انتهيت».</i>"
    )
    keyboard = [
        [InlineKeyboardButton("✅ انتهيت من الرفع", callback_data=f"s2_done_{ep_id}")],
        [InlineKeyboardButton("🗑️ مسح ما رُفع", callback_data=f"s2_clear_{ep_id}")],
        [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
    ]
    try:
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
    except Exception:
        await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)


async def enter_stage3_manual_mode(query, session):
    ep_id = session.get("episode_id")
    session["state"] = "WAITING_STAGE3_AUDIO"
    text = (
        f"📤 <b>رفع ملف صوت خارجي</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"أرسل الآن ملف الصوت (mp3 / wav / m4a / ogg / opus / flac / aac).\n\n"
        f"💡 سيتم التحقق من الملف عبر ffprobe قبل الاعتماد.\n"
        f"⚠️ <i>لا يمكن التحقق من المزامنة تلقائيًا؛ سيمرّ الملف لمرحلة الترجمة "
        f"وسيُبنى التايملاين على أساسه.</i>\n"
        f"❌ لإلغاء الرفع اضغط «إلغاء»."
    )
    keyboard = [[InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")]]
    try:
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
    except Exception:
        await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)


async def prompt_stage4_upload(query, session):
    ep_id = session.get("episode_id")
    total = len(session.get("sentences", []))
    text = (
        f"📸 <b>المرحلة 4 — استقبال الصور</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"أرسل الصور هنا (كصور أو ملفات دفعة واحدة).\n"
        f"سمِّ كل صورة برقمها فقط (1.webp، 2.png، 3.jpg...).\n\n"
        f"🎯 <b>المطلوب:</b> <code>{total}</code> صورة.\n"
        f"<i>عند الانتهاء سيظهر خيار: ترتيب تلقائي أو تعيين يدوي.</i>"
    )
    try:
        await query.edit_message_text(text, parse_mode=ParseMode.HTML)
    except Exception:
        await query.message.reply_text(text, parse_mode=ParseMode.HTML)


async def enter_stage5_manual_mode(query, session):
    ep_id = session.get("episode_id")
    session["state"] = "WAITING_STAGE5_METADATA"
    session["stage5_metadata_buffer"] = None
    text = (
        f"✍️ <b>إدخال بيانات النشر يدويًا</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"أرسل رسالة واحدة بالتنسيق التالي:\n\n"
        f"<pre>TITLE:\nعنوان الحلقة\n\n"
        f"ALT_TITLES:\nعنوان بديل 1\nعنوان بديل 2\n\n"
        f"DESCRIPTION:\nالوصف هنا (عدة أسطر)\n\n"
        f"TAGS:\nتاج1, تاج2, تاج3\n\n"
        f"THUMBNAIL:\nنص الصورة المصغرة</pre>\n\n"
        f"<i>الحقول غير المذكورة تُترك فارغة.</i>"
    )
    keyboard = [
        [InlineKeyboardButton("⏭️ تخطي بيانات النشر", callback_data=f"s5_skip_{ep_id}")],
        [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
    ]
    try:
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
    except Exception:
        await query.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)


# =================================================================
# 4. الرسائل النصية
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
                card_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML
            )
        else:
            await update.message.reply_text(
                f"❌ لم يتم العثور على حلقة برقم ID <code>{_esc(entered_text)}</code>.",
                parse_mode=ParseMode.HTML,
            )
        return

    # ================= Stage 1 manual script =================
    if state in ("WAITING_SCRIPT_REPLACE", "WAITING_SCRIPT_EDIT"):
        if _is_stale(chat_id, session):
            return
        if session.get("episode_status") == "script_approved":
            await update.message.reply_text(
                "⚠️ السكريبت معتمد بالفعل. استخدم /start لبدء حلقة جديدة.",
                parse_mode=ParseMode.HTML,
            )
            return
        raw = update.message.text or ""
        if not raw.strip():
            return
        buf = session.get("script_input_buffer") or []
        buf.append(raw)
        session["script_input_buffer"] = buf
        session["script_input_messages"] = int(session.get("script_input_messages", 0)) + 1
        await show_script_input_progress(chat_id, session, context)
        return

    # ================= Stage 2 manual prompts =================
    if state == "WAITING_STAGE2_PROMPTS":
        if _is_stale(chat_id, session):
            return
        raw = update.message.text or ""
        if not raw.strip():
            return
        buf = session.get("stage2_prompt_buffer") or []
        buf.append(raw)
        session["stage2_prompt_buffer"] = buf
        await _show_stage2_upload_progress(chat_id, session, context)
        return

    # ================= Stage 5 manual metadata =================
    if state == "WAITING_STAGE5_METADATA":
        if _is_stale(chat_id, session):
            return
        parsed = _parse_manual_metadata(update.message.text or "")
        if not parsed.get("title"):
            await update.message.reply_text(
                "⚠️ لم أتمكن من قراءة <code>TITLE:</code>.\n"
                "أرسل الرسالة مرة أخرى بالتنسيق المطلوب.",
                parse_mode=ParseMode.HTML,
            )
            return
        session["stage5_metadata_buffer"] = parsed
        await update.message.reply_text(
            f"✅ <b>تم استلام بيانات النشر.</b>\n"
            f"📝 العنوان: <b>{_esc(parsed.get('title'))}</b>\n"
            f"<i>اضغط «انتهيت» للإرسال أو «إلغاء» للتراجع.</i>",
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ إرسال البيانات", callback_data=f"s5_done_{session.get('episode_id')}")],
                [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
            ]),
        )
        return

    if state == "WAITING_IMAGE_CORRECTION":
        await handle_image_correction_text(update, context)
        return


def _parse_manual_metadata(text: str) -> dict:
    """يقرأ حقول بيانات النشر من نص منسّق."""
    if not text:
        return {}
    sections = {"TITLE": "", "ALT_TITLES": "", "DESCRIPTION": "", "TAGS": "", "THUMBNAIL": ""}
    current = None
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Z_]+)\s*:\s*(.*)$", line)
        if m and m.group(1) in sections:
            current = m.group(1)
            if m.group(2):
                sections[current] += m.group(2) + "\n"
            continue
        if current:
            sections[current] += line + "\n"

    def clean(s):
        return s.strip()

    title = clean(sections["TITLE"])
    alt = [x.strip() for x in sections["ALT_TITLES"].splitlines() if x.strip()]
    desc = clean(sections["DESCRIPTION"])
    tags = [x.strip() for x in re.split(r"[,\n]+", sections["TAGS"]) if x.strip()]
    thumb = clean(sections["THUMBNAIL"])

    return {
        "title": title,
        "alt_titles": alt,
        "description": desc,
        "tags": tags,
        "thumbnail": thumb,
    }


async def _show_stage2_upload_progress(chat_id, session, context):
    buf = session.get("stage2_prompt_buffer") or []
    files = int(session.get("stage2_prompt_files", 0))
    ep_id = session.get("episode_id")
    total = len(session.get("sentences", []))

    approx_blocks = 0
    for b in buf:
        approx_blocks += len([x for x in b.split("\n\n") if x.strip()])

    text = (
        f"📥 <b>وضع رفع أوامر الصور</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>المستلَم:</b>\n"
        f"   • مقاطع نصية: <code>{approx_blocks}</code>\n"
        f"   • ملفات .txt: <code>{files}</code>\n"
        f"🎯 <b>عدد الجمل:</b> <code>{total}</code>\n\n"
        f"👇 استمر بالرفع، أو اضغط «انتهيت» عند الانتهاء."
    )
    keyboard = [
        [InlineKeyboardButton("✅ انتهيت من الرفع", callback_data=f"s2_done_{ep_id}")],
        [InlineKeyboardButton("🗑️ مسح ما رُفع", callback_data=f"s2_clear_{ep_id}")],
        [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
    ]
    try:
        await context.bot.send_message(
            chat_id=chat_id, text=text,
            reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.error(f"فشل عرض تقدّم Stage2: {e}")


# =================================================================
# 5. المرحلة 1
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
    session["awaiting_script_replace"] = False
    session["approved_script"] = None
    session["script_version"] = 0
    session["script_versions"] = []
    session["episode_status"] = "generating_script"
    session["script_input_buffer"] = []
    session["script_input_messages"] = 0
    session["script_input_files"] = 0

    _stage_set(session, 1, input_mode="automatic", status="running",
               source=None, artifact=None, manual_override=False, approved=False)

    if _is_stale(chat_id, session):
        return

    status_card = (
        f"<b>⚙️ [ 1/2 ] جاري تشغيل المرحلة الأولى للحلقة #{_esc(ep_id)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏳ توليد السكربت الإنجليزي + هندسة الجمل القصيرة + الخطة الإبداعية...\n"
        f"<i>لن تبدأ المرحلة الثانية حتى تعتمد السكريبت بنفسك.</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    try:
        status_msg = await query.edit_message_text(status_card, parse_mode=ParseMode.HTML)
    except Exception:
        try:
            status_msg = await query.message.reply_text(status_card, parse_mode=ParseMode.HTML)
        except Exception:
            return

    session["stage1_plan_message_id"] = status_msg.message_id

    try:
        stage1_res = await asyncio.to_thread(generate_stage1_script, episode)
        if _is_stale(chat_id, session):
            return

        s1_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"
        with open(s1_file, "w", encoding="utf-8") as f:
            json.dump(stage1_res, f, ensure_ascii=False, indent=2)

        sentences = _as_list(stage1_res.get("full_script_sentences"))
        session["stage1_result"] = stage1_res
        session["sentences"] = sentences
        session["episode_status"] = "awaiting_script_review"
        session["stage1_approved"] = False
        session["script_version"] = 1

        v_path = _save_script_version(ep_id, 1, sentences)
        session["script_versions"] = [{"version": 1, "sentences": list(sentences), "path": str(v_path)}]

        n = len(sentences)
        in_range = (SENTENCE_MIN <= n <= SENTENCE_MAX)

        # ✅ manual_override لا يُثبّت هنا — يبقى False حتى يضغط المستخدم الزر الصريح
        _stage_set(session, 1,
                   status="awaiting_approval",
                   source="ai",
                   artifact=str(v_path),
                   input_mode="automatic",
                   manual_override=False,
                   )

        summary_text = _build_stage1_summary(ep_id, episode, stage1_res)
        try:
            await status_msg.edit_text(summary_text, parse_mode=ParseMode.HTML)
        except Exception:
            pass

        if not in_range:
            # نتيجة النموذج خارج النطاق → اعرض خيارات يدوية
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"⚠️ <b>النموذج أنتج عددًا خارج النطاق الموصى به</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 <b>عدد الجمل:</b> <code>{n}</code>\n"
                    f"✅ <b>الموصى:</b> <code>{SENTENCE_MIN}</code>–<code>{SENTENCE_MAX}</code> جملة.\n"
                    f"<i>يمكنك اعتماد النتيجة يدويًا، أو تعديلها، أو إعادة التوليد.</i>"
                ),
                parse_mode=ParseMode.HTML,
            )

        await _send_full_script_for_review(
            context, chat_id, ep_id, episode, sentences,
            version=1, manual_override=(not in_range),
        )

    except asyncio.CancelledError:
        logger.info(f"🛑 تم إلغاء المرحلة 1 للحلقة {ep_id}")
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        logger.exception(f"خطأ في المرحلة الأولى للحلقة {ep_id}")
        _stage_mark_failed(session, 1)
        try:
            await status_msg.edit_text(
                f"❌ <b>خطأ أثناء توليد المرحلة الأولى:</b>\n"
                f"<code>{_esc(str(e))}</code>",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass

        # خيارات يدوية عند الفشل
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "🛠️ <b>ماذا تريد أن تفعل؟</b>\n"
                f"• <b>استخدام النتيجة يدويًا</b>: أرسل سكريبتك بنفسك.\n"
                f"• <b>إعادة التوليد</b>: محاولة أخرى تلقائيًا.\n"
                f"• <b>إلغاء</b>: العودة للقائمة."
            ),
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✍️ إدخال السكريبت يدويًا", callback_data=f"s1_man_{ep_id}")],
                [InlineKeyboardButton("🔄 إعادة التوليد", callback_data=f"regenerate_stage1_{ep_id}")],
                [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
            ]),
            parse_mode=ParseMode.HTML,
        )


# =================================================================
# 6. المرحلة 2 — تلقائي + بوابة اعتماد + يدوي
# =================================================================

async def run_stage2_auto(query, context):
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    ep_id = session.get("episode_id")

    if _is_stale(chat_id, session):
        return
    if not _stage_is_approved(session, 1) or not session.get("approved_script"):
        try:
            await query.edit_message_text(
                "⛔ <b>لا يمكن بدء المرحلة الثانية قبل اعتماد السكريبت.</b>",
                parse_mode=ParseMode.HTML,
            )
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

    if not isinstance(stage1_res, dict):
        try:
            await query.edit_message_text("❌ <b>تعذّر العثور على بيانات المرحلة الأولى.</b>", parse_mode=ParseMode.HTML)
        except Exception:
            pass
        return

    sentences = list(session.get("approved_script") or [])
    if not sentences:
        try:
            await query.edit_message_text("❌ <b>لا يوجد approved_script.</b>", parse_mode=ParseMode.HTML)
        except Exception:
            pass
        return

    _stage_mark_running(session, 2, mode="automatic")

    status_card = (
        f"<b>⚙️ المرحلة 2 — توليد أوامر الصور</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏳ جاري صياغة البرومبتات وتجزئتها (1:1)..."
    )
    try:
        status_msg = await query.edit_message_text(status_card, parse_mode=ParseMode.HTML)
    except Exception:
        try:
            status_msg = await query.message.reply_text(status_card, parse_mode=ParseMode.HTML)
        except Exception:
            return

    prompts_dir = OUTPUTS_DIR / f"episode_{ep_id}_prompts"
    # ✅ تنظيف الملفات القديمة قبل إعادة التوليد (تجنّب تراكم الأجزاء القديمة)
    if prompts_dir.exists():
        try:
            shutil.rmtree(prompts_dir, ignore_errors=True)
            logger.info(f"🧹 تم تنظيف مجلد الأوامر القديم: {prompts_dir.name}")
        except Exception as e:
            logger.warning(f"تعذّر حذف {prompts_dir}: {e}")
    prompts_dir.mkdir(parents=True, exist_ok=True)

    try:
        batches = await asyncio.to_thread(
            generate_stage2_prompts_batches, sentences, stage1_result=stage1_res,
        )
        if _is_stale(chat_id, session):
            return

        sent_files = []
        for idx, batch in enumerate(batches, start=1):
            if _is_stale(chat_id, session):
                return
            p_file = prompts_dir / f"prompts_part_{idx:02d}.txt"
            with open(p_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(batch))
            sent_files.append(p_file)
            with open(p_file, "rb") as fp:
                await context.bot.send_document(
                    chat_id=chat_id, document=fp,
                    caption=(
                        f"📦 <b>حزمة أوامر الصور: الجزء [{idx:02d}]</b>\n"
                        f"└ يحتوي على <b>{len(batch)}</b> برومبت."
                    ),
                    parse_mode=ParseMode.HTML,
                )

        _stage_set(
            session, 2,
            status="awaiting_approval",
            source="ai",
            artifact=str(prompts_dir),
            input_mode="automatic",
        )

        total_prompts = sum(len(b) for b in batches)
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"✅ <b>تم توليد أوامر الصور</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📦 <b>عدد الأجزاء:</b> <code>{len(batches)}</code>\n"
                f"🧩 <b>إجمالي البرومبتات:</b> <code>{total_prompts}</code>\n"
                f"🎯 <b>عدد الجمل:</b> <code>{len(sentences)}</code>\n\n"
                f"👇 اعتمد الحزمة للمتابعة إلى المرحلة الثالثة:"
            ),
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ اعتماد أوامر الصور", callback_data=f"s2_approve_{ep_id}")],
                [InlineKeyboardButton("✍️ استبدال بأوامر يدوية", callback_data=f"s2_man_{ep_id}")],
                [InlineKeyboardButton("🔄 إعادة التوليد", callback_data=f"s2_retry_{ep_id}")],
                [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
            ]),
            parse_mode=ParseMode.HTML,
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        logger.exception(f"خطأ في المرحلة الثانية للحلقة {ep_id}")
        _stage_mark_failed(session, 2, source="ai")

        # لا نحذف الملفات الجزئية
        partial_files = sorted(prompts_dir.glob("prompts_part_*.txt"))
        partial_note = ""
        if partial_files:
            partial_note = f"\n📁 <b>ملفات جزئية محفوظة:</b> <code>{len(partial_files)}</code>"

        try:
            await status_msg.edit_text(
                f"❌ <b>خطأ في المرحلة الثانية:</b>\n"
                f"<code>{_esc(str(e))}</code>{partial_note}",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass

        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "🛠️ <b>ماذا تريد أن تفعل؟</b>\n"
                "• <b>رفع أوامر يدوية</b>: أرسل ملفات .txt أو نصًا مباشرًا.\n"
                "• <b>إعادة المحاولة</b>.\n"
                "• <b>إلغاء</b>."
            ),
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✍️ رفع أوامر يدويًا", callback_data=f"s2_man_{ep_id}")],
                [InlineKeyboardButton("🔄 إعادة المحاولة", callback_data=f"s2_retry_{ep_id}")],
                [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
            ]),
            parse_mode=ParseMode.HTML,
        )


async def finalize_stage2_manual(query, session, context):
    """يُحوّل الـ buffer النصي إلى ملفات .txt في مجلد manual ثم يعرض شاشة الاعتماد."""
    ep_id = session.get("episode_id")
    buf = session.get("stage2_prompt_buffer") or []
    files = int(session.get("stage2_prompt_files", 0))

    if not buf and not files:
        try:
            await query.answer("⚠️ لم ترفع أي أوامر بعد.", show_alert=True)
        except Exception:
            pass
        return

    manual_dir = OUTPUTS_DIR / f"episode_{ep_id}_prompts_manual"
    # ✅ تنظيف المجلد القديم قبل الكتابة الجديدة
    if manual_dir.exists():
        try:
            shutil.rmtree(manual_dir, ignore_errors=True)
            logger.info(f"🧹 تم تنظيف مجلد الأوامر اليدوية القديم: {manual_dir.name}")
        except Exception as e:
            logger.warning(f"تعذّر حذف {manual_dir}: {e}")
    manual_dir.mkdir(parents=True, exist_ok=True)

    # كتابة النصوص المتراكمة إلى ملف واحد أو أكثر
    written = []
    for i, chunk in enumerate(buf, start=1):
        p = manual_dir / f"manual_prompts_{i:02d}.txt"
        p.write_text(chunk, encoding="utf-8")
        written.append(p)

    # ✅ حساب عدد الأوامر الفعلي لعرض تحذير إن كان أقل من عدد الجمل
    total_prompts = 0
    for p in written:
        try:
            content = p.read_text(encoding="utf-8")
            blocks = [b for b in re.split(r"\n\s*\n", content) if b.strip()]
            total_prompts += len(blocks)
        except Exception:
            pass

    total_sentences = len(session.get("sentences", []))
    warning_text = ""
    if total_prompts < total_sentences:
        warning_text = (
            f"\n⚠️ <b>تحذير:</b> عدد الأوامر (<code>{total_prompts}</code>) أقل من عدد الجمل "
            f"(<code>{total_sentences}</code>).\n"
            f"<i>قد يؤدي هذا إلى صور مفقودة أو ترتيب غير مكتمل.</i>\n"
        )

    _stage_set(
        session, 2,
        status="awaiting_approval",
        source="user_text" if not files else "user_file",
        artifact=str(manual_dir),
        input_mode="manual",
    )

    session["state"] = "IDLE"
    session["stage2_prompt_buffer"] = []
    session["stage2_prompt_files"] = 0

    await context.bot.send_message(
        chat_id=query.message.chat_id if hasattr(query, "message") else None,
        text=(
            f"✅ <b>تم حفظ {len(written)} ملف أوامر يدوية</b>\n"
            f"📁 <code>{manual_dir.name}</code>\n"
            f"🧩 <b>عدد الأوامر:</b> <code>{total_prompts}</code>\n"
            f"{warning_text}\n"
            f"👇 اعتمد الحزمة للمتابعة إلى المرحلة الثالثة:"
        ),
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("✅ اعتماد أوامر الصور", callback_data=f"s2_approve_{ep_id}")],
            [InlineKeyboardButton("✏️ إعادة الرفع", callback_data=f"s2_man_{ep_id}")],
            [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
        ]),
        parse_mode=ParseMode.HTML,
    )


async def approve_stage2(query, session, context):
    ep_id = session.get("episode_id")
    # اعتماد Stage 2 (بمثابة بوابة بين 2 و 3)
    _stage_mark_approved(session, 2, source=_stage_get(session, 2).get("source") or "ai",
                        artifact=_stage_get(session, 2).get("artifact"))

    # عرض خيار وضع المرحلة الثالثة
    text = (
        f"<b>🔊 المرحلة 3 — التعليق الصوتي</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"اختر طريقة الحصول على الصوت:\n\n"
        f"<b>🤖 تلقائي</b> — عبر Azure Text-to-Speech (مع اختيار الصوت).\n"
        f"<b>📤 يدوي</b> — أنت ترفع ملف صوت (.mp3/.wav/.m4a...).\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    try:
        await query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🤖 توليد الصوت تلقائيًا", callback_data=f"s3_auto_{ep_id}")],
                [InlineKeyboardButton("📤 رفع ملف صوت خارجي", callback_data=f"s3_man_{ep_id}")],
                [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
            ]),
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=text,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🤖 توليد الصوت تلقائيًا", callback_data=f"s3_auto_{ep_id}")],
                [InlineKeyboardButton("📤 رفع ملف صوت خارجي", callback_data=f"s3_man_{ep_id}")],
                [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
            ]),
            parse_mode=ParseMode.HTML,
        )


# =================================================================
# 7. المرحلة 3 — تلقائي + يدوي
# =================================================================

async def present_audio_engine_choice(bot_or_query, chat_id=None):
    text = (
        "🎙️ <b>المرحلة 3: اختيار الصوت التعبيري (Microsoft Azure)</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "يمكنك الاستماع لعينة فورية لأي صوت قبل الاختيار.\n"
        f"<i>الصوت الافتراضي: <code>{DEFAULT_VOICE}</code></i>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    descriptions = {
        "en-US-BrianMultilingualNeural": "الصوت المعتمد — سرد هادئ وإنساني ⭐",
        "en-US-GuyNeural": "تلوين انفعالي كامل 🌟",
        "en-US-DavisNeural": "سرد ناضج وهادئ 📖",
        "en-US-TonyNeural": "حماسي وواثق ⚡",
        "en-US-JasonNeural": "شبابي وسريع 🚀",
    }
    buttons = []
    for v in AZURE_MALE_VOICES:
        label = descriptions.get(v, v)
        buttons.append([
            InlineKeyboardButton(f"🗣️ {label}", callback_data=f"voice_{v}"),
            InlineKeyboardButton("🎧 استمع", callback_data=f"preview_voice_{v}"),
        ])
    reply_markup = InlineKeyboardMarkup(buttons)
    if chat_id:
        await bot_or_query.send_message(chat_id=chat_id, text=text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    else:
        await bot_or_query.edit_message_text(text=text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def run_stage3(query, context):
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    ep_id = session.get("episode_id")
    sentences = session.get("sentences", [])
    engine = "azure"
    voice = session.get("voice") or DEFAULT_VOICE

    if _is_stale(chat_id, session):
        return

    if not _stage_is_approved(session, 2):
        try:
            await query.edit_message_text(
                "⛔ <b>لا يمكن توليد الصوت قبل اعتماد أوامر الصور.</b>",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass
        return

    if session.get("episode_status") != "script_approved":
        try:
            await query.edit_message_text(
                "⛔ <b>لا يمكن توليد الصوت قبل اعتماد السكريبت.</b>",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass
        return

    if voice not in AZURE_MALE_VOICES:
        try:
            await query.edit_message_text(
                f"❌ <b>الصوت المختار غير مُعرَّف:</b> <code>{_esc(voice)}</code>",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass
        return

    _stage_mark_running(session, 3, mode="automatic")

    try:
        wait_msg = await query.edit_message_text(
            f"⏳ <b>جاري توليد ملف الصوت الموحد عبر {engine.upper()}...</b>\n"
            f"🗣️ الصوت: <code>{voice}</code>",
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        return

    try:
        episode_context = session.get("stage1_result")
        if not isinstance(episode_context, dict):
            s1_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"
            if not s1_file.exists():
                raise RuntimeError(f"لا توجد نتيجة مرحلة أولى للحلقة {ep_id}.")
            with open(s1_file, "r", encoding="utf-8") as f:
                episode_context = json.load(f)
            session["stage1_result"] = episode_context

        context_sentences = episode_context.get("full_script_sentences")
        if not isinstance(context_sentences, list) or not context_sentences:
            raise RuntimeError(f"result.stage1 لا يحتوي full_script_sentences.")
        if sentences != context_sentences:
            raise RuntimeError(f"جمل الجلسة لا تطابق full_script_sentences.")

        if _is_stale(chat_id, session):
            return

        audio_path = await asyncio.to_thread(
            generate_stage3_audio,
            episode_id=ep_id, sentences=sentences, engine=engine, voice=voice,
            output_dir=OUTPUTS_DIR, episode_context=episode_context,
        )
        if _is_stale(chat_id, session):
            return

        session["audio_path"] = audio_path
        _stage_mark_approved(session, 3, source="ai", artifact=str(audio_path))

        await wait_msg.edit_text(
            text=(
                f"<b>🎉 تم توليد التعليق الصوتي</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🎙️ <b>الصوت:</b> <code>{voice}</code> ({engine.upper()})\n"
                f"📁 <b>الملف:</b> <code>{Path(audio_path).name}</code>\n\n"
                f"👇 هل تريد المتابعة أو استبدال الصوت؟"
            ),
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ متابعة إلى الصور", callback_data=f"s3_continue_{ep_id}")],
                [InlineKeyboardButton("📤 استبدال الصوت بملف خارجي", callback_data=f"s3_replace_{ep_id}")],
                [InlineKeyboardButton("🔄 توليد من جديد", callback_data=f"s3_retry_{ep_id}")],
            ]),
            parse_mode=ParseMode.HTML,
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        logger.exception(f"خطأ في المرحلة الثالثة للحلقة {ep_id}")
        _stage_mark_failed(session, 3, source="ai")
        try:
            await wait_msg.edit_text(
                text=(
                    f"❌ <b>فشل توليد الصوت:</b>\n<code>{_esc(str(e))}</code>\n\n"
                    f"👇 يمكنك رفع ملف صوت خارجي أو إعادة المحاولة."
                ),
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📤 رفع ملف صوت خارجي", callback_data=f"s3_man_{ep_id}")],
                    [InlineKeyboardButton("🔄 إعادة المحاولة", callback_data=f"s3_retry_{ep_id}")],
                    [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
                ]),
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            pass


# =================================================================
# 8. استقبال الصور والملفات (multiplexed)
# =================================================================

async def handle_media_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    session = get_session(chat_id)

    if _is_stale(chat_id, session):
        return

    state = session.get("state")

    # ============ 1) .txt في وضع سكريبت Stage 1 ============
    if state in ("WAITING_SCRIPT_REPLACE", "WAITING_SCRIPT_EDIT"):
        if session.get("episode_status") == "script_approved":
            await update.message.reply_text(
                "⚠️ السكريبت معتمد بالفعل.",
                parse_mode=ParseMode.HTML,
            )
            return
        doc = update.message.document
        if doc and (
            (doc.mime_type and doc.mime_type.startswith("text/"))
            or (doc.file_name and doc.file_name.lower().endswith(".txt"))
        ):
            try:
                file_obj = await doc.get_file()
                content_bytes = await file_obj.download_as_bytearray()
                content = None
                for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1256", "latin-1"):
                    try:
                        content = bytes(content_bytes).decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if content is None:
                    content = bytes(content_bytes).decode("utf-8", errors="replace")
                buf = session.get("script_input_buffer") or []
                buf.append(content)
                session["script_input_buffer"] = buf
                session["script_input_files"] = int(session.get("script_input_files", 0)) + 1
                await update.message.reply_text(
                    f"📄 <b>تم استلام الملف:</b> <code>{_esc(doc.file_name)}</code>",
                    parse_mode=ParseMode.HTML,
                )
                await show_script_input_progress(chat_id, session, context)
            except Exception as e:
                logger.exception("فشل قراءة ملف السكريبت .txt")
                await update.message.reply_text(
                    f"❌ <b>فشل قراءة الملف:</b>\n<code>{_esc(str(e))}</code>",
                    parse_mode=ParseMode.HTML,
                )
            return
        else:
            await update.message.reply_text(
                "⚠️ في هذا الوضع: <b>نص عادي</b> أو <b>ملف .txt</b> فقط.",
                parse_mode=ParseMode.HTML,
            )
            return

    # ============ 2) .txt في وضع Stage 2 prompts ============
    if state == "WAITING_STAGE2_PROMPTS":
        doc = update.message.document
        if doc and (
            (doc.mime_type and doc.mime_type.startswith("text/"))
            or (doc.file_name and doc.file_name.lower().endswith(".txt"))
        ):
            try:
                file_obj = await doc.get_file()
                content_bytes = await file_obj.download_as_bytearray()
                content = None
                for enc in ("utf-8", "utf-8-sig", "utf-16", "cp1256", "latin-1"):
                    try:
                        content = bytes(content_bytes).decode(enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if content is None:
                    content = bytes(content_bytes).decode("utf-8", errors="replace")
                buf = session.get("stage2_prompt_buffer") or []
                buf.append(content)
                session["stage2_prompt_buffer"] = buf
                session["stage2_prompt_files"] = int(session.get("stage2_prompt_files", 0)) + 1
                await update.message.reply_text(
                    f"📄 <b>تم استلام الملف:</b> <code>{_esc(doc.file_name)}</code>",
                    parse_mode=ParseMode.HTML,
                )
                await _show_stage2_upload_progress(chat_id, session, context)
            except Exception as e:
                logger.exception("فشل قراءة ملف Stage2 .txt")
                await update.message.reply_text(
                    f"❌ <b>فشل قراءة الملف:</b>\n<code>{_esc(str(e))}</code>",
                    parse_mode=ParseMode.HTML,
                )
            return
        else:
            await update.message.reply_text(
                "⚠️ في هذا الوضع: <b>نص عادي</b> أو <b>ملف .txt</b> فقط.",
                parse_mode=ParseMode.HTML,
            )
            return

    # ============ 3) ملف صوتي في وضع Stage 3 manual ============
    if state == "WAITING_STAGE3_AUDIO":
        doc = update.message.document or (update.message.audio if update.message.audio else None)
        # audio يمكن أن يكون voice أو audio
        file_name = None
        file_obj = None
        mime = None

        if update.message.document:
            doc = update.message.document
            file_name = doc.file_name or "uploaded_audio"
            mime = doc.mime_type or ""
            file_obj = await doc.get_file()
        elif update.message.audio:
            aud = update.message.audio
            file_name = aud.file_name or f"audio_{aud.file_unique_id}.mp3"
            mime = aud.mime_type or "audio/mpeg"
            file_obj = await aud.get_file()
        elif update.message.voice:
            vo = update.message.voice
            file_name = f"voice_{vo.file_unique_id}.ogg"
            mime = vo.mime_type or "audio/ogg"
            file_obj = await vo.get_file()

        if not file_obj:
            await update.message.reply_text(
                "⚠️ الرجاء إرسال الملف كـ <b>Document</b> أو <b>Audio</b>.",
                parse_mode=ParseMode.HTML,
            )
            return

        ext = Path(file_name).suffix.lower() if file_name else ""
        if ext not in ALLOWED_AUDIO_EXTS:
            await update.message.reply_text(
                f"⚠️ صيغة غير مدعومة: <code>{_esc(ext or mime)}</code>\n"
                f"المدعوم: {', '.join(sorted(ALLOWED_AUDIO_EXTS))}",
                parse_mode=ParseMode.HTML,
            )
            return

        ep_id = session.get("episode_id")
        target = OUTPUTS_DIR / f"episode_{ep_id}_manual_audio{ext}"
        try:
            await file_obj.download_to_drive(target)
            if not target.exists() or target.stat().st_size == 0:
                raise RuntimeError("الملف المُحمَّل فارغ.")
            # ✅ التحقق الفعلي من صلاحية الصوت عبر ffprobe
            duration = await asyncio.to_thread(_probe_audio_file, target)
            logger.info(f"🎧 تم التحقق من الصوت اليدوي: {target.name} ({duration:.1f}s)")
        except Exception as e:
            logger.exception("فشل تحميل أو التحقق من الصوت اليدوي")
            try:
                if target.exists():
                    target.unlink()
            except Exception:
                pass
            await update.message.reply_text(
                f"❌ <b>الملف مرفوض:</b>\n<code>{_esc(str(e))}</code>\n\n"
                f"<i>أرسل ملفًا صوتيًا سليمًا آخر.</i>",
                parse_mode=ParseMode.HTML,
            )
            return

        session["audio_path"] = str(target)
        session["state"] = "IDLE"
        _stage_mark_approved(session, 3, source="uploaded_audio",
                             artifact=str(target))

        await update.message.reply_text(
            text=(
                f"✅ <b>تم استلام الصوت اليدوي.</b>\n"
                f"📁 <code>{target.name}</code>\n"
                f"⏱️ <b>المدة:</b> <code>{duration:.1f}s</code>\n"
                f"🎯 يمكنك المتابعة إلى رفع الصور."
            ),
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ متابعة إلى الصور", callback_data=f"s3_continue_{ep_id}")],
                [InlineKeyboardButton("📤 استبدال الصوت", callback_data=f"s3_replace_{ep_id}")],
            ]),
            parse_mode=ParseMode.HTML,
        )
        return

    # ============ 4) استقبال الصور ============
    if not session.get("episode_id"):
        return

    ep_id = session.get("episode_id", "201")
    total_expected = len(session.get("sentences", []))

    raw_dir = OUTPUTS_DIR / f"episode_{ep_id}_raw_images"
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_obj = None
    file_name = None
    if update.message.document and update.message.document.mime_type \
            and update.message.document.mime_type.startswith("image/"):
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
        session["image_review_confirmed_flag"] = False
        session["final_image_map"] = {}

        current = session["uploaded_count"]
        if current % 5 == 0 or (total_expected > 0 and current == total_expected):
            pct = int((current / total_expected * 100)) if total_expected > 0 else 0
            await update.message.reply_text(
                f"📥 <b>تم استلام:</b> <code>{current} / {total_expected}</code> ({pct}%)\n"
                f"👇 اختر طريقة الترتيب:",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🤖 ترتيب تلقائي حسب الاسم", callback_data=f"s4_auto_{ep_id}")],
                    [InlineKeyboardButton("✍️ تعيين وترتيب يدوي", callback_data=f"s4_man_{ep_id}")],
                ]),
                parse_mode=ParseMode.HTML,
            )


# =================================================================
# 9. التعيين اليدوي للصور (Manual assign)
# =================================================================

async def start_manual_assign_flow(query, context, session):
    chat_id = query.message.chat_id
    unindexed = session.get("manual_unindexed_files", []) or []
    missing_idx = session.get("manual_missing_indices", []) or []

    if not unindexed and not missing_idx:
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

    session["manual_queue"] = [p for p in unindexed if Path(p).resolve() not in already]
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


async def handle_manual_assign(query, context, slot):
    chat_id = query.message.chat_id
    session = get_session(chat_id)
    current = session.get("manual_current")
    if not current:
        return
    try:
        slot_int = int(slot)
    except (ValueError, TypeError):
        return
    session["manual_map"][slot_int] = current
    session["manual_current"] = None

    try:
        await query.answer(f"✅ تعيين: {slot_int}")
    except Exception:
        pass
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass

    logger.info(f"✅ تعيين يدوي: {Path(current).name} → slot {slot_int}")
    await show_next_manual_image(context, chat_id)


async def _build_map_from_ocr_and_manual(session) -> dict:
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
    review_map = {int(k): str(v) for k, v in (review_map or {}).items() if v}
    session["image_review_map"] = dict(review_map)
    session["image_review_original"] = dict(review_map)
    session["image_review_missing"] = [s for s in range(1, total_expected + 1) if s not in review_map]
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
        _stage_set(session, 4, input_mode="manual", status="succeeded", source="uploaded_images")
        text = (
            f"✅ <b>تم تجميع كل الصور ({found}/{total_expected})</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🧩 <b>تعيين يدوي:</b> <code>{manual_count}</code>\n"
            f"👇 يمكنك مراجعة الترتيب ثم اعتماده."
        )
        keyboard = [
            [InlineKeyboardButton("📋 مراجعة الصور واعتماد الترتيب", callback_data="review_mode_full")],
            [InlineKeyboardButton("📊 عرض الملخص النهائي مباشرة", callback_data="review_finish")],
            [InlineKeyboardButton("❌ إلغاء", callback_data="btn_back_main")],
        ]
    else:
        missing_txt = _format_missing_indices(missing) if missing else "—"
        text = (
            f"⚠️ <b>لسه فيه نواقص</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ <b>المعتمد:</b> <code>{found}</code> / <code>{total_expected}</code>\n"
            f"⚠️ <b>الناقص:</b> <code>{len(missing)}</code>\n"
            f"    └ {_esc(missing_txt)}\n"
            f"🔁 <b>التكرارات:</b> <code>{len(duplicates)}</code>\n\n"
            f"<i>ارفع الصور الناقصة، ثم اضغط «إعادة الترتيب».</i>"
        )
        keyboard = [
            [InlineKeyboardButton("🔄 إعادة الترتيب بعد رفع النواقص", callback_data=f"s4_auto_{session.get('episode_id')}")],
            [InlineKeyboardButton("❌ إلغاء", callback_data="btn_back_main")],
        ]

    await context.bot.send_message(
        chat_id=chat_id, text=text,
        reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML,
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
    remaining_slots = sorted([s for s in session.get("manual_missing", []) if int(s) not in assigned_keys])

    buttons = []
    row = []
    for s in remaining_slots:
        row.append(InlineKeyboardButton(f"#{s}", callback_data=f"manual_assign_{s}"))
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
                chat_id=chat_id, photo=img,
                caption=(
                    f"🖼️ <b>تعيين يدوي — متبقٍ {len(queue)}</b>\n"
                    f"الملف: <code>{_esc(Path(current_path).name)}</code>\n\n"
                    f"👇 اختر رقم الكادر:"
                ),
                reply_markup=InlineKeyboardMarkup(buttons),
                parse_mode=ParseMode.HTML,
            )
    except Exception as e:
        logger.error(f"فشل إرسال الصورة: {e}")
        await show_next_manual_image(context, chat_id)


# =================================================================
# 10. ترتيب الصور التلقائي
# =================================================================

async def run_image_verification(msg_obj, context, force_manual_after: bool = False):
    chat_id = msg_obj.chat_id
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    ep_id = session.get("episode_id", "201")
    sentences = session.get("sentences", [])
    total_expected = len(sentences)

    if total_expected == 0:
        await context.bot.send_message(chat_id=chat_id, text="⚠️ لا توجد جمل.", parse_mode=ParseMode.HTML)
        return

    raw_dir = OUTPUTS_DIR / f"episode_{ep_id}_raw_images"
    clean_dir = OUTPUTS_DIR / f"episode_{ep_id}_clean_frames"

    try:
        progress_msg = await context.bot.send_message(
            chat_id=chat_id,
            text="🔍 <b>جاري ترتيب الصور حسب اسم الملف...</b>",
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
            return

    except MissingAssetsError as m_err:
        if _is_stale(chat_id, session):
            return
        missing_preview = _format_missing_indices(m_err.missing_indices)
        unindexed_files = list(getattr(m_err, "unindexed_files", []) or [])
        missing_indices = [int(x) for x in (m_err.missing_indices or [])]

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
        _reset_image_review_state(session, ocr_map_int, total_expected)

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
        session["image_review_missing"] = [s for s in range(1, total_expected + 1) if s not in merged]
        session["image_review_duplicates"] = _compute_duplicates_from_map(merged)

        _stage_set(session, 4, status="awaiting_manual_input")

        keyboard = []
        if unindexed_files:
            keyboard.append([InlineKeyboardButton(
                f"🗑️ عرض الصور المتجاهلة ({len(unindexed_files)})",
                callback_data="btn_show_ignored",
            )])
        keyboard.append([InlineKeyboardButton(
            "✍️ الدخول إلى التعيين اليدوي",
            callback_data="btn_manual_assign",
        )])
        keyboard.append([InlineKeyboardButton(
            "🔄 إعادة الترتيب بعد رفع النواقص",
            callback_data=f"s4_auto_{ep_id}",
        )])
        keyboard.append([InlineKeyboardButton("❌ إلغاء", callback_data="btn_back_main")])

        await progress_msg.edit_text(
            f"⚠️ <b>ترتيب الصور اكتمل جزئيًا</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ <b>الموجودة:</b> <code>{m_err.found_count}</code> من <code>{m_err.total_expected}</code>\n"
            f"❌ <b>النواقص:</b> <code>{_esc(missing_preview)}</code>\n\n"
            f"📌 ارفع النواقص، أو ادخل للتعيين اليدوي.",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.HTML,
        )
        return

    except asyncio.CancelledError:
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        logger.exception(f"خطأ أثناء ترتيب الصور للحلقة {ep_id}")
        _stage_mark_failed(session, 4)
        await progress_msg.edit_text(
            f"❌ <b>خطأ أثناء معالجة الصور:</b>\n<code>{_esc(str(e))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # نجاح كامل
    review_map = {idx + 1: str(path) for idx, path in enumerate(frames)}
    for slot, path in (session.get("manual_map") or {}).items():
        try:
            sp = int(slot)
        except (ValueError, TypeError):
            continue
        if path and Path(path).exists():
            review_map[sp] = str(path)

    _reset_image_review_state(session, review_map, total_expected)
    session["ocr_indexed_map"] = {int(k): str(v) for k, v in review_map.items()}

    found = len(session["image_review_map"])
    missing_count = len(session["image_review_missing"])
    duplicates_count = len(session["image_review_duplicates"])

    _stage_set(session, 4,
               status="succeeded",
               source="uploaded_images",
               input_mode=_stage_get(session, 4).get("input_mode") or "automatic",
               artifact=str(clean_dir))

    if force_manual_after:
        # المستخدم اختار الوضع اليدوي → افتح التعيين مباشرة
        try:
            await progress_msg.edit_reply_markup(reply_markup=None)
        except Exception:
            pass
        session["manual_unindexed_files"] = []
        session["manual_missing_indices"] = []
        session["state"] = "IDLE"
        # ملاحظة: لأن الترتيب نجح، لا يوجد Unindexed.
        # سنعرض له شاشة المراجعة اليدوية (full) ليتمكن من تغيير الأرقام.
        await start_image_review(chat_id, context, mode="full")
        return

    keyboard = [
        [InlineKeyboardButton("✅ اعتماد الترتيب الحالي", callback_data="review_finish")],
        [InlineKeyboardButton("📋 مراجعة جميع الصور", callback_data="review_mode_full")],
        [InlineKeyboardButton("⚡ مراجعة الصور التي تحتاج تدقيقًا", callback_data="review_mode_quick")],
        [InlineKeyboardButton("🔄 إعادة الترتيب", callback_data=f"s4_auto_{ep_id}")],
    ]
    await progress_msg.edit_text(
        f"✅ <b>تم ترتيب كل الصور بنجاح</b>\n"
        f"📊 العدد: <code>{found}</code> من <code>{total_expected}</code>\n"
        f"⚠️ <b>المفقودة:</b> <code>{missing_count}</code>  |  🔁 <b>التكرارات:</b> <code>{duplicates_count}</code>\n\n"
        f"👇 اختر:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.HTML,
    )


# =================================================================
# 11. نظام مراجعة الصور
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
        await context.bot.send_message(chat_id=chat_id, text="⚠️ لا توجد صور لمراجعتها.", parse_mode=ParseMode.HTML)
        return

    if mode == "quick":
        items = _get_quick_review_items(session)
        if not items:
            await context.bot.send_message(
                chat_id=chat_id,
                text="✅ <b>لا توجد صور تحتاج تدقيقاً.</b>",
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
        status = "❌ مفقودة"
    elif is_excluded:
        status = "🚫 مستبعدة"
    elif is_duplicate:
        status = "⚠️ رقم مكرر"
    elif is_confirmed:
        status = "✅ مُعتمد"
    else:
        status = "⏳ بانتظار المراجعة"
    if was_edited:
        status += " ✏️"

    file_display = Path(path).name if path else "—"
    return (
        f"<b>🖼️ مراجعة الصور — {idx + 1} / {len(items)}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔢 <b>Slot:</b> <code>#{slot}</code>\n"
        f"📁 <b>الملف:</b> <code>{_esc(file_display)}</code>\n"
        f"📌 <b>الحالة:</b> {status}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def _build_review_keyboard(session, slot):
    items = session.get("image_review_items", [])
    idx = session.get("image_review_index", 0)
    total = len(items)
    review_map = session.get("image_review_map", {}) or {}
    path_exists = slot in review_map
    is_excluded = slot in (session.get("image_review_excluded") or [])

    row1 = []
    if path_exists and not is_excluded:
        row1.append(InlineKeyboardButton("✅ تأكيد", callback_data="review_confirm"))
    row1.append(InlineKeyboardButton("✏️ تغيير الرقم", callback_data="review_change_num"))
    row2 = [InlineKeyboardButton("❌ استبعاد" if not is_excluded else "♻️ استرجاع", callback_data="review_skip")]
    row3 = []
    if idx > 0:
        row3.append(InlineKeyboardButton("⬅️", callback_data="review_prev"))
    row3.append(InlineKeyboardButton(f"({idx + 1}/{total})", callback_data="review_noop"))
    if idx < total - 1:
        row3.append(InlineKeyboardButton("➡️", callback_data="review_next"))
    row4 = [InlineKeyboardButton("🏁 إنهاء المراجعة", callback_data="review_finish")]
    return InlineKeyboardMarkup([row1, row2, row3, row4])


async def show_review_image(chat_id, context):
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return
    items = session.get("image_review_items", []) or []
    idx = session.get("image_review_index", 0)
    if not items:
        await context.bot.send_message(chat_id=chat_id, text="ℹ️ لا توجد صور.", parse_mode=ParseMode.HTML)
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
                    chat_id=chat_id, photo=img, caption=caption,
                    reply_markup=keyboard, parse_mode=ParseMode.HTML,
                )
            session["image_review_message_id"] = sent.message_id
            return
        except Exception as e:
            logger.error(f"فشل إرسال صورة المراجعة: {e}")

    sent = await context.bot.send_message(chat_id=chat_id, text=caption, reply_markup=keyboard, parse_mode=ParseMode.HTML)
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
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"⚠️ لا توجد صورة مرتبطة بـ <code>#{slot}</code>.",
            parse_mode=ParseMode.HTML,
        )
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
    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            f"✏️ <b>تعديل رقم الصورة</b>\n"
            f"🔢 الحالي: <code>#{slot}</code>\n"
            f"📥 أرسل الرقم الجديد (1 إلى {total_expected})."
        ),
        parse_mode=ParseMode.HTML,
    )


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
        await update.message.reply_text("⚠️ أرسل رقمًا صحيحًا.", parse_mode=ParseMode.HTML)
        return
    total_expected = len(session.get("sentences", []))
    if new_slot < 1 or new_slot > total_expected:
        await update.message.reply_text(f"⚠️ الرقم خارج النطاق (1–{total_expected}).", parse_mode=ParseMode.HTML)
        return

    review_map = session.get("image_review_map", {}) or {}
    current_path = review_map.get(int(old_slot))
    if not current_path:
        await update.message.reply_text(f"⚠️ لا توجد صورة للرقم <code>#{old_slot}</code>.", parse_mode=ParseMode.HTML)
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
    session["image_review_missing"] = [s for s in range(1, total_expected + 1) if s not in review_map]
    session["state"] = "IMAGE_REVIEW"
    session["image_review_pending_slot"] = None

    await update.message.reply_text(
        f"✅ <b>تم:</b> <code>#{old_slot}</code> → <code>#{new_slot}</code>",
        parse_mode=ParseMode.HTML,
    )
    await show_review_image(chat_id, context)


# =================================================================
# 12. شاشة الملخص
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
        await context.bot.send_message(chat_id=chat_id, text="⚠️ لا توجد صور.", parse_mode=ParseMode.HTML)
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

    duplicates_txt = "—" if not duplicates else " | ".join("، ".join(f"#{s}" for s in grp) for grp in duplicates)
    excluded_txt = "—" if not excluded else ", ".join(f"#{s}" for s in excluded)
    edits_txt = "—" if not manual_edits else ", ".join(f"#{e.get('old')}→#{e.get('new')}" for e in manual_edits)
    missing_txt = "—" if not missing else _format_missing_indices(missing)

    fully_complete = (ordered_count == total_expected) and (not missing) and (not duplicates)
    header = "✅ <b>جاهز للاعتماد</b>" if fully_complete else "⚠️ <b>غير مكتمل — لن تبدأ الرندرة</b>"

    text = (
        f"<b>📊 ملخص مراجعة الصور</b>\n{header}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>المتوقع:</b> <code>{total_expected}</code>\n"
        f"✅ <b>المرتب:</b> <code>{ordered_count}</code>\n"
        f"⚠️ <b>المفقودة:</b> <code>{len(missing)}</code>\n"
        f"    └ {_esc(missing_txt)}\n"
        f"🔁 <b>التكرارات:</b> <code>{len(duplicates)}</code>\n"
        f"    └ {_esc(duplicates_txt)}\n"
        f"🚫 <b>المستبعدة:</b> <code>{len(excluded)}</code>\n"
        f"    └ {_esc(excluded_txt)}\n"
        f"✏️ <b>تعديلات:</b> <code>{len(manual_edits)}</code>\n"
        f"    └ {_esc(edits_txt)}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>⚠️ لن يبدأ المونتاج إلا بعد الاعتماد واكتمال كل الأرقام.</i>"
    )

    row1 = []
    if fully_complete:
        row1.append(InlineKeyboardButton(
            "✅ اعتماد الترتيب وبدء المونتاج",
            callback_data="review_summary_approve",
        ))
    else:
        row1.append(InlineKeyboardButton(
            "🔄 إعادة الترتيب",
            callback_data=f"s4_auto_{session.get('episode_id')}",
        ))
    row2 = [
        InlineKeyboardButton("🔄 العودة للمراجعة", callback_data="review_summary_edit"),
        InlineKeyboardButton("❌ إلغاء", callback_data="review_summary_cancel"),
    ]
    await context.bot.send_message(
        chat_id=chat_id, text=text,
        reply_markup=InlineKeyboardMarkup([row1, row2]), parse_mode=ParseMode.HTML,
    )


# =================================================================
# 13. اعتماد + الرندرة
# =================================================================

def _validate_final_map(final_map, total_expected):
    errors = []
    slots = sorted(final_map.keys())
    expected_set = set(range(1, total_expected + 1))
    actual_set = set(slots)
    if actual_set != expected_set:
        missing_slots = sorted(expected_set - actual_set)
        extra_slots = sorted(actual_set - expected_set)
        if missing_slots:
            errors.append(f"أرقام مفقودة: {_format_missing_indices(missing_slots)}")
        if extra_slots:
            errors.append(f"أرقام خارج النطاق: {_format_missing_indices(extra_slots)}")

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
            errors.append(f"تعذّر قراءة الملف #{slot}")
            continue
        try:
            key = str(pp.resolve())
        except Exception:
            key = str(pp)
        if key in resolved_seen:
            errors.append(f"مسار مكرر بين #{resolved_seen[key]} و #{slot}")
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
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"❌ <b>لا يمكن بدء المونتاج — الخريطة غير مكتملة.</b>\n"
                f"{errors_txt}"
            ),
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 إعادة الترتيب", callback_data=f"s4_auto_{session.get('episode_id')}")],
                [InlineKeyboardButton("📋 العودة للمراجعة", callback_data="review_mode_full")],
                [InlineKeyboardButton("❌ إلغاء", callback_data="review_summary_cancel")],
            ]),
            parse_mode=ParseMode.HTML,
        )
        return

    session["final_image_map"] = final_map
    session["image_review_confirmed_flag"] = True

    _stage_mark_approved(
        session, 4,
        source=_stage_get(session, 4).get("source") or "uploaded_images",
        artifact=_stage_get(session, 4).get("artifact"),
    )

    await run_final_render(msg_obj, context)


async def run_final_render(msg_obj, context):
    chat_id = msg_obj.chat_id
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    ep_id = session.get("episode_id", "201")
    sentences = session.get("sentences", [])
    total_expected = len(sentences)
    final_map = session.get("final_image_map") or {}

    if not final_map:
        await context.bot.send_message(chat_id=chat_id, text="❌ لا توجد صور معتمدة.", parse_mode=ParseMode.HTML)
        return

    ok, errors = _validate_final_map(final_map, total_expected)
    if not ok:
        errors_txt = "\n".join(f"• {_esc(e)}" for e in errors[:8])
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"❌ <b>رفض المونتاج:</b>\n{errors_txt}",
            parse_mode=ParseMode.HTML,
        )
        return

    ordered_slots = sorted(final_map.keys())
    frames = [str(final_map[s]) for s in ordered_slots]

    try:
        progress_msg = await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"<b>🎬 رندرة كاملة — FFmpeg 1080p 60fps</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ {len(frames)} صورة معتمدة.\n"
                f"⏳ المزامنة + الترجمة + Ken Burns...\n"
                f"<i>قد تستغرق 2–4 دقائق...</i>"
            ),
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        return

    try:
        audio_file = session.get("audio_path")
        if not audio_file:
            raise RuntimeError("ملف الصوت غير موجود.")
        ap = Path(audio_file)
        if not ap.exists() or not ap.is_file():
            raise RuntimeError("ملف الصوت غير موجود على القرص.")

        subtitles_ass = OUTPUTS_DIR / f"episode_{ep_id}_subtitles.ass"
        final_video = OUTPUTS_DIR / f"episode_{ep_id}_final_1080p.mp4"

        timeline = await asyncio.to_thread(
            align_audio_and_generate_ass, audio_file, sentences, subtitles_ass
        )
        if _is_stale(chat_id, session):
            return

        await asyncio.to_thread(
            render_final_video, frames, timeline, audio_file, subtitles_ass, final_video
        )
        if _is_stale(chat_id, session):
            return

        file_size_mb = final_video.stat().st_size / (1024 * 1024)
        if file_size_mb < 49:
            with open(final_video, "rb") as fv:
                await context.bot.send_video(
                    chat_id=chat_id, video=fv,
                    caption=(
                        f"🏆 <b>فيديو الحلقة #{ep_id} جاهز!</b>\n"
                        f"1080p 60fps | {file_size_mb:.1f} MB"
                    ),
                    parse_mode=ParseMode.HTML,
                )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🏆 <b>تم تصدير الفيديو على السيرفر!</b>\n"
                    f"📊 {file_size_mb:.1f} MB (أكبر من حد تيليجرام)\n"
                    f"📁 <code>{final_video}</code>"
                ),
                parse_mode=ParseMode.HTML,
            )

        # ✅ الحفاظ على artifact المرحلة الرابعة (مجلد الفريمات)
        # لا نستبدله بمسار الفيديو النهائي — الفيديو ناتج الرندرة وليس صور المرحلة 4
        _stage_mark_approved(session, 4,
                             source=_stage_get(session, 4).get("source") or "uploaded_images",
                             artifact=_stage_get(session, 4).get("artifact"))

    except asyncio.CancelledError:
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        await progress_msg.edit_text(
            f"❌ <b>خطأ أثناء الرندرة:</b>\n<code>{_esc(str(e))}</code>\n\n"
            f"👇 خيارات:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 إعادة الرندرة", callback_data="render_retry")],
                [InlineKeyboardButton("📤 استبدال الصوت", callback_data="render_replace_audio")],
                [InlineKeyboardButton("✍️ تعديل ترتيب الصور", callback_data="render_edit_images")],
                [InlineKeyboardButton("❌ إلغاء", callback_data="soft_cancel")],
            ]),
            parse_mode=ParseMode.HTML,
        )
        return

    # ---------------- Stage 5 ----------------
    await run_stage5(progress_msg, context)


# =================================================================
# 14. المرحلة 5 — بيانات النشر (تلقائي / يدوي / تخطي)
# =================================================================

async def run_stage5(msg_obj, context):
    chat_id = msg_obj.chat_id
    session = get_session(chat_id)
    if _is_stale(chat_id, session):
        return

    ep_id = session.get("episode_id")
    sentences = session.get("sentences", [])

    _stage_mark_running(session, 5, mode="automatic")

    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text="📦 <b>جاري توليد بيانات النشر (عناوين، غلاف، وصف، تاجز)...</b>",
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        pass

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

        _stage_set(session, 5,
                   status="awaiting_approval",
                   source="ai",
                   input_mode="automatic",
                   artifact=None)

        # ✅ تخزين meta التلقائية في الـ buffer (للوصول إليها عند الاعتماد)
        session["stage5_metadata_buffer"] = meta

        # ✅ الزر يستخدم callback منفصل للاعتماد التلقائي
        await context.bot.send_message(
            chat_id=chat_id,
            text="👇 اختر:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✅ اعتماد بيانات النشر",
                                      callback_data=f"s5_approve_auto_{ep_id}")],
                [InlineKeyboardButton("✍️ إدخال يدوي / تعديل",
                                      callback_data=f"s5_man_{ep_id}")],
                [InlineKeyboardButton("🔄 إعادة التوليد",
                                      callback_data=f"s5_retry_{ep_id}")],
                [InlineKeyboardButton("⏭️ تخطي بيانات النشر",
                                      callback_data=f"s5_skip_{ep_id}")],
            ]),
            parse_mode=ParseMode.HTML,
        )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        if _is_stale(chat_id, session):
            return
        logger.exception(f"فشل بيانات النشر للحلقة {ep_id}")
        _stage_mark_failed(session, 5, source="ai")
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"❌ <b>فشل توليد بيانات النشر</b>\n"
                f"<code>{_esc(str(e))}</code>\n\n"
                f"<i>الفيديو النهائي محفوظ على السيرفر ولم يُمس.</i>"
            ),
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("✍️ إدخال البيانات يدويًا", callback_data=f"s5_man_{ep_id}")],
                [InlineKeyboardButton("🔄 إعادة المحاولة", callback_data=f"s5_retry_{ep_id}")],
                [InlineKeyboardButton("⏭️ تخطي بيانات النشر", callback_data=f"s5_skip_{ep_id}")],
            ]),
            parse_mode=ParseMode.HTML,
        )
        return

    # ❌ لا نستدعي _finalize_episode هنا — القرار للمستخدم عبر أزرار الاعتماد/التخطي


async def approve_stage5_auto(query, session, context):
    """اعتماد بيانات النشر التلقائية ثم إغلاق دورة الإنتاج."""
    chat_id = query.message.chat_id
    ep_id = session.get("episode_id")

    meta = session.get("stage5_metadata_buffer") or {}
    # الميتاداتا التلقائية تحتوي على مفاتيح *_message (وليست title اليدوية)
    if not meta.get("titles_message"):
        try:
            await query.answer("⚠️ لا توجد بيانات محفوظة للاعتماد.", show_alert=True)
        except Exception:
            pass
        return

    _stage_mark_approved(session, 5, source="ai", artifact=None)

    try:
        await query.edit_message_text(
            "✅ <b>تم اعتماد بيانات النشر التلقائية.</b>\n"
            "<i>جارٍ إغلاق دورة الإنتاج...</i>",
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        pass

    await _finalize_episode(context, chat_id, session, ep_id)


async def finalize_stage5_manual(query, session, context):
    chat_id = query.message.chat_id
    ep_id = session.get("episode_id")
    meta = session.get("stage5_metadata_buffer") or {}
    if not meta.get("title"):
        try:
            await query.edit_message_text("⚠️ لا توجد بيانات محفوظة.", parse_mode=ParseMode.HTML)
        except Exception:
            pass
        return

    session["state"] = "IDLE"

    titles_lines = [f"• {meta.get('title')}"]
    for alt in meta.get("alt_titles", []) or []:
        titles_lines.append(f"• {alt}")
    titles_msg = "🎯 <b>العناوين (يدوي):</b>\n" + "\n".join(titles_lines)

    thumb_msg = f"🖼️ <b>نص الصورة المصغرة:</b>\n{_esc(meta.get('thumbnail') or '—')}"
    desc_msg = f"📝 <b>الوصف:</b>\n{_esc(meta.get('description') or '—')}"
    tags_list = meta.get("tags") or []
    tags_msg = "🏷️ <b>التاجز:</b>\n" + (", ".join(_esc(t) for t in tags_list) if tags_list else "—")

    try:
        await context.bot.send_message(chat_id=chat_id, text=titles_msg, parse_mode=ParseMode.HTML)
        await context.bot.send_message(chat_id=chat_id, text=thumb_msg, parse_mode=ParseMode.HTML)
        await context.bot.send_message(chat_id=chat_id, text=desc_msg, parse_mode=ParseMode.HTML)
        await context.bot.send_message(chat_id=chat_id, text=tags_msg, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"فشل إرسال بيانات النشر اليدوية: {e}")

    _stage_mark_approved(session, 5, source="user_text", artifact=None)

    await _finalize_episode(context, chat_id, session, ep_id)


async def _finalize_episode(context, chat_id, session, ep_id):
    """تُعلّم الحلقة كـ completed وتصفّر الجلسة."""
    try:
        updated = await mark_episode_completed(ep_id)
        if not updated:
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"⚠️ <b>تمت الحلقة</b> لكن فشل تحديث <code>episodes.json</code> "
                    f"للحلقة <code>#{_esc(ep_id)}</code>. حدّثها يدويًا."
                ),
                parse_mode=ParseMode.HTML,
            )
    except Exception as e:
        logger.exception(f"خطأ في mark_episode_completed: {e}")

    await context.bot.send_message(
        chat_id=chat_id,
        text="🎉 <b>ألف مبروك! اكتملت دورة الإنتاج بنسبة 100%.</b>",
        parse_mode=ParseMode.HTML,
    )

    try:
        if not _is_stale(chat_id, session):
            session["episode_status"] = "completed"
            session["cancelled"] = True
            user_sessions[chat_id] = _fresh_session()
            logger.info(f"♻️ إعادة ضبط الجلسة {chat_id} بعد الحلقة {ep_id}")
    except Exception as e:
        logger.warning(f"تعذّر إعادة ضبط الجلسة: {e}")


# =================================================================
# 15. نقطة التشغيل
# =================================================================

def main():
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN غير موجود!")

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CallbackQueryHandler(handle_callback_query))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    app.add_handler(MessageHandler(
        filters.PHOTO | filters.Document.ALL | filters.AUDIO | filters.VOICE,
        handle_media_upload,
    ))

    print("=" * 60)
    print("🚀 Vot Studio Pro — Stage Manager (auto/manual/override) ready")
    print("=" * 60)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
