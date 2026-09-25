import os
import wave
import struct
import math
import shutil
import subprocess
import array
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("Stage4Composer")


def create_synthetic_whoosh(output_path: Path):
    """
    إنشاء ملف مؤثر هوائي ناعم (Soft Air Whoosh) في الخلفية
    بصوت طبيعي وهادئ (-18dB) في حال لم يكن الملف متوفراً في السيرفر.
    """
    if output_path.exists():
        return

    sample_rate = 44100
    duration = 0.4  # ثانية
    n_samples = int(sample_rate * duration)

    with wave.open(str(output_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for i in range(n_samples):
            t = i / sample_rate
            envelope = math.sin(math.pi * t / duration) ** 2
            freq = 150 + 150 * (t / duration)
            sample = envelope * math.sin(2 * math.pi * freq * t)
            val = int(sample * 32767 * 0.12)
            wav_file.writeframes(struct.pack("<h", val))


def build_whoosh_timeline(
    base_whoosh: Path,
    transition_times: List[float],
    total_duration: float,
    output_path: Path,
    whoosh_volume: float = 0.4,
):
    """
    توليد مسار صوتي موحد (Unified SFX Track) في الذاكرة عبر بايثون الخالص:
    - يقرأ ملف whoosh الأساسي مرة واحدة.
    - ينشئ بُفر صوتي بطول الفيديو الكامل.
    - يزرع نسخة من الـ whoosh عند كل نقطة انتقال بشدة 0.4.
    هذا يُغني عن تمرير عشرات المدخلات لـ FFmpeg ويختصر أمر التنفيذ النهائي.
    """
    if not base_whoosh.exists() or not base_whoosh.is_file():
        raise FileNotFoundError(f"ملف الـ whoosh الأساسي غير موجود: {base_whoosh}")

    if not isinstance(total_duration, (int, float)) or \
            not math.isfinite(total_duration) or total_duration <= 0:
        raise ValueError(f"مدة إجمالية غير صالحة لمسار الـ whoosh: {total_duration!r}")

    with wave.open(str(base_whoosh), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    base_samples = array.array("h")
    base_samples.frombytes(raw)
    base_len = len(base_samples)

    total_samples = int(total_duration * sr)
    if total_samples <= 0:
        raise ValueError(
            f"عدد العينات الكلي غير صالح لمسار الـ whoosh: {total_samples}"
        )

    out = array.array("h", bytes(2 * total_samples))

    for t in transition_times:
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            continue
        start = int(t * sr)
        if start < 0 or start >= total_samples:
            continue
        for i in range(base_len):
            idx = start + i
            if idx >= total_samples:
                break
            v = out[idx] + int(base_samples[i] * whoosh_volume)
            if v > 32767:
                v = 32767
            elif v < -32768:
                v = -32768
            out[idx] = v

    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(out.tobytes())


def get_ken_burns_filter(pattern_index: int, duration: float, fps: int = 60) -> str:
    """
    تطبيق النمط الدوري الذكي لحركة كين بيرنز (6 حركات):
    1. Slow Zoom In
    2. Slow Pan Right
    3. Slow Zoom Out
    4. Slow Pan Up
    5. Slow Zoom In
    6. Slow Pan Left
    """
    if not isinstance(duration, (int, float)) or \
            not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"مدة غير صالحة لفلتر Ken Burns: {duration!r}")

    if not isinstance(fps, int) or fps <= 0:
        raise ValueError(f"معدل إطارات غير صالح لفلتر Ken Burns: {fps!r}")

    frames = int(duration * fps)
    if frames < 1:
        raise ValueError(
            f"المدة ({duration:.6f}s) قصيرة جدًا لإنتاج إطار واحد عند {fps}fps. "
            f"الحد الأدنى المطلوب تقريبًا: {1.0 / fps:.6f}s"
        )

    pattern = pattern_index % 6

    if pattern == 0:
        return f"zoompan=z='min(zoom+0.0008,1.15)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 1:
        return f"zoompan=z='1.1':x='if(lte(on,1),0,x+1.2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 2:
        return f"zoompan=z='if(lte(on,1),1.15,max(1.0,zoom-0.0008))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 3:
        return f"zoompan=z='1.1':x='iw/2-(iw/zoom/2)':y='if(lte(on,1),ih*0.1,max(0,y-0.8))':d={frames}:s=1920x1080:fps={fps}"
    elif pattern == 4:
        return f"zoompan=z='min(zoom+0.0007,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"
    else:
        return f"zoompan=z='1.1':x='if(lte(on,1),iw*0.1,max(0,x-1.2))':y='ih/2-(ih/zoom/2)':d={frames}:s=1920x1080:fps={fps}"


def _probe_media_duration(path: Path) -> float:
    """
    قراءة المدة الفعلية لملف وسائط (صوت أو فيديو) بالثواني عبر ffprobe.
    تُستخدم لجعل الصوت هو المرجع الزمني النهائي للمونتاج.

    ترفض: الناتج الفارغ، NaN، Infinity، الصفر، القيم السالبة.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"فشل ffprobe في قراءة المدة للملف: {path} "
            f"(رمز الخروج: {e.returncode})"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffprobe غير متوفر في بيئة التشغيل. لا يمكن قياس مدة الوسائط."
        ) from e

    raw = (result.stdout or "").strip()
    if not raw:
        raise RuntimeError(f"ffprobe أعاد ناتجًا فارغًا للملف: {path}")

    try:
        duration = float(raw)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"قيمة مدة غير قابلة للتحويل من ffprobe للملف: {path} (الناتج: {raw!r})"
        ) from e

    if not math.isfinite(duration):
        raise RuntimeError(
            f"قيمة مدة غير منتهية أو NaN من ffprobe للملف: {path} ({duration!r})"
        )
    if duration <= 0:
        raise RuntimeError(
            f"قيمة مدة غير موجبة من ffprobe للملف: {path} ({duration})"
        )

    return duration


def _safe_concat_escape(path: Path) -> str:
    """
    تهيئة مسار آمن لصيغة FFmpeg concat demuxer:
    - المسار يجب أن يكون مطلقًا ومحلولًا.
    - نغلّفه بعلامات اقتباس مفردة، ونهرّب أي اقتباس مفرد داخلي.
    """
    resolved = path.resolve().as_posix()
    # Escape single quotes for the ffmpeg concat demuxer syntax.
    escaped = resolved.replace("'", "'\\''")
    return f"file '{escaped}'\n"


def _validate_regular_file(path: Path, label: str, index: int = -1) -> None:
    """التحقق من وجود ملف عادي بأخطاء واضحة."""
    idx_part = f"[{index}] " if index >= 0 else ""
    if not path.exists():
        raise FileNotFoundError(f"{idx_part}{label} غير موجود: {path}")
    if not path.is_file():
        raise ValueError(f"{idx_part}{label} ليس ملفًا عاديًا: {path}")


def _validate_non_empty_file(path: Path, label: str) -> None:
    """التحقق من أن الملف موجود وليس فارغًا."""
    if not path.exists():
        raise RuntimeError(f"{label} غير موجود بعد التنفيذ: {path}")
    if not path.is_file():
        raise RuntimeError(f"{label} ليس ملفًا عاديًا: {path}")
    try:
        if path.stat().st_size <= 0:
            raise RuntimeError(f"{label} فارغ (0 بايت): {path}")
    except OSError as e:
        raise RuntimeError(f"تعذّر فحص حجم {label}: {path}") from e


def render_final_video(
    frames: List[Path],
    timeline: List[Dict[str, Any]],
    audio_file: Path,
    subtitles_ass: Path,
    output_video_path: Path
):
    """
    بناء خط المونتاج الآلي وتصدير الفيديو النهائي عبر FFmpeg:
    - 1080p Full HD / 60fps
    - مؤثرات Ken Burns فقط (بدون Vignette)
    - انتقالات صوتية Soft Whoosh عند كل قطع
    - خلو تام بنسبة 100% من الموسيقى الخلفية

    [سياسة المدة الجديدة - النسخة المُحصَّنة]
    - الصوت (التعليق الصوتي) هو المرجع الزمني الوحيد.
    - نقيس المدة الحقيقية للفيديو بعد الدمج الفعلي (ffprobe على الملف المُجمَّع)
      وليس عبر مجموع النظري للمقاطع، لتجنّب فروق الإطارات الناتجة عن تقريب
      الـ 60fps أثناء الترميز.
    - إذا كان الفيديو أقصر من الصوت: نُمدّد آخر إطار (tpad clone) بالفرق الحقيقي.
    - إذا كان الفيديو أطول: pad_needed=0 ويقتطع -t الزائد من الفيديو فقط.
    - لا نمسّ مسار التعليق الصوتي إطلاقًا.
    - حذف -shortest نهائيًا من خطوة الدمج.
    - amix duration=first آمن لأن المدخل الأول = التعليق بطول audio_duration.

    [البنية الجديدة]
    - المرحلة A: دمج المقاطع فيديو-فقط (بدون صوت) عبر -c:v copy.
    - المرحلة B: قياس المدة الحقيقية + tpad + mux مع التعليق الصوتي و-t audio_duration.

    [العزل]
    - كل استدعاء يحصل على مجلد مؤقت فريد (uuid) داخل مجلد الإخراج.
    - يُنظَّف المجلد المؤقت بعد النجاح أو الفشل، دون المساس بمخرجات سابقة.
    - المخرج النهائي يُكتب أولًا في ملف مرحلي داخل المجلد المؤقت، ثم يُنقل
      ذريًّا إلى المسار النهائي بعد نجاح كل التحققات. هذا يضمن عدم إتلاف أي
      فيديو نهائي سابق إذا فشلت الرندرة.
    """

    # ============================================================
    # [حماية 1] استبعاد العناصر الفارغة + التحقق من تناسق الأعداد
    # ============================================================
    if not isinstance(frames, list) or not isinstance(timeline, list):
        raise TypeError("frames و timeline يجب أن تكونا قائمتين.")

    renderable_timeline = [it for it in timeline if not it.get("empty")]

    if not renderable_timeline:
        raise ValueError(
            "قائمة التوقيت لا تحتوي على أي عنصر قابل للرندرة "
            "(كل العناصر فارغة أو القائمة فارغة أصلًا)."
        )

    if len(frames) == len(renderable_timeline):
        raw_pairs = list(zip(frames, renderable_timeline))
    elif len(frames) == len(timeline):
        raw_pairs = [
            (f, it) for f, it in zip(frames, timeline)
            if not it.get("empty")
        ]
    else:
        raise ValueError(
            f"عدم تناسق: عدد الصور ({len(frames)}) لا يطابق عدد العناصر القابلة "
            f"للرندرة ({len(renderable_timeline)}) ولا العدد الكامل للـ timeline "
            f"({len(timeline)})."
        )

    # ============================================================
    # [حماية 2 + 3] التحقق المبكر من المدة ومن سلامة ملفات الصور
    # ============================================================
    validated_pairs: List[Tuple[Path, Dict[str, Any], float]] = []
    for idx, (frame_path, item) in enumerate(raw_pairs):
        raw_duration = item.get("duration", 0.0)

        try:
            duration = float(raw_duration)
        except (TypeError, ValueError) as e:
            preview = str(item.get("text", ""))[:40]
            raise ValueError(
                f"مدة غير قابلة للتحويل ({raw_duration!r}) للعنصر رقم {idx} "
                f"(النص: {preview!r})."
            ) from e

        if not math.isfinite(duration):
            preview = str(item.get("text", ""))[:40]
            raise ValueError(
                f"مدة غير منتهية/NaN ({duration}) للعنصر رقم {idx} "
                f"(النص: {preview!r})."
            )

        if duration <= 0:
            preview = str(item.get("text", ""))[:40]
            raise ValueError(
                f"مدة غير صالحة ({duration}) للعنصر رقم {idx} "
                f"(النص: {preview!r}). كل عنصر قابل للرندرة يجب أن تكون مدته > 0."
            )

        fp = Path(frame_path)
        _validate_regular_file(fp, "ملف الصورة", index=idx)

        validated_pairs.append((fp, item, duration))

    # ============================================================
    # [حماية 4] التحقق المبكر من مدخلات الصوت والترجمة ومجلد الإخراج
    # ============================================================
    audio_file = Path(audio_file)
    subtitles_ass = Path(subtitles_ass)
    output_video_path = Path(output_video_path)

    _validate_regular_file(audio_file, "ملف التعليق الصوتي")
    _validate_regular_file(subtitles_ass, "ملف الترجمة ASS")

    output_dir = output_video_path.parent
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"تعذّر إنشاء مجلد الإخراج: {output_dir}"
        ) from e

    if not output_dir.is_dir():
        raise RuntimeError(f"مسار الإخراج الأب ليس مجلدًا: {output_dir}")

    # ============================================================
    # [FIX 1] مساحة عمل مؤقتة فريدة لكل استدعاء (uuid)
    # ============================================================
    run_id = uuid.uuid4().hex
    temp_dir = output_dir / f"temp_segments_{run_id}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        raise RuntimeError(
            f"تعذّر إنشاء مجلد العمل المؤقت الفريد: {temp_dir}"
        ) from e

    # الـ whoosh الأساسي يُخزَّن داخل مساحة العمل المؤقتة لتفادي التسابق بين العمليات.
    sfx_whoosh = temp_dir / "whoosh_soft.wav"
    whoosh_timeline = temp_dir / "whoosh_timeline.wav"

    # المخرج النهائي المرحلي: لا نكتب أبدًا مباشرة إلى output_video_path قبل نجاح كل شيء.
    staged_output_video = temp_dir / "final_output_staged.mp4"

    success = False
    try:
        create_synthetic_whoosh(sfx_whoosh)
        if not sfx_whoosh.exists() or sfx_whoosh.stat().st_size <= 0:
            raise RuntimeError(
                f"فشل إنشاء ملف الـ whoosh الأساسي: {sfx_whoosh}"
            )

        segment_files: List[Path] = []
        fps = 60

        logger.info(
            f"🎬 بدء جلسة الرندرة (run_id={run_id}) لعدد "
            f"{len(validated_pairs)} مقطع Ken Burns..."
        )

        # ============================================================
        # 1. رندرة مقطع مستقل لكل لقطة
        # ============================================================
        for idx, (frame_path, _item, duration) in enumerate(validated_pairs):
            kb_filter = get_ken_burns_filter(idx, duration, fps=fps)

            seg_output = temp_dir / f"seg_{idx:03d}.mp4"
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(frame_path),
                "-vf", kb_filter,
                "-t", str(duration),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
                "-an",
                str(seg_output),
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"فشل إنشاء المقطع رقم {idx} (stage=segment_render): "
                    f"exit={e.returncode}"
                ) from e

            _validate_non_empty_file(seg_output, f"المقطع رقم {idx}")
            segment_files.append(seg_output)

        # ============================================================
        # 2. إنشاء قائمة تجميع المقاطع (بمسارات آمنة ومهرَّبة)
        # ============================================================
        concat_list_file = temp_dir / "concat_list.txt"
        try:
            with open(concat_list_file, "w", encoding="utf-8") as f:
                for seg in segment_files:
                    f.write(_safe_concat_escape(seg))
        except OSError as e:
            raise RuntimeError(
                f"تعذّر كتابة ملف قائمة الدمج: {concat_list_file}"
            ) from e

        if not concat_list_file.exists() or concat_list_file.stat().st_size <= 0:
            raise RuntimeError(
                f"ملف قائمة الدمج فارغ أو غير موجود: {concat_list_file}"
            )

        # ============================================================
        # [المرحلة A] دمج الفيديو فقط (بدون صوت)
        # ============================================================
        logger.info("🧩 [A] دمج مقاطع الفيديو (فيديو-فقط) باستخدام stream copy...")

        concat_video_only = temp_dir / "concat_video_only.mp4"
        cmd_concat_only = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list_file),
            "-c:v", "copy",
            "-an",
            str(concat_video_only),
        ]
        try:
            subprocess.run(cmd_concat_only, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"فشل دمج المقاطع (stage=concat_video_only): exit={e.returncode}"
            ) from e

        _validate_non_empty_file(concat_video_only, "الفيديو المُجمَّع (فيديو-فقط)")

        # ============================================================
        # [المرحلة B] قياس المدة الحقيقية + tpad + mux مع التعليق
        # ============================================================
        audio_duration = _probe_media_duration(audio_file)
        real_video_duration = _probe_media_duration(concat_video_only)

        pad_needed = max(0.0, audio_duration - real_video_duration)

        logger.info(
            f"🎧 مدة التعليق الصوتي المرجعية: {audio_duration:.3f}s | "
            f"🎞️ المدة الحقيقية للفيديو المُجمَّع: {real_video_duration:.3f}s | "
            f"🩹 الحشو المطلوب (tpad clone): {pad_needed:.3f}s"
        )

        unsubbed_video = temp_dir / "unsubbed_assembled.mp4"
        logger.info(
            "🎞️ [B] تطبيق tpad clone ودمج التعليق الصوتي (الصوت هو مرجع المدة)..."
        )

        tpad_filter = (
            f"[0:v]tpad=stop_mode=clone:stop_duration={pad_needed:.3f}[vpad]"
        )

        cmd_concat = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(concat_video_only),
            "-i", str(audio_file),
            "-filter_complex", tpad_filter,
            "-map", "[vpad]",
            "-map", "1:a",
            "-t", f"{audio_duration:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-pix_fmt", "yuv420p", "-r", str(fps),
            "-c:a", "aac", "-b:a", "192k",
            str(unsubbed_video),
        ]
        try:
            subprocess.run(cmd_concat, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"فشل دمج الفيديو مع الصوت (stage=pad_and_mux): exit={e.returncode}"
            ) from e

        _validate_non_empty_file(unsubbed_video, "الفيديو الوسيط (بدون ترجمة)")

        # ============================================================
        # 3. توليد مسار whoosh موحد بطول audio_duration
        # ============================================================
        transition_times: List[float] = []
        current_time = 0.0
        for _, _, duration in validated_pairs[:-1]:
            current_time += duration
            transition_times.append(current_time)

        build_whoosh_timeline(
            sfx_whoosh,
            transition_times,
            audio_duration,
            whoosh_timeline,
            whoosh_volume=0.4,
        )
        _validate_non_empty_file(whoosh_timeline, "مسار الـ whoosh الموحد")

        # ============================================================
        # 4. دمج الترجمة وميكس الصوت داخل filter_complex موحد
        #    الكتابة تكون إلى الملف المرحلي فقط.
        # ============================================================
        logger.info(
            "✨ حرق الترجمة الحركية الصفراء الباهتة وتطبيق مؤثرات الانتقال الصوتية..."
        )

        ass_escaped = (
            subtitles_ass.resolve().as_posix().replace(":", "\\:")
        )

        filter_complex = (
            f"[0:v]ass='{ass_escaped}'[vout];"
            f"[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0:weights=1 1[aout]"
        )

        cmd_final = [
            "ffmpeg", "-y", "-v", "info",
            "-i", str(unsubbed_video),
            "-i", str(whoosh_timeline),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k",
            str(staged_output_video),
        ]
        try:
            subprocess.run(cmd_final, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"فشل التصدير النهائي (stage=final_mux): exit={e.returncode}"
            ) from e

        # ============================================================
        # [FIX 6] التحقق من الإخراج المرحلي قبل نقله إلى المسار النهائي
        # ============================================================
        _validate_non_empty_file(staged_output_video, "الفيديو النهائي المرحلي")
        final_duration = _probe_media_duration(staged_output_video)
        if final_duration <= 0:
            raise RuntimeError(
                f"مدة الفيديو النهائي المرحلي غير موجبة: {final_duration}"
            )

        # ============================================================
        # [FIX 7] النقل الذري إلى المسار النهائي بعد نجاح كل التحققات.
        # لا يتم لمس أي فيديو نهائي سابق إذا فشل أي شيء قبل هذه النقطة.
        # ============================================================
        try:
            staged_output_video.replace(output_video_path)
        except OSError as e:
            raise RuntimeError(
                f"تعذّر نقل الفيديو النهائي المرحلي إلى المسار النهائي: "
                f"{staged_output_video} -> {output_video_path}"
            ) from e

        logger.info(
            f"🏆 تم تصدير الفيديو النهائي بنجاح بأعلى دقة: {output_video_path} "
            f"(المدة النهائية = {final_duration:.3f}s | مرجع الصوت = {audio_duration:.3f}s)"
        )

        success = True

    finally:
        # ============================================================
        # [FIX 1] تنظيف المجلد المؤقت الفريد فقط
        # ============================================================
        try:
            if temp_dir.exists() and temp_dir.is_dir():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as cleanup_err:
            logger.warning(
                f"تعذّر تنظيف المجلد المؤقت {temp_dir}: {cleanup_err}"
            )

    if not success:
        # لن نصل هنا إلا في حالة استثناء تم رفعه بالفعل من داخل try.
        raise RuntimeError("فشلت الرندرة دون استثناء صريح (حالة غير متوقعة).")
