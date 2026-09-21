import os
import re
import cv2
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from gemini_engine import call_gemini_vision_with_fallback

logger = logging.getLogger("Stage4Vision")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".jfif", ".bmp"}


class MissingAssetsError(Exception):
    def __init__(
        self,
        missing_indices: List[int],
        total_expected: int,
        found_count: int,
        unindexed_files: Optional[List[Path]] = None
    ):
        self.missing_indices = missing_indices
        self.total_expected = total_expected
        self.found_count = found_count
        self.unindexed_files = unindexed_files or []
        msg = f"⚠️ تم التحقق من {found_count} صورة من أصل {total_expected}."
        super().__init__(msg)


def read_image_safe(path: Path) -> Optional[np.ndarray]:
    """قراءة الصورة بأمان من الذاكرة لتفادي مشاكل رموز ويندوز مثل النقاط …"""
    try:
        with open(path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        logger.error(f"تعذر فتح الملف {path.name}: {e}")
        return None


def write_image_safe(path: Path, img: np.ndarray):
    """حفظ الصورة بأمان تام متوافق مع كافة الرموز والامتدادات"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    success, encoded_img = cv2.imencode(ext, img)
    if success:
        with open(path, "wb") as f:
            f.write(encoded_img)
    else:
        raise IOError(f"فشل تشفير وحفظ الصورة: {path}")


def extract_index_using_gemini_vision(image_path: Path) -> Optional[int]:
    """قص مؤقت في الرام للركن الأيمن وقراءة الرقم عبر Gemini Vision مع تحسينات"""
    try:
        img = read_image_safe(image_path)
        if img is None:
            return None

        h, w, _ = img.shape

        # [FIX 3] توسيع هامش الأمان إلى 18% لضمان بقاء الرقم كاملاً داخل الكادر
        crop_y = int(h * 0.82)
        crop_x = int(w * 0.82)
        corner_crop = img[crop_y:h, crop_x:w]

        # [FIX 1] تكبير 3x فقط بدون تحويل للرمادي أو رفع تباين حاد
        # Gemini Vision شبكة بصرية تفهم الألوان والظلال الطبيعية، والفلتر الكلاسيكي
        # (cvtColor + convertScaleAbs) يمحو ملامح الأرقام الباهتة ويحولها لتشويش رقمي.
        corner_crop = cv2.resize(corner_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        _, buffer = cv2.imencode(".jpg", corner_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        crop_bytes = buffer.tobytes()

        prompt = (
            "Look at this image crop from the bottom-right corner. "
            "There is a small, faint number printed there. "
            "What is this integer number? Return ONLY the integer digits (e.g. 1, 2, 45). "
            "If there is absolutely no number visible, reply with NONE."
        )

        response_text = call_gemini_vision_with_fallback(
            image_bytes=crop_bytes,
            mime_type="image/jpeg",
            user_prompt=prompt
        )

        numbers = re.findall(r"\b\d+\b", response_text)
        if numbers:
            detected = int(numbers[0])
            # التحقق من أن الرقم في النطاق المنطقي (1-500)
            if 1 <= detected <= 500:
                return detected
        return None

    except Exception as e:
        logger.warning(f"تعذر قراءة الصورة {image_path.name}: {e}")
        return None


def apply_seamless_inpainting(input_img_path: Path, output_img_path: Path):
    """إخفاء الرقم الباهت بالرقعة الذكية من الصورة الكاملة الأصلية"""
    img = read_image_safe(input_img_path)
    if img is None:
        raise ValueError(f"تعذر فتح الصورة: {input_img_path}")

    h, w, _ = img.shape
    start_y = int(h * 0.88)
    end_y = int(h * 0.98)
    start_x = int(w * 0.88)
    end_x = int(w * 0.98)

    sample_region = img[start_y - 20 : start_y, start_x:end_x]
    mean_color = np.mean(sample_region, axis=(0, 1)).astype(np.uint8)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[start_y:end_y, start_x:end_x] = 255

    feathered_mask = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0
    feathered_mask = np.repeat(feathered_mask[:, :, np.newaxis], 3, axis=2)

    solid_bg = np.full_like(img, mean_color)
    blended = (feathered_mask * solid_bg + (1.0 - feathered_mask) * img).astype(np.uint8)
    refined = cv2.inpaint(blended, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    write_image_safe(output_img_path, refined)


def _scan_single_image(img_path: Path) -> Tuple[Path, Optional[int]]:
    num = extract_index_using_gemini_vision(img_path)
    return img_path, num


def rename_images_with_detected_numbers(
    uploaded_images_dir: Path,
    output_frames_dir: Path,
    expected_total: int
) -> Tuple[Dict[int, Path], List[Path], List[Tuple[int, Path, Path]]]:
    """
    يقرأ كل الصور، يكتشف الرقم من الركن الأيمن، ويعيد قاموساً بالأرقام المكتشفة.

    [FIX 4] لم يعد يتم نسخ الصور إلى مجلد renamed_images؛ بل نحتفظ بالمسارات
    الأصلية في الذاكرة ليطبق عليها الـ Inpainting لاحقاً مباشرة داخل الكادر
    المستهدف، مما يلغي عملية كتابة قرص كاملة (60% توفير في I/O).

    Returns:
        - Dict[int, Path]: صور مرتبة بالأرقام المكتشفة (مسارات أصلية)
        - List[Path]: صور لم يتم التعرف على رقمها
        - List[Tuple[int, Path, Path]]: أرقام متكررة (الرقم، الصورة الأولى، الصورة الثانية)
    """
    uploaded_files = [
        p for p in uploaded_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not uploaded_files:
        raise FileNotFoundError("لم يتم العثور على أي صور في مجلد الرفع!")

    indexed_images: Dict[int, Path] = {}
    unindexed_files: List[Path] = []
    conflicts: List[Tuple[int, Path, Path]] = []

    logger.info(f"🔍 بدء فحص {len(uploaded_files)} صورة بالتوازي...")

    # OCR متوازي عبر 5 مسارات
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_scan_single_image, p): p for p in uploaded_files}

        for future in as_completed(futures):
            img_path = futures[future]
            try:
                _, detected_index = future.result()

                if detected_index is not None and 1 <= detected_index <= expected_total:
                    if detected_index in indexed_images:
                        # رقم متكرر - نحتفظ بالأولى ونضيف الثانية للـ conflicts + unindexed
                        conflicts.append((detected_index, indexed_images[detected_index], img_path))
                        unindexed_files.append(img_path)  # ← التعديل: الصورة المكررة تدخل الوضع اليدوي
                        logger.warning(f"⚠️ رقم متكرر {detected_index}: {img_path.name} → أُرسلت للتعيين اليدوي")
                    else:
                        indexed_images[detected_index] = img_path
                        logger.info(f"✅ تم التعرف على {img_path.name} → رقم {detected_index}")
                else:
                    unindexed_files.append(img_path)
                    logger.warning(f"❌ لم يتم التعرف على رقم: {img_path.name}")

            except Exception as e:
                logger.error(f"خطأ في معالجة {img_path.name}: {e}")
                unindexed_files.append(img_path)

    # [FIX 4] الاحتفاظ بالمسارات الأصلية مباشرة - لا كتابة قرص مكررة
    final_indexed: Dict[int, Path] = indexed_images

    logger.info(f"📊 النتائج: {len(final_indexed)} مكتشفة | {len(unindexed_files)} غير مكتشفة | {len(conflicts)} متكررة")

    return final_indexed, unindexed_files, conflicts


def process_and_verify_images(
    uploaded_images_dir: Path,
    output_frames_dir: Path,
    expected_total: int,
    allow_partial: bool = False,
    manual_assignments: Optional[Dict[int, Path]] = None
) -> List[Path]:
    """
    المعالجة الرئيسية مع دعم الوضع اليدوي.

    Args:
        manual_assignments: تخصيصات يدوية من المستخدم {رقم: مسار_الصورة}
    """
    if not uploaded_images_dir.exists():
        raise FileNotFoundError(f"المجلد غير موجود: {uploaded_images_dir}")

    # استخدام الدالة الجديدة لإعادة التسمية
    indexed_images, unindexed_files, conflicts = rename_images_with_detected_numbers(
        uploaded_images_dir, output_frames_dir, expected_total
    )

    # [FIX 2] عزل مجلد renamed لكل حلقة على حدة لتفادي التداخل مع الحلقات الأخرى
    # وتسهيل حذفه تلقائياً عند (/start) عبر bot.py
    renamed_dir = output_frames_dir.parent / f"{output_frames_dir.name}_renamed"

    # دمج التخصيصات اليدوية إن وجدت
    if manual_assignments:
        for idx, img_path in manual_assignments.items():
            if 1 <= idx <= expected_total:
                new_path = renamed_dir / f"img_{idx:03d}.png"
                img = read_image_safe(img_path)
                if img is not None:
                    write_image_safe(new_path, img)
                    indexed_images[idx] = new_path
                    logger.info(f"🔧 تم إضافة تخصيص يدوي: {img_path.name} → رقم {idx}")

                    # إزالة من غير المكتشفة لو موجودة
                    if img_path in unindexed_files:
                        unindexed_files.remove(img_path)

    # تسكين الصور غير المقروءة في الأماكن الفارغة (تلقائياً)
    if unindexed_files and not manual_assignments:
        empty_slots = [i for i in range(1, expected_total + 1) if i not in indexed_images]
        for slot, fallback_img in zip(empty_slots, unindexed_files):
            new_path = renamed_dir / f"img_{slot:03d}.png"
            img = read_image_safe(fallback_img)
            if img is not None:
                write_image_safe(new_path, img)
                indexed_images[slot] = new_path
                logger.info(f"📥 تم تسكين {fallback_img.name} في المكان الفارغ {slot}")

    found_count = len(indexed_images)
    missing_numbers = [i for i in range(1, expected_total + 1) if i not in indexed_images]

    # لو في صور ناقصة ولم يُطلب التجاوز
    if missing_numbers and not allow_partial:
        raise MissingAssetsError(
            missing_indices=missing_numbers,
            total_expected=expected_total,
            found_count=found_count,
            unindexed_files=unindexed_files  # 👈 تمرير الصور غير المقروءة للوضع اليدوي في البوت
        )

    if not indexed_images:
        raise RuntimeError("فشل تجهيز أي كادر صالح للرندرة!")

    output_frames_dir.mkdir(parents=True, exist_ok=True)
    verified_frames: List[Path] = []

    # [FIX 4] تطبيق الـ Inpainting مرة واحدة فقط مباشرة داخل الكادر المستهدف
    # frame_xxx.png وإلغاء ملف clean_raw_xxx.png الوسيط تماماً.
    first_available_idx = min(indexed_images.keys())

    # كتابة الكادرات الحقيقية بترتيب ثابت
    for idx in sorted(indexed_images.keys()):
        raw_p = indexed_images[idx]
        final_frame_path = output_frames_dir / f"frame_{idx:03d}.png"
        apply_seamless_inpainting(raw_p, final_frame_path)
        logger.info(f"💾 تم توليد الكادر النهائي: {final_frame_path.name}")

    # تطبيق ملء الفراغات التلقائي (Forward/Backward-Fill) بدون إعادة كتابة
    for frame_idx in range(1, expected_total + 1):
        final_frame_path = output_frames_dir / f"frame_{frame_idx:03d}.png"

        if frame_idx not in indexed_images:
            candidates_before = [i for i in indexed_images if i < frame_idx]
            source_idx = max(candidates_before) if candidates_before else first_available_idx
            source_path = output_frames_dir / f"frame_{source_idx:03d}.png"
            shutil.copyfile(source_path, final_frame_path)
            logger.info(f"🔄 تم ملء الكادر {frame_idx:03d} من الكادر {source_idx:03d}")

        verified_frames.append(final_frame_path)

    return verified_frames
