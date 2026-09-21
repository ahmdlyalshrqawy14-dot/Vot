import os
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from gemini_engine import call_gemini_vision_with_fallback

logger = logging.getLogger("Stage4Vision")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".jfif", ".bmp"}


class MissingAssetsError(Exception):
    """استثناء عند وجود صور مفقودة لإخطار المستخدم بأرقامها الحقيقية بدقة"""
    def __init__(self, missing_indices: List[int], total_expected: int, found_count: int):
        self.missing_indices = missing_indices
        self.total_expected = total_expected
        self.found_count = found_count
        msg = (
            f"⚠️ تم التحقق من {found_count} صورة من أصل {total_expected}.\n"
            f"الصور المفقودة المطلوب رفعها: {missing_indices}"
        )
        super().__init__(msg)


def extract_index_using_gemini_vision(image_path: Path) -> Optional[int]:
    """
    قص الركن السفلي الأيمن وإرساله لـ Gemini Vision لقراءة الرقم الباهت بعينيه
    دون الاعتماد على اسم الملف أو تاريخ الرفع إطلاقاً.
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            return None

        h, w, _ = img.shape
        # اقتطاع آخر 20% من الارتفاع والعرض (الركن السفلي الأيمن فقط)
        crop_y = int(h * 0.80)
        crop_x = int(w * 0.80)
        corner_crop = img[crop_y:h, crop_x:w]

        # تحويل الاقتطاع إلى صيغة JPEG خفيفة وسريعة
        _, buffer = cv2.imencode(".jpg", corner_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        crop_bytes = buffer.tobytes()

        prompt = (
            "Look at this image crop of the bottom-right corner. "
            "There is a small, faint, subtle number or digits. "
            "What is this integer number? Return ONLY the integer digits (e.g. 1, 2, 45). "
            "If there is absolutely no number, reply with NONE."
        )

        response_text = call_gemini_vision_with_fallback(
            image_bytes=crop_bytes,
            mime_type="image/jpeg",
            user_prompt=prompt
        )

        # استخراج الأرقام من رد النموذج
        numbers = re.findall(r"\b\d+\b", response_text)
        if numbers:
            recognized_num = int(numbers[0])
            logger.info(f"👁️ [Gemini Vision] تم فحص {image_path.name} ➔ الرقم المكتشف: [{recognized_num}]")
            return recognized_num
        else:
            logger.warning(f"⚠️ [Gemini Vision] لم يتم العثور على رقم في {image_path.name} (رد: {response_text.strip()})")
            return None

    except Exception as e:
        logger.error(f"❌ خطأ أثناء فحص الصورة {image_path.name} بـ Gemini Vision: {e}")
        return None


def apply_seamless_inpainting(input_img_path: Path, output_img_path: Path):
    """إخفاء الرقم تماماً برقعة مطابقة للون الخلفية الرمادية"""
    img = cv2.imread(str(input_img_path), cv2.IMREAD_COLOR)
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

    output_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_img_path), refined)


def _scan_single_image(img_path: Path) -> Tuple[Path, Optional[int]]:
    """دالة مساعدة للفحص المتوازي"""
    num = extract_index_using_gemini_vision(img_path)
    return img_path, num


def process_and_verify_images(
    uploaded_images_dir: Path,
    output_frames_dir: Path,
    expected_total: int
) -> List[Path]:
    """
    دورة الفحص الذكية الشاملة:
    1. قراءة كافة الصور المرفوعة.
    2. فحص الركن السفلي الأيمن لكل صورة عبر Gemini Vision بالتوازي (Parallel Threads).
    3. ربط كل صورة برقمها الفعلي بغض النظر عن اسمها أو توقيتها.
    4. إعادة تسمية الصور إلى: frame_001.png, frame_002.png...
    5. تطبيق الرقعة الذكية لإخفاء الأرقام.
    """
    if not uploaded_images_dir.exists():
        raise FileNotFoundError(f"المجلد غير موجود: {uploaded_images_dir}")

    uploaded_files = [
        p for p in uploaded_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not uploaded_files:
        raise FileNotFoundError("لم يتم العثور على أي صور في مجلد الرفع!")

    logger.info(f"🚀 بدء فحص {len(uploaded_files)} صورة بالذكاء الاصطناعي (Gemini Vision) بالتوازي...")

    indexed_images: Dict[int, Path] = {}
    unindexed_files: List[Path] = []

    # فحص الصور بالتوازي عبر 5 مسارات للاستفادة من المفاتيح الأربعة والسرعة القصوى
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_scan_single_image, img_p) for img_p in uploaded_files]
        for future in as_completed(futures):
            img_path, detected_index = future.result()
            if detected_index is not None and 1 <= detected_index <= expected_total:
                indexed_images[detected_index] = img_path
            else:
                unindexed_files.append(img_path)

    found_count = len(indexed_images)
    logger.info(f"📊 نتيجة الفحص: تم التعرف بنجاح على {found_count} صورة من أصل {expected_total}.")

    # كشف النواقص الحقيقية
    missing_numbers = [i for i in range(1, expected_total + 1) if i not in indexed_images]
    if missing_numbers:
        logger.warning(f"🚨 الصور المفقودة فعلياً: {missing_numbers}")
        raise MissingAssetsError(
            missing_indices=missing_numbers,
            total_expected=expected_total,
            found_count=found_count
        )

    # إذا كانت كل الصور متوفرة بنسبة 100%، نقوم بإعادة تسميتها وتطبيق الرقعة الذكية
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    verified_frames: List[Path] = []

    for idx in range(1, expected_total + 1):
        raw_img = indexed_images[idx]
        target_path = output_frames_dir / f"frame_{idx:03d}.png"
        apply_seamless_inpainting(raw_img, target_path)
        verified_frames.append(target_path)

    logger.info(f"🏆 تم ترتيب وتجهيز كافة الصور ({len(verified_frames)} صورة) بنجاح بنسبة 100%!")
    return verified_frames
