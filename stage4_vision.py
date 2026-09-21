import os
import re
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger("Stage4Vision")


class MissingAssetsError(Exception):
    """استثناء مخصص عند وجود صور مفقودة لإيقاف الرندرة وتنبيه المستخدم"""
    def __init__(self, missing_indices: List[int], total_expected: int, found_count: int):
        self.missing_indices = missing_indices
        self.total_expected = total_expected
        self.found_count = found_count
        msg = (
            f"⚠️ تم التحقق من {found_count} صورة من أصل {total_expected} صورة.\n"
            f"الصور المفقودة التي لم تُرفع هي: {missing_indices}."
        )
        super().__init__(msg)


def extract_index_from_bottom_right(image_path: Path) -> Optional[int]:
    """
    فحص الركن السفلي الأيمن من الصورة لاستخراج الرقم التسلسلي الباهت.
    يعتمد أولاً على قراءة اسم الملف إذا كان يحتوي رقماً، ثم الفحص البصري.
    """
    # 1. محاولة قراءة الرقم من اسم الملف المرفوع إذا كان مسمى بترتيبه (مثل 1.png أو prompt_01)
    file_numbers = re.findall(r"\d+", image_path.stem)
    if file_numbers:
        return int(file_numbers[-1])

    # 2. الفحص البصري عبر معالجة الركن السفلي الأيمن باستخدام OpenCV
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w, _ = img.shape
    # اقتطاع الركن السفلي الأيمن (آخر 15% من العرض والارتفاع)
    crop_y = int(h * 0.85)
    crop_x = int(w * 0.85)
    corner = img[crop_y:h, crop_x:w]

    # تحويل لرمادي وتطبيق Threshold لإبراز الأرقام الباهتة
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # محاولة استخدام pytesseract إن وجد
    try:
        import pytesseract
        text = pytesseract.image_to_string(thresh, config="--psm 8 -c tessedit_char_whitelist=0123456789")
        nums = re.findall(r"\d+", text)
        if nums:
            return int(nums[0])
    except Exception:
        pass

    return None


def apply_seamless_inpainting(input_img_path: Path, output_img_path: Path):
    """
    الرقعة الذكية غير المرئية (Seamless Background Inpainting):
    سحب عينة لونية دقيقة من البكسلات الرمادية المجاورة للرقم
    وتطبيق رقعة متطابقة 100% مع تنعيم حواف ناعم جداً (Edge Feathering & Blending)
    لإخفاء الرقم تماماً دون ترك أي أثر مرئي.
    """
    img = cv2.imread(str(input_img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"تعذر فتح الصورة: {input_img_path}")

    h, w, _ = img.shape

    # إحداثيات مستطيل الرقم في الركن السفلي الأيمن
    start_y = int(h * 0.88)
    end_y = int(h * 0.98)
    start_x = int(w * 0.88)
    end_x = int(w * 0.98)

    # سحب عينة لونية من البكسلات الملاصقة مباشرة أعلى المستطيل
    sample_region = img[start_y - 20 : start_y, start_x:end_x]
    mean_color = np.mean(sample_region, axis=(0, 1)).astype(np.uint8)

    # إنشاء قناع للرقعة
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[start_y:end_y, start_x:end_x] = 255

    # تطبيق تنعيم الحواف (Edge Feathering عبر Gaussian Blur)
    feathered_mask = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0
    feathered_mask = np.repeat(feathered_mask[:, :, np.newaxis], 3, axis=2)

    # إنشاء مصفوفة لون الخلفية الموحد
    solid_bg = np.full_like(img, mean_color)

    # دمج الرقعة بسلاسة مع الصورة الأصلية
    blended = (feathered_mask * solid_bg + (1.0 - feathered_mask) * img).astype(np.uint8)

    # استخدام inpaint كصمام أمان إضافي لإزالة أي أطياف متبقية
    refined = cv2.inpaint(blended, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    output_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_img_path), refined)


def process_and_verify_images(
    uploaded_images_dir: Path,
    output_frames_dir: Path,
    expected_total: int
) -> List[Path]:
    """
    دورة المعالجة الكاملة:
    1. الفحص البصري وترتيب الصور.
    2. كشف النواقص والتنبيه الدقيق.
    3. تطبيق الرقعة الذكية وإعادة التسمية الموحدة (frame_001.png).
    """
    uploaded_files = list(uploaded_images_dir.glob("*.png")) + list(uploaded_images_dir.glob("*.jpg"))
    if not uploaded_files:
        raise FileNotFoundError(f"لم يتم العثور على أي صور في المجلد: {uploaded_images_dir}")

    # فهرسة الصور بناءً على الرقم المستخرج
    indexed_images: Dict[int, Path] = {}
    unindexed = []

    for img_p in uploaded_files:
        idx = extract_index_from_bottom_right(img_p)
        if idx and 1 <= idx <= expected_total:
            indexed_images[idx] = img_p
        else:
            unindexed.append(img_p)

    # إذا كان هناك صور غير مرقمة ولكن عدد الصور يطابق بالتمام، نرتبها حسب الاسم
    if len(indexed_images) < expected_total and (len(indexed_images) + len(unindexed)) == expected_total:
        unindexed_sorted = sorted(unindexed, key=lambda p: p.name)
        free_indices = [i for i in range(1, expected_total + 1) if i not in indexed_images]
        for f_idx, path in zip(free_indices, unindexed_sorted):
            indexed_images[f_idx] = path

    # ثانياً: كاشف النواقص الذكي
    missing = [i for i in range(1, expected_total + 1) if i not in indexed_images]
    if missing:
        raise MissingAssetsError(
            missing_indices=missing,
            total_expected=expected_total,
            found_count=len(indexed_images)
        )

    # ثالثاً: تطبيق الرقعة الذكية وتصدير الصور المرتبة
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    verified_frames: List[Path] = []

    for idx in range(1, expected_total + 1):
        raw_img = indexed_images[idx]
        target_path = output_frames_dir / f"frame_{idx:03d}.png"
        apply_seamless_inpainting(raw_img, target_path)
        verified_frames.append(target_path)

    logger.info(f"✅ تم فحص وترتيب وتطبيق الرقعة الذكية على {len(verified_frames)} صورة بنجاح.")
    return verified_frames
