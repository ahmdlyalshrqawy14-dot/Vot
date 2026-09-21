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
    def __init__(self, missing_indices: List[int], total_expected: int, found_count: int):
        self.missing_indices = missing_indices
        self.total_expected = total_expected
        self.found_count = found_count
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
    """قص مؤقت في الرام للركن الأيمن وقراءة الرقم عبر Gemini Vision"""
    try:
        img = read_image_safe(image_path)
        if img is None:
            return None

        h, w, _ = img.shape
        crop_y = int(h * 0.80)
        crop_x = int(w * 0.80)
        corner_crop = img[crop_y:h, crop_x:w]

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

        numbers = re.findall(r"\b\d+\b", response_text)
        if numbers:
            return int(numbers[0])
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


def process_and_verify_images(
    uploaded_images_dir: Path,
    output_frames_dir: Path,
    expected_total: int,
    allow_partial: bool = False
) -> List[Path]:
    if not uploaded_images_dir.exists():
        raise FileNotFoundError(f"المجلد غير موجود: {uploaded_images_dir}")

    uploaded_files = [
        p for p in uploaded_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not uploaded_files:
        raise FileNotFoundError("لم يتم العثور على أي صور في مجلد الرفع!")

    indexed_images: Dict[int, Path] = {}
    unindexed_files: List[Path] = []

    # فحص متوازي عبر 5 مسارات للاستفادة من المفاتيح الأربعة بسرعة
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_scan_single_image, img_p) for img_p in uploaded_files]
        for future in as_completed(futures):
            img_path, detected_index = future.result()
            if detected_index is not None and 1 <= detected_index <= expected_total:
                indexed_images[detected_index] = img_path
            else:
                unindexed_files.append(img_path)

    # تسكين الصور غير المقروءة في الأماكن الفارغة
    if unindexed_files:
        empty_slots = [i for i in range(1, expected_total + 1) if i not in indexed_images]
        for slot, fallback_img in zip(empty_slots, unindexed_files):
            indexed_images[slot] = fallback_img

    found_count = len(indexed_images)
    missing_numbers = [i for i in range(1, expected_total + 1) if i not in indexed_images]

    # لو في صور ناقصة ولم يُطلب التجاوز
    if missing_numbers and not allow_partial:
        raise MissingAssetsError(
            missing_indices=missing_numbers,
            total_expected=expected_total,
            found_count=found_count
        )

    output_frames_dir.mkdir(parents=True, exist_ok=True)
    verified_frames: List[Path] = []

    cleaned_cache: Dict[int, Path] = {}
    for idx, raw_p in indexed_images.items():
        clean_target = output_frames_dir / f"clean_raw_{idx:03d}.png"
        apply_seamless_inpainting(raw_p, clean_target)
        cleaned_cache[idx] = clean_target

    if not cleaned_cache:
        raise RuntimeError("فشل تجهيز أي كادر صالح للرندرة!")

    # تطبيق ملء الفراغات التلقائي (Forward-Fill)
    first_available_idx = min(cleaned_cache.keys())
    last_valid_frame = cleaned_cache[first_available_idx]

    for frame_idx in range(1, expected_total + 1):
        if frame_idx in cleaned_cache:
            last_valid_frame = cleaned_cache[frame_idx]

        final_frame_path = output_frames_dir / f"frame_{frame_idx:03d}.png"
        if last_valid_frame != final_frame_path:
            shutil.copyfile(last_valid_frame, final_frame_path)

        verified_frames.append(final_frame_path)

    return verified_frames
