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
    expected_total: int,
    skip_paths: Optional[set] = None,
) -> Tuple[Dict[int, Path], List[Path], List[Tuple[int, Path, Path]]]:
    """
    يقرأ كل الصور، يكتشف الرقم من الركن الأيمن، ويعيد قاموساً بالأرقام المكتشفة.

    [FIX-MANUAL-SKIP] أي مسار موجود في skip_paths (تعيين يدوي نهائي للمستخدم)
    لا يُمرَّر إلى Gemini إطلاقًا، وبذلك لا يُسجَّل كفشل OCR ولا يُعاد فحصه.
    """
    skip_paths = skip_paths or set()

    uploaded_files: List[Path] = []
    for p in uploaded_images_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in VALID_EXTENSIONS:
            continue
        # تخطّي أي صورة المستخدم عيّنها يدويًا — لا OCR عليها أبدًا
        try:
            if p.resolve() in skip_paths:
                logger.info(f"🔒 تخطّي OCR (تعيين يدوي نهائي): {p.name}")
                continue
        except Exception:
            pass
        uploaded_files.append(p)

    if not uploaded_files:
        # قد يكون كل شيء معيَّنًا يدويًا — لا نرفع استثناء، نعيد قوائم فارغة
        logger.info("ℹ️ لا توجد صور بحاجة إلى OCR (كل الملفات معيَّنة يدويًا أو لا ملفات صالحة).")
        return {}, [], []

    indexed_images: Dict[int, Path] = {}
    unindexed_files: List[Path] = []
    conflicts: List[Tuple[int, Path, Path]] = []

    logger.info(f"🔍 بدء فحص {len(uploaded_files)} صورة بالتوازي...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_scan_single_image, p): p for p in uploaded_files}

        for future in as_completed(futures):
            img_path = futures[future]
            try:
                _, detected_index = future.result()

                if detected_index is not None and 1 <= detected_index <= expected_total:
                    if detected_index in indexed_images:
                        conflicts.append((detected_index, indexed_images[detected_index], img_path))
                        unindexed_files.append(img_path)
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

    logger.info(f"📊 النتائج: {len(indexed_images)} مكتشفة | {len(unindexed_files)} غير مكتشفة | {len(conflicts)} متكررة")

    return indexed_images, unindexed_files, conflicts


def process_and_verify_images(
    uploaded_images_dir: Path,
    output_frames_dir: Path,
    expected_total: int,
    allow_partial: bool = False,
    manual_assignments: Optional[Dict[int, Path]] = None
) -> List[Path]:
    """
    المعالجة الرئيسية مع دعم الوضع اليدوي.

    [FIX-GUESSING] تم إلغاء التسكين التلقائي للصور غير المقروءة، وإلغاء
    Forward/Backward-Fill للخانات الناقصة. الدالة ترجع فقط الكادرات المؤكدة
    (Gemini + تخصيص يدوي)، وتترك النواقص لنظام المراجعة اليدوي في bot.py.

    [FIX-MANUAL-FINAL] أي مسار موجود في manual_assignments يُعتبر اختيارًا نهائيًا
    من المستخدم:
      - يُستبعد من OCR تمامًا (لا يُمرَّر إلى Gemini).
      - لا يظهر مطلقًا في unindexed_files.
      - رقمه ثابت ولا يُعاد فحصه حتى مع إعادة التشغيل.

    Args:
        manual_assignments: تخصيصات يدوية من المستخدم {رقم: مسار_الصورة}
    """
    if not uploaded_images_dir.exists():
        raise FileNotFoundError(f"المجلد غير موجود: {uploaded_images_dir}")

    # ---- 1) تطبيع التعيين اليدوي كمصدر نهائي ----
    manual_assignments = manual_assignments or {}
    manual_final: Dict[int, Path] = {}
    manual_paths_skip: set = set()  # مسارات لا تُمرَّر إلى Gemini أبدًا

    for raw_idx, raw_path in manual_assignments.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            logger.warning(f"⚠️ مفتاح تعيين يدوي غير صالح: {raw_idx!r}")
            continue
        if not (1 <= idx <= expected_total):
            logger.warning(f"⚠️ رقم تعيين يدوي خارج النطاق ({idx}) — تم تجاهله")
            continue
        p = Path(raw_path)
        if not p.exists() or not p.is_file():
            logger.warning(f"⚠️ مسار تعيين يدوي غير موجود: {p}")
            continue
        manual_final[idx] = p
        try:
            manual_paths_skip.add(p.resolve())
        except Exception:
            pass

    # ---- 2) OCR فقط للصور غير المعيَّنة يدويًا ----
    indexed_images, unindexed_files, conflicts = rename_images_with_detected_numbers(
        uploaded_images_dir,
        output_frames_dir,
        expected_total,
        skip_paths=manual_paths_skip,
    )

    # أي ملف ظهر في unindexed وهو أصلًا معيَّن يدويًا → احذفه من قائمة الفشل
    # (احتياط مزدوج؛ القائمة أساسًا لا تحتوي عليها بفضل skip_paths)
    def _safe_resolve(x: Path) -> Optional[Path]:
        try:
            return x.resolve()
        except Exception:
            return None

    unindexed_files = [
        p for p in unindexed_files
        if _safe_resolve(p) not in manual_paths_skip
    ]

    # عزل مجلد renamed لكل حلقة على حدة
    renamed_dir = output_frames_dir.parent / f"{output_frames_dir.name}_renamed"
    renamed_dir.mkdir(parents=True, exist_ok=True)

    # ---- 3) تثبيت التعيين اليدوي فوق نتيجة Gemini (أولوية المستخدم) ----
    for idx, img_path in manual_final.items():
        new_path = renamed_dir / f"img_{idx:03d}.png"
        img = read_image_safe(img_path)
        if img is not None:
            write_image_safe(new_path, img)
            indexed_images[idx] = new_path
            logger.info(f"🔒 تعيين يدوي نهائي: {img_path.name} → رقم {idx}")
        else:
            # حتى لو القراءة فشلت، نحتفظ بالمسار الأصلي كاختيار مستخدم
            indexed_images[idx] = img_path
            logger.warning(f"⚠️ تعيين يدوي بدون إعادة ترميز: {img_path.name} → {idx}")

    # [FIX-GUESSING-1] لا تسكين تلقائي. النواقص تُترك لنظام المراجعة اليدوي.

    found_count = len(indexed_images)
    missing_numbers = [i for i in range(1, expected_total + 1) if i not in indexed_images]

    if missing_numbers and not allow_partial:
        raise MissingAssetsError(
            missing_indices=missing_numbers,
            total_expected=expected_total,
            found_count=found_count,
            unindexed_files=unindexed_files
        )

    if not indexed_images:
        raise RuntimeError("فشل تجهيز أي كادر صالح للرندرة!")

    output_frames_dir.mkdir(parents=True, exist_ok=True)
    verified_frames: List[Path] = []

    # كتابة الكادرات المؤكدة فقط (بدون رقعة)
    for idx in sorted(indexed_images.keys()):
        raw_p = indexed_images[idx]
        final_frame_path = output_frames_dir / f"frame_{idx:03d}.png"

        img = read_image_safe(raw_p)
        if img is not None:
            write_image_safe(final_frame_path, img)
        else:
            shutil.copyfile(raw_p, final_frame_path)

        logger.info(f"💾 تم توليد الكادر النهائي: {final_frame_path.name}")
        verified_frames.append(final_frame_path)

    # [FIX-GUESSING-2] لا Forward/Backward-Fill. النواقص تُترك لنظام المراجعة.

    if missing_numbers:
        logger.warning(
            f"⚠️ كادرات ناقصة متروكة للمراجعة اليدوية (لم تُملأ تلقائيًا): {missing_numbers}"
        )

    if unindexed_files:
        logger.warning(
            f"⚠️ صور فشل قراءة رقمها ولم تُسكَّن تلقائيًا: "
            f"{[p.name for p in unindexed_files]}"
        )

    return verified_frames
