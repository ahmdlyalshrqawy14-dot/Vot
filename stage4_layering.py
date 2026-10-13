"""
stage4_layering.py
──────────────────
طبقة Auto-Layering أوتوماتيكية باستخدام rembg (U2-Net).

الوظيفة:
- قص العنصر الرئيسي (Foreground Cutout)
- تحويل الخلفية إلى نسخة أرشيفية مظلمة + تأثير Halftone خفيف
- إضافة حافة ورق بيضاء + ظل درامي حول العنصر المقصوص
- إرجاع مسارات الطبقات جاهزة للاستخدام في stage4_composer

الاستخدام:
    from stage4_layering import process_layers_batch

    layers_map = process_layers_batch(
        frames_dir=Path("outputs/episode_201/frames"),
        output_dir=Path("outputs/episode_201/layers"),
        max_workers=4
    )
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

logger = logging.getLogger("Stage4Layering")

# ============================================================
# إعدادات قابلة للتعديل
# ============================================================

# حجم الحافة البيضاء حول العنصر المقصوص (بالبكسل)
PAPER_EDGE_WIDTH = 14

# قوة الظل
SHADOW_OFFSET = (18, 22)
SHADOW_BLUR = 18
SHADOW_OPACITY = 110          # 0-255

# درجة تعتيم الخلفية
BG_BRIGHTNESS = 0.42
BG_CONTRAST = 0.85
BG_BLUR_RADIUS = 6

# هل نطبق تأثير Halftone بسيط؟
ENABLE_HALFTONE = True
HALFTONE_DOT_SIZE = 3


# ============================================================
# أدوات مساعدة
# ============================================================

def _safe_open(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception as e:
        logger.error(f"فشل فتح الصورة {path.name}: {e}")
        return None


def _apply_halftone(img: Image.Image, dot_size: int = 3) -> Image.Image:
    """تأثير Halftone بسيط جدًا (نقاط) بدون مكتبات إضافية."""
    img = img.convert("L")  # رمادي
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    # تقليل الدقة ثم تكبير تاني عشان يطلع شكل نقاط
    small = Image.fromarray(arr.astype(np.uint8)).resize(
        (w // dot_size, h // dot_size), Image.Resampling.BILINEAR
    )
    small = small.resize((w, h), Image.Resampling.NEAREST)

    # نحول للنقاط البيضاء/السوداء
    result = ImageOps.autocontrast(small)
    result = result.convert("RGB")
    return result


def _add_paper_edge_and_shadow(
    fg_rgba: Image.Image,
    edge_width: int = PAPER_EDGE_WIDTH,
    shadow_offset: Tuple[int, int] = SHADOW_OFFSET,
    shadow_blur: int = SHADOW_BLUR,
    shadow_opacity: int = SHADOW_OPACITY,
) -> Image.Image:
    """
    يضيف:
    1. حافة ورق بيضاء حول الـ alpha
    2. ظل درامي أسفل يمين
    ويرجع صورة RGBA جديدة أكبر شوية.
    """
    # نوسع الكانفس عشان الظل والحافة
    pad = max(edge_width * 2, shadow_offset[0] + shadow_blur, shadow_offset[1] + shadow_blur) + 10
    new_w = fg_rgba.width + pad * 2
    new_h = fg_rgba.height + pad * 2

    # ---- الظل ----
    shadow = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    # نأخذ الـ alpha ونحوله لظل أسود
    alpha = fg_rgba.split()[-1]
    shadow_mask = Image.new("L", fg_rgba.size, 0)
    shadow_mask.paste(alpha, (0, 0))
    shadow_mask = shadow_mask.point(lambda p: shadow_opacity if p > 10 else 0)

    shadow_img = Image.new("RGBA", fg_rgba.size, (0, 0, 0, shadow_opacity))
    shadow_img.putalpha(shadow_mask)
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=shadow_blur))

    sx = pad + shadow_offset[0]
    sy = pad + shadow_offset[1]
    shadow.paste(shadow_img, (sx, sy), shadow_img)

    # ---- الحافة البيضاء ----
    # نوسع الـ alpha شوية
    edge_mask = alpha.filter(ImageFilter.MaxFilter(edge_width * 2 + 1))
    edge = Image.new("RGBA", fg_rgba.size, (255, 255, 255, 255))
    edge.putalpha(edge_mask)

    # ---- تجميع ----
    canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    canvas.paste(shadow, (0, 0), shadow)

    # نضع الحافة البيضاء
    canvas.paste(edge, (pad, pad), edge)

    # نضع العنصر الأصلي فوق الحافة
    canvas.paste(fg_rgba, (pad, pad), fg_rgba)

    return canvas


def create_layered_assets(
    input_path: Path,
    output_dir: Path,
    idx: int,
) -> Optional[Dict[str, Path]]:
    """
    يعالج صورة واحدة ويرجع مسارات الطبقات.
    """
    try:
        from rembg import remove
    except ImportError:
        logger.error("مكتبة rembg غير مثبتة! نفذ: pip install \"rembg[cpu]\"")
        return None

    original = _safe_open(input_path)
    if original is None:
        return None

    # ---- 1. قص العنصر الرئيسي ----
    with open(input_path, "rb") as f:
        input_bytes = f.read()

    try:
        output_bytes = remove(input_bytes)
    except Exception as e:
        logger.error(f"فشل rembg على {input_path.name}: {e}")
        return None

    fg = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    # ---- 2. الخلفية الأرشيفية ----
    bg = original.convert("RGB")
    bg = bg.filter(ImageFilter.GaussianBlur(radius=BG_BLUR_RADIUS))
    bg = ImageEnhance.Brightness(bg).enhance(BG_BRIGHTNESS)
    bg = ImageEnhance.Contrast(bg).enhance(BG_CONTRAST)

    if ENABLE_HALFTONE:
        # نخلط الـ halftone مع الصورة الأصلية بنسبة
        ht = _apply_halftone(original, HALFTONE_DOT_SIZE)
        bg = Image.blend(bg, ht, alpha=0.35)

    # ---- 3. حافة ورق + ظل ----
    fg_with_edge = _add_paper_edge_and_shadow(fg)

    # ---- حفظ ----
    output_dir.mkdir(parents=True, exist_ok=True)

    fg_path = output_dir / f"fg_{idx:03d}.png"
    bg_path = output_dir / f"bg_{idx:03d}.png"
    mask_path = output_dir / f"mask_{idx:03d}.png"

    fg_with_edge.save(fg_path, "PNG")
    bg.save(bg_path, "PNG")
    # نحفظ الـ alpha الأصلي كـ mask لو احتجناه بعدين
    fg.split()[-1].save(mask_path, "PNG")

    logger.info(f"✅ طبقات جاهزة → {idx:03d} | {input_path.name}")

    return {
        "fg": fg_path,
        "bg": bg_path,
        "mask": mask_path,
        "original": input_path,
        "index": idx,
    }


def process_layers_batch(
    frames: List[Path],
    output_dir: Path,
    max_workers: int = 4,
) -> Dict[int, Dict[str, Path]]:
    """
    يعالج قائمة صور بالتوازي.

    Parameters
    ----------
    frames : List[Path]
        قائمة مسارات الصور المرتبة (frame_001.png, frame_002.png ...)
    output_dir : Path
        مجلد حفظ الطبقات
    max_workers : int
        عدد الثريدات

    Returns
    -------
    Dict[int, Dict[str, Path]]
        {1: {"fg": Path, "bg": Path, "mask": Path}, 2: {...}, ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[int, Dict[str, Path]] = {}

    def _worker(item: Tuple[int, Path]) -> Optional[Dict[str, Path]]:
        idx, path = item
        return create_layered_assets(path, output_dir, idx)

    # نستخرج الرقم من اسم الملف (frame_001.png → 1)
    tasks = []
    for p in frames:
        stem = p.stem  # frame_001
        try:
            num = int(stem.split("_")[-1])
        except Exception:
            logger.warning(f"تخطي ملف بدون رقم واضح: {p.name}")
            continue
        tasks.append((num, p))

    logger.info(f"🚀 بدء Auto-Layering على {len(tasks)} صورة (workers={max_workers})")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_worker, t): t[0] for t in tasks}

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
                if res:
                    results[idx] = res
            except Exception as e:
                logger.error(f"خطأ في الصورة رقم {idx}: {e}")

    logger.info(f"🏁 انتهى الـ Layering: {len(results)}/{len(tasks)} نجحت")
    return results


# ============================================================
# اختبار سريع (اختياري)
# ============================================================
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 3:
        print("استخدام: python stage4_layering.py <مجلد_الصور> <مجلد_الإخراج>")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    frames = sorted(
        [p for p in src.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    )
    print(f"وجدت {len(frames)} صورة")

    result = process_layers_batch(frames, dst, max_workers=3)
    print(f"تم إنشاء طبقات لـ {len(result)} صورة في: {dst}")
