from pathlib import Path
from PIL import Image

def slice_8_grid(sheet_path: Path, start_idx: int, output_dir: Path) -> list:
    """
    تقطيع صورة شيت 2K مكونة من 8 كادرات (صفين × 4 أعمدة)
    وحفظها كصور مستقلة بالترقيم الصحيح (مثلاً من 1 إلى 8)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    with Image.open(sheet_path) as img:
        sheet_w, sheet_h = img.size

        # أبعاد المربع الواحد (شبكة 4 أعمدة × صفين)
        cols = 4
        rows = 2
        cell_w = sheet_w // cols
        cell_h = sheet_h // rows

        current_id = start_idx
        for r in range(rows):
            for c in range(cols):
                left = c * cell_w
                top = r * cell_h
                right = left + cell_w
                bottom = top + cell_h

                # قص الكادر
                cell_crop = img.crop((left, top, right, bottom))

                # حفظ الصورة باسم الكادر المناظر للصوت
                out_name = output_dir / f"beat_{current_id:03d}.png"
                cell_crop.save(out_name, format="PNG", quality=95)
                saved_paths.append(out_name)
                
                current_id += 1

    return saved_paths

def save_single_frame(image_path: Path, beat_id: int, output_dir: Path) -> Path:
    """حفظ الصورة الفردية المباشرة (للمشاهد من 41 إلى 45)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = output_dir / f"beat_{beat_id:03d}.png"
    
    with Image.open(image_path) as img:
        img.save(out_name, format="PNG", quality=95)
        
    return out_name
