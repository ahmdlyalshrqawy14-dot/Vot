import json
from pathlib import Path
from config.settings import CHARACTER_COLOR

def generate_google_flow_files(script_data: dict, metadata_text: str, episode_dir: Path) -> list:
    beats = script_data.get("beats", [])
    files_dir = episode_dir / "prompts_txt"
    files_dir.mkdir(parents=True, exist_ok=True)

    base_template = (
        "Create a 2D muscular character illustration in the exact style of the provided example. "
        f"The figure should be {CHARACTER_COLOR}, with a smooth head, large white oval eyes, no mouth, "
        "and a defined muscular body wearing black shorts. Maintain the same proportions and facial features "
        "as in the reference. {action} "
        "In the bottom-right corner of the image, render a small, clean, unobtrusive badge text '#{b_id:02d}'. "
        "Background must be plain white. Clean, cel-shaded, and expressive style. "
        "Keep the style consistent across all generated images."
    )

    output_packages = []

    # دفعة 1 (01 إلى 20)
    b1_lines = [base_template.format(action=b.get("visual_prompt", "athletic pose").replace("\n", " ").strip(), b_id=b["id"]) for b in beats[0:20]]
    file_b1 = files_dir / "01_Batch_1_Beats_01_to_20.txt"
    file_b1.write_text("\n".join(b1_lines), encoding="utf-8")
    output_packages.append({"path": file_b1, "caption": "📄 <b>الدفعة الأولى: المشاهد (01 إلى 20)</b>\n20 سطراً لـ Google Flow."})

    # دفعة 2 (21 إلى 40)
    b2_lines = [base_template.format(action=b.get("visual_prompt", "athletic pose").replace("\n", " ").strip(), b_id=b["id"]) for b in beats[20:40]]
    file_b2 = files_dir / "02_Batch_2_Beats_21_to_40.txt"
    file_b2.write_text("\n".join(b2_lines), encoding="utf-8")
    output_packages.append({"path": file_b2, "caption": "📄 <b>الدفعة الثانية: المشاهد (21 إلى 40)</b>\n20 سطراً لـ Google Flow."})

    # دفعة 3 (41 إلى 45)
    b3_lines = [base_template.format(action=b.get("visual_prompt", "athletic pose").replace("\n", " ").strip(), b_id=b["id"]) for b in beats[40:45]]
    file_b3 = files_dir / "03_Batch_3_Beats_41_to_45.txt"
    file_b3.write_text("\n".join(b3_lines), encoding="utf-8")
    output_packages.append({"path": file_b3, "caption": "📄 <b>الدفعة الثالثة: المشاهد (41 إلى 45)</b>\n5 أسطر لـ Google Flow."})

    # الأغلفة
    thumbs = [
        ("Shocking Curiosity", f"Extremely shocked and horrified, hands on head, wide eyes, {CHARACTER_COLOR} muscular character leaning towards camera."),
        ("Pain-Point Angle", f"Intense agony, tightly clutching knee joint, dynamic strain, {CHARACTER_COLOR} muscular character in black shorts."),
        ("Ultimate Result", f"Triumphant victory pose flexing biceps, muscular dominance, {CHARACTER_COLOR} muscular character standing tall.")
    ]
    t_lines = [f"[{title}]\nYouTube Clickable Thumbnail, 16:9, 2D style, {CHARACTER_COLOR} muscular character in black shorts. {act} Plain white background." for title, act in thumbs]
    file_t = files_dir / "04_Thumbnails.txt"
    file_t.write_text("\n\n".join(t_lines), encoding="utf-8")
    output_packages.append({"path": file_t, "caption": "🖼️ <b>برومبتات الأغلفة الثلاثة (Thumbnails)</b>"})

    return output_packages\n