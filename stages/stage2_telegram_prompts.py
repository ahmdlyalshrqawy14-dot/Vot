import json
from pathlib import Path
from config.settings import CHARACTER_COLOR

def generate_13_telegram_messages(script_data: dict, metadata_text: str, episode_dir: Path) -> list:
    """
    تقسيم البرومبتات إلى 13 رسالة منفصلة:
    - 5 رسائل لشيتات مجمعة (كل شيت 8 كادرات 4x2 مع ترقيم داخلي واضح).
    - 5 رسائل لكادرات فردية (المشاهد 41 إلى 45 مع ترقيم داخلي).
    - 3 رسائل للغلاف (Thumbnails) مطابقة تماماً لشروط الزوايا الثلاث.
    """
    beats = script_data.get("beats", [])
    messages = []

    # ==========================================
    # 1. الشيتات الخمسة المجمعة (المشاهد 1 إلى 40)
    # ==========================================
    for sheet_idx in range(5):
        start_beat = sheet_idx * 8 + 1
        end_beat = start_beat + 7
        sheet_beats = beats[start_beat - 1 : end_beat]

        panels_text = []
        for idx, b in enumerate(sheet_beats, start=start_beat):
            panels_text.append(
                f"- Panel #{idx:02d}: A distinct square showing the muscular {CHARACTER_COLOR} character in black shorts. "
                f"Action: {b['visual_prompt']}. "
                f"Crucial detail: Draw a bold, high-contrast corner badge with the clear number '#{idx:02d}'."
            )

        panels_block = "\n".join(panels_text)

        prompt_body = (
            f"A professional 2K high-resolution 4x2 grid containing exactly 8 distinct comic panels with clean thin white borders. "
            f"Consistent art style across all panels: 2D cel-shaded athletic character illustration, {CHARACTER_COLOR} smooth skin, "
            f"large oval white eyes, no mouth, muscular build, plain white background, bold dynamic lighting.\n\n"
            f"The 8 panels are arranged as follows:\n"
            f"{panels_block}\n\n"
            f"Every single panel must have its designated number badge clearly visible in its top corner."
        )

        msg = (
            f"📋 <b>الشيت رقم {sheet_idx + 1}/5 (المشاهد من #{start_beat:02d} إلى #{end_beat:02d})</b>\n\n"
            f"انسخ البرومبت التالي لتوليد شيت الـ 8 مربعات بدقة 2K:\n\n"
            f"<code>{prompt_body}</code>"
        )
        messages.append({"type": "sheet", "sheet_id": sheet_idx + 1, "text": msg})

    # ==========================================
    # 2. الكادرات الفردية الخمسة (المشاهد 41 إلى 45)
    # ==========================================
    for b_idx in range(40, len(beats)):
        beat = beats[b_idx]
        b_num = beat["id"]

        single_prompt = (
            f"Cinematic 2D athletic character illustration in clean cel-shaded style. "
            f"The figure is {CHARACTER_COLOR} with a smooth head, large white oval eyes, no mouth, athletic muscular body wearing black shorts. "
            f"Action and emotion: {beat['visual_prompt']}. "
            f"Plain white background, high-contrast studio lighting, sharp 8k details. "
            f"Important: In the top-left corner, render a bold modern badge showing the text '#{b_num:02d}'."
        )

        msg = (
            f"🖼️ <b>المشهد الفردي رقم #{b_num:02d}</b>\n\n"
            f"انسخ البرومبت التالي لإنتاج الصورة الفردية:\n\n"
            f"<code>{single_prompt}</code>"
        )
        messages.append({"type": "single", "beat_id": b_num, "text": msg})

    # ==========================================
    # 3. برومبتات الغلاف الثلاثة (Thumbnails)
    # ==========================================
    thumb_angles = [
        {
            "num": 1,
            "title": "زاوية الصدمة والفضول (Shocking Curiosity)",
            "action": f"Extremely shocked and horrified, holding head with both hands, eyes wide open, muscular {CHARACTER_COLOR} body in black shorts leaning forward with dramatic perspective looking directly at the camera. Highly exaggerated expressive pose visible clearly at tiny sizes."
        },
        {
            "num": 2,
            "title": "زاوية نقطة الألم (Pain-Point Angle)",
            "action": f"Intense agony and frustration, grabbing the hurting joint/muscle tightly, teeth-clenching tension (conveyed through aggressive posture), muscular {CHARACTER_COLOR} character in black shorts bent in pain. Urgent high-contrast drama."
        },
        {
            "num": 3,
            "title": "زاوية النتيجة والقوة (Ultimate Result / Gain)",
            "action": f"Explosive celebration and absolute triumph, flexing both biceps victoriously with raw athletic dominance, glowing energetic aura, muscular {CHARACTER_COLOR} character in black shorts standing tall. Inspiring and powerful presence."
        }
    ]

    for angle in thumb_angles:
        t_prompt = (
            f"Create a 2D muscular character illustration in the exact signature style. "
            f"The figure must be {CHARACTER_COLOR}, with a smooth head, large white oval eyes, no mouth, and a defined muscular body wearing black shorts. "
            f"{angle['action']} "
            f"Background must be plain white. Clean, cel-shaded, and hyper-expressive style. "
            f"Ultra-clear dynamic silhouette designed to be instantly recognizable on small mobile thumbnail screens."
        )

        msg = (
            f"🎨 <b>برومبت الغلاف رقم {angle['num']}/3 - {angle['title']}</b>\n\n"
            f"⚠️ <i>ملاحظة: هذه الصورة مخصصة للغلاف ولن تدخل في مونتاج الفيديو.</i>\n\n"
            f"<code>{t_prompt}</code>"
        )
        messages.append({"type": "thumbnail", "thumb_id": angle['num'], "text": msg})

    # حفظ الرسائل في ملف نصي للرجوع إليها في أي وقت
    prompts_cache_file = episode_dir / "telegram_prompts.json"
    with open(prompts_cache_file, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)

    return messages
