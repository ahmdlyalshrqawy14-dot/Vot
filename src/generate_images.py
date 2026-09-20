import os
import sys
import time
import requests
from io import BytesIO
from PIL import Image
from openai import AzureOpenAI

# 1. أوامر مشاهد شريط الفيديو الـ 45 فقط (مأخوذة من حركات السكربت)
TIMELINE_PROMPTS = {
    1: "pointing an accusing finger forward with a questioning posture",
    2: "arms folded tightly across chest, shaking head in firm disapproval",
    3: "holding a heavy sledgehammer/mallet raised, ready to strike downward",
    4: "standing proud, flexing defined leg muscles with confidence",
    5: "giving a double thumbs-up with an encouraging nod",
    6: "holding a small classic pink piggy bank carefully with both hands",
    7: "happily juggling three glowing gold coins in the air",
    8: "holding a sturdy metallic defensive shield firmly in front of chest",
    9: "raising one hand in a worried open-palm stop warning gesture",
    10: "demonstrating a smooth, graceful, controlled deep squat descent",
    11: "looking down at knees with a worried, anxious expression",
    12: "peering closely through a large magnifying glass with intense curiosity",
    13: "standing in a rock-solid, wide-stance balanced athletic pose",
    14: "pointing directly at inner quadriceps muscle with focused attention",
    15: "crossing arms tightly over chest mimicking a locked seatbelt buckle",
    16: "halting abruptly mid-squat with wide, startled eyes",
    17: "tapping temple thoughtfully next to a floating glowing brain icon",
    18: "measuring growing bicep with a tailor's tape measure",
    19: "bouncing a basketball with energetic athletic movement",
    20: "standing relaxed while clean sparkle effects float around joints",
    21: "pointing a firm warning finger toward knee and quadriceps",
    22: "standing tall and heroic with an imaginary flowing cape pose",
    23: "classic frustrated facepalm gesture with one hand over face",
    24: "holding a large floating question mark sign overhead",
    25: "leaning forward expectantly with an intense, curious gaze",
    26: "holding a wrench and a hammer in a ready-to-work posture",
    27: "demonstrating a clean biomechanical hip hinge movement pattern",
    28: "pointing downward at feet to emphasize knee-over-toe alignment",
    29: "raising both hands in a strict stop sign gesture",
    30: "stacking small barbell weight plates carefully one by one",
    31: "pointing toward a wall calendar graphic to emphasize patience",
    32: "running hurriedly with an anxious and hurried expression",
    33: "moving fluidly through a complete deep squat movement cycle",
    34: "holding a drafting compass drawing a clean geometric arc",
    35: "looking down into a small hole in the ground with a disappointed frown",
    36: "closing an imaginary padlock with a firm decisive nod",
    37: "deep full squat at bottom range in a powerful confident stance",
    38: "stretching arms out wide as radiant glowing light bursts around",
    39: "pointing enthusiastically toward a ringing notification bell graphic",
    40: "waving hello warmly with a friendly welcoming hand gesture",
    41: "pointing downward toward an empty chat bubble graphic",
    42: "tilting head in deep reflection with a questioning chin hand gesture",
    43: "gesturing downward invitingly toward the comments section",
    44: "typing actively and enthusiastically on an imaginary keyboard",
    45: "bowing slightly in respectful gratitude with open arms"
}

# 2. البرومبت الأساسي الحرفي طبقاً للدليل (يعدل اللون والحركة فقط)
BASE_PROMPT_TEMPLATE = (
    "Create a 2D muscular character illustration in the exact style of the provided example. "
    "The figure should be orange, with a smooth head, large white oval eyes, no mouth, "
    "and a defined muscular body wearing black shorts. "
    "Maintain the same proportions and facial features as in the reference. "
    "{pose} "
    "Background must be plain white. Clean, cel-shaded, and expressive style. "
    "Keep the style consistent across all generated images."
)

def get_target_path(index):
    return f"images/timeline/{index:03d}.png"

def is_valid_image(file_path):
    if not file_path or not os.path.exists(file_path):
        return False
    if os.path.getsize(file_path) < 10240:
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def init_azure_client():
    endpoint = os.getenv("AZURE_IMAGE_ENDPOINT")
    api_key = os.getenv("AZURE_IMAGE_KEY")
    deployment = os.getenv("AZURE_IMAGE_DEPLOYMENT", "gpt-image-2.5-flare")

    if not endpoint or not api_key:
        print("[ERROR] Missing AZURE_IMAGE_ENDPOINT or AZURE_IMAGE_KEY.")
        sys.exit(1)

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2024-02-01"
    )
    return client, deployment

def generate_with_retry(client, deployment, prompt, size="1024x1024", max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            response = client.images.generate(
                model=deployment,
                prompt=prompt,
                size=size,
                quality="hd",
                n=1
            )
            image_url = response.data[0].url
            resp = requests.get(image_url, timeout=60)
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"[WARN] API attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                backoff_time = attempt * 5
                print(f"[SAFETY] Cooling down for {backoff_time}s...")
                time.sleep(backoff_time)
    return None

def slice_2x2_sheet(sheet_img, start_idx, end_idx):
    sheet_w, sheet_h = sheet_img.size
    cols = 2
    rows = 2
    cell_w = sheet_w / cols
    cell_h = sheet_h / rows

    pad_x = int(cell_w * 0.03)
    pad_y = int(cell_h * 0.04)

    current_idx = start_idx
    for r in range(rows):
        for c in range(cols):
            if current_idx > end_idx:
                break

            left = int(c * cell_w) + pad_x
            top = int(r * cell_h) + pad_y
            right = int((c + 1) * cell_w) - pad_x
            bottom = int((r + 1) * cell_h) - pad_y

            cell = sheet_img.crop((left, top, right, bottom))
            cell = cell.resize((1280, 720), Image.Resampling.LANCZOS)
            
            target_path = get_target_path(current_idx)
            cell.save(target_path, "PNG", quality=95)
            print(f"[SLICED] Scene {current_idx:03d} -> {target_path}")
            current_idx += 1

def run_timeline_pipeline(client, deployment):
    """توليد 11 لوحة مجمعة (كل لوحة 4 مشاهد) بإجمالي 44 مشهداً + المشهد 45 منفرداً"""
    print("\n--- [STAGE 1] Generating Timeline Grids (Scenes 001 to 044) ---")
    
    # 11 دفعة رباعية
    batches = [(i, i + 3) for i in range(1, 45, 4)]

    for batch_num, (start_idx, end_idx) in enumerate(batches, 1):
        already_done = all(is_valid_image(get_target_path(i)) for i in range(start_idx, end_idx + 1))
        if already_done:
            print(f"[SKIP] Batch {batch_num}/11 (Scenes {start_idx:02d}-{end_idx:02d}) complete.")
            continue

        print(f"[REQUEST] Generating Grid {batch_num}/11 (Scenes {start_idx:02d} to {end_idx:02d})...")
        panels_desc = " ".join([f"Panel {i:02d}: {TIMELINE_PROMPTS[i]}." for i in range(start_idx, end_idx + 1)])
        
        grid_prompt = (
            f"A 2D animation storyboard sheet strictly containing exactly 4 equal rectangular panels in a clean 2x2 grid. "
            f"Solid white background (#FFFFFF), thin dark borders between panels. "
            f"Orange muscular humanoid character, smooth bald head, white oval eyes, no mouth, wearing black shorts. "
            f"Panels: {panels_desc}"
        )

        sheet_img = generate_with_retry(client, deployment, grid_prompt, size="1024x1024")
        if sheet_img:
            slice_2x2_sheet(sheet_img, start_idx, end_idx)
        else:
            print(f"[FAIL] Batch {batch_num} failed. Left for Fallback Worker.")

        time.sleep(4)

    # المشهد رقم 45 يُولد منفرداً بالبرومبت الأساسي الحرفي
    path_45 = get_target_path(45)
    if not is_valid_image(path_45):
        print("\n[REQUEST] Generating Final Scene 045 individually...")
        prompt_45 = BASE_PROMPT_TEMPLATE.format(pose=f"{TIMELINE_PROMPTS[45]}.")
        img_45 = generate_with_retry(client, deployment, prompt_45, size="1024x1024")
        if img_45:
            img_45 = img_45.resize((1280, 720), Image.Resampling.LANCZOS)
            img_45.save(path_45, "PNG", quality=95)
            print(f"[SAVED] Scene 045 -> {path_45}")
        time.sleep(4)

def run_fallback_worker(client, deployment):
    """المرحلة 2: الفحص الصارم وتوليد أي مشهد ناقص فردياً بالبرومبت الأساسي الحرفي"""
    print("\n--- [STAGE 2] Timeline Integrity Check & Fallback Worker ---")
    
    missing_timeline = [i for i in range(1, 46) if not is_valid_image(get_target_path(i))]
    if not missing_timeline:
        print("[SUCCESS] All 45 video timeline frames are complete and valid!")
        return

    print(f"[FALLBACK] Recovering {len(missing_timeline)} missing frames individually: {missing_timeline}")
    for count, idx in enumerate(missing_timeline, 1):
        target_path = get_target_path(idx)
        # استخدام البرومبت الأساسي الحرفي من الدليل
        single_prompt = BASE_PROMPT_TEMPLATE.format(pose=f"{TIMELINE_PROMPTS[idx]}.")

        print(f"[FALLBACK {count}/{len(missing_timeline)}] Generating Frame {idx:03d}...")
        img = generate_with_retry(client, deployment, single_prompt, size="1024x1024")
        if img:
            img = img.resize((1280, 720), Image.Resampling.LANCZOS)
            img.save(target_path, "PNG", quality=95)
            print(f"[RECOVERED] Scene {idx:03d} -> {target_path}")
        else:
            print(f"[CRITICAL] Frame {idx:03d} failed after retries!")

        time.sleep(5)

def main():
    # التأكد من وجود مجلد مشاهد الفيديو فقط
    os.makedirs("images/timeline", exist_ok=True)

    client, deployment = init_azure_client()

    # 1. توليد المشاهد عبر الشبكات الرباعية
    run_timeline_pipeline(client, deployment)

    # 2. خطة الأمان الفردية
    run_fallback_worker(client, deployment)

    # التحقق النهائي
    total_valid = sum(1 for i in range(1, 46) if is_valid_image(get_target_path(i)))
    print(f"\n==========================================")
    print(f"TIMELINE READY: {total_valid}/45 frames ready in images/timeline/")
    print(f"==========================================")

    if total_valid < 45:
        sys.exit(1)

if __name__ == "__main__":
    main()
