import os
import sys
import time
import requests
from io import BytesIO
from PIL import Image
from openai import AzureOpenAI

# 1. أوامر مشاهد الفيديو الـ 45
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

# 2. برومبتات الأغلفة الـ 3 (مخصصة لجذب الانتباه ومطابقة لمعايير يوتيوب الاحترافية)
THUMBNAIL_CONFIG = {
    "thumb_curiosity": (
        "YouTube master thumbnail composition, high visual impact, rule of thirds. "
        "Athletic orange humanoid character on the right side leaning forward with hand on chin, "
        "staring with intense scientific curiosity and shock at a giant glowing holographic knee joint diagram floating on the left. "
        "High contrast, cinematic clean 2D vector cel-shaded style, ample empty negative space on top for bold title text, "
        "pure solid white background, vibrant saturated orange skin, strictly NO mouth, NO nose, solid black gym shorts."
    ),
    "thumb_pain_point": (
        "High-CTR YouTube thumbnail composition, dramatic contrast. "
        "Athletic orange humanoid character in sudden sports distress, kneeling and clasping its knees with expressive body language. "
        "Dramatic red glowing warning aura and lightning energy radiating directly from the kneecap joint. "
        "Clean minimal 2D vector cel-shaded art, bold thick outlines, pure seamless white background, "
        "large empty negative space for headline text. Strictly NO mouth, NO nose, black shorts."
    ),
    "thumb_outcome_gain": (
        "Heroic high-conversion YouTube thumbnail. "
        "Athletic orange muscular character in an explosive, flawless deep squat stance, flexing both biceps in a triumphant victory pose. "
        "Brilliant golden solar flare energy bursting behind, sparkling diamond-solid knee joints indicating invincible bulletproof joints. "
        "Ultra-crisp 2D vector animation style, high saturation, pure white background, empty space for title overlay. Strictly NO mouth, NO nose."
    )
}

CHAR_DNA = (
    "2D modern cel-shaded animation style. Athletic orange humanoid figure, "
    "vibrant orange skin, completely smooth round bald head, large expressive white oval eyes. "
    "Strictly NO mouth, NO nose, NO facial hair. Defined muscular build, solid black athletic workout shorts. "
    "Pure solid seamless white background (#FFFFFF), clean bold outlines, flat minimal studio lighting."
)

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
            print(f"[WARN] API call attempt {attempt}/{max_retries} failed: {e}")
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
            
            target_path = f"images/timeline/{current_idx:03d}.png"
            cell.save(target_path, "PNG", quality=95)
            print(f"[SLICED] Video Scene {current_idx:03d} -> {target_path}")
            current_idx += 1

def run_timeline_pipeline(client, deployment):
    """توليد مشاهد الفيديو الـ 45 (11 لوحة مجمعة + مشهد 45 منفصل)"""
    print("\n--- [STAGE 1] Generating Video Timeline Frames (001 to 045) ---")
    
    # 11 دفعة رباعية للمشاهد من 1 إلى 44
    batches = [(i, i + 3) for i in range(1, 45, 4)]

    for batch_num, (start_idx, end_idx) in enumerate(batches, 1):
        already_done = all(is_valid_image(f"images/timeline/{i:03d}.png") for i in range(start_idx, end_idx + 1))
        if already_done:
            print(f"[SKIP] Video Batch {batch_num}/11 (Scenes {start_idx:02d}-{end_idx:02d}) complete.")
            continue

        print(f"[REQUEST] Generating Video Grid {batch_num}/11 (Scenes {start_idx:02d} to {end_idx:02d})...")
        panels_desc = " ".join([f"Panel {i:02d}: {TIMELINE_PROMPTS[i]}." for i in range(start_idx, end_idx + 1)])
        
        grid_prompt = (
            f"A 2D animation storyboard contact sheet strictly containing exactly 4 equal rectangular panels in a clean 2x2 grid (2 rows, 2 columns). "
            f"Solid seamless white background (#FFFFFF), thin dark borders between panels. {CHAR_DNA} Action Panels: {panels_desc}"
        )

        sheet_img = generate_with_retry(client, deployment, grid_prompt, size="1024x1024")
        if sheet_img:
            slice_2x2_sheet(sheet_img, start_idx, end_idx)
        else:
            print(f"[FAIL] Batch {batch_num} failed. Left for Fallback Worker.")

        time.sleep(4)

    # المشهد رقم 45 (يولد مفرداً بدقة 16:9 مباشرة)
    path_45 = "images/timeline/045.png"
    if not is_valid_image(path_45):
        print("[REQUEST] Generating Final Video Frame (Scene 045) directly in 16:9...")
        prompt_45 = f"{CHAR_DNA} Action pose: The character is {TIMELINE_PROMPTS[45]}. 16:9 widescreen composition."
        img_45 = generate_with_retry(client, deployment, prompt_45, size="1024x1024")
        if img_45:
            img_45 = img_45.resize((1280, 720), Image.Resampling.LANCZOS)
            img_45.save(path_45, "PNG", quality=95)
            print(f"[SAVED] Scene 045 -> {path_45}")
        time.sleep(4)

def run_thumbnails_pipeline(client, deployment):
    """توليد الأغلفة الـ 3 المستقلة بأعلى دقة واحترافية تسويقية"""
    print("\n--- [STAGE 2] Generating Standalone 16:9 Master Thumbnails (3 Variants) ---")

    for thumb_name, thumb_prompt in THUMBNAIL_CONFIG.items():
        out_path = f"images/thumbnails/{thumb_name}.png"
        if is_valid_image(out_path):
            print(f"[SKIP] Thumbnail '{thumb_name}' already exists and valid.")
            continue

        print(f"[REQUEST] Generating High-Impact Master Thumbnail: {thumb_name}...")
        thumb_img = generate_with_retry(client, deployment, thumb_prompt, size="1024x1024")
        if thumb_img:
            thumb_img = thumb_img.resize((1280, 720), Image.Resampling.LANCZOS)
            thumb_img.save(out_path, "PNG", quality=95)
            print(f"[SAVED] Thumbnail created -> {out_path}")
        else:
            print(f"[FAIL] Failed to generate thumbnail: {thumb_name}")

        time.sleep(4)

def run_fallback_worker(client, deployment):
    """المرحلة 3: فحص أمان صارم يعوض أي مشهد مفقود فردياً"""
    print("\n--- [STAGE 3] Security Integrity Check & Fallback Worker ---")
    
    # فحص مشاهد التايم لاين
    missing_timeline = [i for i in range(1, 46) if not is_valid_image(f"images/timeline/{i:03d}.png")]
    if missing_timeline:
        print(f"[FALLBACK] Recovering {len(missing_timeline)} missing timeline frames: {missing_timeline}")
        for idx in missing_timeline:
            path = f"images/timeline/{idx:03d}.png"
            prompt = f"{CHAR_DNA} Action pose: The character is {TIMELINE_PROMPTS[idx]}. 16:9 widescreen composition."
            img = generate_with_retry(client, deployment, prompt, size="1024x1024")
            if img:
                img = img.resize((1280, 720), Image.Resampling.LANCZOS)
                img.save(path, "PNG", quality=95)
                print(f"[RECOVERED] Scene {idx:03d} -> {path}")
            time.sleep(5)

    # فحص الأغلفة
    for thumb_name, thumb_prompt in THUMBNAIL_CONFIG.items():
        path = f"images/thumbnails/{thumb_name}.png"
        if not is_valid_image(path):
            print(f"[FALLBACK] Recovering missing thumbnail: {thumb_name}...")
            img = generate_with_retry(client, deployment, thumb_prompt, size="1024x1024")
            if img:
                img = img.resize((1280, 720), Image.Resampling.LANCZOS)
                img.save(path, "PNG", quality=95)
            time.sleep(5)

def main():
    os.makedirs("images/timeline", exist_ok=True)
    os.makedirs("images/thumbnails", exist_ok=True)

    client, deployment = init_azure_client()

    # 1. إنتاج مشاهد الفيديو الـ 45 بنظام 2x2
    run_timeline_pipeline(client, deployment)

    # 2. إنتاج الأغلفة الـ 3 الاحترافية المنفصلة
    run_thumbnails_pipeline(client, deployment)

    # 3. التحقق الاحتياطي النهائي وضمان اكتمال كل ملف
    run_fallback_worker(client, deployment)

    valid_timeline = sum(1 for i in range(1, 46) if is_valid_image(f"images/timeline/{i:03d}.png"))
    valid_thumbs = sum(1 for t in THUMBNAIL_CONFIG if is_valid_image(f"images/thumbnails/{t}.png"))

    print(f"\n==========================================")
    print(f"STATUS: Timeline Frames: {valid_timeline}/45 | Thumbnails: {valid_thumbs}/3")
    print(f"Total Ready Assets: {valid_timeline + valid_thumbs}/48")
    print(f"==========================================")

    if valid_timeline < 45 or valid_thumbs < 3:
        sys.exit(1)

if __name__ == "__main__":
    main()
