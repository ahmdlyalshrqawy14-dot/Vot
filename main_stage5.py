import json
import sys
from pathlib import Path
from stage5_metadata import generate_stage5_metadata

BASE_DIR = Path(__file__).resolve().parent
EPISODES_FILE = BASE_DIR / "episodes.json"
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_episode_and_script(target_id: str):
    """جلب بيانات الحلقة من episodes.json والسكربت من مخرجات المرحلة الأولى"""
    if not EPISODES_FILE.exists():
        raise FileNotFoundError(f"ملف الحلقات غير موجود: {EPISODES_FILE}")

    with open(EPISODES_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    episodes = raw if isinstance(raw, list) else raw.get("episodes", [])

    matched_ep = None
    for ep in episodes:
        if str(ep.get("id")) == str(target_id):
            matched_ep = ep
            break

    if not matched_ep:
        matched_ep = episodes[0]
        target_id = str(matched_ep.get("id", "201"))

    # قراءة مخرجات المرحلة الأولى
    stage1_file = OUTPUTS_DIR / f"stage1_episode_{target_id}.json"
    sentences = []
    if stage1_file.exists():
        with open(stage1_file, "r", encoding="utf-8") as f:
            sentences = json.load(f).get("full_script_sentences", [])

    return matched_ep, sentences


def main():
    target_id = sys.argv[1] if len(sys.argv) > 1 else "201"
    print(f"🔍 جاري قراءة بيانات الحلقة [{target_id}]...")

    episode_data, script_sentences = load_episode_and_script(target_id)
    ep_id = episode_data.get("id", target_id)

    print(f"🎬 موضوع الحلقة: '{episode_data.get('topic')}'")
    print(f"📝 جمل السكربت المتوفرة: {len(script_sentences)}")

    print("\n🚀 توليد ميتاداتا النشر الرقمي عبر Gemini...")
    metadata = generate_stage5_metadata(episode_data, script_sentences)

    # حفظ كائن البيانات في outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUTS_DIR / f"stage5_episode_{ep_id}_metadata.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print("✅ اكتملت المرحلة الخامسة بنجاح! معاينة رسائل التليجرام الأربعة:")
    print("=" * 65)

    print("\n--- [الرسالة الأولى: العناوين] ---")
    print(metadata.get("titles_message"))

    print("\n--- [الرسالة الثانية: أوامر الغلاف] ---")
    print(metadata.get("thumbnails_message"))

    print("\n--- [الرسالة الثالثة: الوصف] ---")
    print(metadata.get("description_message"))

    print("\n--- [الرسالة الرابعة: التاجز] ---")
    print(metadata.get("tags_message"))

    print("\n" + "=" * 65)
    print(f"💾 تم حفظ مخرجات المرحلة في: {out_file}")
    print("=" * 65)


if __name__ == "__main__":
    main()
