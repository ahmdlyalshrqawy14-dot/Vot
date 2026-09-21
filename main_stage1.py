import json
import sys
from pathlib import Path
from stage1_generator import generate_stage1_script

BASE_DIR = Path(__file__).resolve().parent
EPISODES_FILE = BASE_DIR / "episodes.json"
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_episode(episode_id=None):
    if not EPISODES_FILE.exists():
        raise FileNotFoundError(f"ملف الحلقات غير موجود: {EPISODES_FILE}")

    with open(EPISODES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # معالجة الملف سواء كان مصفوفة أو كائن يحتوي قائمة
    episodes = data if isinstance(data, list) else data.get("episodes", [])

    if not episodes:
        raise ValueError("ملف episodes.json لا يحتوي على أي حلقات!")

    if episode_id is not None:
        for ep in episodes:
            if str(ep.get("id")) == str(episode_id):
                return ep
        raise ValueError(f"لم يتم العثور على الحلقة برقم ID: {episode_id}")

    # إذا لم يُحدد ID، سحب أول حلقة بحالة pending
    for ep in episodes:
        if ep.get("status") == "pending":
            return ep

    # إذا لم توجد حلقة pending، سحب أول حلقة في الملف كافتراضي
    return episodes[0]


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # قراءة رقم الحلقة إذا تم تمريره من سطر الأوامر (مثل: python main_stage1.py 101)
    target_id = sys.argv[1] if len(sys.argv) > 1 else None

    print("🔍 جاري سحب بيانات الحلقة من episodes.json...")
    episode = load_episode(target_id)
    ep_id = episode.get("id", "unknown")
    print(f"🎬 تم العثور على الحلقة ID [{ep_id}]: '{episode.get('topic')}'")

    print("\n🚀 بدء تشغيل المرحلة الأولى وتوليد السكربت عبر Gemini...")
    result = generate_stage1_script(episode)

    output_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 55)
    print("✅ تم توليد سكربت المرحلة الأولى بنجاح!")
    print(f"🆔 رقم الحلقة: {result.get('id')}")
    print(f"📌 الموضوع: {result.get('topic')}")
    print(f"📊 إجمالي الكلمات: {result.get('total_word_count')}")
    print(f"🔢 إجمالي الجمل المنفصلة: {len(result.get('full_script_sentences', []))}")
    print(f"💾 حُفظ الملف في: {output_file}")
    print("=" * 55)


if __name__ == "__main__":
    main()
