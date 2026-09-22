import json
import sys
import traceback
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


def _safe_get(d, *keys, default=None):
    """جلب قيمة متداخلة بأمان دون رفع استثناء."""
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k)
        if current is None:
            return default
    return current


def _print_generation_summary(result, output_file):
    """عرض ملخص نتيجة التوليد في الطرفية."""
    print("\n" + "=" * 55)
    print("✅ تم توليد سكربت المرحلة الأولى بنجاح!")
    print(f"🆔 رقم الحلقة: {result.get('id')}")
    print(f"📌 الموضوع: {result.get('topic')}")
    print(f"📊 إجمالي الكلمات: {result.get('total_word_count')}")
    print(f"🔢 إجمالي الجمل المنفصلة: {len(result.get('full_script_sentences', []) or [])}")

    # حقول جديدة من stage1_generator موجودة داخل creative_brief
    creative_brief = result.get("creative_brief") or {}
    if not isinstance(creative_brief, dict):
        creative_brief = {}

    core_idea = creative_brief.get("core_idea")
    if core_idea is not None:
        print(f"💡 الفكرة الجوهرية (core_idea): {core_idea}")

    unique_angle = creative_brief.get("unique_angle")
    if unique_angle is not None:
        print(f"🎯 الزاوية الفريدة (unique_angle): {unique_angle}")

    episode_format = creative_brief.get("episode_format")
    if episode_format is not None:
        print(f"🎞️  صيغة الحلقة (episode_format): {episode_format}")

    hook = result.get("hook")
    if hook is not None:
        print(f"🪝 الخطاف (hook): {hook}")

    scene_plan = result.get("scene_plan") or []
    print(f"🎬 عدد المشاهد في scene_plan: {len(scene_plan)}")

    approved = _safe_get(result, "quality_report", "approved")
    print(f"🧪 حالة quality_report.approved: {approved}")

    # خطة الاحتفاظ بالمشاهد (retention_plan)
    retention = result.get("retention_plan") or {}
    if isinstance(retention, dict) and retention:
        print("\n--- خطة الاحتفاظ بالمشاهد (retention_plan) ---")
        hook_strategy = retention.get("hook_strategy")
        if hook_strategy is not None:
            print(f"  • hook_strategy: {hook_strategy}")

        open_loops = retention.get("open_loops") or []
        print(f"  • عدد open_loops: {len(open_loops)}")

        pattern_interrupts = retention.get("pattern_interrupts") or []
        print(f"  • عدد pattern_interrupts: {len(pattern_interrupts)}")

        payoff = retention.get("payoff")
        if payoff is not None:
            print(f"  • payoff: {payoff}")

    print(f"\n💾 حُفظ الملف في: {output_file}")
    print("=" * 55)


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # قراءة رقم الحلقة إذا تم تمريره من سطر الأوامر (مثل: python main_stage1.py 101)
    target_id = sys.argv[1] if len(sys.argv) > 1 else None

    # 1) اختيار الحلقة
    try:
        print("🔍 جاري سحب بيانات الحلقة من episodes.json...")
        episode = load_episode(target_id)
    except Exception as e:
        print("❌ فشل في تحميل بيانات الحلقة من episodes.json")
        print(f"   السبب: {e}")
        traceback.print_exc()
        sys.exit(1)

    ep_id = episode.get("id", "unknown")
    print(f"🎬 تم العثور على الحلقة ID [{ep_id}]: '{episode.get('topic')}'")

    # 2) توليد السكربت
    print("\n🚀 بدء تشغيل المرحلة الأولى وتوليد السكربت عبر Gemini...")
    try:
        result = generate_stage1_script(episode)
    except Exception as e:
        print("❌ فشل توليد سكربت المرحلة الأولى عبر stage1_generator.")
        print(f"   السبب: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not isinstance(result, dict):
        print("❌ الناتج من generate_stage1_script ليس كائن JSON صالحًا.")
        sys.exit(1)

    # 3) حفظ الناتج
    output_file = OUTPUTS_DIR / f"stage1_episode_{ep_id}.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("❌ فشل حفظ ملف الناتج JSON.")
        print(f"   المسار: {output_file}")
        print(f"   السبب: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4) عرض الملخص
    _print_generation_summary(result, output_file)


if __name__ == "__main__":
    main()
