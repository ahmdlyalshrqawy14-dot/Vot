import json
import sys
from pathlib import Path
from stage2_generator import generate_stage2_prompts_batches

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_stage1_script(episode_id: str):
    """البحث عن ملف مخرجات المرحلة الأولى للحلقة المحددة"""
    # البحث المباشر باسم الملف المحدد
    stage1_file = OUTPUTS_DIR / f"stage1_episode_{episode_id}.json"
    if stage1_file.exists():
        with open(stage1_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    # في حال لم يحدد ID، البحث عن أول ملف متاح للمرحلة الأولى
    available_files = list(OUTPUTS_DIR.glob("stage1_episode_*.json"))
    if available_files:
        with open(available_files[0], "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError(f"لم يتم العثور على ملف المرحلة الأولى في: {OUTPUTS_DIR}")


def main():
    target_id = sys.argv[1] if len(sys.argv) > 1 else None

    print("🔍 جاري قراءة سكربت المرحلة الأولى...")
    stage1_data = load_stage1_script(target_id)
    ep_id = stage1_data.get("id", "201")
    sentences = stage1_data.get("full_script_sentences", [])

    if not sentences:
        print("❌ خطأ: مصفوفة 'full_script_sentences' فارغة!")
        return

    print(f"🎬 الحلقة ID [{ep_id}]: إجمالي الجمل المطلوب توليد صور لها = {len(sentences)}")

    # إنشاء مجلد مخصص لأوامر الصور الخاصة بالحلقة
    prompts_dir = OUTPUTS_DIR / f"episode_{ep_id}_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    print("\n🚀 بدء توليد أوامر الصور وتجزئتها آلياً...")
    batches = generate_stage2_prompts_batches(sentences)

    total_prompts = sum(len(b) for b in batches)
    print("\n💾 جاري حفظ ملفات الدفعات (.txt)...")

    for idx, batch_prompts in enumerate(batches, start=1):
        filename = prompts_dir / f"prompts_part_{idx:02d}.txt"
        # فاصل سطر فارغ واحد فقط بين كل برومبت والآخر بدون أي نصوص إضافية
        content = "\n\n".join(batch_prompts)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"   📄 تم حفظ {filename.name} ({len(batch_prompts)} برومبت)")

    print("\n" + "=" * 55)
    print("✅ اكتملت المرحلة الثانية بنجاح!")
    print(f"🆔 رقم الحلقة: {ep_id}")
    print(f"📊 إجمالي جمل السكربت: {len(sentences)}")
    print(f"🖼️ إجمالي أوامر الصور المولدة: {total_prompts}")
    print(f"📦 عدد ملفات الدفعات (حد أقصى 24 لكل ملف): {len(batches)}")
    print(f"📁 مسار الملفات: {prompts_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
