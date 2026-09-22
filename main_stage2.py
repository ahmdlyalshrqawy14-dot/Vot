import json
import sys
from pathlib import Path

from stage2_generator import generate_stage2_prompts_batches


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_stage1_script(episode_id: str | None = None) -> dict:
    """
    تحميل مخرجات المرحلة الأولى ككائن كامل (Dict).

    - إذا تم تمرير episode_id: يُقرأ outputs/stage1_episode_{episode_id}.json
    - إذا لم يُمرَّر: يُستخدم أول ملف outputs/stage1_episode_*.json
    - إذا لم توجد أي ملفات: يُرفع FileNotFoundError واضح.
    """
    if episode_id:
        stage1_file = OUTPUTS_DIR / f"stage1_episode_{episode_id}.json"
        if not stage1_file.exists():
            raise FileNotFoundError(
                f"لم يتم العثور على ملف المرحلة الأولى للحلقة [{episode_id}]: {stage1_file}"
            )
        try:
            with open(stage1_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"فشل تحليل JSON في ملف المرحلة الأولى: {stage1_file}\nالتفاصيل: {e}"
            ) from e
        except (OSError, PermissionError) as e:
            raise OSError(
                f"فشل قراءة ملف المرحلة الأولى: {stage1_file}\nالتفاصيل: {e}"
            ) from e
        if not isinstance(data, dict):
            raise ValueError(
                f"محتوى ملف المرحلة الأولى ليس كائن JSON صالحًا: {stage1_file}"
            )
        return data

    # لا يوجد ID: ابحث عن أول ملف متاح
    available_files = sorted(OUTPUTS_DIR.glob("stage1_episode_*.json"))
    if not available_files:
        raise FileNotFoundError(
            f"لم يتم العثور على أي ملفات للمرحلة الأولى في: {OUTPUTS_DIR}"
        )

    first_file = available_files[0]
    try:
        with open(first_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"فشل تحليل JSON في ملف المرحلة الأولى: {first_file}\nالتفاصيل: {e}"
        ) from e
    except (OSError, PermissionError) as e:
        raise OSError(
            f"فشل قراءة ملف المرحلة الأولى: {first_file}\nالتفاصيل: {e}"
        ) from e
    if not isinstance(data, dict):
        raise ValueError(
            f"محتوى ملف المرحلة الأولى ليس كائن JSON صالحًا: {first_file}"
        )
    return data


def _safe_len(value) -> int:
    """إرجاع الطول بأمان إذا كانت القيمة قابلة للقياس، وإلا 0."""
    try:
        return len(value) if value is not None else 0
    except TypeError:
        return 0


def main():
    target_id = sys.argv[1] if len(sys.argv) > 1 else None

    print("🔍 جاري قراءة سكربت المرحلة الأولى...")

    # 1) تحميل كائن المرحلة الأولى كاملًا
    try:
        stage1_result = load_stage1_script(target_id)
    except (FileNotFoundError, OSError, PermissionError, ValueError) as e:
        print(f"❌ خطأ أثناء تحميل مخرجات المرحلة الأولى: {e}")
        return

    if not isinstance(stage1_result, dict):
        print("❌ خطأ: مخرجات المرحلة الأولى ليست كائن JSON صالحًا.")
        return

    # 2) استخراج المعرّف والجمل
    ep_id = stage1_result.get("id") or stage1_result.get("episode_id") or "unknown"

    sentences = stage1_result.get("full_script_sentences", [])
    if not isinstance(sentences, list) or len(sentences) == 0:
        print("❌ خطأ: مصفوفة 'full_script_sentences' غير موجودة أو فارغة أو ليست قائمة.")
        print("   لن تبدأ المرحلة الثانية، ولن يتم إنشاء أي ملفات Prompts.")
        return

    # 3) بيانات سياقية للعرض فقط (قراءة آمنة)
    scene_plan = stage1_result.get("scene_plan")
    visual_bible = stage1_result.get("visual_bible")
    creative_brief = stage1_result.get("creative_brief")

    has_scene_plan = isinstance(scene_plan, (list, dict)) and len(scene_plan) > 0
    has_visual_bible = bool(visual_bible)
    has_creative_brief = bool(creative_brief)
    scene_plan_count = _safe_len(scene_plan)

    print(f"🎬 الحلقة ID [{ep_id}]: إجمالي الجمل المطلوب توليد صور لها = {len(sentences)}")
    print(f"🧩 scene_plan: {'موجود' if has_scene_plan else 'غير موجود'} "
          f"({scene_plan_count} مشهد)")
    print(f"🎨 visual_bible: {'موجود' if has_visual_bible else 'غير موجود'}")
    print(f"📝 creative_brief: {'موجود' if has_creative_brief else 'غير موجود'}")

    # 4) إنشاء مجلد مخصص لأوامر الصور الخاصة بالحلقة
    prompts_dir = OUTPUTS_DIR / f"episode_{ep_id}_prompts"
    try:
        prompts_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        print(f"❌ خطأ أثناء إنشاء مجلد الإخراج: {prompts_dir}\nالتفاصيل: {e}")
        return

    # 5) توليد أوامر الصور (تمرير stage1_result كامل)
    print("\n🚀 بدء توليد أوامر الصور وتجزئتها آلياً...")
    try:
        batches = generate_stage2_prompts_batches(
            sentences,
            stage1_result=stage1_result,
        )
    except Exception as e:
        print(f"❌ خطأ أثناء تشغيل generate_stage2_prompts_batches:\nالتفاصيل: {e}")
        return

    if not isinstance(batches, list) or len(batches) == 0:
        print("❌ خطأ: لم يتم إنتاج أي دفعات من المرحلة الثانية.")
        return

    total_prompts = sum(len(b) for b in batches if b)
    if total_prompts == 0:
        print("❌ خطأ: إجمالي أوامر الصور المولدة = 0. لن يتم حفظ أي ملفات.")
        return

    # 5.1) التحقق من تطابق عدد البرومبتات مع عدد الجمل
    if total_prompts != len(sentences):
        print("❌ خطأ: عدد أوامر الصور المولدة لا يطابق عدد جمل السكربت.")
        print(f"   - عدد الجمل: {len(sentences)}")
        print(f"   - عدد أوامر الصور المولدة: {total_prompts}")
        print("   لن يتم حفظ أي ملفات Prompts.")
        return

    # 6) حفظ ملفات الدفعات (.txt)
    print("\n💾 جاري حفظ ملفات الدفعات (.txt)...")
    saved_count = 0
    for idx, batch_prompts in enumerate(batches, start=1):
        if not batch_prompts:
            continue
        filename = prompts_dir / f"prompts_part_{idx:02d}.txt"
        # فاصل سطر فارغ واحد فقط بين كل برومبت والآخر بدون أي نصوص إضافية
        content = "\n\n".join(batch_prompts)
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
        except (OSError, PermissionError) as e:
            print(f"❌ فشل حفظ الملف: {filename}\nالتفاصيل: {e}")
            return
        saved_count += 1
        print(f"   📄 تم حفظ {filename.name} ({len(batch_prompts)} برومبت)")

    if saved_count == 0:
        print("❌ لم يتم حفظ أي ملفات Prompts.")
        return

    # 7) العرض النهائي
    print("\n" + "=" * 55)
    print("✅ اكتملت المرحلة الثانية بنجاح!")
    print(f"🆔 رقم الحلقة: {ep_id}")
    print(f"📊 إجمالي جمل السكربت: {len(sentences)}")
    print(f"🖼️ إجمالي أوامر الصور المولدة: {total_prompts}")
    print(f"📦 عدد ملفات الدفعات: {saved_count}")
    print("🧠 السياق المستخدم:")
    print(f"   - scene_plan: {'نعم' if has_scene_plan else 'لا'} ({scene_plan_count} مشهد)")
    print(f"   - visual_bible: {'نعم' if has_visual_bible else 'لا'}")
    print(f"   - creative_brief: {'نعم' if has_creative_brief else 'لا'}")
    print(f"📁 مسار الملفات: {prompts_dir}")
    print("=" * 55)


if __name__ == "__main__":
    main()
