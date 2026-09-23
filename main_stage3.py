import json
import sys
from pathlib import Path

from stage3_audio import generate_stage3_audio
from config import AZURE_MALE_VOICES


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_stage1_sentences(episode_id=None):
    """
    يحمّل ملف المرحلة الأولى كاملًا ويعيد:
        (ep_id, sentences, stage1_data)

    - sentences مأخوذة فقط من full_script_sentences.
    - stage1_data هو كامل كائن JSON بدون أي تعديل أو حذف.
    """

    if episode_id:
        file_path = OUTPUTS_DIR / f"stage1_episode_{episode_id}.json"
        if not file_path.exists():
            raise FileNotFoundError(
                f"لم يتم العثور على ملف المرحلة الأولى المطلوب: {file_path}"
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                stage1_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"ملف JSON غير صالح في {file_path}: {e}")

        if not isinstance(stage1_data, dict):
            raise ValueError(
                f"محتوى ملف المرحلة الأولى ليس dict كما هو متوقع: {file_path}"
            )

        ep_id = str(stage1_data.get("id", episode_id))
        sentences = stage1_data.get("full_script_sentences", [])

        return ep_id, sentences, stage1_data

    available_files = list(OUTPUTS_DIR.glob("stage1_episode_*.json"))
    if not available_files:
        raise FileNotFoundError(
            "لم يتم العثور على أي مخرجات للمرحلة الأولى في مجلد outputs!"
        )

    chosen_file = available_files[0]
    try:
        with open(chosen_file, "r", encoding="utf-8") as f:
            stage1_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"ملف JSON غير صالح في {chosen_file}: {e}")

    if not isinstance(stage1_data, dict):
        raise ValueError(
            f"محتوى ملف المرحلة الأولى ليس dict كما هو متوقع: {chosen_file}"
        )

    ep_id = str(stage1_data.get("id", "201"))
    sentences = stage1_data.get("full_script_sentences", [])

    return ep_id, sentences, stage1_data


def main():
    """
    الاستخدام:
    python main_stage3.py [episode_id] [engine: azure] [voice_name]
    الافتراضي: azure مع en-US-GuyNeural

    ملاحظة: المرحلة الثالثة تدعم Microsoft Azure فقط.
    """

    args = sys.argv[1:]
    target_id = args[0] if len(args) > 0 else None
    engine = args[1].lower() if len(args) > 1 else "azure"
    voice = args[2] if len(args) > 2 else "en-US-GuyNeural"

    # ------------------------------------------------------------------
    # 1) التحقق من المحرك: Azure فقط
    # ------------------------------------------------------------------
    if engine != "azure":
        raise SystemExit("❌ المرحلة الثالثة تدعم Microsoft Azure فقط.")

    # ------------------------------------------------------------------
    # 2) التحقق من الصوت: يجب أن يكون ضمن AZURE_MALE_VOICES
    # ------------------------------------------------------------------
    if voice not in AZURE_MALE_VOICES:
        raise SystemExit(
            f"❌ الصوت '{voice}' غير موجود في قائمة AZURE_MALE_VOICES."
        )

    print("🔍 جاري قراءة سكربت المرحلة الأولى...")
    ep_id, sentences, stage1_data = load_stage1_sentences(target_id)

    # ------------------------------------------------------------------
    # 3) التحققات الصارمة قبل بدء الإنتاج
    # ------------------------------------------------------------------
    if not isinstance(stage1_data, dict):
        raise SystemExit("❌ stage1_data ليس dict صالحًا.")

    if not isinstance(sentences, list):
        raise SystemExit("❌ full_script_sentences ليست قائمة list صالحة.")

    if not sentences:
        raise SystemExit("❌ خطأ: لا توجد جمل في السكربت!")

    # التأكد من أن sentences هي نفس القيمة الأصلية داخل stage1_data
    if sentences is not stage1_data.get("full_script_sentences", []):
        # مقارنة بالقيمة (وليست بالهوية) للتأكد من عدم إعادة بنائها
        if sentences != stage1_data.get("full_script_sentences", []):
            raise SystemExit(
                "❌ sentences لا تطابق full_script_sentences داخل stage1_data."
            )

    print(f"🎬 الحلقة ID [{ep_id}]: عدد الجمل = {len(sentences)}")
    print(f"⚙️ المحرك المختار: {engine.upper()}")
    print(f"🗣️ الصوت الرجالي المختار: {voice}")

    print("\n🚀 بدء إنتاج التعليق الصوتي التعبيري والتحقق البرمجي الصارم...")

    audio_path = generate_stage3_audio(
        episode_id=ep_id,
        sentences=sentences,
        engine=engine,
        voice=voice,
        output_dir=OUTPUTS_DIR,
        episode_context=stage1_data,
    )

    if audio_path is None:
        raise SystemExit("❌ فشل generate_stage3_audio: لم يُعد مسار ملف الصوت.")

    print("\n" + "=" * 55)
    print("✅ اكتملت المرحلة الثالثة بنجاح!")
    print(f"🆔 رقم الحلقة: {ep_id}")
    print(f"🎧 المحرك: {engine}")
    print(f"🎙️ الصوت: {voice}")
    print(f"📁 ملف الصوت الموحد جاهز في: {audio_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
