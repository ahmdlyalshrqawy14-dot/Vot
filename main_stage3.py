import json
import sys
from pathlib import Path
from stage3_audio import generate_stage3_audio
from config import AZURE_MALE_VOICES, GOOGLE_MALE_VOICES

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_stage1_sentences(episode_id=None):
    if episode_id:
        file_path = OUTPUTS_DIR / f"stage1_episode_{episode_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return str(episode_id), json.load(f).get("full_script_sentences", [])

    available_files = list(OUTPUTS_DIR.glob("stage1_episode_*.json"))
    if not available_files:
        raise FileNotFoundError("لم يتم العثور على أي مخرجات للمرحلة الأولى في مجلد outputs!")

    chosen_file = available_files[0]
    with open(chosen_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return str(data.get("id", "201")), data.get("full_script_sentences", [])


def main():
    """
    الاستخدام:
    python main_stage3.py [episode_id] [engine: azure|google] [voice_name]
    الافتراضي: azure مع en-US-GuyNeural
    """
    args = sys.argv[1:]
    target_id = args[0] if len(args) > 0 else None
    engine = args[1].lower() if len(args) > 1 else "azure"
    voice = args[2] if len(args) > 2 else ("en-US-GuyNeural" if engine == "azure" else "en-US-Journey-D")

    print("🔍 جاري قراءة سكربت المرحلة الأولى...")
    ep_id, sentences = load_stage1_sentences(target_id)

    if not sentences:
        print("❌ خطأ: لا توجد جمل في السكربت!")
        return

    print(f"🎬 الحلقة ID [{ep_id}]: عدد الجمل = {len(sentences)}")
    print(f"⚙️ المحرك المختار: {engine.upper()}")
    print(f"🗣️ الصوت الرجالي المختار: {voice}")

    print("\n🚀 بدء إنتاج التعليق الصوتي التعبيري والتحقق البرمجي الصارم...")
    audio_path = generate_stage3_audio(
        episode_id=ep_id,
        sentences=sentences,
        engine=engine,
        voice=voice,
        output_dir=OUTPUTS_DIR
    )

    print("\n" + "=" * 55)
    print("✅ اكتملت المرحلة الثالثة بنجاح!")
    print(f"🆔 رقم الحلقة: {ep_id}")
    print(f"🎧 المحرك: {engine}")
    print(f"🎙️ الصوت: {voice}")
    print(f"📁 ملف الصوت الموحد جاهز في: {audio_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
