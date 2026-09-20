import json
import sys
from pathlib import Path
from config.settings import DATA_FILE, OUTPUT_DIR
from stages.stage1_script import run_stage_1
from stages.stage2_images import run_stage_2
from stages.stage3_audio import run_stage_3
from stages.stage4_metadata import run_stage_4

def load_episodes():
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found!")
        sys.exit(1)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_episodes(episodes):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)

def process_single_episode(episode: dict):
    ep_id = episode.get("id", 1)
    print(f"\n==================================================")
    print(f"   STARTING PIPELINE FOR EPISODE #{ep_id}: {episode.get('topic')}")
    print(f"==================================================")

    # إنشاء مجلد مخرجات خاص بالحلقة
    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    # المرحلة 1: إنتاج السكربت
    script_text = run_stage_1(episode, episode_dir)

    # المرحلة 2: توليد أوامر الصور لكل جملة
    image_prompts = run_stage_2(script_text, episode_dir)

    # المرحلة 3: التعليق الصوتي والترميز العاطفي وتوليد الصوت
    audio_data = run_stage_3(script_text, episode_dir)

    # المرحلة 4: العناوين، الثامبنيل، الوصف، والكلمات الدلالية
    metadata_text = run_stage_4(episode, script_text, episode_dir)

    print(f"\n==================================================")
    print(f"   SUCCESSFULLY COMPLETED EPISODE #{ep_id}")
    print(f"   Outputs stored at: {episode_dir}")
    print(f"==================================================")

def main():
    episodes = load_episodes()
    pending_episodes = [ep for ep in episodes if ep.get("status") == "pending"]

    if not pending_episodes:
        print("No pending episodes found to process.")
        return

    print(f"Found {len(pending_episodes)} pending episode(s).")

    for episode in pending_episodes:
        try:
            process_single_episode(episode)
            # تحديث حالة الحلقة وحفظ الملف
            episode["status"] = "completed"
            save_episodes(episodes)
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed on Episode #{episode.get('id')}: {str(e)}")
            # التوقف عند حدوث خطأ لمراجعة السبب
            break

if __name__ == "__main__":
    main()
