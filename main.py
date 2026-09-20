import os
import json
import sys
from pathlib import Path
from config.settings import DATA_FILE, OUTPUT_DIR
from stages.stage1_script import run_stage_1
from stages.stage2_images import run_stage_2
from stages.stage2_render_images import render_all_images
from stages.stage3_audio import run_stage_3
from stages.stage4_metadata import run_stage_4
from stages.stage5_video import run_stage_5
from stages.telegram_notifier import send_telegram_results

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

    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    # 1. السكربت الإنجليزي
    script_text = run_stage_1(episode, episode_dir)

    # 2. أوامر الصور
    image_prompts = run_stage_2(script_text, episode_dir)

    # 2.5 توليد وفحص الصور
    render_all_images(image_prompts, episode_dir)

    # 3. الصوت البشري
    audio_data = run_stage_3(script_text, episode_dir)

    # 4. الميتاداتا
    metadata_text = run_stage_4(episode, script_text, episode_dir)

    # 5. المونتاج ورندرة الفيديو (1080p + نصوص تفاعلية)
    run_stage_5(episode_dir)

    # 6. إرسال الفيديو والملخص على تيليجرام
    send_telegram_results(episode, episode_dir)

    print(f"\n==================================================")
    print(f"   SUCCESSFULLY FINISHED EPISODE #{ep_id}")
    print(f"   Outputs saved in: {episode_dir}")
    print(f"==================================================")

def main():
    episodes = load_episodes()
    target_id_env = os.getenv("TARGET_EPISODE_ID", "").strip()

    target_episode = None

    # التحقق إذا حدد المستخدم حلقة معينة
    if target_id_env:
        try:
            target_id = int(target_id_env)
            target_episode = next((ep for ep in episodes if ep.get("id") == target_id), None)
            if not target_episode:
                print(f"[ERROR] Episode #{target_id} not found in {DATA_FILE.name}!")
                sys.exit(1)
            print(f"Targeting requested Episode: #{target_id}")
        except ValueError:
            print(f"[ERROR] Invalid Episode ID format: '{target_id_env}'")
            sys.exit(1)
    else:
        # السحب التلقائي لأول حلقة pending
        pending_episodes = [ep for ep in episodes if ep.get("status") == "pending"]
        if not pending_episodes:
            print("All episodes are already completed!")
            return
        target_episode = pending_episodes[0]
        print(f"Auto-selected next pending Episode: #{target_episode.get('id')}")

    ep_id = target_episode.get("id")

    try:
        process_single_episode(target_episode)
        target_episode["status"] = "completed"
        save_episodes(episodes)
        print(f"\nEpisode #{ep_id} marked as COMPLETED. Execution finished.")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed on Episode #{ep_id}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
