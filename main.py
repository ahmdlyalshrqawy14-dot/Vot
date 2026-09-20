import os
import json
import sys
from config.settings import DATA_FILE, OUTPUT_DIR
from stages.stage1_script import run_stage_1
from stages.stage2_images import run_stage_2
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
    print(f"\n{'='*55}\n   STARTING PRODUCTION: EPISODE #{ep_id} ({episode.get('topic')})\n{'='*55}")

    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    # 1. السكربت الطويل وتقسيم المشاهد
    script_data = run_stage_1(episode, episode_dir)

    # 2. توليد الصور بنسبة 16:9 عريضة
    run_stage_2(script_data, episode_dir)

    # 3. توليد الصوت البشري لكل مشهد
    run_stage_3(script_data, episode_dir)

    # 4. الميتاداتا
    run_stage_4(episode, script_data, episode_dir)

    # 5. المونتاج ورندرة الـ 5 دقائق مع BGM و الترجمة
    run_stage_5(script_data, episode_dir)

    # 6. الإشعار عبر تيليجرام
    send_telegram_results(episode, episode_dir)
    print(f"\n{'='*55}\n   COMPLETED EPISODE #{ep_id}\n{'='*55}")

def main():
    episodes = load_episodes()
    target_id_env = os.getenv("TARGET_EPISODE_ID", "").strip()

    if target_id_env:
        target_episode = next((ep for ep in episodes if str(ep.get("id")) == target_id_env), None)
        if not target_episode:
            print(f"Episode #{target_id_env} not found!")
            sys.exit(1)
    else:
        pending = [ep for ep in episodes if ep.get("status") == "pending"]
        if not pending:
            print("All episodes are already completed!")
            return
        target_episode = pending[0]

    process_single_episode(target_episode)
    target_episode["status"] = "completed"
    save_episodes(episodes)

if __name__ == "__main__":
    main()
