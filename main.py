import json
import sys
from pathlib import Path
from config.settings import DATA_FILE, OUTPUT_DIR
from stages.stage1_script import run_stage_1
from stages.stage2_images import run_stage_2
from stages.stage2_render_images import render_all_images
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

    episode_dir = OUTPUT_DIR / f"episode_{ep_id:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    # 1. إنتاج السكربت الإنجليزي
    script_text = run_stage_1(episode, episode_dir)

    # 2. استخراج أوامر الصور لكل جملة
    image_prompts = run_stage_2(script_text, episode_dir)

    # 2.5 توليد وفحص الصور تسلسلياً (Azure + Pollinations Fallback)
    render_all_images(image_prompts, episode_dir)

    # 3. النص المرمز عاطفياً وتوليد الصوت البشري
    audio_data = run_stage_3(script_text, episode_dir)

    # 4. الميتاداتا والثامبنيل والكلمات الدلالية
    metadata_text = run_stage_4(episode, script_text, episode_dir)

    print(f"\n==================================================")
    print(f"   SUCCESSFULLY FINISHED EPISODE #{ep_id}")
    print(f"   Outputs saved in: {episode_dir}")
    print(f"==================================================")

def main():
    episodes = load_episodes()
    pending_episodes = [ep for ep in episodes if ep.get("status") == "pending"]

    if not pending_episodes:
        print("All episodes are already completed! Nothing to run.")
        return

    # استهداف حلقة واحدة فقط في كل تشغيل
    target_episode = pending_episodes[0]
    ep_id = target_episode.get("id")
    print(f"Targeting single episode: #{ep_id} ({target_episode.get('topic')})")

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
