import json
import sys
from pathlib import Path
from stage4_vision import process_and_verify_images, MissingAssetsError
from stage4_subtitles import align_audio_and_generate_ass
from stage4_composer import render_final_video

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"


def main():
    target_id = sys.argv[1] if len(sys.argv) > 1 else "201"

    stage1_file = OUTPUTS_DIR / f"stage1_episode_{target_id}.json"
    audio_file = OUTPUTS_DIR / f"episode_{target_id}_audio.mp3"
    uploaded_images_dir = OUTPUTS_DIR / f"episode_{target_id}_raw_images"

    if not stage1_file.exists():
        print(f"❌ خطأ: ملف المرحلة الأولى غير موجود ({stage1_file})!")
        return

    if not audio_file.exists():
        print(f"❌ خطأ: ملف الصوت الموحد من المرحلة الثالثة غير موجود ({audio_file})!")
        return

    with open(stage1_file, "r", encoding="utf-8") as f:
        stage1_data = json.load(f)

    sentences = stage1_data.get("full_script_sentences", [])
    total_expected = len(sentences)

    print("\n" + "=" * 60)
    print(f"🚀 بدء تشغيل المرحلة الرابعة (المونتاج والدمج الآلي) للحلقة [{target_id}]")
    print(f"📊 إجمالي جمل السكربت والصور المستهدفة: {total_expected}")
    print("=" * 60)

    # 1. الفحص البصري وكشف النواقص والرقعة الذكية
    clean_frames_dir = OUTPUTS_DIR / f"episode_{target_id}_clean_frames"
    try:
        frames = process_and_verify_images(
            uploaded_images_dir=uploaded_images_dir,
            output_frames_dir=clean_frames_dir,
            expected_total=total_expected
        )
    except MissingAssetsError as missing_err:
        print("\n🚨 توقف مسار الرندرة مؤقتاً بسبب نقص الأصول:")
        print(str(missing_err))
        return

    # 2. المزامنة الزمنية وتوليد الترجمة الحركية
    subtitles_ass = OUTPUTS_DIR / f"episode_{target_id}_subtitles.ass"
    print("\n⏱️ جاري مزامنة الصوت وتوليد الترجمة الحركية الصفراء الباهتة...")
    timeline = align_audio_and_generate_ass(
        audio_path=audio_file,
        sentences=sentences,
        output_ass_path=subtitles_ass
    )

    # 3. الرندرة والمونتاج وتطبيق حركات Ken Burns والمؤثرات
    final_video = OUTPUTS_DIR / f"episode_{target_id}_final_1080p.mp4"
    print("\n🎬 بدء الرندرة الآلية للفيديو عبر FFmpeg (1080p 60fps)...")
    render_final_video(
        frames=frames,
        timeline=timeline,
        audio_file=audio_file,
        subtitles_ass=subtitles_ass,
        output_video_path=final_video
    )

    print("\n" + "=" * 60)
    print("🏆 اكتملت المرحلة الرابعة بنجاح وتم تصدير الفيديو النهائي!")
    print(f"🆔 رقم الحلقة: {target_id}")
    print(f"📹 دقة الفيديو: 1080p Full HD @ 60fps")
    print(f"🎨 الترجمة: أصفر باهت (#FDE047) - Word-by-Word بدون صندوق خلفي")
    print(f"🎥 حركات الكاميرا: Ken Burns 6-Pattern Cycle متجددة")
    print(f"🔇 الموسيقى الخلفية: محظورة 100% (Zero BGM)")
    print(f"💾 مسار الفيديو النهائي: {final_video}")
    print("=" * 60)


if __name__ == "__main__":
    main()
