import os
import re
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk
from config.models import generate_with_fallback
from config.settings import (
    PROMPTS_DIR,
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    AZURE_VOICE_NAME,
    AZURE_VOICE_RATE
)

def run_stage_3(script_text: str, episode_dir: Path) -> dict:
    """
    المرحلة الثالثة: توليد النص الصوتي المرمز بالمشاعر ثم التوليد الصوتي عبر Azure Speech
    """
    prompt_file = PROMPTS_DIR / "03_audio.txt"
    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    user_prompt = f"English Script:\n{script_text}"

    print(f"\n--- [Stage 3] Generating Emotional Voiceover Script ---")
    tagged_script = generate_with_fallback(
        stage_name="audio",
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # حفظ النص المرمز بالانفعالات
    tagged_file = episode_dir / "voiceover_script.txt"
    with open(tagged_file, "w", encoding="utf-8") as f:
        f.write(tagged_script)

    print(f"--- [Stage 3] Emotional script saved to: {tagged_file} ---")

    # تنقية النص الصوتي الموجه لـ Azure (إزالة وسومات المشاعر والأقواس لنطق صافٍ)
    spoken_lines = []
    for line in tagged_script.splitlines():
        line_clean = re.sub(r"\[.*?\]", "", line).strip()
        line_clean = line_clean.strip('"').strip("'")
        if line_clean:
            spoken_lines.append(line_clean)

    cleaned_spoken_text = " ".join(spoken_lines)

    # توليد الصوت عبر Azure Speech
    audio_output_path = episode_dir / "voiceover.wav"
    if AZURE_SPEECH_KEY and AZURE_SPEECH_REGION:
        print(f"--- [Stage 3] Synthesizing speech with voice: {AZURE_VOICE_NAME} ---")
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_SPEECH_REGION
        )
        speech_config.speech_synthesis_voice_name = AZURE_VOICE_NAME
        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(audio_output_path))
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        # صياغة SSML لضبط معدل السرعة الرياضي المحدد في الإعدادات
        ssml_text = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{AZURE_VOICE_NAME}">
                <prosody rate="{AZURE_VOICE_RATE}">
                    {cleaned_spoken_text}
                </prosody>
            </voice>
        </speak>
        """.strip()

        result = synthesizer.speak_ssml_async(ssml_text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"--- [Stage 3] Audio generated successfully: {audio_output_path} ---")
        else:
            print(f"--- [Stage 3] Warning: Speech synthesis issue: {result.reason} ---")
    else:
        print("--- [Stage 3] Azure Speech credentials missing. Skipping audio rendering. ---")

    return {
        "tagged_script": tagged_script,
        "audio_path": str(audio_output_path)
    }
