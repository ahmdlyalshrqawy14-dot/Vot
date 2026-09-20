import html
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk
from config.settings import (
    AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION,
    AZURE_VOICE_NAME,
    AZURE_VOICE_RATE
)

EMOTION_MAP = {
    "urgent": "shouting",
    "serious": "serious",
    "energetic": "cheerful",
    "calm": "chat",
    "curious": "curious"
}

def run_stage_3(script_data: dict, episode_dir: Path):
    audio_dir = episode_dir / "audio_beats"
    audio_dir.mkdir(parents=True, exist_ok=True)
    beats = script_data.get("beats", [])

    print(f"\n--- [Stage 3] Synthesizing Expressive Audio Beats ---")

    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_SPEECH_KEY,
        region=AZURE_SPEECH_REGION
    )
    speech_config.speech_synthesis_voice_name = AZURE_VOICE_NAME

    for beat in beats:
        b_id = beat["id"]
        audio_path = audio_dir / f"beat_{b_id:03d}.wav"

        if audio_path.exists() and audio_path.stat().st_size > 1024:
            continue

        safe_text = html.escape(beat["spoken_text"])
        azure_style = EMOTION_MAP.get(beat.get("emotion", "energetic"), "cheerful")

        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{AZURE_VOICE_NAME}">
                <mstts:express-as style="{azure_style}" styledegree="1.3">
                    <prosody rate="{AZURE_VOICE_RATE}">
                        {safe_text}
                    </prosody>
                </mstts:express-as>
            </voice>
        </speak>
        """.strip()

        audio_config = speechsdk.audio.AudioOutputConfig(filename=str(audio_path))
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        res = synthesizer.speak_ssml_async(ssml).get()

        if res.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            fallback_ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{AZURE_VOICE_NAME}">{safe_text}</voice>
            </speak>
            """.strip()
            synthesizer.speak_ssml_async(fallback_ssml).get()

    print(f"--- [Stage 3] Generated all audio beats successfully! ---")
