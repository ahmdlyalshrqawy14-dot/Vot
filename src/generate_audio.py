import os
import json
import sys
import re
from pydub import AudioSegment
import azure.cognitiveservices.speech as speechsdk

# ربط انفعالات الدليل بأساليب أزور العصبية (Azure Express-As Styles)
STYLE_MAPPING = {
    "shock": "excited",
    "urgency": "shouting",
    "excitement": "excited",
    "enthusiasm": "cheerful",
    "determination": "hopeful",
    "trust": "friendly",
    "calm": "friendly",
    "interest": "whispering",
    "curiosity": "chat",
    "warning": "unfriendly",
    "concern": "sad",
    "frustration": "angry",
    "disapproval": "unfriendly",
    "disbelief": "excited",
    "sarcasm": "chat",
    "pride": "cheerful",
    "satisfaction": "cheerful",
    "joy": "cheerful",
    "relief": "friendly",
    "whispers": "whispering"
}

def remove_emojis(text: str) -> str:
    """إزالة الرموز التعبيرية لمنع نطقها حرفياً عبر الـ TTS"""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # وجوه تعبيرية
        "\U0001F300-\U0001F5FF"  # رموز وأشكال
        "\U0001F680-\U0001F6FF"  # وسائل نقل
        "\U0001F1E0-\U0001F1FF"  # أعلام
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # إيموجي مكملة
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r"", text).strip()

def build_ssml(text: str, emotion_tag: str, voice_name="en-US-AndrewMultilingualNeural") -> str:
    """بناء كود SSML احترافي يغير أسلوب الإلقاء حسب الانفعال المكتوب"""
    clean_text = remove_emojis(text)
    style = STYLE_MAPPING.get(emotion_tag.lower(), "friendly")
    
    return f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
    <voice name="{voice_name}">
        <mstts:express-as style="{style}">
            <prosody rate="1.05">
                {clean_text}
            </prosody>
        </mstts:express-as>
    </voice>
</speak>"""

def synthesize_timeline(episode_data, speech_key, speech_region):
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    
    os.makedirs("audio/parts", exist_ok=True)
    timeline = episode_data.get("timeline", [])
    
    sync_map = []
    combined_audio = AudioSegment.empty()
    current_time_ms = 0
    pause_between_sentences = AudioSegment.silent(duration=200) # وقفة خفيفة (0.2 ثانية)

    for item in timeline:
        sentence_id = item["id"]
        voice_tts_raw = item.get("voice_tts", "")
        
        # استخراج أول انفعال من النص
        emotion_match = re.search(r"\[(.*?)\]", voice_tts_raw)
        emotion = emotion_match.group(1).strip() if emotion_match else "friendly"
        
        # إزالة الأقواس للحصول على النص المنطوق
        spoken_text = re.sub(r"^(\s*\[.*?\]\s*)+", "", voice_tts_raw).strip()
        
        ssml_content = build_ssml(spoken_text, emotion)
        part_path = f"audio/parts/{sentence_id:03d}.mp3"
        
        file_config = speechsdk.audio.AudioOutputConfig(filename=part_path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
        
        print(f"Synthesizing [{sentence_id}/{len(timeline)}]: ({emotion}) -> {spoken_text[:40]}...")
        result = synthesizer.speak_ssml_async(ssml_content).get()
        
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Error synthesizing sentence {sentence_id}: {result.reason}")
            sys.exit(1)
            
        segment = AudioSegment.from_file(part_path)
        duration_ms = len(segment)
        
        start_sec = round(current_time_ms / 1000.0, 3)
        end_sec = round((current_time_ms + duration_ms) / 1000.0, 3)
        
        sync_map.append({
            "id": sentence_id,
            "image_filename": f"{sentence_id:03d}.png",
            "audio_part": part_path,
            "start_time": start_sec,
            "end_time": end_sec,
            "duration": round(duration_ms / 1000.0, 3),
            "script_sentence": item.get("script_sentence", "")
        })
        
        combined_audio += segment + pause_between_sentences
        current_time_ms += duration_ms + 200

    full_audio_path = "audio/full_voiceover.mp3"
    combined_audio.export(full_audio_path, format="mp3")
    print(f"Exported full voiceover: {full_audio_path} (Total Duration: {round(current_time_ms/1000.0, 2)}s)")
    
    sync_map_path = "audio/sync_map.json"
    with open(sync_map_path, "w", encoding="utf-8") as f:
        json.dump(sync_map, f, indent=2, ensure_ascii=False)
    print(f"Saved synchronization map: {sync_map_path}")

def main():
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    
    if not speech_key or not speech_region:
        print("Error: AZURE_SPEECH_KEY or AZURE_SPEECH_REGION environment variables are missing.")
        sys.exit(1)
        
    if not os.path.exists("current_episode.json"):
        print("Error: current_episode.json not found. Run phase 1 first.")
        sys.exit(1)
        
    with open("current_episode.json", "r", encoding="utf-8") as f:
        episode_data = json.load(f)
        
    synthesize_timeline(episode_data, speech_key, speech_region)

if __name__ == "__main__":
    main()
