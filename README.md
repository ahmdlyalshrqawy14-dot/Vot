# VOT (Voice of Truth) - Automated Video Production Pipeline

VOT is a fully automated, multi-stage Python pipeline designed to generate professional, AI-directed documentary-style videos. It handles everything from script generation to voiceover synthesis, image acquisition, and complex video rendering with dynamic motion graphics, timed kinetic subtitles, and contextual visual assets.

## 🚀 Features

* **AI-Driven Narrative Generation**: Uses advanced language models (Gemini) to construct compelling hooks, revelations, and calls to action.
* **Automated Voice Synthesis**: High-quality TTS integration (Azure/Google).
* **AI Video Director**: Intelligently plans camera pacing (punch-ins, slow pans, drifts), transitions, and scene durations based on narrative roles.
* **Kinetic Word-Level Subtitles**: Generates precise, karaoke-style active word highlighting timed perfectly to the audio using FFmpeg and ASS subtitles.
* **Contextual Visual Overlays**: Automatically searches, downloads, and caches royalty-free transparent icons (via Wikimedia Commons) to emphasize key concepts.
* **High-Fidelity FFmpeg Rendering**: Enforces strict, production-standard encoding (`libx264`, `crf 20`, `yuv420p`, `192k aac`) for optimal web streaming and clean compression.

## 🏗️ Architecture (The 5 Stages)

The pipeline is modular and executes in sequential stages:

1. **Stage 1 (Generator - `main_stage1.py`)**: Script and prompt generation. Uses Gemini to create structured narrative arcs and visual prompts for each scene.
2. **Stage 2 (Vision - `main_stage2.py`)**: (Optional/External) Acquires visual assets (images) based on Stage 1 prompts.
3. **Stage 3 (Audio - `main_stage3.py`)**: Generates high-quality TTS audio for the script and calculates exact word-level timing and metadata boundaries.
4. **Stage 4 (Composer - `main_stage4.py`)**: The core video rendering engine.
   * `stage4_director.py`: Evaluates the script and audio to plan camera movements, pacing, and visual overlays.
   * `stage4_assets.py`: Fetches and caches transparent PNG/SVG icons for visual emphasis.
   * `stage4_subtitles.py`: Builds precise ASS subtitle files with dynamic word-level highlighting.
   * `stage4_composer.py`: Compiles frames, audio, subtitles, and overlays into a final 1080p 60fps MP4 using complex FFmpeg filter graphs.
5. **Stage 5 (Metadata - `main_stage5.py`)**: Prepares final metadata for publishing and archival.

## 🛠️ Setup & Installation

### Prerequisites
* Python 3.9+
* [FFmpeg](https://ffmpeg.org/download.html) (Ensure it is added to your system PATH)
* Required API keys (Gemini, Azure Speech, etc.)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd vot-pipeline
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `requests` and `Pillow` installed for the asset pipeline).*

3. Configure Environment:
   * Copy `.env.example` to `.env` (if applicable) or create a `.env` file in the root directory.
   * Add your API keys:
     ```env
     GEMINI_API_KEY=your_gemini_key
     AZURE_SPEECH_KEY=your_azure_key
     AZURE_SPEECH_REGION=uaenorth
     ```

## 🎥 Usage

The pipeline is typically executed stage by stage.

**Example execution for an episode ID (e.g., 201):**

```bash
# Generate Script
python3 main_stage1.py 201

# (Acquire images for Stage 2 externally or run Stage 2 script)
python3 main_stage2.py 201

# Generate Audio & Timings
python3 main_stage3.py 201

# Render Final Video
python3 main_stage4.py 201
```

All artifacts, including intermediate JSON plans, clean frames, cached assets, and final `.mp4` videos, are saved in the `outputs/` directory.

## ⚙️ Configuration & Customization

* **Typography & Subtitles**: Modify the ASS style header in `stage4_subtitles.py` to change fonts (default is Roboto), colors, or positioning.
* **Camera Pacing**: Adjust the prompt logic and fallback timing thresholds in `stage4_director.py` and `stage4_composer.py`.
* **API Providers**: Model logic and rate limit management are handled safely in `gemini_engine.py` and `config.py`.
