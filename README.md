# Voiceover Alchemist

![изображение](https://github.com/user-attachments/assets/40633119-a594-41dc-b9aa-8666f408ed62)


The **Voiceover Alchemist** is an audio dubbing editor designed to quickly record voiceovers, remove voice from original video, and change voices via RVC post-processing.

## Features

- **Audio Segments:** Record and process audio segments for voiceover.
- **Voice Model Integration:** Apply different voice models to audio segments.
- **Voice Removal:** Remove voices from the underlying video.
- **Video Export:** Combine video and processed audio into a single output file.
- **Real-Time Playback:** Synchronize audio and video during playback.

---

## Installation

### Requirements

- Python 3.10
- Pip 24.0
- FFmpeg


### Dependencies

Install the following libraries using pip:

- `rvc` - Currently, there is a requirement of python & pip versions, see requirements
- `numpy`
- `PySide6`
- `pyaudio`
- `scipy`
- `librosa`
- `onnxruntime`
- `soundfile`
- `python-dotenv`
- `audio-separator`

Additionally, ensure **FFmpeg** is installed and available in your system's PATH.

### Steps to Install

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd multi-voice-dubbing-editor
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python3.10 -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```

3. Install pip 24.0:

   ```bash
   python -m ensurepip
   python -m pip install --upgrade pip
   ```

4. Initialize/Install models

   ```bash
   python full_init.py
   ```

---

## Usage

### Running the Application

1. Ensure your virtual environment is activated.
2. Start the application:
   ```bash
   python app.py
   ```

### Workflow

1. **Create or Open a Project:** Use the toolbar to create a new project or load an existing one.
2. **Import Video:** Import a video file to extract audio and begin editing.
4. **Choose Voice Models:** Select voice models and process the audio. The RVC models are stored in the assets/models directory.
3. **Record Audio Segments:** Record audio segments in synchronization with the video. The voice from the video will be removed and replaced with your voice or any RVC-based voice
5. **Preview and Export:** Preview the combined audio and export the video.

**Note:** Some of the features when used at the first time might require downloading of additional models, please be patient and read the logs in the console to see what's going on!

---

## Project Structure

Upon creating a project, the following structure is generated:

```
project_folder/
  ├── recordings/        # Recorded audio files
  ├── processed/         # Processed audio files
  ├── assets/models/     # Voice model files
  ├── project.json       # Project metadata
  ├── original.wav       # Original extracted audio
  ├── preview.wav        # Combined preview audio
  └── video.mp4          # Imported video file
```

---

## Development

### Adding New Features

1. Fork and clone the repository.
2. Implement your changes.
3. Run tests to ensure functionality.
4. Submit a pull request for review.


## Contributions

We welcome contributions!

---

## Acknowledgements

- **PySide6** for the GUI framework.
- **FFmpeg** for video and audio processing.
- **RVC** for voice conversion.
- **audio-separator** for vocal removal.
