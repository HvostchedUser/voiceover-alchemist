import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from scipy.io import wavfile
from PyQt5 import QtCore, QtGui, QtWidgets, QtMultimedia, QtMultimediaWidgets

from rvc.modules.vc.modules import VC

import pyaudio
import wave
import subprocess

class DubbingApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Voice Dubbing Tool")

        # State variables
        self.video_path = None
        self.segments = []  # List of segments: {start, end, model, raw_audio, processed_audio}
        self.current_model = None
        self.models = self.load_models()
        self.recording = False
        self.record_start_time = None
        self.record_end_time = None

        # RVC Initialization
        load_dotenv(".env")
        self.vc = VC()

        self.setup_ui()

    def setup_ui(self):
        # Central widget
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Video player
        self.video_player = QtMultimediaWidgets.QVideoWidget()
        layout.addWidget(self.video_player)

        # Media player
        self.media_player = QtMultimedia.QMediaPlayer(None, QtMultimedia.QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_player)

        # Timeline slider
        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline_slider.setRange(0,0)
        self.timeline_slider.sliderMoved.connect(self.seek_video)
        layout.addWidget(self.timeline_slider)

        # Model selection dropdown
        self.model_combo = QtWidgets.QComboBox()
        for m in self.models:
            self.model_combo.addItem(m)
        self.model_combo.currentIndexChanged.connect(self.select_model)
        layout.addWidget(self.model_combo)

        # Recording buttons
        record_layout = QtWidgets.QHBoxLayout()
        self.record_button = QtWidgets.QPushButton("Record Segment")
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button = QtWidgets.QPushButton("Stop Recording")
        self.stop_record_button.clicked.connect(self.stop_recording)
        record_layout.addWidget(self.record_button)
        record_layout.addWidget(self.stop_record_button)
        layout.addLayout(record_layout)

        # Segment list & color-coded representation
        self.segment_list = QtWidgets.QListWidget()
        self.segment_list.itemClicked.connect(self.jump_to_segment)
        layout.addWidget(self.segment_list)

        # Load and save buttons
        load_button = QtWidgets.QPushButton("Load Video")
        load_button.clicked.connect(self.load_video)
        save_button = QtWidgets.QPushButton("Save Dubbed Video")
        save_button.clicked.connect(self.save_dubbed_video)
        control_layout = QtWidgets.QHBoxLayout()
        control_layout.addWidget(load_button)
        control_layout.addWidget(save_button)
        layout.addLayout(control_layout)

        self.setCentralWidget(central_widget)

        # Media player signals
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.positionChanged.connect(self.position_changed)

    def load_models(self):
        # Assuming models are in assets/models/ folder
        model_base = Path("assets/models/")
        models = [p.name for p in model_base.iterdir() if p.is_dir()]
        return models

    def select_model(self, index):
        self.current_model = self.model_combo.itemText(index)

    def load_video(self):
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setNameFilters(["Video files (*.mp4 *.mov *.avi)"])
        if file_dialog.exec_():
            self.video_path = file_dialog.selectedFiles()[0]
            self.media_player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(self.video_path)))
            self.media_player.play()

    def duration_changed(self, duration):
        self.timeline_slider.setRange(0, duration)

    def position_changed(self, position):
        # Update the timeline slider
        self.timeline_slider.setValue(position)
        # Could also update segment highlighting if needed

    def seek_video(self, position):
        self.media_player.setPosition(position)

    def start_recording(self):
        if not self.current_model:
            QtWidgets.QMessageBox.warning(self, "No Model Selected", "Please select a voice model before recording.")
            return

        self.recording = True
        self.record_start_time = self.media_player.position()
        self.raw_audio_path = "temp_record.wav"
        self.start_audio_capture(self.raw_audio_path)

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.stop_audio_capture()
        self.record_end_time = self.media_player.position()

        # Process the recorded segment with the chosen RVC model
        processed_audio_path = self.process_recorded_audio(self.raw_audio_path, self.current_model)
        
        # Add segment to list
        seg_info = {
            "start": self.record_start_time,
            "end": self.record_end_time,
            "model": self.current_model,
            "raw_audio": self.raw_audio_path,
            "processed_audio": processed_audio_path
        }
        self.segments.append(seg_info)
        self.update_segment_list()

    def start_audio_capture(self, filename):
        # Simple pyaudio recording setup
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 22050
        self.chunk = 1024
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.audio_format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)
        self.frames = []
        # Instead of a blocking call, we could use a QTimer to periodically read from stream
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.record_frame)
        self.timer.start(50)  # record chunks every 50 ms

    def record_frame(self):
        data = self.stream.read(self.chunk, exception_on_overflow=False)
        self.frames.append(data)

    def stop_audio_capture(self):
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        wf = wave.open(self.raw_audio_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def process_recorded_audio(self, input_wav, model_name):
        # Load the corresponding model
        model_dir = Path(f"assets/models/{model_name}/")

        pth_files = list(model_dir.glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError("No .pth file found.")
        pth_file = str(pth_files[0])

        index_files = list(model_dir.glob("*.index"))
        if not index_files:
            raise FileNotFoundError("No index file found.")
        index_file = str(index_files[0])

        self.vc.get_vc(pth_file)
        tgt_sr, audio_opt, times, _ = self.vc.vc_single(1, Path(input_wav), index_file=index_file)
        output_wav = f"processed_{model_name}_{os.path.basename(input_wav)}"
        wavfile.write(output_wav, tgt_sr, audio_opt)

        return output_wav

    def update_segment_list(self):
        self.segment_list.clear()
        for seg in self.segments:
            duration_sec = (seg["end"] - seg["start"]) / 1000.0
            item = QtWidgets.QListWidgetItem(f"{seg['model']} segment: {duration_sec:.2f}s at {seg['start']/1000:.2f}s")
            # Optionally color-code by model (just an example)
            color = QtGui.QColor(hash(seg["model"]) % 256, (hash(seg["model"]) >> 8) % 256, (hash(seg["model"]) >> 16) % 256)
            brush = QtGui.QBrush(color)
            item.setBackground(brush)
            self.segment_list.addItem(item)

    def jump_to_segment(self, item):
        # Jump to segment start in the video
        index = self.segment_list.row(item)
        seg = self.segments[index]
        self.media_player.setPosition(seg["start"])

    def save_dubbed_video(self):
        # Merge processed segments into one audio track, then blend with original audio
        # For simplicity, assume we create a combined audio track from segments:
        # Strategy:
        # 1. Extract original audio from video (using ffmpeg).
        # 2. Overlay processed segments onto original audio.
        # 3. Remux with the original video.

        # Extract original audio
        base_video = self.video_path
        extracted_audio = "original_audio.wav"
        subprocess.run(["ffmpeg", "-i", base_video, "-q:a", "0", "-map", "a", extracted_audio, "-y"])

        # Prepare overlay command
        # We need to mix processed segments at their correct time offset.
        # This could be done by using ffmpeg filters (e.g. amerge, adelay).
        # For multiple segments, we might have to do multiple filter_complex steps.
        
        # Let's say we create a filter_complex command:
        # For each segment:
        #   - add a delay (adelay) for that segment start
        # Then amerge all together and mix with original using amix
        # This is a rough concept and might need refinement.
        
        inputs = [extracted_audio]
        filters = []
        input_count = 1
        for seg in self.segments:
            inputs.append(seg["processed_audio"])
            start_sec = seg["start"] / 1000.0
            filters.append(f"[{input_count}:0]adelay={int(start_sec*1000)}|{int(start_sec*1000)}[a{input_count}];")
            input_count += 1
        
        # Now we merge all delayed processed tracks and the original with amix:
        # We'll have N processed tracks plus the original track (the first input).
        # Connect them all: [0:a] + all aX 
        # Something like: [0:0][a1][a2]...amix=...:dropout_transition=0
        amix_inputs = "[0:0]" + "".join([f"[a{i}]" for i in range(1, input_count)])
        amix_filter = f"{''.join(filters)}{amix_inputs}amix=inputs={input_count}:dropout_transition=0[aout]"
        
        final_audio = "final_audio.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i", base_video
        ]
        for segpath in inputs[1:]:
            cmd += ["-i", segpath]
        cmd += [
            "-filter_complex", amix_filter,
            "-map", "[aout]",
            final_audio
        ]
        subprocess.run(cmd)

        # Now combine final_audio with video (no re-encoding of video)
        final_video = "dubbed_video.mp4"
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", base_video,
            "-i", final_audio,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            final_video
        ])

        QtWidgets.QMessageBox.information(self, "Done", f"Dubbed video saved as: {final_video}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DubbingApp()
    window.show()
    sys.exit(app.exec_())
