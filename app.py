from contextlib import contextmanager
import sys
import os
import time
import json
import threading
import shutil
from pathlib import Path
from dotenv import load_dotenv
from subprocess import run, PIPE
import numpy as np
import queue
import tempfile
import uuid

from PySide6.QtCore import (
    Qt, QUrl, Signal, Slot, QObject, QMetaObject, Q_ARG
)
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QListWidget,
    QSplitter, QToolBar, QFileDialog, QDialog, QPlainTextEdit,
    QSpinBox, QMessageBox, QMenuBar, QSizePolicy
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

import pyaudio
import wave
from scipy.io import wavfile
import librosa
import soundfile as sf

load_dotenv(".env")

# Adjust these imports to match your environment.
from rvc.modules.vc.modules import VC
from audio_separator.separator import Separator

FFMPEG = "ffmpeg"
PROJECT_FILE = "project.json"
VIDEO_FILE = "video.mp4"
ORIGINAL_AUDIO = "original.wav"
PREVIEW_AUDIO = "preview.wav"
RECORDINGS_DIR = "recordings"
PROCESSED_DIR = "processed"
MODELS_DIR = "assets/models"
ORIGINAL_VOICE_OPTION = "<original voice>"

file_lock = threading.Lock()

@contextmanager
def locked_file_operation():
    """Thread-safe file operation context."""
    with file_lock:
        yield

def safe_wavfile_write(filepath, sample_rate, audio_data):
    with locked_file_operation():
        wavfile.write(filepath, sample_rate, audio_data)
        print(f"File written successfully: {filepath}")

def safe_wavfile_read(filepath):
    with locked_file_operation():
        sr, data = wavfile.read(filepath)
        print(f"File read successfully: {filepath}")
        return sr, data

separator = Separator()
vc = VC()

def process_audio(input_file, models, final_output_name="output_final.wav"):
    """
    Process an audio file with a chain of models sequentially.
    """
    global separator
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")

    current_file = input_file
    try:
        for i, (model, output_index) in enumerate(models):
            separator.load_model(model)
            output_files = separator.separate(current_file)
            print(f"Output files from {model}: {output_files}")
            if output_index >= len(output_files):
                raise ValueError(f"Invalid output index {output_index} for model {model}.")
            current_file = output_files[output_index]

            for f in output_files:
                temp_path = os.path.join(temp_dir, os.path.basename(f))
                shutil.move(f, temp_path)

            current_file = os.path.join(temp_dir, os.path.basename(current_file))
            print(f"Processed with {model}: {current_file}")

        final_output_path = os.path.join(os.path.dirname(input_file), final_output_name)
        shutil.copy(current_file, final_output_path)
        print(f"Final output saved as: {final_output_path}")
        return final_output_path
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Temporary directory {temp_dir} deleted.")

def remove_voice_from_segment(segment_data_wav_path, model_path="Kim_Vocal_2.onnx"):
    """
    Remove voice from a portion of the original video audio.
    """
    temp_output = os.path.join(tempfile.gettempdir(), f"voice_removed_{uuid.uuid4().hex}.wav")
    models_to_apply = [(model_path, 0)]
    final_output = process_audio(segment_data_wav_path, models_to_apply, final_output_name=os.path.basename(temp_output))
    if final_output != temp_output:
        shutil.copy(final_output, temp_output)
    return temp_output

def resample_audio(audio, original_sr, target_sr):
    """
    Resample audio array from original_sr to target_sr if needed.
    """
    if original_sr == target_sr:
        return audio
    audio = audio.astype(np.float32) / 32768.0
    if audio.ndim == 1:
        resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    else:
        resampled_channels = []
        for c in range(audio.shape[1]):
            channel_data = audio[:, c]
            resampled_channel = librosa.resample(channel_data, orig_sr=original_sr, target_sr=target_sr)
            resampled_channels.append(resampled_channel)
        resampled = np.stack(resampled_channels, axis=-1)
    resampled = (resampled * 32768.0).clip(-32768, 32767).astype(np.int16)
    return resampled

def extract_audio_from_video(video_path, audio_path):
    """
    Extract raw PCM audio from a given video file.
    """
    temp_path = audio_path + ".temp.wav"
    run([FFMPEG, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", temp_path], check=True)
    sr, data = safe_wavfile_read(temp_path)
    shutil.move(temp_path, audio_path)
    channels = data.shape[1] if data.ndim > 1 else 1
    return sr, channels

class Segment:
    def __init__(self, start_time, end_time, recording_path, model_name,
                 processed=False, processing=False, leave_original=False,
                 processed_path=None, voice_removed_path=None, pitch=0):
        self.start_time = start_time
        self.end_time = end_time
        self.recording_path = recording_path
        self.model_name = model_name
        self.processed = processed
        self.processing = processing
        self.leave_original = leave_original
        self.processed_path = processed_path
        self.voice_removed_path = voice_removed_path
        self.pitch = pitch

    def duration(self):
        return self.end_time - self.start_time

    def __str__(self):
        status = "Processing" if self.processing else ("Done" if self.processed else "Pending")
        return f"{self.start_time:.2f}s - {self.end_time:.2f}s | {self.model_name} | {status}"

    def to_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "recording_path": self.recording_path,
            "model_name": self.model_name,
            "processed": self.processed,
            "processing": self.processing,
            "leave_original": self.leave_original,
            "processed_path": self.processed_path,
            "voice_removed_path": self.voice_removed_path,
            "pitch": self.pitch
        }

    @staticmethod
    def from_dict(d):
        return Segment(
            d["start_time"],
            d["end_time"],
            d["recording_path"],
            d["model_name"],
            d["processed"],
            d["processing"],
            d["leave_original"],
            d.get("processed_path", None),
            d.get("voice_removed_path", None),
            d.get("pitch", 0)
        )

def save_project(project_dir, segments):
    """
    Save the current list of segments to project.json in a thread-safe manner.
    """
    project_file = os.path.join(project_dir, PROJECT_FILE)
    temp_file = f"{project_file}.tmp"
    data = {"segments": [s.to_dict() for s in segments]}
    try:
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=4)
        os.replace(temp_file, project_file)
        print(f"Project saved successfully: {project_file}")
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"Error saving project: {e}")

def load_project(project_dir):
    """
    Load the project.json and create a list of Segment objects.
    """
    with open(os.path.join(project_dir, PROJECT_FILE), "r") as f:
        data = json.load(f)
    segments = [Segment.from_dict(d) for d in data["segments"]]
    return segments

def ensure_project_structure(project_dir):
    """
    Create required sub-directories in the project if not existing.
    """
    os.makedirs(os.path.join(project_dir, RECORDINGS_DIR), exist_ok=True)
    os.makedirs(os.path.join(project_dir, PROCESSED_DIR), exist_ok=True)

class RecordingWorker(threading.Thread):
    """
    Thread that records audio from microphone until stop is called.
    """
    def __init__(self, output_path, record_sr, record_channels, finished_callback, error_callback):
        super().__init__()
        self.output_path = output_path
        self.record_sr = record_sr
        self.record_channels = record_channels
        self._stop_flag = threading.Event()
        self.finished_callback = finished_callback
        self.error_callback = error_callback

    def run(self):
        try:
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = self.record_channels
            RATE = self.record_sr
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
            frames = []
            while not self._stop_flag.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open(self.output_path, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            self.finished_callback(True)
        except Exception as e:
            self.error_callback(str(e))
            self.finished_callback(False)

    def stop(self):
        self._stop_flag.set()

class TaskType:
    PROCESS_SEGMENT = 1
    EXPORT = 2

def combine_audio(original_wav_path, segments, project_dir, preview_path, orig_sr, orig_channels):
    """
    Rebuild the preview audio from the original WAV plus all processed segments.
    """
    print("Combine audio with smooth transitions and consistent sample rates")
    if not os.path.exists(original_wav_path):
        print("Original WAV file does not exist!")
        return
    sr, original_data = safe_wavfile_read(original_wav_path)
    if sr != orig_sr:
        original_data = resample_audio(original_data, sr, orig_sr)
        sr = orig_sr
    original_data = original_data.astype(np.float32)
    if original_data.ndim == 1 and orig_channels > 1:
        original_data = np.tile(original_data[:, None], (1, orig_channels))

    for seg in segments:
        # Only integrate fully processed segments
        if not seg.processed or seg.voice_removed_path is None:
            continue

        start_sample = int(seg.start_time * orig_sr)
        end_sample = int(seg.end_time * orig_sr)
        if end_sample > len(original_data):
            end_sample = len(original_data)

        # Insert "voice-removed" portion
        sr_vr, vr_data = safe_wavfile_read(os.path.join(project_dir, seg.voice_removed_path))
        if sr_vr != orig_sr:
            vr_data = resample_audio(vr_data, sr_vr, orig_sr)
        vr_data = vr_data.astype(np.float32)
        if vr_data.ndim == 1 and orig_channels > 1:
            vr_data = np.tile(vr_data[:, None], (1, orig_channels))
        elif vr_data.ndim > 1 and orig_channels == 1:
            vr_data = np.mean(vr_data, axis=1)
        elif vr_data.ndim > 1 and vr_data.shape[1] != orig_channels:
            if vr_data.shape[1] < orig_channels:
                vr_data = np.pad(vr_data, ((0,0),(0, orig_channels - vr_data.shape[1])), mode='constant')
            else:
                vr_data = vr_data[:, :orig_channels]

        if vr_data.shape[0] != (end_sample - start_sample):
            diff = (end_sample - start_sample) - vr_data.shape[0]
            if diff > 0:
                vr_data = np.pad(vr_data, ((0,diff),(0,0)) if vr_data.ndim>1 else ((0,diff)), mode='constant')
            else:
                vr_data = vr_data[:end_sample - start_sample]

        original_data[start_sample:end_sample] = vr_data

        # Overlay the processed voice
        seg_audio_path = os.path.join(project_dir, seg.processed_path)
        if os.path.exists(seg_audio_path):
            sr2, seg_data = safe_wavfile_read(seg_audio_path)
            if sr2 != orig_sr:
                seg_data = resample_audio(seg_data, sr2, orig_sr)
            seg_data = seg_data.astype(np.float32)
            if seg_data.ndim == 1 and orig_channels > 1:
                seg_data = np.tile(seg_data[:, None], (1, orig_channels))
            elif seg_data.ndim > 1 and orig_channels == 1:
                seg_data = np.mean(seg_data, axis=1)
            elif seg_data.ndim > 1 and seg_data.shape[1] != orig_channels:
                if seg_data.shape[1] < orig_channels:
                    seg_data = np.pad(seg_data, ((0,0),(0, orig_channels - seg_data.shape[1])), mode='constant')
                else:
                    seg_data = seg_data[:, :orig_channels]

            if seg_data.shape[0] != (end_sample - start_sample):
                diff = (end_sample - start_sample) - seg_data.shape[0]
                if diff > 0:
                    seg_data = np.pad(seg_data, ((0,diff),(0,0)) if seg_data.ndim>1 else ((0,diff)), mode='constant')
                else:
                    seg_data = seg_data[:end_sample - start_sample]

            original_data[start_sample:start_sample + seg_data.shape[0]] += seg_data

    # Normalize
    max_val = np.max(np.abs(original_data))
    if max_val > 32767.0:
        original_data *= (32767.0 / max_val)

    original_data = original_data.astype(np.int16)
    safe_wavfile_write(preview_path, orig_sr, original_data)
    if os.path.exists(preview_path):
        print(f"Combined audio written to {preview_path}")
    else:
        print("Error: Combined audio file was not created!")

class ProcessingTask:
    def __init__(self, task_type, segment=None, project_dir=None, orig_sr=None, orig_channels=None, video_path=None, preview_audio_path=None, output_path=None):
        self.task_type = task_type
        self.segment = segment
        self.project_dir = project_dir
        self.orig_sr = orig_sr
        self.orig_channels = orig_channels
        self.video_path = video_path
        self.preview_audio_path = preview_audio_path
        self.output_path = output_path

class EventEmitter(QObject):
    """
    Used to relay events from background threads to the main thread (GUI).
    """
    processingEvent = Signal(str, dict)

class AudioProcessingManager(threading.Thread):
    """
    Thread that processes tasks (like segment processing or video export) one-by-one in a queue.
    """
    def __init__(self, callback, error_callback):
        super().__init__(daemon=True)
        self.task_queue = queue.Queue()
        self.callback = callback
        self.error_callback = error_callback
        self.stop_flag = threading.Event()
        self.eventEmitter = EventEmitter()

    def run(self):
        while not self.stop_flag.is_set():
            try:
                task = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task.task_type == TaskType.PROCESS_SEGMENT:
                self.handle_process_segment(task)
            elif task.task_type == TaskType.EXPORT:
                self.handle_export(task)

            self.task_queue.task_done()

    def add_task(self, task):
        self.task_queue.put(task)

    def stop(self):
        self.stop_flag.set()

    def emit_event(self, event_type, data):
        self.eventEmitter.processingEvent.emit(event_type, data)

    def handle_process_segment(self, task):
        """
        Process a single segment: run voice conversion if needed, remove voice from original,
        then combine. Note that we do NOT remove any other queued segments from the project
        or memory; we only modify this specific segment.
        """
        seg = task.segment
        project_dir = task.project_dir
        orig_sr = task.orig_sr
        orig_channels = task.orig_channels

        try:
            # We do not reload the entire list from disk here; we only:
            # 1) update seg
            # 2) re-save the project
            # 3) combine

            # If segment is "original voice," skip voice conversion
            if seg.leave_original:
                seg.processed_path = seg.recording_path
            else:
                model_name = seg.model_name
                model_dir = Path(MODELS_DIR, model_name)
                pth_files = list(model_dir.glob("*.pth"))
                if not pth_files:
                    raise FileNotFoundError("No .pth file found in model directory.")
                pth_file = pth_files[0]
                vc.get_vc(str(pth_file))
                index_files = list(model_dir.glob("*.index"))
                if not index_files:
                    raise FileNotFoundError("No index file (.index) found in model directory.")
                index_file = index_files[0]

                input_wav = os.path.join(project_dir, seg.recording_path)
                tgt_sr, audio_opt, times, _ = vc.vc_single(
                    sid=1,
                    input_audio_path=Path(input_wav),
                    index_file=str(index_file),
                    filter_radius=0,
                    rms_mix_rate=0.0,
                    protect=0.5,
                    f0_up_key=seg.pitch
                )

                if orig_sr != tgt_sr:
                    audio_opt = resample_audio(audio_opt, tgt_sr, orig_sr)
                    tgt_sr = orig_sr

                if audio_opt.ndim == 1 and orig_channels > 1:
                    audio_opt = np.tile(audio_opt[:, None], (1, orig_channels))
                elif audio_opt.ndim == 2 and audio_opt.shape[1] != orig_channels:
                    if audio_opt.shape[1] < orig_channels:
                        audio_opt = np.pad(audio_opt, ((0,0),(0, orig_channels - audio_opt.shape[1])), mode='constant')
                    else:
                        audio_opt = audio_opt[:, :orig_channels]

                processed_path = os.path.join(PROCESSED_DIR, f"processed_{int(time.time()*1000)}.wav")
                full_processed_path = os.path.join(project_dir, processed_path)
                if audio_opt.dtype != np.int16:
                    max_val = np.max(np.abs(audio_opt))
                    if max_val > 32767:
                        audio_opt = (audio_opt.astype(np.float32)/max_val)*32767
                    audio_opt = audio_opt.astype(np.int16)
                safe_wavfile_write(full_processed_path, tgt_sr, audio_opt)
                seg.processed_path = processed_path

            # Remove original voice from video portion for this segment
            seg.voice_removed_path = None
            orig_audio_path = os.path.join(project_dir, ORIGINAL_AUDIO)
            sr_o, orig_data = safe_wavfile_read(orig_audio_path)
            if sr_o != orig_sr:
                orig_data = resample_audio(orig_data, sr_o, orig_sr)
            start_sample = int(seg.start_time * orig_sr)
            end_sample = int(seg.end_time * orig_sr)
            if end_sample > len(orig_data):
                end_sample = len(orig_data)
            segment_data = orig_data[start_sample:end_sample]
            temp_segment_file = os.path.join(tempfile.gettempdir(), f"original_segment_{uuid.uuid4().hex}.wav")
            safe_wavfile_write(temp_segment_file, orig_sr, segment_data)
            voice_removed_temp = remove_voice_from_segment(temp_segment_file)
            voice_removed_path_rel = os.path.join(PROCESSED_DIR, f"voice_removed_{int(time.time()*1000)}.wav")
            voice_removed_path = os.path.join(project_dir, voice_removed_path_rel)
            shutil.move(voice_removed_temp, voice_removed_path)
            seg.voice_removed_path = voice_removed_path_rel

            # Mark the segment as processed
            seg.processing = False
            seg.processed = True

            # Save the updated segment changes to project
            # We do NOT remove other queued segments or anything else.
            # We must find the same segment in the main list and update it as well.
            # The main window will do the final preview update on event.
            self.emit_event("segment_done", {"segment": seg})

        except Exception as e:
            self.error_callback(str(e))

    def handle_export(self, task):
        """
        Export the final video.
        """
        try:
            result = run([
                FFMPEG, "-y",
                "-i", task.video_path,
                "-i", task.preview_audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                task.output_path
            ], stdout=PIPE, stderr=PIPE, text=True)

            if result.returncode != 0:
                error_msg = result.stderr
                self.error_callback(error_msg)
                self.emit_event("export_done", {"success": False, "output_path": task.output_path})
            else:
                self.emit_event("export_done", {"success": True, "output_path": task.output_path})
        except Exception as e:
            self.error_callback(str(e))
            self.emit_event("export_done", {"success": False, "output_path": task.output_path})

class HotkeyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hotkeys Help")
        layout = QVBoxLayout(self)
        edit = QPlainTextEdit(self)
        edit.setReadOnly(True)
        help_text = """Hotkeys:
Space: Play/Pause
Left Arrow: Seek -5s
Right Arrow: Seek +5s
R: Start/Stop Recording
Backspace or Delete: Remove selected segment
P: Reprocess selected segment
O: Open project
N: New project
S: Save project
I: Import video
E: Export video
H: Show this help
"""
        edit.setPlainText(help_text)
        layout.addWidget(edit)
        btn = QPushButton("Close", self)
        btn.clicked.connect(self.close)
        layout.addWidget(btn)
        self.resize(400,300)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voiceover Alchemist")

        self.project_dir = None
        self.segments = []
        self.orig_sr = None
        self.orig_channels = None

        self.is_recording = False
        self.recording_worker = None

        self.processing_manager = AudioProcessingManager(
            callback=self.on_processing_callback,
            error_callback=self.on_worker_error
        )
        self.processing_manager.start()

        # Connect events from worker to main thread
        self.processing_manager.eventEmitter.processingEvent.connect(self.on_processing_callback_main_thread)

        self.player_video = QMediaPlayer()
        self.player_audio = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player_audio.setAudioOutput(self.audio_output)
        self.video_widget = QVideoWidget()
        self.player_video.setVideoOutput(self.video_widget)

        # Toggle button for play/pause
        self.play_pause_button = QPushButton("Play")
        # Toggle button for start/stop recording
        self.record_toggle_button = QPushButton("Start Recording")

        self.remove_button = QPushButton("Remove Selected Segment")
        self.reprocess_button = QPushButton("Reprocess Segment")

        self.model_combo = QComboBox()
        self.pitch_spinbox = QSpinBox()
        self.pitch_spinbox.setRange(-48,48)
        self.pitch_spinbox.setValue(0)
        self.pitch_spinbox.setToolTip("Pitch shift in semitones (-48 to 48)")

        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 1000)
        self.segment_list = QListWidget()

        # Time counter label
        self.time_label = QLabel("0.00s")
        self.time_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.record_toggle_button.clicked.connect(self.toggle_recording)
        self.remove_button.clicked.connect(self.remove_selected_segment)
        self.reprocess_button.clicked.connect(self.reprocess_selected_segment)

        self.player_video.positionChanged.connect(self.on_position_changed)
        self.player_video.durationChanged.connect(self.on_duration_changed)
        self.timeline_slider.sliderMoved.connect(self.on_slider_moved)
        self.segment_list.itemClicked.connect(self.on_segment_selected)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(QLabel("Voice Model:"))
        controls_layout.addWidget(self.model_combo)
        controls_layout.addWidget(QLabel("Pitch:"))
        controls_layout.addWidget(self.pitch_spinbox)
        controls_layout.addWidget(self.record_toggle_button)
        controls_layout.addWidget(self.remove_button)
        controls_layout.addWidget(self.reprocess_button)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.timeline_slider)
        slider_layout.addWidget(self.time_label)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_widget)
        left_layout.addLayout(slider_layout)
        left_layout.addLayout(controls_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Recorded Segments:"))
        right_layout.addWidget(self.segment_list)
        right_layout.addStretch()

        splitter = QSplitter()
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 10)

        container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.addWidget(splitter)
        container.setLayout(container_layout)
        self.setCentralWidget(container)

        # Toolbar setup
        toolbar = QToolBar("Project Toolbar")
        self.addToolBar(toolbar)

        create_action = toolbar.addAction("Create Project")
        open_action = toolbar.addAction("Open Project")
        save_action = toolbar.addAction("Save Project")
        import_action = toolbar.addAction("Import Video")
        export_action = toolbar.addAction("Export Video")
        hotkey_action = toolbar.addAction("Hotkeys Help")

        create_action.triggered.connect(self.new_project)
        open_action.triggered.connect(self.open_project)
        save_action.triggered.connect(self.save_project_action)
        import_action.triggered.connect(self.import_video)
        export_action.triggered.connect(self.export_video_action)
        hotkey_action.triggered.connect(self.show_hotkeys)

        self.player_video.pause()
        self.player_audio.pause()

        # Start playback sync in background
        self.sync_thread = threading.Thread(target=self.sync_playback_loop, daemon=True)
        self.sync_queue = queue.Queue()
        self.sync_thread.start()

        self.player_audio.mediaStatusChanged.connect(self.on_media_status_changed)
        self.pending_position = None
        self.was_playing = False

        # Start the "timer" loop for sync
        self.timer_event()

        self.setFocusPolicy(Qt.StrongFocus)

    def show_hotkeys(self):
        dlg = HotkeyDialog(self)
        dlg.exec()

    def on_processing_callback(self, event_type, data):
        """
        Called inside worker thread, do NOT do UI updates here.
        """
        pass

    @Slot(str, dict)
    def on_processing_callback_main_thread(self, event_type, data):
        """
        Called in main thread to handle the events from worker.
        """
        if event_type == "segment_done":
            seg = data["segment"]
            # Update the segment in memory
            for local_seg in self.segments:
                if local_seg.recording_path == seg.recording_path:
                    local_seg.processed = seg.processed
                    local_seg.processing = seg.processing
                    local_seg.processed_path = seg.processed_path
                    local_seg.voice_removed_path = seg.voice_removed_path
                    local_seg.pitch = seg.pitch
                    break
            # Save the updated project
            if self.project_dir:
                save_project(self.project_dir, self.segments)
            # Re-run combine to reflect the new changes
            self.regenerate_preview_audio(forced=False)
            self.update_segment_list()

        elif event_type == "export_done":
            success = data["success"]
            output_path = data["output_path"]
            if not success:
                self.show_error_msg("Failed to export video.")
            else:
                QMessageBox.information(self, "Exported", f"Video exported to {output_path}")

    @Slot(str)
    def show_error_msg(self, msg):
        """
        Show an error message in a QPlainTextEdit for easy copy/paste.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle("Error")
        layout = QVBoxLayout(dlg)
        edit = QPlainTextEdit(dlg)
        edit.setReadOnly(True)
        edit.setPlainText(msg)
        layout.addWidget(edit)
        btn = QPushButton("Close", dlg)
        btn.clicked.connect(dlg.close)
        layout.addWidget(btn)
        dlg.resize(600,400)
        dlg.exec()

    def on_worker_error(self, error_msg):
        """
        Called by worker on error in background thread.
        Use invokeMethod to show error in main thread.
        """
        QMetaObject.invokeMethod(self, "show_error_msg", Qt.QueuedConnection, Q_ARG(str, error_msg))

    def sync_playback_loop(self):
        """
        Background loop to sync audio/video every 0.5s.
        """
        while True:
            time.sleep(0.5)
            self.sync_queue.put(None)

    def process_sync_queue(self):
        while not self.sync_queue.empty():
            self.sync_queue.get()
            self.sync_playback()

    def regenerate_preview_audio(self, forced=True):
        """
        Rebuild the preview audio from segments if needed.
        """
        if not self.project_dir or self.orig_sr is None:
            return
        preview_audio_path = os.path.join(self.project_dir, PREVIEW_AUDIO)

        # Re-combine using all segments (some might be processed)
        orig_audio_path = os.path.join(self.project_dir, ORIGINAL_AUDIO)
        combine_audio(orig_audio_path, self.segments, self.project_dir, preview_audio_path, self.orig_sr, self.orig_channels)
        self.update_audio_player(preview_audio_path)

    def update_audio_player(self, preview_path):
        self.player_audio.mediaStatusChanged.disconnect(self.on_media_status_changed)
        self.player_audio.mediaStatusChanged.connect(self.on_media_status_changed)
        self.player_audio.stop()
        self.player_audio.setSource(QUrl.fromLocalFile(preview_path))
        print(f"Set new preview audio source: {preview_path}")

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            if self.pending_position is not None:
                self.player_audio.setPosition(self.pending_position)
                self.pending_position = None
            if self.was_playing and self.player_video.playbackState() == QMediaPlayer.PlayingState:
                self.player_audio.play()

    def load_models(self):
        self.model_combo.clear()
        self.model_combo.addItem(ORIGINAL_VOICE_OPTION)
        if os.path.isdir(MODELS_DIR):
            dirs = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
            self.model_combo.addItems(dirs)

    def is_original_voice_selected(self):
        return self.model_combo.currentText() == ORIGINAL_VOICE_OPTION

    def toggle_play_pause(self):
        """
        Toggle between playing and pausing the video/audio.
        """
        if self.player_video.playbackState() == QMediaPlayer.PlayingState:
            # Pause
            self.player_video.pause()
            self.player_audio.pause()
            # If recording, stop recording on pause
            if self.is_recording:
                self.stop_recording()
                self.record_toggle_button.setText("Start Recording")
            self.play_pause_button.setText("Play")
        else:
            # Play
            self.player_video.play()
            self.player_audio.play()
            self.play_pause_button.setText("Pause")

    def toggle_recording(self):
        """
        Toggle between starting and stopping the recording.
        """
        if not self.is_recording:
            # Scroll 1s back if possible
            pos = self.player_video.position() - 1000
            if pos < 0:
                pos = 0
            self.player_video.setPosition(pos)
            self.player_audio.setPosition(pos)

            self.start_recording()
            self.record_toggle_button.setText("Stop Recording")
        else:
            self.stop_recording()
            self.record_toggle_button.setText("Start Recording")

    def on_duration_changed(self, duration):
        if duration > 0:
            self.timeline_slider.setMaximum(duration)

    def on_position_changed(self, position):
        if self.player_video.duration() > 0:
            self.timeline_slider.setValue(position)
        self.time_label.setText(f"{position/1000:.2f}s")

    def on_slider_moved(self, position):
        self.player_video.setPosition(position)
        self.player_audio.setPosition(position)
        self.time_label.setText(f"{position/1000:.2f}s")

    def on_segment_selected(self, item):
        idx = self.segment_list.row(item)
        if idx < 0 or idx >= len(self.segments):
            return
        seg = self.segments[idx]
        self.player_video.pause()
        self.player_audio.pause()
        pos_ms = int(seg.start_time * 1000)
        self.player_video.setPosition(pos_ms)
        self.player_audio.setPosition(pos_ms)

    def start_recording(self):
        if not self.project_dir or self.orig_sr is None:
            QMessageBox.warning(self, "No Project", "Please create or open a project first.")
            return
        if self.is_recording:
            return
        self.is_recording = True
        self.record_start_time = self.player_video.position() / 1000.0
        rec_path = os.path.join(RECORDINGS_DIR, f"record_{int(time.time() * 1000)}.wav")
        full_rec_path = os.path.join(self.project_dir, rec_path)

        def recording_finished(success):
            QMetaObject.invokeMethod(self, "on_recording_finished", Qt.QueuedConnection,
                                     Q_ARG(bool, success), Q_ARG(str, rec_path))

        def recording_error(error_msg):
            QMetaObject.invokeMethod(self, "show_error_msg", Qt.QueuedConnection, Q_ARG(str, error_msg))

        self.recording_worker = RecordingWorker(
            full_rec_path,
            self.orig_sr,
            self.orig_channels,
            recording_finished,
            recording_error
        )
        self.recording_worker.start()

        if self.player_video.playbackState() != QMediaPlayer.PlayingState:
            self.player_video.play()
            self.player_audio.play()
            self.play_pause_button.setText("Pause")

    @Slot(bool, str)
    def on_recording_finished(self, success, rec_path):
        if not success:
            QMessageBox.warning(self, "Recording Error", "Recording failed.")
            self.is_recording = False
            self.record_toggle_button.setText("Start Recording")
            return

        end_time = self.player_video.position() / 1000.0
        start_time = self.record_start_time
        duration = end_time - start_time
        if duration <= 0:
            QMessageBox.warning(self, "Error", "Recorded duration invalid.")
        else:
            model_name = self.model_combo.currentText() or ORIGINAL_VOICE_OPTION
            leave_original = self.is_original_voice_selected()

            seg = Segment(
                start_time,
                end_time,
                rec_path,
                model_name,
                processed=False,
                processing=True,
                leave_original=leave_original,
                processed_path=None,
                voice_removed_path=None,
                pitch=self.pitch_spinbox.value()
            )

            self.segments.append(seg)
            self.update_segment_list()
            save_project(self.project_dir, self.segments)

            # Add to queue
            task = ProcessingTask(
                TaskType.PROCESS_SEGMENT,
                segment=seg,
                project_dir=self.project_dir,
                orig_sr=self.orig_sr,
                orig_channels=self.orig_channels
            )
            self.processing_manager.add_task(task)

        self.is_recording = False
        self.record_toggle_button.setText("Start Recording")

    def stop_recording(self):
        """
        Stop the current recording if active.
        """
        if not self.is_recording:
            return
        self.is_recording = False
        if self.recording_worker:
            self.recording_worker.stop()

    def reprocess_selected_segment(self):
        idx = self.segment_list.currentRow()
        if idx < 0:
            return
        seg = self.segments[idx]
        model_name = self.model_combo.currentText()
        leave_original = self.is_original_voice_selected()

        seg.model_name = model_name
        seg.leave_original = leave_original
        seg.processed = False
        seg.processing = True
        seg.processed_path = None
        seg.voice_removed_path = None
        seg.pitch = self.pitch_spinbox.value()

        self.update_segment_list()
        save_project(self.project_dir, self.segments)

        task = ProcessingTask(
            TaskType.PROCESS_SEGMENT,
            segment=seg,
            project_dir=self.project_dir,
            orig_sr=self.orig_sr,
            orig_channels=self.orig_channels
        )
        self.processing_manager.add_task(task)

    def remove_selected_segment(self):
        idx = self.segment_list.currentRow()
        if idx < 0:
            return
        self.segments.pop(idx)
        save_project(self.project_dir, self.segments)
        self.update_segment_list()

        # Re-combine to ensure the removed segment is no longer in preview
        if self.project_dir and self.orig_sr is not None:
            orig_audio_path = os.path.join(self.project_dir, ORIGINAL_AUDIO)
            preview_path = os.path.join(self.project_dir, PREVIEW_AUDIO)
            combine_audio(orig_audio_path, self.segments, self.project_dir, preview_path, self.orig_sr, self.orig_channels)
            self.update_audio_player(preview_path)

    def new_project(self):
        self.close_all_threads()
        project_dir = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not project_dir:
            return
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.mov *.mkv)")
        if not video_file:
            return
        target_video = os.path.join(project_dir, VIDEO_FILE)
        shutil.copyfile(video_file, target_video)
        ensure_project_structure(project_dir)
        orig_audio_path = os.path.join(project_dir, ORIGINAL_AUDIO)
        self.orig_sr, self.orig_channels = extract_audio_from_video(target_video, orig_audio_path)
        self.project_dir = project_dir
        self.segments = []
        self.load_models()
        shutil.copyfile(orig_audio_path, os.path.join(project_dir, PREVIEW_AUDIO))
        self.player_video.setSource(QUrl.fromLocalFile(target_video))
        self.player_video.pause()
        self.player_audio.pause()
        self.update_segment_list()
        save_project(self.project_dir, self.segments)
        self.regenerate_preview_audio(forced=False)
        print(f"New project created: {self.project_dir}")

    def open_project(self):
        self.close_all_threads()
        project_dir = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not project_dir:
            return
        project_file = os.path.join(project_dir, PROJECT_FILE)
        if not os.path.exists(project_file):
            self.show_error_msg("No project.json found in selected folder.")
            return
        self.project_dir = project_dir
        self.segments = load_project(project_dir)
        orig_audio_path = os.path.join(project_dir, ORIGINAL_AUDIO)
        if not os.path.exists(orig_audio_path):
            self.show_error_msg("No original audio file in project folder.")
            return
        sr, data = safe_wavfile_read(orig_audio_path)
        self.orig_sr = sr
        self.orig_channels = data.shape[1] if data.ndim > 1 else 1
        self.load_models()
        video_path = os.path.join(project_dir, VIDEO_FILE)
        if not os.path.exists(video_path):
            self.show_error_msg("No video file in project folder.")
            return
        self.player_video.setSource(QUrl.fromLocalFile(video_path))
        self.player_video.pause()
        self.player_audio.pause()
        self.update_segment_list()
        self.regenerate_preview_audio(forced=False)
        print(f"Project loaded: {self.project_dir}")

    def close_all_threads(self):
        if self.recording_worker and self.recording_worker.is_alive():
            self.recording_worker.stop()
            self.recording_worker.join()
        self.recording_worker = None

    def save_project_action(self):
        if not self.project_dir:
            self.show_error_msg("No project loaded or created.")
            return
        save_project(self.project_dir, self.segments)
        QMessageBox.information(self, "Saved", "Project saved successfully.")

    def import_video(self):
        if not self.project_dir:
            self.show_error_msg("No project loaded or created.")
            return
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.mov *.mkv)")
        if not video_file:
            return
        target_video = os.path.join(self.project_dir, VIDEO_FILE)
        shutil.copyfile(video_file, target_video)
        sr, channels = extract_audio_from_video(target_video, os.path.join(self.project_dir, ORIGINAL_AUDIO))
        self.orig_sr = sr
        self.orig_channels = channels
        self.segments = []
        self.update_segment_list()
        self.player_video.setSource(QUrl.fromLocalFile(target_video))
        shutil.copyfile(os.path.join(self.project_dir, ORIGINAL_AUDIO), os.path.join(self.project_dir, PREVIEW_AUDIO))
        self.regenerate_preview_audio(forced=False)
        self.player_video.pause()
        self.player_audio.pause()

    def export_video_action(self):
        if not self.project_dir or self.orig_sr is None:
            self.show_error_msg("No project loaded or created.")
            return
        output_path, _ = QFileDialog.getSaveFileName(self, "Export Video", "", "MKV Files (*.mkv);;MP4 Files (*.mp4)")
        if not output_path:
            return
        video_path = os.path.join(self.project_dir, VIDEO_FILE)
        preview_audio_path = os.path.join(self.project_dir, PREVIEW_AUDIO)

        task = ProcessingTask(
            TaskType.EXPORT,
            project_dir=self.project_dir,
            video_path=video_path,
            preview_audio_path=preview_audio_path,
            output_path=output_path
        )
        self.processing_manager.add_task(task)

    def sync_playback(self):
        if self.player_video.playbackState() == QMediaPlayer.PlayingState:
            vid_pos = self.player_video.position()
            aud_pos = self.player_audio.position()
            if abs(aud_pos - vid_pos) > 100:
                self.player_audio.setPosition(vid_pos)

    def timer_event(self):
        self.process_sync_queue()
        QMetaObject.invokeMethod(self, "timer_event", Qt.QueuedConnection)

    def closeEvent(self, event):
        self.close_all_threads()
        self.processing_manager.stop()
        self.processing_manager.join()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self.timer_event()

    def update_segment_list(self):
        # Sort segments by start time
        self.segments.sort(key=lambda seg: seg.start_time)
        self.segment_list.clear()
        for seg in self.segments:
            self.segment_list.addItem(str(seg))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_play_pause()
        elif event.key() == Qt.Key_Left:
            pos = self.player_video.position() - 5000
            if pos < 0:
                pos = 0
            self.player_video.setPosition(pos)
            self.player_audio.setPosition(pos)
        elif event.key() == Qt.Key_Right:
            pos = self.player_video.position() + 5000
            if pos > self.player_video.duration():
                pos = self.player_video.duration()
            self.player_video.setPosition(pos)
            self.player_audio.setPosition(pos)
        elif event.key() == Qt.Key_R:
            self.toggle_recording()
        elif event.key() in [Qt.Key_Backspace, Qt.Key_Delete]:
            self.remove_selected_segment()
        elif event.key() == Qt.Key_P:
            self.reprocess_selected_segment()
        elif event.key() == Qt.Key_O:
            self.open_project()
        elif event.key() == Qt.Key_N:
            self.new_project()
        elif event.key() == Qt.Key_S:
            self.save_project_action()
        elif event.key() == Qt.Key_I:
            self.import_video()
        elif event.key() == Qt.Key_E:
            self.export_video_action()
        elif event.key() == Qt.Key_H:
            self.show_hotkeys()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 700)
    window.show()
    sys.exit(app.exec())
