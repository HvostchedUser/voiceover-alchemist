from contextlib import contextmanager
import sys
import os
import time
import json
import threading
import shutil
from pathlib import Path
from dotenv import load_dotenv
from subprocess import run
import numpy as np
import queue

from PySide6.QtCore import (
    Qt, QUrl, QTimer, QThread, Signal, Slot, QObject
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QListWidget, QListWidgetItem,
    QSplitter, QToolBar, QFileDialog, QMessageBox
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import QProcess

import pyaudio
import wave
from scipy.io import wavfile
import librosa
import soundfile as sf

load_dotenv(".env")
from rvc.modules.vc.modules import VC

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
    """Context manager for locking file operations."""
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

class Segment:
    def __init__(self, start_time, end_time, recording_path, model_name,
                 processed=False, processing=False, leave_original=False,
                 processed_path=None):
        self.start_time = start_time
        self.end_time = end_time
        self.recording_path = recording_path
        self.model_name = model_name
        self.processed = processed
        self.processing = processing
        self.leave_original = leave_original
        self.processed_path = processed_path

    def duration(self):
        return self.end_time - self.start_time

    def __str__(self):
        status = "Processing" if self.processing else ("Done" if self.processed or self.leave_original else "Pending")
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
            "processed_path": self.processed_path
        }

    @staticmethod
    def from_dict(d):
        return Segment(d["start_time"], d["end_time"], d["recording_path"],
                       d["model_name"], d["processed"], d["processing"],
                       d["leave_original"], d.get("processed_path", None))

def resample_audio(audio, original_sr, target_sr):
    if original_sr == target_sr:
        return audio
    audio = audio.astype(np.float32) / 32768.0
    if audio.ndim == 1:
        resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    else:
        # per-channel
        resampled_channels = []
        for c in range(audio.shape[1]):
            channel_data = audio[:, c]
            resampled_channels.append(librosa.resample(channel_data, orig_sr=original_sr, target_sr=target_sr))
        resampled = np.stack(resampled_channels, axis=-1)

    resampled = (resampled * 32768.0).astype(np.int16)
    return resampled

def extract_audio_from_video(video_path, audio_path):
    temp_path = audio_path + ".temp.wav"
    run([FFMPEG, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", temp_path], check=True)
    sr, data = safe_wavfile_read(temp_path)
    shutil.move(temp_path, audio_path)
    channels = data.shape[1] if data.ndim > 1 else 1
    return sr, channels

def combine_audio(original_wav_path, segments, project_dir, preview_path, orig_sr, orig_channels):
    print("combine_audio")
    if not os.path.exists(original_wav_path):
        print("Original WAV file does not exist!")
        return
    sr, original_data = safe_wavfile_read(original_wav_path)
    original_data = original_data.astype(np.float32)

    if original_data.ndim == 1 and orig_channels > 1:
        original_data = np.tile(original_data[:, None], (1, orig_channels))

    for seg in segments:
        print("seg")
        seg_audio_path = None
        if seg.leave_original:
            seg_audio_path = os.path.join(project_dir, seg.recording_path)
        else:
            if seg.processed and seg.processed_path:
                seg_audio_path = os.path.join(project_dir, seg.processed_path)

        if seg_audio_path and os.path.exists(seg_audio_path):
            sr2, seg_data = safe_wavfile_read(seg_audio_path)
            seg_data = seg_data.astype(np.float32)
            if sr2 != orig_sr:
                seg_data = resample_audio(seg_data, sr2, orig_sr)

            # match channels
            if original_data.ndim == 1 and seg_data.ndim == 2:
                seg_data = np.mean(seg_data, axis=1)
            elif original_data.ndim == 2 and seg_data.ndim == 1:
                seg_data = np.tile(seg_data[:, None], (1, orig_channels))

            start_sample = int(seg.start_time * orig_sr)
            end_sample = start_sample + len(seg_data)
            if end_sample > len(original_data):
                seg_data = seg_data[:len(original_data) - start_sample]

            if original_data.ndim == 1:
                original_data[start_sample:end_sample] += seg_data
            else:
                original_data[start_sample:end_sample, :] += seg_data

    max_val = np.max(np.abs(original_data))
    if max_val > 32767.0:
        original_data *= (32767.0 / max_val)
    original_data = original_data.astype(np.int16)

    safe_wavfile_write(preview_path, orig_sr, original_data)
    if os.path.exists(preview_path):
        print(f"Combined audio written to {preview_path}")
        # Large file checks
        file_size = os.path.getsize(preview_path)
        print(f"Preview file size: {file_size} bytes")

        # Optional: read back if needed (just for debug)
        sr_check, data_check = safe_wavfile_read(preview_path)
        print(f"Combined audio sample rate: {sr_check}, shape: {data_check.shape}, max: {np.max(data_check)}, min: {np.min(data_check)}")

        # If arrays are huge, consider warning
        if data_check.size > 100_000_000:
            print("Warning: Very large audio array, consider shorter segments.")
    else:
        print("Error: Combined audio file was not created!")

def export_video(project_dir, orig_sr, callback):
    video_path = os.path.join(project_dir, VIDEO_FILE)
    preview_path = os.path.join(project_dir, PREVIEW_AUDIO)
    output_path = os.path.join(project_dir, "exported.mp4")

    process = QProcess()
    process.setProgram(FFMPEG)
    process.setArguments(["-y", "-i", video_path, "-i", preview_path, "-ar", str(orig_sr), "-c:v", "copy", "-c:a", "pcm_s16le", output_path])

    def finished(exitCode, exitStatus):
        callback(output_path, exitCode == 0)
        process.deleteLater()

    process.finished.connect(finished)
    process.start()

def save_project(project_dir, segments):
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
        QMessageBox.warning(None, "Error", f"Failed to save project: {e}")

def load_project(project_dir):
    with open(os.path.join(project_dir, PROJECT_FILE), "r") as f:
        data = json.load(f)
    segments = [Segment.from_dict(d) for d in data["segments"]]
    return segments

def ensure_project_structure(project_dir):
    os.makedirs(os.path.join(project_dir, RECORDINGS_DIR), exist_ok=True)
    os.makedirs(os.path.join(project_dir, PROCESSED_DIR), exist_ok=True)

class RecordingWorker(QObject):
    finished = Signal(bool)

    def __init__(self, output_path, record_sr, record_channels):
        super().__init__()
        self.output_path = output_path
        self._stop_flag = False
        self.record_sr = record_sr
        self.record_channels = record_channels

    def stop(self):
        self._stop_flag = True

    def run(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = self.record_channels
        RATE = self.record_sr
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        frames = []
        while not self._stop_flag:
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
        self.finished.emit(True)

class ProcessingThreadWorker(QObject):
    finished = Signal(str, str)  # (processed_path, rec_path)
    error = Signal(str)

    def __init__(self, project_dir, orig_sr, orig_channels):
        super().__init__()
        self.project_dir = project_dir
        self.orig_sr = orig_sr
        self.orig_channels = orig_channels
        self.running = True
        self.task_queue = queue.Queue()

    def add_task(self, segment):
        self.task_queue.put(segment)

    def stop(self):
        self.running = False

    def process_segment(self, segment):
        try:
            if segment.leave_original:
                processed_path = segment.recording_path
            else:
                vc = VC()
                model_name = segment.model_name
                model_dir = Path(MODELS_DIR, model_name)

                pth_files = list(model_dir.glob("*.pth"))
                if not pth_files:
                    raise FileNotFoundError("No .pth file found in model directory.")
                pth_file = pth_files[0]
                vc.get_vc(str(pth_file))

                index_files = list(model_dir.glob("*.index"))
                if not index_files:
                    raise FileNotFoundError("No index file found in model directory.")
                index_file = index_files[0]

                input_wav = os.path.join(self.project_dir, segment.recording_path)
                tgt_sr, audio_opt, times, _ = vc.vc_single(
                    sid=1,
                    input_audio_path=Path(input_wav),
                    index_file=str(index_file)
                )

                if self.orig_sr != tgt_sr:
                    audio_opt = resample_audio(audio_opt, tgt_sr, self.orig_sr)
                    tgt_sr = self.orig_sr

                # channel adjust
                if audio_opt.ndim == 1 and self.orig_channels > 1:
                    audio_opt = np.tile(audio_opt[:, None], (1, self.orig_channels))
                elif audio_opt.ndim == 2 and audio_opt.shape[1] != self.orig_channels:
                    if audio_opt.shape[1] < self.orig_channels:
                        audio_opt = np.pad(audio_opt, ((0,0),(0,self.orig_channels - audio_opt.shape[1])), mode='constant')
                    else:
                        audio_opt = audio_opt[:, :self.orig_channels]

                processed_path = os.path.join(PROCESSED_DIR, f"processed_{int(time.time()*1000)}.wav")
                full_processed_path = os.path.join(self.project_dir, processed_path)
                if audio_opt.dtype != np.int16:
                    max_val = np.max(np.abs(audio_opt))
                    if max_val > 32767:
                        audio_opt = (audio_opt.astype(np.float32)/max_val)*32767
                    audio_opt = audio_opt.astype(np.int16)

                safe_wavfile_write(full_processed_path, tgt_sr, audio_opt)

            self.finished.emit(processed_path, segment.recording_path)
        except Exception as e:
            self.error.emit(str(e))

    def run(self):
        while self.running:
            try:
                segment = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not self.running:
                break
            self.process_segment(segment)

class CombineThreadWorker(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(self, project_dir, orig_sr, orig_channels):
        super().__init__()
        self.project_dir = project_dir
        self.orig_sr = orig_sr
        self.orig_channels = orig_channels
        self.running = True
        self.task_queue = queue.Queue()

    def add_task(self, segments):
        print("add_task")
        self.task_queue.put(segments)

    def stop(self):
        self.running = False

    def process_combine(self, segments):
        try:
            print("process_combine")
            orig_path = os.path.join(self.project_dir, ORIGINAL_AUDIO)
            prev_path = os.path.join(self.project_dir, PREVIEW_AUDIO)
            combine_audio(orig_path, segments, self.project_dir, prev_path, self.orig_sr, self.orig_channels)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def run(self):
        while self.running:
            try:
                segments = self.task_queue.get(timeout=0.1)
                print("get")
            except queue.Empty:
                continue
            if not self.running:
                break
            self.process_combine(segments)

class ProcessingManager:
    def __init__(self, project_dir, orig_sr, orig_channels):
        self.thread = QThread()
        self.worker = ProcessingThreadWorker(project_dir, orig_sr, orig_channels)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

        # callback maps
        self.callbacks = {}
        self.error_callbacks = {}

        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

    def add_to_queue(self, segment, callback, error_callback):
        self.callbacks[segment.recording_path] = callback
        self.error_callbacks[segment.recording_path] = error_callback
        self.worker.add_task(segment)

    def on_finished(self, processed_path, rec_path):
        cb = self.callbacks.get(rec_path, None)
        if cb:
            del self.callbacks[rec_path]
            del self.error_callbacks[rec_path]
            cb(processed_path, rec_path)

    def on_error(self, error_msg):
        for ecb in self.error_callbacks.values():
            ecb(error_msg)
        self.callbacks.clear()
        self.error_callbacks.clear()

    def stop(self):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()

class CombineManager:
    def __init__(self, project_dir, orig_sr, orig_channels):
        self.thread = QThread()
        self.worker = CombineThreadWorker(project_dir, orig_sr, orig_channels)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

        self.tasks = queue.Queue()

        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

    def add_to_queue(self, segments, callback, error_callback):
        print("add_to_queue_combine")
        self.tasks.put((callback, error_callback))
        self.worker.add_task(segments)

    def on_finished(self):
        if not self.tasks.empty():
            cb, ecb = self.tasks.get()
            cb()

    def on_error(self, error_msg):
        if not self.tasks.empty():
            cb, ecb = self.tasks.get()
            print("on_error")
            ecb(error_msg)

    def stop(self):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Voice Dubbing Editor")

        self.project_dir = None
        self.segments = []
        self.orig_sr = None
        self.orig_channels = None
        self.processing_manager = None
        self.combine_manager = None

        self.is_recording = False
        self.player_video = QMediaPlayer()
        self.player_audio = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player_audio.setAudioOutput(self.audio_output)

        self.video_widget = QVideoWidget()
        self.player_video.setVideoOutput(self.video_widget)

        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.record_button = QPushButton("Record Segment")
        self.stop_record_button = QPushButton("Stop Recording")
        self.remove_button = QPushButton("Remove Selected Segment")
        self.reprocess_button = QPushButton("Reprocess Segment")

        self.model_combo = QComboBox()
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 1000)
        self.segment_list = QListWidget()

        self.play_button.clicked.connect(self.on_play)
        self.pause_button.clicked.connect(self.on_pause)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.remove_button.clicked.connect(self.remove_selected_segment)
        self.reprocess_button.clicked.connect(self.reprocess_selected_segment)

        self.player_video.positionChanged.connect(self.on_position_changed)
        self.player_video.durationChanged.connect(self.on_duration_changed)
        self.timeline_slider.sliderMoved.connect(self.on_slider_moved)
        self.segment_list.itemClicked.connect(self.on_segment_selected)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(QLabel("Voice Model:"))
        controls_layout.addWidget(self.model_combo)
        controls_layout.addWidget(self.record_button)
        controls_layout.addWidget(self.stop_record_button)
        controls_layout.addWidget(self.remove_button)
        controls_layout.addWidget(self.reprocess_button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_widget)
        left_layout.addWidget(self.timeline_slider)
        left_layout.addLayout(controls_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Recorded Segments:"))
        right_layout.addWidget(self.segment_list)
        right_layout.addStretch()

        splitter = QSplitter()
        left_widget = QWidget(); left_widget.setLayout(left_layout)
        right_widget = QWidget(); right_widget.setLayout(right_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 10)

        container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.addWidget(splitter)
        container.setLayout(container_layout)
        self.setCentralWidget(container)

        toolbar = QToolBar("Project Toolbar")
        self.addToolBar(toolbar)
        create_action = toolbar.addAction("Create Project")
        open_action = toolbar.addAction("Open Project")
        save_action = toolbar.addAction("Save Project")
        import_action = toolbar.addAction("Import Video")
        export_action = toolbar.addAction("Export Video")

        create_action.triggered.connect(self.new_project)
        open_action.triggered.connect(self.open_project)
        save_action.triggered.connect(self.save_project_action)
        import_action.triggered.connect(self.import_video)
        export_action.triggered.connect(self.export_video_action)

        self.player_video.pause()
        self.stop_record_button.setEnabled(False)

        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.sync_playback)
        self.playback_timer.start(500)

    def on_worker_error(self, error_msg):
        QMessageBox.warning(self, "Error", error_msg)

    def sync_playback(self):
        if self.player_video.playbackState() == QMediaPlayer.PlayingState:
            vid_pos = self.player_video.position()
            aud_pos = self.player_audio.position()
            if abs(aud_pos - vid_pos) > 50:
                self.player_audio.setPosition(vid_pos)

    def regenerate_preview_audio(self):
        print("Regenerating preview audio")
        if not self.project_dir or self.orig_sr is None:
            return
        orig_audio_path = os.path.join(self.project_dir, ORIGINAL_AUDIO)
        preview_audio_path = os.path.join(self.project_dir, PREVIEW_AUDIO)
        if not os.path.exists(orig_audio_path):
            print("Original audio file does not exist!")
            return
        if not self.segments:
            shutil.copyfile(orig_audio_path, preview_audio_path)
        else:
            combine_audio(orig_audio_path, self.segments, self.project_dir, preview_audio_path, self.orig_sr, self.orig_channels)
        self.player_audio.stop()
        self.player_audio.setSource(QUrl.fromLocalFile(preview_audio_path))
        self.player_audio.play()
        print(f"Regenerated preview audio: {preview_audio_path}")

    def load_models(self):
        self.model_combo.clear()
        self.model_combo.addItem(ORIGINAL_VOICE_OPTION)
        if os.path.isdir(MODELS_DIR):
            dirs = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
            self.model_combo.addItems(dirs)

    def is_original_voice_selected(self):
        return self.model_combo.currentText() == ORIGINAL_VOICE_OPTION

    def on_play(self):
        self.player_video.play()
        self.player_audio.play()

    def on_pause(self):
        self.player_video.pause()
        self.player_audio.pause()

    def on_duration_changed(self, duration):
        if duration > 0:
            self.timeline_slider.setMaximum(duration)

    def on_position_changed(self, position):
        if self.player_video.duration() > 0:
            self.timeline_slider.setValue(position)

    def on_slider_moved(self, position):
        self.player_video.setPosition(position)
        self.player_audio.setPosition(position)

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

    def on_combine_finished(self):
        # Immediately regenerate preview audio to ensure player is updated
        print("on_combine_finished")
        self.regenerate_preview_audio()

        was_playing = (self.player_video.playbackState() == QMediaPlayer.PlayingState)
        current_position = self.player_video.position()
        preview_audio_path = os.path.join(self.project_dir, PREVIEW_AUDIO)

        # Stop and reload player audio from the updated preview
        self.player_audio.stop()
        self.player_audio.setSource(QUrl.fromLocalFile(preview_audio_path))

        # Restore playback position
        self.player_video.setPosition(current_position)
        self.player_audio.setPosition(current_position)

        if was_playing:
            self.player_video.play()
            self.player_audio.play()
        else:
            self.player_video.pause()
            self.player_audio.pause()

        self.update_segment_list()
        print("Preview audio updated after combine.")

    def start_recording(self):
        if not self.project_dir or self.orig_sr is None:
            QMessageBox.warning(self, "No Project", "Please create or open a project first.")
            return
        if self.is_recording:
            return
        self.is_recording = True
        self.record_start_time = self.player_video.position()/1000.0
        rec_path = os.path.join(RECORDINGS_DIR, f"record_{int(time.time()*1000)}.wav")
        full_rec_path = os.path.join(self.project_dir, rec_path)

        self.record_thread = QThread()
        self.record_worker = RecordingWorker(full_rec_path, self.orig_sr, self.orig_channels)
        self.record_worker.moveToThread(self.record_thread)
        self.record_thread.started.connect(self.record_worker.run)
        self.record_worker.finished.connect(self.on_recording_finished)
        self.record_thread.start()
        self.stop_record_button.setEnabled(True)

        if self.player_video.playbackState() != QMediaPlayer.PlayingState:
            self.player_video.play()
            self.player_audio.play()
        self.current_recording_path = rec_path

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.stop_record_button.setEnabled(False)
        if self.record_worker:
            self.record_worker.stop()

    def on_recording_finished(self, success):
        end_time = self.player_video.position()/1000.0
        start_time = self.record_start_time
        duration = end_time - start_time
        if duration <= 0:
            QMessageBox.warning(self, "Error", "Recorded duration invalid.")
        else:
            model_name = self.model_combo.currentText() or ORIGINAL_VOICE_OPTION
            leave_original = self.is_original_voice_selected()

            seg = Segment(start_time, end_time, self.current_recording_path, model_name,
                          processed=leave_original,
                          processing=not leave_original,
                          leave_original=leave_original,
                          processed_path=(self.current_recording_path if leave_original else None))

            self.segments.append(seg)
            self.update_segment_list()
            save_project(self.project_dir, self.segments)

            if leave_original:
                self.combine_manager.add_to_queue(self.segments, self.on_combine_finished, self.on_worker_error)
            else:
                self.processing_manager.add_to_queue(seg, self.on_segment_processed, self.on_worker_error)

        self.record_thread.quit()
        self.record_thread.wait()
        self.record_thread = None
        self.record_worker = None

    def reprocess_selected_segment(self):
        idx = self.segment_list.currentRow()
        if idx < 0:
            return
        seg = self.segments[idx]
        model_name = self.model_combo.currentText()
        leave_original = self.is_original_voice_selected()

        seg.model_name = model_name
        seg.leave_original = leave_original
        seg.processed = leave_original
        seg.processing = not leave_original
        seg.processed_path = seg.recording_path if leave_original else None

        self.update_segment_list()
        save_project(self.project_dir, self.segments)

        if leave_original:
            self.combine_manager.add_to_queue(self.segments, self.on_combine_finished, self.on_worker_error)
        else:
            self.processing_manager.add_to_queue(seg, self.on_segment_processed, self.on_worker_error)

        
    def on_segment_processed(self, processed_path, rec_path):
        was_playing = (self.player_video.playbackState() == QMediaPlayer.PlayingState)
        current_position = self.player_video.position()

        # Update the processed segment
        for seg in self.segments:
            if seg.recording_path == rec_path:
                seg.processed = True
                seg.processing = False
                seg.processed_path = processed_path
                break

        self.update_segment_list()
        save_project(self.project_dir, self.segments)

        # After processing is done, combine the segments again so that the preview is updated
        # Use a small delay to avoid immediate re-entrancy issues
        print("add_to_queue")
        self.combine_manager.add_to_queue(self.segments, self.on_combine_finished, self.on_worker_error)

        self.player_video.setPosition(current_position)
        self.player_audio.setPosition(current_position)

        if was_playing:
            self.player_video.play()
            self.player_audio.play()
        else:
            self.player_video.pause()
            self.player_audio.pause()

        print(f"Processing complete for segment: {rec_path}")
        
    def update_segment_list(self):
        self.segment_list.clear()
        for seg in self.segments:
            self.segment_list.addItem(str(seg))

    def remove_selected_segment(self):
        idx = self.segment_list.currentRow()
        if idx < 0:
            return
        self.segments.pop(idx)
        self.update_segment_list()
        save_project(self.project_dir, self.segments)
        self.combine_manager.add_to_queue(self.segments, self.on_combine_finished, self.on_worker_error)

    def new_project(self):
        self.close_managers()
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
        self.setup_managers()
        self.regenerate_preview_audio()
        self.player_video.setSource(QUrl.fromLocalFile(target_video))
        self.player_video.pause()
        self.player_audio.pause()
        self.update_segment_list()
        save_project(self.project_dir, self.segments)
        print(f"New project created: {self.project_dir}")

    def open_project(self):
        self.close_managers()
        project_dir = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not project_dir:
            return
        project_file = os.path.join(project_dir, PROJECT_FILE)
        if not os.path.exists(project_file):
            QMessageBox.warning(self, "Error", "No project.json found in selected folder.")
            return
        self.project_dir = project_dir
        self.segments = load_project(project_dir)
        orig_audio_path = os.path.join(project_dir, ORIGINAL_AUDIO)
        if not os.path.exists(orig_audio_path):
            QMessageBox.warning(self, "Error", "No original audio file in project folder.")
            return
        sr, data = safe_wavfile_read(orig_audio_path)
        self.orig_sr = sr
        self.orig_channels = data.shape[1] if data.ndim > 1 else 1
        self.load_models()
        video_path = os.path.join(project_dir, VIDEO_FILE)
        if not os.path.exists(video_path):
            QMessageBox.warning(self, "Error", "No video file in project folder.")
            return
        self.setup_managers()
        self.player_video.setSource(QUrl.fromLocalFile(video_path))
        self.regenerate_preview_audio()
        self.player_video.pause()
        self.player_audio.pause()
        self.update_segment_list()
        print(f"Project loaded: {self.project_dir}")

    def close_managers(self):
        if self.processing_manager:
            self.processing_manager.stop()
            self.processing_manager = None
        if self.combine_manager:
            self.combine_manager.stop()
            self.combine_manager = None

    def setup_managers(self):
        self.processing_manager = ProcessingManager(self.project_dir, self.orig_sr, self.orig_channels)
        self.combine_manager = CombineManager(self.project_dir, self.orig_sr, self.orig_channels)

    def save_project_action(self):
        if not self.project_dir:
            QMessageBox.warning(self, "Error", "No project loaded or created.")
            return
        save_project(self.project_dir, self.segments)
        QMessageBox.information(self, "Saved", "Project saved successfully.")

    def import_video(self):
        if not self.project_dir:
            QMessageBox.warning(self, "Error", "No project loaded or created.")
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
        self.combine_manager.add_to_queue(self.segments, self.on_combine_finished, self.on_worker_error)
        self.player_video.pause()

    def export_video_action(self):
        if not self.project_dir or self.orig_sr is None:
            QMessageBox.warning(self, "Error", "No project loaded or created.")
            return

        output_path = os.path.join(self.project_dir, "exported.mkv")
        video_path = os.path.join(self.project_dir, VIDEO_FILE)
        preview_audio_path = os.path.join(self.project_dir, PREVIEW_AUDIO)

        result = run([
            FFMPEG, "-y",
            "-i", video_path,
            "-i", preview_audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            output_path
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FFmpeg Error: {result.stderr}")
            QMessageBox.warning(self, "Error", "Failed to export video.")
        else:
            print(f"Exported video successfully: {output_path}")
            QMessageBox.information(self, "Exported", f"Video exported to {output_path}")

    def closeEvent(self, event):
        self.close_managers()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 700)
    window.show()
    sys.exit(app.exec())