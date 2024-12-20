from contextlib import contextmanager
from multiprocessing import Queue
from platform import release
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
from threading import Lock
from threading import Thread  # Add this import statement

from PySide6.QtCore import (
    Qt, QUrl, QThread, Signal, Slot, QObject, QMetaObject, QTimer
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QListWidget, QListWidgetItem,
    QSplitter, QToolBar, QFileDialog, QMessageBox
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

import pyaudio
import wave
from scipy.io import wavfile
import librosa
import soundfile as sf

load_dotenv(".env")
# Ensure that the following imports are correct and accessible
from rvc.modules.vc.modules import VC  # Ensure this path is correct
from audio_separator.separator import Separator  # Ensure this path is correct

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

separator = Separator()
vc = VC()

def process_audio(input_file, models, final_output_name="output_final.wav"):
    """
    Process an audio file using a sequence of machine learning models, keeping the input intact
    and storing intermediate files in a temporary folder. Allows specifying which output file to use.

    Args:
        input_file (str): Path to the input audio file.
        models (list of tuple): List of (model filename, output index) to apply in sequence.
        final_output_name (str): Name of the final output file.

    Returns:
        str: Path to the final processed file.
    """
    # Use the globally instantiated Separator
    global separator

    # Create a temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")

    current_file = input_file

    try:
        # Apply models sequentially
        for i, (model, output_index) in enumerate(models):
            separator.load_model(model)
            output_files = separator.separate(current_file)
            print(f"Output files from {model}: {output_files}")

            # Select the specified output file
            if output_index >= len(output_files):
                raise ValueError(f"Invalid output index {output_index} for model {model}.")
            current_file = output_files[output_index]

            # Move intermediate files to the temp directory
            for file in output_files:
                temp_path = os.path.join(temp_dir, os.path.basename(file))
                shutil.move(file, temp_path)

            current_file = os.path.join(temp_dir, os.path.basename(current_file))
            print(f"Processed with {model}: {current_file}")

        # Save the final output with a clean name in the same directory as the input file
        final_output_path = os.path.join(os.path.dirname(input_file), final_output_name)
        shutil.copy(current_file, final_output_path)

        print(f"Final output saved as: {final_output_path}")
        return final_output_path

    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Temporary directory {temp_dir} deleted.")

def remove_voice_from_segment(input_audio_path, model_path="Kim_Vocal_2.onnx"):
    """
    Remove voice from the provided audio segment using the given ONNX model.
    Returns the path to the voice-removed audio segment.
    """
    temp_output = os.path.join(tempfile.gettempdir(), f"voice_removed_{uuid.uuid4().hex}.wav")
    models_to_apply = [(model_path, 0)]
    
    # Corrected: Pass input_audio_path instead of input_file
    final_output = process_audio(input_audio_path, models_to_apply, final_output_name=os.path.basename(temp_output))

    # The process_audio function saves the output to the same directory as input_file
    # If final_output is not in tempfile, copy it:
    if final_output != temp_output:
        shutil.copy(final_output, temp_output)

    return temp_output

class Segment:
    def __init__(self, start_time, end_time, recording_path, model_name,
                 processed=False, processing=False, leave_original=False,
                 processed_path=None, voice_removed_path=None):
        self.start_time = start_time
        self.end_time = end_time
        self.recording_path = recording_path
        self.model_name = model_name
        self.processed = processed
        self.processing = processing
        self.leave_original = leave_original
        self.processed_path = processed_path
        self.voice_removed_path = voice_removed_path

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
            "processed_path": self.processed_path,
            "voice_removed_path": self.voice_removed_path
        }

    @staticmethod
    def from_dict(d):
        return Segment(d["start_time"], d["end_time"], d["recording_path"],
                       d["model_name"], d["processed"], d["processing"],
                       d["leave_original"], d.get("processed_path", None),
                       d.get("voice_removed_path", None))

def resample_audio(audio, original_sr, target_sr):
    if original_sr == target_sr:
        return audio

    # Convert to float32 for processing
    audio = audio.astype(np.float32) / 32768.0

    if audio.ndim == 1:
        resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    else:
        # Resample each channel separately
        resampled_channels = []
        for c in range(audio.shape[1]):
            channel_data = audio[:, c]
            resampled_channel = librosa.resample(channel_data, orig_sr=original_sr, target_sr=target_sr)
            resampled_channels.append(resampled_channel)
        resampled = np.stack(resampled_channels, axis=-1)

    # After resampling, convert back to int16
    resampled = (resampled * 32768.0).clip(-32768, 32767).astype(np.int16)
    return resampled


def extract_audio_from_video(video_path, audio_path):
    temp_path = audio_path + ".temp.wav"
    run([FFMPEG, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", temp_path], check=True)
    sr, data = safe_wavfile_read(temp_path)
    shutil.move(temp_path, audio_path)
    channels = data.shape[1] if data.ndim > 1 else 1
    return sr, channels

def combine_audio(original_wav_path, segments, project_dir, preview_path, orig_sr, orig_channels, crossfade_duration=0.2):
    import uuid  # Ensure uuid is imported
    print("combine_audio with smooth transitions and consistent sample rates")
    if not os.path.exists(original_wav_path):
        print("Original WAV file does not exist!")
        return
    sr, original_data = safe_wavfile_read(original_wav_path)

    # Ensure original audio sample rate matches orig_sr (if not, resample)
    if sr != orig_sr:
        original_data = resample_audio(original_data, sr, orig_sr)
        sr = orig_sr  # Update sample rate after resampling

    # Convert to float32 for processing
    original_data = original_data.astype(np.float32)

    if original_data.ndim == 1 and orig_channels > 1:
        original_data = np.tile(original_data[:, None], (1, orig_channels))

    # Copy the base data to allow restoration where no segments exist
    base_data = original_data.copy()

    for seg in segments:
        print(f"Processing segment: {seg}")
        start_sample = int(seg.start_time * orig_sr)
        end_sample = int(seg.end_time * orig_sr)
        crossfade_samples = int(crossfade_duration * orig_sr)

        if end_sample > len(original_data):
            end_sample = len(original_data)

        # Ensure voice removal is applied to the corresponding portion of the original audio
        if not seg.voice_removed_path or not os.path.exists(os.path.join(project_dir, seg.voice_removed_path)):
            # Extract the segment portion of the original audio
            segment_data = base_data[start_sample:end_sample]
            temp_segment_file = os.path.join(tempfile.gettempdir(), f"original_segment_{uuid.uuid4().hex}.wav")
            safe_wavfile_write(temp_segment_file, orig_sr, segment_data.astype(np.int16))

            # Remove voice from the segment portion
            voice_removed_temp = remove_voice_from_segment(temp_segment_file)
            voice_removed_path_rel = os.path.join(PROCESSED_DIR, f"voice_removed_{int(time.time()*1000)}.wav")
            voice_removed_path = os.path.join(project_dir, voice_removed_path_rel)
            shutil.move(voice_removed_temp, voice_removed_path)

            # Store the result in the segment object
            seg.voice_removed_path = voice_removed_path_rel
            save_project(project_dir, segments)

        # Read the voice-removed audio for this segment and ensure correct sample rate
        sr_vr, vr_data = safe_wavfile_read(os.path.join(project_dir, seg.voice_removed_path))
        # Resample if necessary
        if sr_vr != orig_sr:
            vr_data = resample_audio(vr_data, sr_vr, orig_sr)
            sr_vr = orig_sr  # Update sample rate after resampling

        # Convert to float32
        vr_data = vr_data.astype(np.float32)

        # Match the number of channels
        if vr_data.ndim == 1 and orig_channels > 1:
            vr_data = np.tile(vr_data[:, None], (1, orig_channels))
        elif vr_data.ndim > 1 and orig_channels == 1:
            vr_data = np.mean(vr_data, axis=1)
        elif vr_data.ndim > 1 and orig_channels > 1:
            if vr_data.shape[1] != orig_channels:
                # Adjust channels if mismatched
                if vr_data.shape[1] < orig_channels:
                    vr_data = np.pad(vr_data, ((0,0),(0, orig_channels - vr_data.shape[1])), mode='constant')
                else:
                    vr_data = vr_data[:, :orig_channels]

        if vr_data.shape[0] != (end_sample - start_sample):
            # Resize if needed
            diff = (end_sample - start_sample) - vr_data.shape[0]
            if diff > 0:
                # pad with zeros
                vr_data = np.pad(vr_data, ((0,diff), (0, 0) if vr_data.ndim > 1 else (0,)), mode='constant')
            else:
                vr_data = vr_data[:end_sample - start_sample]

        # Apply crossfade at the start and end of the segment
        if crossfade_samples > 0:
            # Determine actual crossfade samples to avoid exceeding boundaries
            actual_crossfade_start = min(crossfade_samples, start_sample)
            actual_crossfade_end = min(crossfade_samples, len(original_data) - end_sample)

            # Create fade curves
            fade_out = np.linspace(1.0, 0.0, actual_crossfade_start, endpoint=False, dtype=np.float32).reshape(-1, 1)
            fade_in = np.linspace(0.0, 1.0, actual_crossfade_start, endpoint=False, dtype=np.float32).reshape(-1, 1)

            # Apply fade-out to the end of the region before the segment
            if actual_crossfade_start > 0:
                original_fade_out_region = original_data[start_sample - actual_crossfade_start:start_sample]
                original_data[start_sample - actual_crossfade_start:start_sample] *= fade_out

                # Apply fade-in to the beginning of the voice-removed segment
                vr_fade_in_region = vr_data[:actual_crossfade_start]
                vr_data[:actual_crossfade_start] *= fade_in

                # Overlap the faded out original with faded in new audio
                original_data[start_sample - actual_crossfade_start:start_sample] += vr_data[:actual_crossfade_start]

            # Similarly handle the end of the segment
            if actual_crossfade_end > 0:
                fade_out_end = np.linspace(1.0, 0.0, actual_crossfade_end, endpoint=False, dtype=np.float32).reshape(-1, 1)
                fade_in_end = np.linspace(0.0, 1.0, actual_crossfade_end, endpoint=False, dtype=np.float32).reshape(-1, 1)

                # Apply fade-out to the end of the voice-removed segment
                vr_fade_out_region = vr_data[-actual_crossfade_end:]
                vr_data[-actual_crossfade_end:] *= fade_out_end

                # Apply fade-in to the beginning of the original data after the segment
                original_fade_in_region = original_data[end_sample:end_sample + actual_crossfade_end]
                original_data[end_sample:end_sample + actual_crossfade_end] *= fade_in_end

                # Overlap the faded out new audio with faded in original audio
                original_data[end_sample:end_sample + actual_crossfade_end] += vr_data[-actual_crossfade_end:]

        # Replace the original audio portion with the voice-removed version for the non-crossfade region
        if crossfade_samples > 0:
            non_crossfade_start = crossfade_samples
            non_crossfade_end = vr_data.shape[0] - crossfade_samples if crossfade_samples < vr_data.shape[0] else vr_data.shape[0]
            original_data[start_sample + non_crossfade_start:end_sample - crossfade_samples] = vr_data[non_crossfade_start:non_crossfade_end]
        else:
            original_data[start_sample:end_sample] = vr_data

        # Overlay the recorded voice (either processed or original)
        if seg.recording_path:
            seg_audio_path = os.path.join(project_dir, seg.processed_path or seg.recording_path)
            if os.path.exists(seg_audio_path):
                sr2, seg_data = safe_wavfile_read(seg_audio_path)
                # Resample if necessary
                if sr2 != orig_sr:
                    seg_data = resample_audio(seg_data, sr2, orig_sr)
                    sr2 = orig_sr  # Update sample rate after resampling

                # Convert to float32
                seg_data = seg_data.astype(np.float32)

                # Match the number of channels
                if seg_data.ndim == 1 and orig_channels > 1:
                    seg_data = np.tile(seg_data[:, None], (1, orig_channels))
                elif seg_data.ndim > 1 and orig_channels == 1:
                    seg_data = np.mean(seg_data, axis=1)
                elif seg_data.ndim > 1 and orig_channels > 1:
                    if seg_data.shape[1] != orig_channels:
                        # Adjust channels if mismatched
                        if seg_data.shape[1] < orig_channels:
                            seg_data = np.pad(seg_data, ((0,0),(0, orig_channels - seg_data.shape[1])), mode='constant')
                        else:
                            seg_data = seg_data[:, :orig_channels]

                if seg_data.shape[0] != (end_sample - start_sample):
                    # Resize if needed
                    diff = (end_sample - start_sample) - seg_data.shape[0]
                    if diff > 0:
                        seg_data = np.pad(seg_data, ((0, diff), (0, 0) if seg_data.ndim > 1 else (0,)), mode='constant')
                    else:
                        seg_data = seg_data[:end_sample - start_sample]

                # Overlay the recorded audio onto the voice-removed portion
                seg_len = min(len(seg_data), end_sample - start_sample)
                if seg_len > 0:
                    if original_data.ndim == 1:
                        original_data[start_sample:start_sample + seg_len] += seg_data[:seg_len]
                    else:
                        original_data[start_sample:start_sample + seg_len, :] += seg_data[:seg_len, :]

    # After processing all segments
    # Normalize the audio to avoid clipping
    max_val = np.max(np.abs(original_data))
    if max_val > 32767.0:
        original_data *= (32767.0 / max_val)
    original_data = original_data.astype(np.int16)

    # Write the combined audio to the preview path
    safe_wavfile_write(preview_path, orig_sr, original_data)
    if os.path.exists(preview_path):
        print(f"Combined audio written to {preview_path}")
    else:
        print("Error: Combined audio file was not created!")


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

class RecordingWorker(threading.Thread):
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

class ProcessingThread(Thread):
    def __init__(self, project_dir, orig_sr, orig_channels, segment, callback, error_callback):
        super().__init__()
        self.project_dir = project_dir
        self.orig_sr = orig_sr
        self.orig_channels = orig_channels
        self.segment = segment
        self.callback = callback
        self.error_callback = error_callback

    def run(self):
        try:
            if self.segment.leave_original:
                processed_path = self.segment.recording_path
            else:
                model_name = self.segment.model_name
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

                input_wav = os.path.join(self.project_dir, self.segment.recording_path)
                tgt_sr, audio_opt, times, _ = vc.vc_single(
                    sid=1,
                    input_audio_path=Path(input_wav),
                    index_file=str(index_file),
                    filter_radius=0,
                    rms_mix_rate=0.0,
                    protect=0.5
                )

                if self.orig_sr != tgt_sr:
                    audio_opt = resample_audio(audio_opt, tgt_sr, self.orig_sr)
                    tgt_sr = self.orig_sr

                # Channel adjust
                if audio_opt.ndim == 1 and self.orig_channels > 1:
                    audio_opt = np.tile(audio_opt[:, None], (1, self.orig_channels))
                elif audio_opt.ndim == 2 and audio_opt.shape[1] != self.orig_channels:
                    if audio_opt.shape[1] < self.orig_channels:
                        audio_opt = np.pad(audio_opt, ((0, 0), (0, self.orig_channels - audio_opt.shape[1])), mode='constant')
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

            self.callback(processed_path, self.segment.recording_path)
        except Exception as e:
            self.error_callback(f"Error processing segment: {str(e)}")


class ProcessingThreadQueue:
    def __init__(self):
        self.queue = []  # Use a list to store tasks
        self.lock = Lock()
        self.active_thread = None

    def add_to_queue(self, project_dir, orig_sr, orig_channels, segment, callback, error_callback):
        with self.lock:
            self.queue.append((project_dir, orig_sr, orig_channels, segment, callback, error_callback))
        self._start_next()

    def _start_next(self):
        with self.lock:
            if self.active_thread is not None and self.active_thread.is_alive():
                return  # A thread is already running

            if self.queue:  # Start the next task in the queue
                project_dir, orig_sr, orig_channels, segment, callback, error_callback = self.queue.pop(0)
                self.active_thread = ProcessingThread(
                    project_dir, orig_sr, orig_channels, segment, callback, error_callback
                )
                self.active_thread.start()
    def _on_thread_complete(self, callback, error_callback):
        try:
            callback()  # Call the success callback
        except Exception as e:
            error_callback(str(e))  # Handle errors
        finally:
            with self.lock:
                self.active_thread = None  # Reset active thread
                self._start_next()  # Start the next task

    
class CombineThread(Thread):
    def __init__(self, project_dir, orig_sr, orig_channels, segments, callback):
        super().__init__()
        self.project_dir = project_dir
        self.orig_sr = orig_sr
        self.orig_channels = orig_channels
        self.segments = segments
        self.callback = callback

    def run(self):
        try:
            orig_path = os.path.join(self.project_dir, ORIGINAL_AUDIO)
            preview_path = os.path.join(self.project_dir, PREVIEW_AUDIO)
            combine_audio(orig_path, self.segments, self.project_dir, preview_path, self.orig_sr, self.orig_channels)
            self.callback()
        except Exception as e:
            self.error_callback(str(e))



class CombineThreadQueue:
    def __init__(self):
        self.queue = []  # Use a list to store tasks
        self.lock = Lock()
        self.active_thread = None

    def add_to_queue(self, project_dir, orig_sr, orig_channels, segments, callback, error_callback):
        with self.lock:
            self.queue.append((project_dir, orig_sr, orig_channels, segments, callback, error_callback))
        self._start_next()

    def _start_next(self):
        with self.lock:
            if self.active_thread is not None and self.active_thread.is_alive():
                return  # A thread is already running, do nothing

            if self.queue:  # If the queue has pending tasks
                project_dir, orig_sr, orig_channels, segments, callback, error_callback = self.queue.pop(0)
                self.active_thread = CombineThread(
                    project_dir, orig_sr, orig_channels, segments,
                    lambda: self._on_thread_complete(callback, error_callback)
                )
                self.active_thread.start()

    def _on_thread_complete(self, callback, error_callback):
        try:
            callback()  # Call the success callback
        except Exception as e:
            error_callback(str(e))  # Handle errors
        finally:
            with self.lock:
                self.active_thread = None  # Reset active thread
                self._start_next()  # Start the next task

class ExportVideoWorker(threading.Thread):
    def __init__(self, video_path, preview_audio_path, orig_sr, output_path, callback, error_callback):
        super().__init__()
        self.video_path = video_path
        self.preview_audio_path = preview_audio_path
        self.orig_sr = orig_sr
        self.output_path = output_path
        self.callback = callback
        self.error_callback = error_callback

    def run(self):
        try:
            # Using subprocess.run instead of QProcess
            result = run([
                FFMPEG, "-y",
                "-i", self.video_path,
                "-i", self.preview_audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                self.output_path
            ], stdout=PIPE, stderr=PIPE, text=True)

            if result.returncode != 0:
                error_msg = result.stderr
                self.error_callback(error_msg)
                self.callback(False, self.output_path)
            else:
                self.callback(True, self.output_path)
        except Exception as e:
            self.error_callback(str(e))
            self.callback(False, self.output_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Voice Dubbing Editor")

        self.project_dir = None
        self.segments = []
        self.orig_sr = None
        self.orig_channels = None
        self.processing_threads = []
        self.combine_thread = None

        self.is_recording = False
        self.recording_worker = None

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

        # Connect buttons to their respective methods
        self.play_button.clicked.connect(self.on_play)
        self.pause_button.clicked.connect(self.on_pause)
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.remove_button.clicked.connect(self.remove_selected_segment)
        self.reprocess_button.clicked.connect(self.reprocess_selected_segment)

        # Connect media player signals
        self.player_video.positionChanged.connect(self.on_position_changed)
        self.player_video.durationChanged.connect(self.on_duration_changed)
        self.timeline_slider.sliderMoved.connect(self.on_slider_moved)
        self.segment_list.itemClicked.connect(self.on_segment_selected)

        # Setup CombineThreadQueue
        self.combine_thread_queue = CombineThreadQueue()
        self.processing_thread_queue = ProcessingThreadQueue()

        # Layout setup
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

        # Connect toolbar actions
        create_action.triggered.connect(self.new_project)
        open_action.triggered.connect(self.open_project)
        save_action.triggered.connect(self.save_project_action)
        import_action.triggered.connect(self.import_video)
        export_action.triggered.connect(self.export_video_action)

        # Initial UI state
        self.player_video.pause()
        self.player_audio.pause()
        self.stop_record_button.setEnabled(False)

        # Replace QTimer with Python's threading
        self.sync_thread = threading.Thread(target=self.sync_playback_loop, daemon=True)
        self.sync_queue = queue.Queue()
        self.sync_thread.start()

        # Connect mediaStatusChanged to handle media load completion
        self.player_audio.mediaStatusChanged.connect(self.on_media_status_changed)
        self.pending_position = None
        self.was_playing = False

    def on_worker_error(self, error_msg):
        QMessageBox.warning(self, "Error", error_msg)

    def sync_playback_loop(self):
        """Background thread to synchronize playback."""
        while True:
            time.sleep(0.5)  # Equivalent to QTimer interval of 500ms
            self.sync_queue.put(None)  # Placeholder, can be used to send data if needed

    def process_sync_queue(self):
        """Process items from the sync queue."""
        while not self.sync_queue.empty():
            self.sync_playback()

    def regenerate_preview_audio(self):
        print("Regenerating preview audio")
        if not self.project_dir or self.orig_sr is None:
            return
        orig_audio_path = os.path.join(self.project_dir, ORIGINAL_AUDIO)
        preview_audio_path = os.path.join(self.project_dir, PREVIEW_AUDIO)
        if not os.path.exists(orig_audio_path):
            print("Original audio file does not exist!")
            return

        # Save current playback state
        self.was_playing = self.player_audio.playbackState() == QMediaPlayer.PlayingState
        self.current_position = self.player_audio.position()

        if not self.segments:
            shutil.copyfile(orig_audio_path, preview_audio_path)
            print(f"Copied original audio to preview: {preview_audio_path}")
        else:
            # Trigger combine_audio via the CombineThread
            if self.combine_thread and self.combine_thread.is_alive():
                print("Combine thread is already running.")
            else:
                self.combine_thread_queue.add_to_queue(
                    self.project_dir,
                    self.orig_sr,
                    self.orig_channels,
                    self.segments,
                    self.on_combine_finished,
                    self.on_worker_error
                )

        # Update the audio player
        self.update_audio_player(preview_audio_path)

    def update_audio_player(self, preview_path):
        # Disconnect previous signal to avoid multiple connections
        self.player_audio.mediaStatusChanged.disconnect(self.on_media_status_changed)

        # Connect the signal to set position after media is loaded
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
        # Re-sync playback after combining
        print("Combine finished, regenerating preview audio.")
        self.regenerate_preview_audio()
        QMetaObject.invokeMethod(self, "update_segment_list", Qt.QueuedConnection)


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

        # Define callbacks
        def recording_finished(success):
            self.on_recording_finished(success, rec_path)

        def recording_error(error_msg):
            self.on_worker_error(error_msg)

        # Initialize and start the recording thread
        self.recording_worker = RecordingWorker(
            full_rec_path,
            self.orig_sr,
            self.orig_channels,
            recording_finished,
            recording_error
        )
        self.recording_worker.start()
        self.stop_record_button.setEnabled(True)

        # Start playback if not already playing
        if self.player_video.playbackState() != QMediaPlayer.PlayingState:
            self.player_video.play()
            self.player_audio.play()
        self.current_recording_path = rec_path

    def on_recording_finished(self, success, rec_path):
        if not success:
            QMessageBox.warning(self, "Recording Error", "Recording failed.")
            self.is_recording = False
            self.stop_record_button.setEnabled(False)
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
                processed=leave_original,
                processing=not leave_original,
                leave_original=leave_original,
                processed_path=(rec_path if leave_original else None)
            )

            self.segments.append(seg)
            QMetaObject.invokeMethod(self, "update_segment_list", Qt.QueuedConnection)

            save_project(self.project_dir, self.segments)

            if leave_original:
                self.combine_thread_queue.add_to_queue(
                    self.project_dir,
                    self.orig_sr,
                    self.orig_channels,
                    self.segments,
                    self.on_combine_finished,
                    self.on_worker_error
                )

            else:
                self.processing_thread_queue.add_to_queue(
                    self.project_dir,
                    self.orig_sr,
                    self.orig_channels,
                    seg,
                    self.on_segment_processed,
                    self.on_worker_error
                )

        self.is_recording = False
        self.stop_record_button.setEnabled(False)

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.stop_record_button.setEnabled(False)
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
        seg.processed = leave_original
        seg.processing = not leave_original
        seg.processed_path = seg.recording_path if leave_original else None

        QMetaObject.invokeMethod(self, "update_segment_list", Qt.QueuedConnection)

        save_project(self.project_dir, self.segments)

        if leave_original:
            self.combine_thread_queue.add_to_queue(
                self.project_dir,
                self.orig_sr,
                self.orig_channels,
                self.segments,
                self.on_combine_finished,
                self.on_worker_error
            )
        else:
            self.processing_thread_queue.add_to_queue(
                self.project_dir,
                self.orig_sr,
                self.orig_channels,
                seg,
                self.on_segment_processed,
                self.on_worker_error
            )

    def on_segment_processed(self, processed_path, rec_path):
        # Update the processed segment
        for seg in self.segments:
            if seg.recording_path == rec_path:
                seg.processed = True
                seg.processing = False
                seg.processed_path = processed_path
                break

        QMetaObject.invokeMethod(self, "update_segment_list", Qt.QueuedConnection)

        save_project(self.project_dir, self.segments)

        # Recombine segments to update preview audio
        self.combine_thread_queue.add_to_queue(
            self.project_dir,
            self.orig_sr,
            self.orig_channels,
            self.segments,
            self.on_combine_finished,
            self.on_worker_error
        )

    @Slot()
    def update_segment_list(self):
        self.segment_list.clear()
        for seg in self.segments:
            self.segment_list.addItem(str(seg))

    def remove_selected_segment(self):
        idx = self.segment_list.currentRow()
        if idx < 0:
            return
        self.segments.pop(idx)
        QMetaObject.invokeMethod(self, "update_segment_list", Qt.QueuedConnection)

        save_project(self.project_dir, self.segments)
        self.combine_thread_queue.add_to_queue(
            self.project_dir,
            self.orig_sr,
            self.orig_channels,
            self.segments,
            self.on_combine_finished,
            self.on_worker_error
        )

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
        self.regenerate_preview_audio()
        self.player_video.setSource(QUrl.fromLocalFile(target_video))
        self.player_video.pause()
        self.player_audio.pause()
        QMetaObject.invokeMethod(self, "update_segment_list", Qt.QueuedConnection)

        save_project(self.project_dir, self.segments)
        print(f"New project created: {self.project_dir}")

    def open_project(self):
        self.close_all_threads()
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
        self.player_video.setSource(QUrl.fromLocalFile(video_path))
        self.regenerate_preview_audio()
        self.player_video.pause()
        self.player_audio.pause()
        QMetaObject.invokeMethod(self, "update_segment_list", Qt.QueuedConnection)

        print(f"Project loaded: {self.project_dir}")

    def close_all_threads(self):
        # Stop all processing threads
        for thread in self.processing_threads:
            if thread.is_alive():
                # Threads will exit naturally
                pass
        self.processing_threads.clear()

        # Stop combine thread
        if self.combine_thread and self.combine_thread.is_alive():
            # Threads will exit naturally
            pass
        self.combine_thread = None

        # Stop recording thread
        if self.recording_worker and self.recording_worker.is_alive():
            self.recording_worker.stop()
            self.recording_worker.join()
        self.recording_worker = None

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
        QMetaObject.invokeMethod(self, "update_segment_list", Qt.QueuedConnection)

        self.player_video.setSource(QUrl.fromLocalFile(target_video))
        self.regenerate_preview_audio()
        self.player_video.pause()
        self.player_audio.pause()

    def export_video_action(self):
        if not self.project_dir or self.orig_sr is None:
            QMessageBox.warning(self, "Error", "No project loaded or created.")
            return

        output_path, _ = QFileDialog.getSaveFileName(self, "Export Video", "", "MKV Files (*.mkv);;MP4 Files (*.mp4)")
        if not output_path:
            return

        video_path = os.path.join(self.project_dir, VIDEO_FILE)
        preview_audio_path = os.path.join(self.project_dir, PREVIEW_AUDIO)

        # Perform export in a separate thread to avoid blocking the GUI
        export_thread = ExportVideoWorker(
            video_path,
            preview_audio_path,
            self.orig_sr,
            output_path,
            self.on_export_finished,
            self.on_worker_error
        )
        export_thread.start()

    def on_export_finished(self, success, output_path):
        if not success:
            QMessageBox.warning(self, "Error", "Failed to export video.")
        else:
            QMessageBox.information(self, "Exported", f"Video exported to {output_path}")

    def sync_playback(self):
        """Synchronize audio and video playback."""
        if self.player_video.playbackState() == QMediaPlayer.PlayingState:
            vid_pos = self.player_video.position()
            aud_pos = self.player_audio.position()
            if abs(aud_pos - vid_pos) > 50:
                self.player_audio.setPosition(vid_pos)

    def timer_event(self):
        """Custom method to handle synchronization without QTimer."""
        self.process_sync_queue()
        # Schedule the next call
        QMetaObject.invokeMethod(self, "timer_event", Qt.QueuedConnection)

    def closeEvent(self, event):
        self.close_all_threads()
        super().closeEvent(event)

    def showEvent(self, event):
        """Override showEvent to start the synchronization loop."""
        super().showEvent(event)
        # Start processing the sync queue periodically using a timer-like approach
        # Since we removed QTimer, use invokeMethod to repeatedly call timer_event
        self.timer_event()

    def on_combine_finished(self):
        # Re-sync playback after combining
        print("Combine finished, regenerating preview audio.")
        self.regenerate_preview_audio()
        QMetaObject.invokeMethod(self, "update_segment_list", Qt.QueuedConnection)


    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            if self.pending_position is not None:
                self.player_audio.setPosition(self.pending_position)
                self.pending_position = None
            if self.was_playing and self.player_video.playbackState() == QMediaPlayer.PlayingState:
                self.player_audio.play()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 700)
    window.show()
    sys.exit(app.exec())
