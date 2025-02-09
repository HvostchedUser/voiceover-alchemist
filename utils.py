import os
import time
import threading
import tempfile
import shutil
import uuid
import gc
import logging

from contextlib import contextmanager

import numpy as np
import wave
from scipy.io import wavfile
import librosa

from audio_separator.separator import Separator

# Initialize logging if needed
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

# A lock for thread-safe file operations.
file_lock = threading.Lock()

@contextmanager
def locked_file_operation():
    """Context manager for thread-safe file operations."""
    with file_lock:
        yield

def safe_wavfile_write(filepath, sample_rate, audio_data):
    """Write a WAV file in a thread-safe manner."""
    with locked_file_operation():
        wavfile.write(filepath, sample_rate, audio_data)
        logging.debug(f"File written successfully: {filepath}")

def safe_wavfile_read(filepath):
    """Read a WAV file in a thread-safe manner."""
    with locked_file_operation():
        sr, data = wavfile.read(filepath)
        logging.debug(f"File read successfully: {filepath}")
        return sr, data

def resample_audio(audio, original_sr, target_sr):
    """
    Resample audio array from original_sr to target_sr if needed.
    Audio is normalized to float32 during processing.
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

def process_audio(input_file, models, final_output_name="output_final.wav"):
    """
    Process an audio file with a chain of models sequentially.
    Uses the Separator from audio_separator.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "assets", "uvr5_weights")
    separator = Separator(model_file_dir=target_dir)
    temp_dir = tempfile.mkdtemp()
    logging.debug(f"Using temporary directory: {temp_dir}")

    current_file = input_file
    try:
        for i, (model, output_index) in enumerate(models):
            separator.load_model(model)
            output_files = separator.separate(current_file)
            logging.debug(f"Output files from {model}: {output_files}")
            if output_index >= len(output_files):
                raise ValueError(f"Invalid output index {output_index} for model {model}.")
            current_file = output_files[output_index]

            for f in output_files:
                temp_path = os.path.join(temp_dir, os.path.basename(f))
                shutil.move(f, temp_path)

            current_file = os.path.join(temp_dir, os.path.basename(current_file))
            logging.debug(f"Processed with {model}: {current_file}")

        final_output_path = os.path.join(os.path.dirname(input_file), final_output_name)
        shutil.copy(current_file, final_output_path)
        logging.debug(f"Final output saved as: {final_output_path}")
        return final_output_path
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.debug(f"Temporary directory {temp_dir} deleted.")
        del separator
        gc.collect()

def remove_voice_from_segment(segment_data_wav_path, model_path="Kim_Vocal_2.onnx"):
    """
    Remove voice from a portion of the original video audio.
    Uses the process_audio chain.
    """
    temp_output = os.path.join(tempfile.gettempdir(), f"voice_removed_{uuid.uuid4().hex}.wav")
    models_to_apply = [(model_path, 0)]
    final_output = process_audio(segment_data_wav_path, models_to_apply, final_output_name=os.path.basename(temp_output))
    if final_output != temp_output:
        shutil.copy(final_output, temp_output)
    return temp_output
