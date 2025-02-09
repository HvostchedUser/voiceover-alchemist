#!/usr/bin/env python
import os
import time
import tempfile
import shutil
import uuid
import logging
import subprocess
import gc

# Import shared utility functions from utils.py
from utils import safe_wavfile_read, safe_wavfile_write, resample_audio, remove_voice_from_segment

def process_segment_with_seedvc(seg, project_dir, orig_sr, orig_channels):
    """
    Process a segment using SeedVC.

    Expected: seg.model_name is of the form "SeedVC: <reference_name>".
    The function uses a reference voice sample from assets/seedvc_references.
    """
    # Extract the reference voice name.
    try:
        reference_name = seg.model_name.split("SeedVC:")[1].strip()
    except IndexError:
        raise ValueError("Invalid SeedVC model name format; should be 'SeedVC: <reference_name>'")
    
    # Determine the reference sample file.
    seedvc_ref_dir = os.path.join("assets", "seedvc_references")
    ref_file = None
    for ext in [".wav", ".mp3", ".flac"]:
        candidate = os.path.join(seedvc_ref_dir, reference_name + ext)
        if os.path.exists(candidate):
            ref_file = candidate
            break
    if not ref_file:
        raise FileNotFoundError(f"SeedVC reference voice sample not found for '{reference_name}' in {seedvc_ref_dir}")

    # Prepare the source audio file (recorded segment).
    source_wav = os.path.join(project_dir, seg.recording_path)
    if not os.path.exists(source_wav):
        raise FileNotFoundError(f"Segment source file not found: {source_wav}")

    # Create a temporary directory for SeedVC output.
    temp_output_dir = tempfile.mkdtemp()
    temp_output_file = os.path.join(temp_output_dir, "converted.wav")

    # Define the SeedVC model checkpoint and configuration.
    SEEDVC_CHECKPOINT = os.path.join("seedvc", "DiT_uvit_tat_xlsr_ema.pth")
    SEEDVC_CONFIG = os.path.join("seedvc", "configs", "presets", "config_dit_mel_seed_uvit_xlsr_tiny.yml")

    # Build the SeedVC command.
    cmd = [
        "python", os.path.join("seedvc", "inference.py"),
        "--source", source_wav,
        "--target", ref_file,
        "--output", temp_output_dir,
        "--diffusion-steps", "25",
        "--length-adjust", "1.0",
        "--inference-cfg-rate", "0.7",
        "--f0-condition", "False",
        "--auto-f0-adjust", "False",
        "--semi-tone-shift", str(seg.pitch),
        "--checkpoint", SEEDVC_CHECKPOINT,
        "--config", SEEDVC_CONFIG,
        "--fp16", "True"
    ]
    logging.info("Running SeedVC conversion: " + " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logging.error("SeedVC conversion failed: " + result.stderr)
        shutil.rmtree(temp_output_dir)
        raise RuntimeError("SeedVC conversion failed")
    
    # Assume that the SeedVC inference writes a file called converted.wav.
    if not os.path.exists(temp_output_file):
        shutil.rmtree(temp_output_dir)
        raise FileNotFoundError("Expected SeedVC output file not found.")

    # Copy the converted file to the project’s processed directory.
    processed_filename = f"seedvc_processed_{int(time.time()*1000)}.wav"
    processed_rel_path = os.path.join("processed", processed_filename)
    processed_full_path = os.path.join(project_dir, processed_rel_path)
    shutil.copy(temp_output_file, processed_full_path)
    seg.processed_path = processed_rel_path

    # Also generate a voice-removed version for preview.
    voice_removed_temp = remove_voice_from_segment(source_wav)
    voice_removed_filename = f"seedvc_voice_removed_{int(time.time()*1000)}.wav"
    voice_removed_rel_path = os.path.join("processed", voice_removed_filename)
    voice_removed_full_path = os.path.join(project_dir, voice_removed_rel_path)
    shutil.copy(voice_removed_temp, voice_removed_full_path)
    seg.voice_removed_path = voice_removed_rel_path

    seg.processing = False
    seg.processed = True

    shutil.rmtree(temp_output_dir)
    gc.collect()
    return seg
