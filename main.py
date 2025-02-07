from pathlib import Path
from dotenv import load_dotenv
from scipy.io import wavfile
from rvc.modules.vc.modules import VC
from rvc.modules.vc.modules import uvr5
import glob
import gc  # For garbage collection

load_dotenv(".env")
vc = VC()

try:
    model_name = "model"

    # Load the only .pth file in the folder
    pth_files = list(Path(f"assets/models/{model_name}/").glob("*.pth"))
    if pth_files:
        pth_file = pth_files[0]
    else:
        raise FileNotFoundError("No .pth file found in the specified directory.")

    vc.get_vc(str(pth_file))

    # Load the only index file in the folder
    index_files = list(Path(f"assets/models/{model_name}/").glob("*.index"))
    if index_files:
        index_file = index_files[0]
    else:
        raise FileNotFoundError("No index file found in the specified directory.")

    # Process the audio
    tgt_sr, audio_opt, times, _ = vc.vc_single(1, Path("input.wav"), index_file=str(index_file), filter_radius=10, rms_mix_rate=0.0, protect=0.5)
    
    # Save the output with proper cleanup
    try:
        wavfile.write(f"output.wav", tgt_sr, audio_opt)
    except Exception as e:
        print(f"Error saving audio: {e}")
        raise
    finally:
        # Clear any references to the audio data
        del audio_opt
        gc.collect()

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Cleanup
    if 'vc' in locals():
        del vc
    gc.collect()
