import os
from pathlib import Path
import requests

RVC_DOWNLOAD_LINK = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/"
BASE_DIR = Path(os.getcwd())


def dl_model(link, model_name, dir_name):
    with requests.get(f"{link}{model_name}") as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(dir_name / model_name), exist_ok=True)
        with open(dir_name / model_name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

print("Downloading hubert_base.pt...")
dl_model(RVC_DOWNLOAD_LINK, "hubert_base.pt", BASE_DIR / "assets/hubert")
print("Downloading rmvpe.pt...")
dl_model(RVC_DOWNLOAD_LINK, "rmvpe.pt", BASE_DIR / "assets/rmvpe")
print("Downloading vocals.onnx...")
dl_model(
    RVC_DOWNLOAD_LINK + "uvr5_weights/onnx_dereverb_By_FoxJoy/",
    "vocals.onnx",
    BASE_DIR / "assets/uvr5_weights/onnx_dereverb_By_FoxJoy",
)

rvc_models_dir = BASE_DIR / "assets/pretrained"

print("Downloading pretrained models:")

model_names = [
    "D32k.pth",
    "D40k.pth",
    "D48k.pth",
    "G32k.pth",
    "G40k.pth",
    "G48k.pth",
    "f0D32k.pth",
    "f0D40k.pth",
    "f0D48k.pth",
    "f0G32k.pth",
    "f0G40k.pth",
    "f0G48k.pth",
]

for model in model_names:
    print(f"Downloading {model}...")
    dl_model(RVC_DOWNLOAD_LINK + "pretrained/", model, rvc_models_dir)

rvc_models_dir = BASE_DIR / "assets/pretrained_v2"

print("Downloading uvr5_weights:")

rvc_models_dir = BASE_DIR / "assets/uvr5_weights"

model_names = [
    "HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth",
    "HP2_all_vocals.pth",
    "HP3_all_vocals.pth",
    "HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth",
    "HP5_only_main_vocal.pth",
    "VR-DeEchoAggressive.pth",
    "VR-DeEchoDeReverb.pth",
    "VR-DeEchoNormal.pth",
]

for model in model_names:
    print(f"Downloading {model}...")
    dl_model(RVC_DOWNLOAD_LINK + "uvr5_weights/", model, rvc_models_dir)

print("All models downloaded!")

import click
import os
env_file_path = os.path.join(os.getcwd(), ".env")

if not os.path.exists(env_file_path):
    default_values = {
        "weight_root": "",
        "weight_uvr5_root": "assets/uvr5_weights",
        "index_root": "",
        "rmvpe_root": "assets/rmvpe",
        "hubert_path": "assets/hubert/hubert_base.pt",
        "save_uvr_path": "",
        "TEMP": "",
        "pretrained": "assets/pretrained",
    }

    with open(env_file_path, "w") as env_file:
        for key, value in default_values.items():
            env_file.write(f"{key}={value}\n")

    click.echo(f"{env_file_path} created successfully.")
else:
    click.echo(f"{env_file_path} already exists, no change")


from audio_separator.separator import Separator
script_dir = os.path.dirname(os.path.abspath(__file__))

target_dir = os.path.join(script_dir, "assets", "uvr5_weights")
separator = Separator(model_file_dir = target_dir)
separator.load_model("Kim_Vocal_2.onnx")