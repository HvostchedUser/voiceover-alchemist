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
