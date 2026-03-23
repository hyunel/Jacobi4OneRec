import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl")
    # .add_local_python_source('inference_engine')
    .add_local_python_source('modal_app')
    .add_local_file("01_generate.py", remote_path="/root/01_generate.py")
    .add_local_file("patch.py", remote_path="/root/patch.py")
    .add_local_file("03_train.py", remote_path="/root/03_train.py")
    .add_local_file("trainer.py", remote_path="/root/trainer.py")
    .add_local_file("ds_config.json", remote_path="/root/ds_config.json")
)

app = modal.App(
    "onerec",
    image=image,
    secrets=[
        modal.Secret.from_name('wandb')
    ],
    volumes={
        '/mnt': modal.Volume.from_name('onerec')
    }
)