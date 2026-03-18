import modal
import modal.experimental

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch~=2.5.1", "numpy~=2.2.1", "transformers", "datasets", "accelerate", "peft", "wandb")
    .add_local_dir(
        "training", remote_path="/root/training"
    )
)
app = modal.App("onerec-torch-cluster", image=image)

n_nodes = 4

@app.function(
    gpu=f"H100:8",
    timeout=60 * 60 * 24,
)
@modal.experimental.clustered(size=n_nodes)
def launch_torchrun():
    from torch.distributed.run import parse_args, run

    cluster_info = modal.experimental.get_cluster_info()

    run(
        parse_args(
            [
                f"--nnodes={n_nodes}",
                f"--node_rank={cluster_info.rank}",
                f"--master_addr={cluster_info.container_ips[0]}",
                f"--nproc-per-node=8",
                "--master_port=1234",
                "training/train.py",
            ]
        )
    )