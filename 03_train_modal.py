from modal_app import app

N_GPU = 2

@app.function(
    gpu=f"A100-80GB:{N_GPU}",
    timeout=60 * 60 * 1,
)
# @modal.experimental.clustered(size=n_nodes)
def launch_torchrun():
    from torch.distributed.run import parse_args, run

    run(
        parse_args(
            [
                f"--nnodes=1",
                f"--nproc-per-node={N_GPU}",
                "--master_port=1234",
                "03_train.py",
                "--target_model_path", "/mnt/OpenOneRec/data/OneRec-1.7B/",
                "--data_path", "/mnt/outputs/fixed_windowed/*.parquet",
                "--output_dir", "/mnt/outputs/OneRec-1.7B-n64w8",
                "--max_new_tokens", "64",
                "--bf16",
                "--report_to", "wandb",
                "--do_train",
                "--num_train_epochs", "1",
                "--per_device_train_batch_size", "1",
                "--gradient_accumulation_steps", "1",
                "--gradient_checkpointing",
                "--save_strategy", "steps",
                "--save_steps", "500",
                "--save_total_limit", "8",
                "--learning_rate", "1e-5",
                "--weight_decay", "0.",
                "--warmup_ratio", "0.03",
                "--lr_scheduler_type", "cosine",
                "--logging_steps", "10",
                "--model_max_length", "16384",
                "--qlora", "False",
                "--deepspeed", "ds_config.json",
            ]
        )
    )