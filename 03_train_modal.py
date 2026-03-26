from modal_app import app

N_GPU = 2

@app.function(
    gpu=f"A100-80GB:{N_GPU}",
    timeout=60 * 60 * 24,
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
                "--model_path", "/mnt/OpenOneRec/data/OneRec-1.7B/",
                "--data_path", "/mnt/outputs/trajectory_n32w16s1/*.parquet",

                "--max_blocks", "32",
                "--output_dir", "/mnt/outputs/OneRec-1.7B-n32w16m05",

                # "--max_blocks", "32",
                # "--output_dir", "/mnt/outputs/OneRec-1.7B-n32w16-lora",

                # "--use_lora", "True",
                # "--lora_r", "32",
                # "--lora_alpha", "16",
                # "--lora_dropout", "0.05",

                "--per_device_train_batch_size", "2",
                "--consistency_scale", "1.0",
                "--ar_loss_weight", "10.0",
                "--distill_temperature", "1.0",
                "--label_smoothing_factor", "0.1",

                "--bf16",
                "--report_to", "wandb",
                "--do_train",
                "--num_train_epochs", "1",
                "--save_strategy", "steps",
                "--save_steps", "5000",
                "--save_total_limit", "8",
                "--learning_rate", "1e-5",
                "--weight_decay", "0.",
                "--warmup_ratio", "0.03",
                "--lr_scheduler_type", "cosine",
                "--logging_steps", "10",
                "--deepspeed", "ds_config.json",
            ]
        )
    )