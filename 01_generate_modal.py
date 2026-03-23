import subprocess
from modal_app import app
from pathlib import Path

DATASETS = [
    # "/mnt/OpenOneRec/output/sft_product_rec.parquet",
    # "/mnt/OpenOneRec/output/sft_video_rec.parquet",
    # "/mnt/OpenOneRec/output/sft_ad_rec.parquet",
    # "/mnt/OpenOneRec/output/sft_interactive_rec.parquet",
    # "/mnt/OpenOneRec/output/sft_rec_reason.parquet",
    # "/mnt/OpenOneRec/raw_data/general_text/sft/filtered_0_000000.parquet",
    # "/mnt/OpenOneRec/raw_data/general_text/sft/filtered_1_000000.parquet",
    # "/mnt/OpenOneRec/raw_data/general_text/sft/filtered_2_000000.parquet",
    # "/mnt/OpenOneRec/raw_data/general_text/sft/filtered_3_000000.parquet",
    "/mnt/OpenOneRec/raw_data/general_text/sft/filtered_4_000000.parquet",
    "/mnt/OpenOneRec/raw_data/general_text/sft/filtered_5_000000.parquet",
    "/mnt/OpenOneRec/raw_data/general_text/sft/filtered_6_000000.parquet",
    "/mnt/OpenOneRec/raw_data/general_text/sft/filtered_7_000000.parquet",
]

OUTPUT_DIR = Path("/mnt/outputs/")

@app.function(
    gpu="A100-40GB",
    timeout=60 * 60 * 24,
)
def launch_generation():
    args = [
        "python", "01_generate.py",
        "--model", "/mnt/OpenOneRec/data/OneRec-1.7B/",
        "--n_token_seq_len", "32",
        "--max_prompt_len", "2048",
        "--max_new_seq_len", "1024",
        "--batch_size", "32",
        "--temperature", "0.75",
        "--top_k", "20",
        "--top_p", "0.95",
    ]

    for dataset in DATASETS:
        path = Path(dataset)
        assert path.exists(), f"Dataset path {dataset} does not exist."

        path_output = OUTPUT_DIR / f"jacobi_{path.stem}.parquet"
        # if path_output.exists():
        #     print(f"Output for dataset {dataset} already exists at {path_output}, skipping generation.")
        #     continue

        subprocess.run([
            *args,
            "--dataset", dataset,
            "-o", path_output,
        ], check=False)