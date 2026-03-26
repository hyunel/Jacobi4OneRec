import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from rich.progress import track

SCHEMA = pa.schema([
    ("uuid", pa.string()),
    ("source", pa.string()),
    ("prompt_ids", pa.list_(pa.int32())),
    ("blocks", pa.list_(pa.list_(pa.int32()))),
    ("answers", pa.list_(pa.list_(pa.int32()))),
])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--block_slide_step", type=int, default=1)
    parser.add_argument("--min_noisy_ratio", type=float, default=0.0)
    parser.add_argument("--max_noisy_ratio", type=float, default=1.0)
    parser.add_argument("--n_token_seq_len", type=int, default=32)
    parser.add_argument("--strategy", default="progressive")

    return parser.parse_args()

def process_block(args, prompt_ids, blocks, noise_ratios):
    n_blocks = len(blocks)

    # Determine which trajectory to select based on the strategy
    if args.strategy == "random":
        noise_idx = np.random.randint(0, args.window_size, size=n_blocks)
    elif args.strategy == "progressive":
        # (args.min_noisy_ratio, args.max_noisy_ratio]
        schedule = np.linspace(args.max_noisy_ratio, args.min_noisy_ratio, args.window_size)[::-1]
        noise_idx = [
            int(np.argmin(np.abs(noise_ratios[j] - schedule[j % args.window_size])))
            for j in range(n_blocks)
        ]

    return {
        "prompt_ids": prompt_ids,
        "blocks": [blocks[j][noise_idx[j]] for j in range(n_blocks)],
        "answers": [block[-1] for block in blocks],
    }

def main(args):
    df = pd.read_parquet(args.input_path)

    with pq.ParquetWriter(args.output_path, SCHEMA) as writer:
        for row in track(df.itertuples(index=False), total=len(df), description="Processing trajectories"):
            n_blocks = len(row.blocks)

            # Calculate noise ratios for each block
            noise_ratios = [
                [
                    1.0 - next((i for i, (a, b) in enumerate(zip(traj, block[-1])) if a != b), len(traj)) / len(traj)
                    for traj in block
                ]
                for block in row.blocks
            ]

            # Process each block
            results = []
            prompt_ids = row.prompt_ids
            for i in range(min(n_blocks, args.block_slide_step)):
                # For each block, prompt is the concatenation of the original prompt and the previous blocks
                if i > 0:
                    prompt_ids = np.concatenate([prompt_ids, row.blocks[i-1][-1]])

                results.append(process_block(args, prompt_ids, row.blocks[i:], noise_ratios[i:]))

            # Write the selected trajectories to the output Parquet file
            writer.write_table(pa.table({
                "uuid": [row.uuid] * len(results),
                "source": [row.source] * len(results),
                "prompt_ids": [result["prompt_ids"] for result in results],
                "blocks": [result["blocks"] for result in results],
                "answers": [result["answers"] for result in results],
            }, schema=SCHEMA))


if __name__ == "__main__":
    main(parse_args())
