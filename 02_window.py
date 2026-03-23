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
    parser.add_argument("--min_noisy_ratio", type=float, default=0.0)
    parser.add_argument("--max_noisy_ratio", type=float, default=1.0)
    parser.add_argument("--n_token_seq_len", type=int, default=32)
    parser.add_argument("--strategy", default="progressive")

    return parser.parse_args()

def main(args):
    df = pd.read_parquet(args.input_path)

    with pq.ParquetWriter(args.output_path, SCHEMA) as writer:
        for row in track(df.itertuples(index=False), total=len(df), description="Processing trajectories"):

            # Calculate noise ratios for each block
            noise_ratios = [
                [
                    1.0 - next((i for i, (a, b) in enumerate(zip(traj, block[-1])) if a != b), len(traj)) / len(traj)
                    for traj in block
                ]
                for block in row.blocks
            ]

            # Determine which trajectory to select based on the strategy
            if args.strategy == "random":
                noise_idx = np.random.randint(0, args.window_size, size=len(row.blocks))
            elif args.strategy == "progressive":
                schedule = np.linspace(args.min_noisy_ratio, args.max_noisy_ratio, args.window_size)
                noise_idx = [
                    int(np.argmin(np.abs(np.asarray(block_noise_ratios) - schedule[i % args.window_size])))
                    for i, block_noise_ratios in enumerate(noise_ratios)
                ]

            # Write the selected trajectories to the output Parquet file
            writer.write_table(pa.table({
                "uuid": [row.uuid],
                "source": [row.source],
                "prompt_ids": [row.prompt_ids],
                "blocks": [[row.blocks[i][noise_idx[i]] for i in range(len(row.blocks))]],
                "answers": [[block[-1] for block in row.blocks]],
            }, schema=SCHEMA))

if __name__ == "__main__":
    main(parse_args())
