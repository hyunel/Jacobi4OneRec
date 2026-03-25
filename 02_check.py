#%%
# Load and inspect the generated trajectories
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

PATH = Path('outputs/trajectory_n32w16/jacobi_filtered_0_000000.parquet')
MODEL_PATH = Path('outputs/OneRec-1.7B')

df = pd.read_parquet(PATH)
print(f"Loaded {len(df)} trajectories from {PATH}")

#%%
# Roll 5 samples for inspection
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

rows = df.sample(5).itertuples(index=False)
for row in rows:
    print(f"UUID: {row.uuid}, Source: {row.source}")
    print(f"Prompt: {tokenizer.decode(row.prompt_ids, skip_special_tokens=False)}".replace("\n", "\\n"))
    for i, (block, answer) in enumerate(zip(row.blocks[:5], row.answers[:5])):
        print(f"    Trajectory {i+1}/{len(row.blocks)}:")
        print(f"      Block: {tokenizer.decode(block, skip_special_tokens=False)}".replace("\n", "\\n"))
        print(f"      Answer: {tokenizer.decode(answer, skip_special_tokens=False)}".replace("\n", "\\n"))
    print()

# %%
# Plot distribution of block lengths
block_lengths = [len(row.blocks) for row in df.itertuples(index=False)]
tick_values = np.arange(min(block_lengths), max(block_lengths) + 1)
plt.hist(block_lengths, bins=np.arange(tick_values[0] - 0.5, tick_values[-1] + 1.5), edgecolor='black')
plt.xticks(tick_values[::max(1, len(tick_values) // 8)])
plt.title("Distribution of Block Lengths")
plt.xlabel("Block Length")
plt.ylabel("Frequency")
plt.show()

# %%
# Plot distribution of prompt lengths
prompt_lengths = [len(row.prompt_ids) for row in df.itertuples(index=False)]
tick_values = np.arange(min(prompt_lengths), max(prompt_lengths) + 1)
plt.hist(prompt_lengths, bins=np.arange(tick_values[0] - 0.5, tick_values[-1] + 1.5))
plt.xticks(tick_values[::max(1, len(tick_values) // 8)])
plt.title("Distribution of Prompt Lengths")
plt.xlabel("Prompt Length")
plt.ylabel("Frequency")
plt.show()

# %%
# Plot total length distribution under different max block counts
block_token_length = len(df.iloc[0].blocks[0])
max_block_counts = [8, 16, 32]
max_prompt_length = max(prompt_lengths)

for max_blocks in max_block_counts:
    total_lengths = [prompt_length + max_blocks * block_token_length for prompt_length in prompt_lengths]
    plt.hist(total_lengths, bins=30, histtype='step', linewidth=2, label=f"max_blocks={max_blocks}")
    print(f"max_blocks={max_blocks}: max_total_length={max_prompt_length + max_blocks * block_token_length}")

plt.title("Distribution of Total Lengths")
plt.xlabel("Total Length")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# %%
