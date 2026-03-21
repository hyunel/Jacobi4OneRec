import pandas as pd
from pathlib import Path

PATH = Path('outputs/fixed/jacobi_filtered_0_000000_fix.parquet')
MODEL_PATH = Path('outputs/OneRec-1.7B')

#%%
df = pd.read_parquet(PATH)
print(df.columns)

#%%
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

tokenizer.decode(df['prompt_ids'][0])
