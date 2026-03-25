import argparse
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer
from functools import partial

disable_caching()

SYSTEM_PROMPT = [
    151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n_token_seq_len", type=int, required=True)
    parser.add_argument("--trim_system_prompt", action="store_true")
    parser.add_argument("--add_think_token", action="store_true")

    return parser.parse_args()

def map_fn(args, append_tokens, row):
    n_prompt = row["prompt_ids_len"]
    input_ids = row["complete_training_sequence_ids"]
    prompt_ids, remaining_ids = input_ids[:n_prompt], input_ids[n_prompt:]
    assert len(prompt_ids) == n_prompt, f"Expected prompt_ids length {n_prompt}, but got {len(prompt_ids)}"
    assert len(remaining_ids) % args.n_token_seq_len == 0, f"Remaining ids length {len(remaining_ids)} is not divisible by n_token_seq_len {args.n_token_seq_len}"

    blocks, answers = [], []
    for i_start in range(0, len(remaining_ids), args.n_token_seq_len * 2):
        block_ids = remaining_ids[i_start:i_start + args.n_token_seq_len]
        answer_ids = remaining_ids[i_start + args.n_token_seq_len:i_start + args.n_token_seq_len * 2]

        blocks.append(block_ids)
        answers.append(answer_ids)

    if args.trim_system_prompt:
        assert prompt_ids[:len(SYSTEM_PROMPT)] == SYSTEM_PROMPT, "The beginning of prompt_ids does not match the expected system prompt"
        prompt_ids = prompt_ids[len(SYSTEM_PROMPT):]

    return {
        "source": row["data_id"],
        "prompt_ids": prompt_ids + append_tokens,
        "blocks": blocks,
        "answers": answers,
    }

def main(args):
    dataset = load_dataset("parquet", data_files=args.input_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    append_tokens = []
    if args.add_think_token:
        append_tokens = tokenizer("<think>\n\n</think>\n\n", add_special_tokens=False).input_ids

    dataset.map(
        partial(map_fn, args, append_tokens),
        num_proc=4,
        remove_columns=dataset.column_names,
    ).to_parquet(args.output_path)

if __name__ == "__main__":
    main(parse_args())