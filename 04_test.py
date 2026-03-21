from collections import defaultdict
import json
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Features, Value
from rich.console import Console
from rich.progress import Progress
from inference_engine.llm import LLM
from inference_engine.sampling_params import SamplingParams

console = Console()

def convert_messages(messages):
    msg_list = []
    for msg in messages:
        content = msg['content']
        if isinstance(content, str):
            msg_list.append({
                'role': msg['role'],
                'content': content
            })
        elif isinstance(content, dict) and 'type' in content and content['type'] == 'text':
            msg_list.append({
                'role': msg['role'],
                'content': content['text']
            })
        elif isinstance(content, list) and len(content) > 0:
            content_text = ""
            for c in content:
                if isinstance(c, dict) and 'type' in c and c['type'] == 'text':
                    content_text += c['text']
                elif isinstance(c, str):
                    content_text += c
                else:
                    continue
            msg_list.append({
                'role': msg['role'],
                'content': content_text
            })
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    return msg_list

def load_data(dataset_paths):
    datasets = [
        load_dataset("parquet", data_files=[str(path)], split="train", streaming=True)
            .select_columns(["messages", "uuid", "source"])
            .cast(Features({
                "messages": Value("large_string"),
                "uuid": Value("string"),
                "source": Value("string")
            }))
        for path in dataset_paths
    ]

    return (concatenate_datasets(datasets, axis=0)
        .map(lambda x: {
            **x,
            "messages": convert_messages(json.loads(x["messages"])[:-1])
        })
        .shuffle(seed=42)
    )

def main(args: argparse.Namespace):
    llm = LLM(
        args.model,
        enforce_eager=False,
        tensor_parallel_size=1,

        max_num_seqs=max(16, args.batch_size),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,

        jacobi_enabled=True,
        jacobi_block_len=args.n_token_seq_len,
        jacobi_max_block_len=min(args.n_token_seq_len, args.max_model_len) + 1
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        decode_strategy="jacobi",
        jacobi_on_policy=False,

        max_tokens=args.max_blocks * args.n_token_seq_len,
        jacobi_block_len=args.n_token_seq_len,
        jacobi_max_iterations=args.max_blocks,
    )

    def _tokenize(x):
        prompts = tokenizer.apply_chat_template(
            x['messages'],
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer(
            prompts,
            padding=False,
            truncation=True,
        )

        return {
            **x,
            "input_ids": model_inputs["input_ids"]
        }

    def _filter(x):
        if len(x['input_ids']) <= args.max_prompt_len:
            return True
        console.print(f"[yellow]Skipping sample with UUID {x['uuid']} due to prompt length {len(x['input_ids'])} exceeding max_prompt_len {args.max_prompt_len}[/yellow]")
        return False

    dataset = (load_data(args.dataset)
        .map(_tokenize)
        .filter(_filter)
        .batch(args.batch_size)
    )

    for batch in dataset:
        for i in range(len(batch['uuid'])):
            console.print(f"[dim] Messages: {batch['messages'][i]}[/dim]")

        outputs = llm.generate(batch['input_ids'], sampling_params)
        for output in outputs:
            console.print(f"[green] Generated: {output['text']}[/green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", help="Path(s) to input dataset(s) in Parquet format.")
    parser.add_argument("--n_token_seq_len", type=int, default=64)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--max_blocks", type=int, default=32)
    parser.add_argument("--max_prompt_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    main(args)