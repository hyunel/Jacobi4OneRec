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
SCHEMA = pa.schema([
    ("source", pa.string()),
    ("data_id", pa.string()),
    ("diffusion_itr_id", pa.string()),
    ("prompt_ids", pa.list_(pa.int32())),
    ("answer_trajectory_ids", pa.list_(pa.list_(pa.int32()))),
    ("teacher_output_ids", pa.list_(pa.int32()))
])

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

def load_data(dataset_paths, batch_size):
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
        jacobi_on_policy=True,

        max_tokens=args.max_blocks * args.n_token_seq_len,
        jacobi_block_len=args.n_token_seq_len,
        jacobi_max_iterations=args.max_blocks,
    )

    schema = SCHEMA.with_metadata({
        'model': args.model,
        'n_token_seq_len': str(args.n_token_seq_len),
        'max_model_len': str(args.max_model_len),
        'max_blocks': str(args.max_blocks),
        'top_k': str(args.top_k),
        'top_p': str(args.top_p),
        'temperature': str(args.temperature)
    })

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

    dataset = (load_data(args.dataset, args.batch_size)
        .map(_tokenize)
        .filter(_filter)
        .batch(args.batch_size)
    )

    progress = Progress(console=console)
    with pq.ParquetWriter(args.output, schema) as writer, progress:
        task = progress.add_task("Generating...", total=None)
        for batch in dataset:
            for i in range(len(batch['uuid'])):
                console.print(f"[dim] Messages: {batch['messages'][i]}[/dim]")

            progress.advance(task)

            try:
                rollouts = llm.generate(batch['input_ids'], sampling_params)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    raise e
                console.print(f"[red]Generation error for batch with UUIDs {[u for u in batch['uuid']]}: {e}[/red]")
                continue

            for rollout, uuid, source in zip(rollouts, batch['uuid'], batch['source']):
                blocks = defaultdict(list)
                for i_block in range(len(rollout)):
                    for k, v in rollout[i_block].items():
                        blocks[k].append(v)

                writer.write_table(pa.table({
                    "source": [source] * len(rollout),
                    "data_id": [uuid] * len(rollout),
                    "diffusion_itr_id": blocks['diffusion_itr_id'],
                    "prompt_ids": blocks["prompt_ids"],
                    "answer_trajectory_ids": blocks["answer_trajectory_ids"],
                    "teacher_output_ids": blocks["teacher_output_ids"]
                }, schema=schema))

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", help="Path(s) to input dataset(s) in Parquet format.")
    parser.add_argument("--output", "-o", type=Path, default='output.parquet', help="Path to output Parquet file.")
    parser.add_argument("--n_token_seq_len", type=int, default=32)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_blocks", type=int, default=32)
    parser.add_argument("--max_prompt_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--top_p", type=float, default=0.95)

    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_args())