import json
import torch
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets, Features, Value
from rich.console import Console
from rich.progress import Progress
from patch import forward_jacobi

console = Console()
SCHEMA = pa.schema([
    ("uuid", pa.string()),
    ("source", pa.string()),
    ("prompt_ids", pa.list_(pa.int32())),
    ("blocks", pa.list_(pa.list_(pa.list_(pa.int32())))),
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
        .batch(batch_size)
        .shuffle(seed=42)
    )

def main(args: argparse.Namespace):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.forward_jacobi = forward_jacobi.__get__(model) # Monkey patching the method to the model instance

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"

    schema = SCHEMA.with_metadata({
        'model': args.model,
        'n_token_seq_len': str(args.n_token_seq_len),
        'max_new_seq_len': str(args.max_new_seq_len),
    })

    dataset = load_data(args.dataset, args.batch_size)
    progress = Progress(console=console)
    with pq.ParquetWriter(args.output, schema) as writer, progress:
        task = progress.add_task("Generating...", total=None)
        try:
            for batch in dataset:
                if not any(u == '91b35870-a4ec-11f0-8f4b-946daee91278' for u in batch['uuid']):
                    continue
                for i in range(len(batch['uuid'])):
                    console.print(f"[dim] Messages: {batch['messages'][i]}[/dim]")

                progress.advance(task)
                prompts = tokenizer.apply_chat_template(
                    batch['messages'],
                    tokenize=False,
                    add_generation_prompt=True
                )

                model_inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(model.device)

                input_ids = model_inputs["input_ids"]
                attention_mask = model_inputs["attention_mask"]

                B, _ = input_ids.shape

                # input_ids = model_inputs["input_ids"]
                # print('Input IDs length:', len(input_ids[0]))
                # while True:
                #     out = model(input_ids)
                #     next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
                #     input_ids = torch.cat((input_ids, next_token.unsqueeze(-1)), dim=-1)
                #     print(tokenizer.decode(next_token[0].item(), skip_special_tokens=False), end="")
                #     if (next_token == tokenizer.eos_token_id).any():
                #         break
                # continue

                # Prefill Phase
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
                past_key_values, next_token = model.forward_jacobi(
                    input_ids=input_ids,
                    attention_mask=attention_mask, # For padding tokens in the prompt
                    position_ids=position_ids,
                    past_key_values=None,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    accept_threshold=args.accept_threshold,
                )

                # Generation Phase
                seq_length = attention_mask.sum(dim=-1)
                trajectories = [[] for _ in range(B)]
                for i_block in range((args.max_new_seq_len + args.n_token_seq_len - 1) // args.n_token_seq_len):
                    active_idx = [
                        i for i, traj in enumerate(trajectories)
                        if (not traj) or (not (traj[-1][-1] == tokenizer.eos_token_id).any())
                    ]

                    if not active_idx:
                        break

                    q_sampled = torch.randint(0, tokenizer.vocab_size, (B, args.n_token_seq_len-1), device=model.device)

                    input_ids = torch.cat((next_token, q_sampled), dim=-1)
                    position_ids = torch.arange(args.n_token_seq_len, device=model.device).unsqueeze(0) + seq_length.unsqueeze(1)
                    position_ids += i_block * args.n_token_seq_len

                    attention_mask = torch.cat((
                        attention_mask,
                        torch.ones(B, args.n_token_seq_len, device=attention_mask.device, dtype=attention_mask.dtype)
                    ), dim=-1)

                    past_key_values, next_token, trajectory = model.forward_jacobi(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        accept_threshold=args.accept_threshold,
                    )

                    for i in active_idx:
                        trajectories[i].append(trajectory[i])
                        decoded = tokenizer.decode(trajectory[i][-1], skip_special_tokens=False)
                        console.print(f"[dim][Sample {i}][/dim] {decoded}")

                # Write batch to parquet
                masks = model_inputs["attention_mask"].bool()
                writer.write_table(pa.table({
                    "prompt_ids": [p[m].tolist() for p, m in zip(model_inputs["input_ids"], masks)],
                    "blocks": [
                        [[step.tolist() for step in block] for block in blocks]
                        for blocks in trajectories
                    ],
                    "uuid": batch["uuid"],
                    "source": batch["source"],
                }, schema=schema))
        except Exception as e:
            console.print("[red]Generation interrupted. Saving progress...[/red]")
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, action="append", help="Path(s) to input dataset(s) in Parquet format.")
    parser.add_argument("--output", "-o", type=Path, default='output.parquet', help="Path to output Parquet file.")
    parser.add_argument("--n_token_seq_len", type=int, default=64)
    parser.add_argument("--max_new_seq_len", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--accept_threshold", type=float, default=0.98,
                        help="Draft acceptance threshold for non-greedy decoding.")
    args = parser.parse_args()

    main(args)