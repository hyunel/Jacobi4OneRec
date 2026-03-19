import json
import torch
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, AutoProcessor, Qwen3ForCausalLM
from datasets import load_dataset, concatenate_datasets, Features, Value
from rich.progress import track
from rich.console import Console
from patch import forward_jacobi

# ~40M input tokens

console = Console()
SCHEMA = pa.schema([])

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

def load_data():
    data1 = load_dataset("parquet", data_files=[
        "data/OpenOneRec/output/sft_rec_reason.parquet",
        "data/OpenOneRec/output/sft_video_rec.parquet",
    ], split="train", streaming=True).select_columns(["messages"])

    data2 = load_dataset("parquet", data_files=[
        "data/OpenOneRec/raw_data/general_text/sft/filtered_0_000000.parquet",
        "data/OpenOneRec/raw_data/general_text/sft/filtered_1_000000.parquet",
        "data/OpenOneRec/raw_data/general_text/sft/filtered_2_000000.parquet",
        "data/OpenOneRec/raw_data/general_text/sft/filtered_3_000000.parquet",
        "data/OpenOneRec/raw_data/general_text/sft/filtered_4_000000.parquet",
    ], split="train", streaming=True).select_columns(["messages"]).cast(Features({"messages": Value("large_string")}))

    data = (concatenate_datasets([data1, data2], axis=0)
        .map(lambda x: {
            "messages": convert_messages(json.loads(x["messages"])[:-1])
        })
        .batch(2)
        .shuffle())
    return data

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

    dataset = load_data()
    with pq.ParquetWriter(args.output, schema) as writer:
        for batch in dataset:
            prompts = tokenizer.apply_chat_template(
                batch['messages'], # TODO: support batch size > 1
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model.device)

            B, L = model_inputs["input_ids"].shape

            # Prefill Phase
            past_key_values, next_token = model.forward_jacobi(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"], # For padding tokens in the prompt
                past_key_values=None,
                n_token_seq_len=args.n_token_seq_len,
                tokenizer=tokenizer,
            )

            # Generation Phase
            generated_ids = model_inputs["input_ids"]
            for n_block in range((args.max_new_seq_len + args.n_token_seq_len - 1) // args.n_token_seq_len):
                active_idx = (~(generated_ids[:, L:] == tokenizer.eos_token_id).any(dim=1)).nonzero(as_tuple=True)[0]
                if len(active_idx) == 0:
                    break

                q_sampled = torch.randint(0, tokenizer.vocab_size, (B, args.n_token_seq_len-1), device=model.device)
                past_key_values, next_token, trajectory = model.forward_jacobi(
                    input_ids=torch.cat((next_token, q_sampled),dim=-1),
                    attention_mask=None,
                    past_key_values=past_key_values,
                    n_token_seq_len=args.n_token_seq_len,

                    tokenizer=tokenizer,
                )
                generated_ids = torch.cat((generated_ids, torch.stack([traj[-1] for traj in trajectory])), dim=-1) # concat blocks

                for i in active_idx:
                    generated_str = ''.join(tokenizer.decode(trajectory[i][-1], skip_special_tokens=False))
                    console.print('batch_idx:', i.item(), 'n_block:', n_block, '\n', generated_str, end='\n\n')

                    # dict_lst.append({
                    #     "n_block": n_block,
                    #     "n_batch": i,
                    #     "prompt_ids": generated_ids[:, :-args.n_token_seq_len].cpu(),
                    #     "trajectory": [step[0].cpu() for step in answer_trajectory_ids],
                    #     "teacher_output_ids": generated_ids[i].cpu()
                    # })

    print(f'Finishing diffusion decoding...', flush=True)
    grouped_by_data_id = defaultdict(list)
    for dic in dict_lst:
        grouped_by_data_id[dic["data_id"]].append(dic)

    new_data = []
    for data_id, group in grouped_by_data_id.items():
        best_teacher_output = max(group, key=lambda x: len(x["teacher_output_ids"]))["teacher_output_ids"]
        for dic in group:
            dic["teacher_output_ids"] = best_teacher_output
            # Now convert to list for JSON
            dic["prompt_ids"] = dic["prompt_ids"].tolist()
            dic["answer_trajectory_ids"] = [a.tolist() for a in dic["answer_trajectory_ids"]]
            dic["teacher_output_ids"] = dic["teacher_output_ids"].tolist()
            new_data.append(dic)

    with open(new_file_path, "w") as f:
        json.dump(new_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default='output')
    parser.add_argument("--n_token_seq_len", type=int, default=64)
    parser.add_argument("--max_new_seq_len", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--use_labels", action="store_true")
    args = parser.parse_args()

    main(args)
    # main(args.filename, model, tokenizer, args.n_token_seq_len, args.max_new_seq_len,
    #      args.use_labels, args.data_bos_id, args.data_eos_id, args.batch_size, args.save_path)
