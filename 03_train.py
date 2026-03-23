from pathlib import Path
from dataclasses import dataclass, field
from datasets import load_dataset

import peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments
from torch.nn.attention.flex_attention import create_block_mask

@dataclass
class Arguments(TrainingArguments):
    use_lora : bool = field(default=False, metadata={"help": "Whether to use LoRA for fine-tuning"})
    lora_r: int = field(default=32, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})

    max_blocks: int = field(default=32, metadata={"help": "Maximum number of blocks to train on"})
    model_path: str = field(default=None, metadata={"help": "Path or HF id for the base model"})
    data_path: str = field(default=None, metadata={"help": "Path to the training data"})

    def __post_init__(self):
        super().__post_init__()
        self.remove_unused_columns = False

class JacobiTrainer(Trainer):
    def __init__(
            self,
            tokenizer,
            n_token_seq_len,
            distill_temperature=1.0,
            ar_loss_weight=10.0,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer
        self.n_token_seq_len = n_token_seq_len
        self.distill_temperature = distill_temperature
        self.ar_loss_weight = ar_loss_weight

        self._custom_metrics = {}
        self.create_block_mask = torch.compile(create_block_mask)

    def _create_mask(
            self,
            n_pads: torch.Tensor,
            n_prompt: torch.Tensor,
            n_sequence: torch.Tensor,
            n_batch_size: torch.Tensor,
        ):
        def mask_mod(b, h, q_idx, kv_idx):
            n_pad = n_pads[b]

            causal_mask = q_idx >= kv_idx
            pad_mask = kv_idx >= n_pad

            # block alignment mask
            q_block_idx = (q_idx - n_prompt) // self.n_token_seq_len
            kv_block_idx = (kv_idx - n_prompt) // self.n_token_seq_len
            alignment_mask = (q_block_idx % 2) == (kv_block_idx % 2)

            # combine all masks
            is_prompt_kv = kv_idx < n_prompt
            valid_mask = is_prompt_kv | alignment_mask

            return causal_mask & pad_mask & valid_mask

        return self.create_block_mask(
            mask_mod, B=n_batch_size, H=None, Q_LEN=n_sequence, KV_LEN=n_sequence, device=self.args.device,
        )

    def _compute_ar_loss(self, logits, input_ids, answer_mask):
        batch_ids = torch.arange(logits.size(0), device=logits.device)[:, None].expand_as(answer_mask)[answer_mask]
        answer_logits = logits[answer_mask]
        answer_targets = input_ids[answer_mask]
        keep = batch_ids[:-1] == batch_ids[1:]
        if not keep.any():
            return logits.sum() * 0
        return torch.nn.functional.cross_entropy(
            answer_logits[:-1][keep].float(),
            answer_targets[1:][keep],
        )

    def _compute_consistency_loss(self, logits, n_blocks, block_mask, answer_mask):
        T = self.distill_temperature
        student = logits[block_mask].float() / T
        teacher = logits[answer_mask].detach().float() / T # no gradient to teacher
        if student.numel() == 0:
            return logits.sum() * 0

        per_token = -(teacher.softmax(dim=-1) * student.log_softmax(dim=-1)).sum(dim=-1)
        batch_ids = torch.arange(logits.size(0), device=logits.device)[:, None].expand_as(block_mask)[block_mask]
        loss_per_sample = torch.zeros(logits.size(0), device=logits.device, dtype=per_token.dtype)
        loss_per_sample.scatter_add_(0, batch_ids, per_token)
        loss_per_sample = loss_per_sample / n_blocks.clamp_min(1)
        return loss_per_sample.mean() * (T * T) / self.n_token_seq_len

    def compute_loss(self, model, inputs, return_outputs = False, **kwargs):
        B, L = inputs["input_ids"].shape

        attention_mask = self._create_mask(
            n_pads=inputs["n_pads"],
            n_prompt=inputs["n_prompt"],
            n_sequence=L,
            n_batch_size=B,
        )

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            position_ids=inputs["position_ids"],
            attn_implementation="flex_attention",
        )

        # Extract the relevant logits
        device = inputs["input_ids"].device
        seq_idx = torch.arange(L, device=device).unsqueeze(0)

        tail_offset = seq_idx - inputs["n_prompt"]
        valid_tail = (tail_offset >= 0) & (
            tail_offset < (2 * inputs["n_blocks"].unsqueeze(1) * self.n_token_seq_len)
        )

        is_answer_chunk = ((tail_offset // self.n_token_seq_len) % 2) == 1

        answer_mask = valid_tail & is_answer_chunk
        block_mask = valid_tail & (~is_answer_chunk)

        # Calculate losses
        logits = outputs.logits
        loss_ar = self._compute_ar_loss(logits, inputs["input_ids"], answer_mask)
        loss_consistency = self._compute_consistency_loss(logits, inputs["n_blocks"], block_mask, answer_mask)

        loss = loss_ar * self.ar_loss_weight + loss_consistency

        self._custom_metrics = {
            "loss_ar": loss_ar.detach().item(),
            "loss_consistency": loss_consistency.detach().item(),
        }

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs:
            logs.update(self._custom_metrics)
        return super().log(logs, *args, **kwargs)


class JacobiDataCollator:
    def __init__(self, n_token_seq_len, max_blocks=None):
        self.n_token_seq_len = n_token_seq_len
        self.max_blocks = max_blocks

    @staticmethod
    def _write_chunk(dst_ids, i_row, i_start, seq):
        seq_len = len(seq)
        if seq_len == 0:
            return
        end_idx = i_start + seq_len
        dst_ids[i_row, i_start:end_idx] = torch.tensor(seq, dtype=torch.long)

    def __call__(self, features):
        # Example: ([PR] = prompt, [B#] = block #, [A#] = answer for block #, [PD] = padding)
        # [PR] [PR] [PR] [PR] [B1] [A1] [B2] [A2] [B3] [A3] [PD] [PD]
        #  0    1    2    3    4    4    5    5    6    6    *    *
        # [PD] [PR] [PR] [PR] [B1] [A1] [B2] [A2] [B3] [A3] [B4] [A4]
        #  *    0    1    2    3    3    4    4    5    5    6    6

        B = len(features)

        n_blocks = [len(feature["blocks"][:self.max_blocks]) for feature in features]
        max_n_blocks = max(n_blocks)
        max_n_prompt = max(len(feature["prompt_ids"]) for feature in features)

        total_len = max_n_prompt + 2 * max_n_blocks * self.n_token_seq_len

        n_pads = torch.empty(B, dtype=torch.long)
        input_ids = torch.zeros((B, total_len), dtype=torch.long)
        position_ids = torch.zeros((B, total_len), dtype=torch.long)

        for i_row, feature in enumerate(features):
            prompt_ids = feature["prompt_ids"]
            blocks = feature["blocks"][:self.max_blocks]
            answers = feature["answers"][:self.max_blocks]

            n_prompt = len(prompt_ids)
            n_pad = max_n_prompt - n_prompt
            n_pads[i_row] = n_pad

            self._write_chunk(input_ids, i_row, n_pad, prompt_ids)
            position_ids[i_row, n_pad:max_n_prompt] = torch.arange(n_prompt, dtype=torch.long)

            for block_idx, (block, answer) in enumerate(zip(blocks, answers)):
                assert len(block) == len(answer) == self.n_token_seq_len, f"Expected block and answer to have length {self.n_token_seq_len}, but got {len(block)} and {len(answer)}"

                i_start = max_n_prompt + 2 * block_idx * self.n_token_seq_len
                block_pos = n_prompt + block_idx * self.n_token_seq_len

                self._write_chunk(input_ids, i_row, i_start, block)
                self._write_chunk(input_ids, i_row, i_start + self.n_token_seq_len, answer)
                block_pos_ids = torch.arange(block_pos, block_pos + self.n_token_seq_len, dtype=torch.long)
                position_ids[i_row, i_start:i_start + self.n_token_seq_len] = block_pos_ids
                position_ids[i_row, i_start + self.n_token_seq_len:i_start + 2 * self.n_token_seq_len] = block_pos_ids

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "n_pads": n_pads,
            "n_blocks": torch.tensor(n_blocks, dtype=torch.long),
            "n_prompt": torch.tensor(max_n_prompt, dtype=torch.long),
        }

def main(args: Arguments):
    # Load the dataset
    dataset = load_dataset("parquet", data_files=args.data_path, split="train")

    # Load the model+tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation='flex_attention',
        dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path
    )

    if args.use_lora:
        lora_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        model = peft.get_peft_model(model, lora_config)

    # Trainer
    n_token_seq_len = len(next(iter(dataset))["blocks"][0])
    trainer = JacobiTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=dataset,
        data_collator=JacobiDataCollator(
            n_token_seq_len=n_token_seq_len,
            max_blocks=args.max_blocks,
        ),
        n_token_seq_len=n_token_seq_len,
    )

    if any(Path(args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args_into_dataclasses()[0])
