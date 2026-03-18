import logging

import re
from typing import Optional, List, Dict, Any

import torch
from transformers import AutoConfig, Qwen3Config


logger = logging.getLogger(__name__)

class Qwen3Processor:
    def __init__(
            self,
            base_model_dir: str,
            tokenizer,
            add_think_pattern: bool = False,
        ):

        try:
            model_config = AutoConfig.from_pretrained(base_model_dir)
        except Exception:
            model_config = Qwen3Config.from_pretrained(base_model_dir)

        self.pad_token_id = model_config.pad_token_id

        # Chat template tokens
        self.im_start_token = "<|im_start|>"
        self.im_end_token = "<|im_end|>"
        self.im_start_token_id = tokenizer.encode(self.im_start_token)[0]
        self.im_end_token_id = tokenizer.encode(self.im_end_token)[0]

        # Derive chat template patterns from tokenizer instead of hardcoded token ids.
        self.assistant_start_pattern = tokenizer.encode(
            f"{self.im_start_token}assistant\n",
            add_special_tokens=False,
        )
        self.im_end_pattern = tokenizer.encode(
            f"{self.im_end_token}\n",
            add_special_tokens=False,
        )
        if not self.im_end_pattern:
            self.im_end_pattern = tokenizer.encode(
                self.im_end_token,
                add_special_tokens=False,
            )

        if add_think_pattern:
            logger.info(f"Thinking pattern enabled: add_think_pattern={add_think_pattern}")

    def _convert_messages(self, messages):
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
        
        if self.add_think_pattern:
            # Process thinking pattern: add /think or /no_think suffix to user messages
            # based on whether assistant message contains reasoning content
            for i in range(len(msg_list)):
                if msg_list[i]['role'] == 'assistant':
                    assistant_content = msg_list[i]['content']
                    
                    # Find corresponding user message (typically the previous one)
                    user_idx = i - 1
                    if user_idx < 0 or msg_list[user_idx]['role'] != 'user':
                        continue
                    
                    # Check if assistant content contains <think> tags
                    pattern = r'<think>(.*?)</think>'
                    match = re.search(pattern, assistant_content, re.DOTALL)
                    
                    if match is None:
                        # No reasoning tags found: add empty tags and mark as /no_think
                        msg_list[user_idx]['content'] += "/no_think"
                        msg_list[i]['content'] = "<think>\n</think>\n" + assistant_content
                    else:
                        # Reasoning tags found: check if they contain actual content
                        reasoning_content = match.group(1)
                        if reasoning_content.strip():
                            # Has reasoning content: mark as /think
                            msg_list[user_idx]['content'] += "/think"
                        else:
                            # Empty reasoning tags: mark as /no_think
                            msg_list[user_idx]['content'] += "/no_think"
            
        return msg_list

    def _get_assistant_mask(self, batch_input_ids: torch.Tensor,
                       start_pattern: Optional[List[int]],
                       end_pattern: Optional[List[int]]):
        """
        Generate mask for assistant tokens in chat format.
        
        Args:
            batch_input_ids: Input token IDs
            start_pattern: Pattern to identify start of assistant response
            end_pattern: Pattern to identify end of assistant response
        
        Returns:
            mask: Boolean mask indicating which tokens to compute loss on
        """
        if not start_pattern:
            start_pattern = self.assistant_start_pattern
        if not end_pattern:
            end_pattern = self.im_end_pattern

        masks = []
        for input_ids in batch_input_ids:
            ids = input_ids.tolist()
            mask = [0] * len(ids)
            start_len = len(start_pattern)
            end_len = len(end_pattern)
            i = 0

            while i <= len(ids) - start_len:
                if ids[i:i + start_len] != start_pattern:
                    i += 1
                    continue

                content_start = i + start_len
                j = content_start
                found_end = False
                while j <= len(ids) - end_len:
                    if ids[j:j + end_len] == end_pattern:
                        found_end = True
                        break
                    j += 1

                if not found_end:
                    for k in range(content_start, len(ids)):
                        mask[k] = 1
                    break

                for k in range(content_start, j):
                    mask[k] = 1

                i = j + end_len

            masks.append(mask)
        return torch.tensor(masks, dtype=torch.long)
    
    def _get_rope_index_qwen3(
                                self,
                                input_ids: torch.LongTensor,
                            ) -> torch.Tensor:
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
        return position_ids
    
    def _process_completion(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process segments format data into model inputs.
        
        Args:
            sample: Sample containing segments with pre-tokenized tokens
            
        Returns:
            Dictionary containing input_ids, attention_mask, labels, etc.
        """
        segments = sample["json"]["segments"]

        segments_text = ""

        for segment in segments:
            if segment["type"] == "text":
                segments_text += segment["text"]
            else:
                logger.error(f"segment type is not text, skip: {segment}")
                continue
        
        # Note: do not use self.tokenizer.eos_token as it's always set to <im_end>
        # References: 
        # 1. https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json#L232
        # 2. https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html#control-tokens
        segments_text += self.tokenizer.pad_token
        
        # Tokenize
        inputs = self.tokenizer(
            segments_text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        input_ids = inputs["input_ids"]
        
        # Check length
        if input_ids.shape[-1] > self.max_length:
            raise ValueError(f"Sample too long: {input_ids.shape[-1]} > {self.max_length}")
        
        # Mask EOS token
        inputs["loss_mask"] = torch.ones_like(input_ids)
        inputs["loss_mask"][..., -1] = 0
        
        # Generate position IDs
        inputs["position_ids"] = self._get_rope_index_qwen3(input_ids)
        
        return inputs

    def _process_chat(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process messages format data into model inputs.
        
        Args:
            sample: Sample containing messages in the new format
            
        Returns:
            Dictionary containing input_ids, attention_mask, labels, etc.
        """
        msg_key = "message" if "message" in sample["json"] else "messages"
        messages = sample["json"][msg_key]

        msg_converted = self._convert_messages(messages)
        
        # Convert messages to text using chat template
        text = self.tokenizer.apply_chat_template(
            msg_converted, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Add EOS token
        text += self.tokenizer.pad_token

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        input_ids = inputs["input_ids"]        
        # Check length
        if input_ids.shape[-1] > self.max_length:
            raise ValueError(f"Sample too long: {input_ids.shape[-1]} > {self.max_length}")
        
        inputs["loss_mask"] = self._get_assistant_mask(
            input_ids,
            start_pattern=self.assistant_start_pattern,
            end_pattern=self.im_end_pattern,
        )
        
        # Mask EOS token
        inputs["loss_mask"][..., -1] = 0

        # itemic id index mask
        itemic_id_mask = torch.zeros_like(input_ids)
        if self.itemic_id_range is not None:
            itemic_id_mask[(input_ids >= self.itemic_id_range[0]) & (input_ids <= self.itemic_id_range[1])] = 1
        inputs["itemic_id_mask"] = itemic_id_mask
        
        # Generate position IDs
        inputs["position_ids"] = self._get_rope_index_qwen3(input_ids)
        
        return inputs

    def process(self, sample, source_name=None):
        if "segments" in sample["json"] and sample["json"]["segments"] is not None:
            inputs = self._process_completion(sample)
        else:
            inputs = self._process_chat(sample)

        inputs['epoch_idx'] = sample['epoch_idx']
        if not inputs:
            raise ValueError("Empty inputs, skip")
        
        # Check if sample exceeds max_sample_length (always <= max_length)
        if inputs["input_ids"].shape[-1] > self.max_sample_length:
            logger.warning(f"Sample exceeds max_sample_length={self.max_sample_length}, length={inputs['input_ids'].shape[-1]}")
            raise ValueError(
                f"Unable to generate sample within max_sample_length={self.max_sample_length}"
            )
        
        return inputs
