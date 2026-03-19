import torch
from typing import Optional, List

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
from transformers.generation.logits_process import LogitsProcessorList


# ---------------------------------------------------------------------------
# Standalone KV cache helpers (new transformers Cache API: cache.layers[i])
# ---------------------------------------------------------------------------

def _clone_layer(layer):
    if layer.is_sliding:
        new = DynamicSlidingWindowLayer(layer.sliding_window)
    else:
        new = DynamicLayer()
    new.is_initialized = True
    return new

def select_batch(cache: Cache, indices: List[int]) -> DynamicCache:
    new_cache = DynamicCache()
    idx = torch.tensor(indices, device=cache.layers[0].keys.device)
    for layer in cache.layers:
        new_layer = _clone_layer(layer)
        new_layer.keys = layer.keys.index_select(0, idx)
        new_layer.values = layer.values.index_select(0, idx)
        new_cache.layers.append(new_layer)
    return new_cache

def merge_batch(dst: Cache, src: Cache, indices: List[int]):
    if not indices:
        return
    idx = torch.tensor(indices, device=src.layers[0].keys.device)
    for dl, sl in zip(dst.layers, src.layers):
        src_seq = sl.keys.shape[2]
        dl.keys[idx, :, :src_seq, :] = sl.keys.clone()
        dl.values[idx, :, :src_seq, :] = sl.values.clone()

def empty_like(cache: Cache, extra_seq: int) -> DynamicCache:
    new_cache = DynamicCache()
    for layer in cache.layers:
        B, H, S, D = layer.keys.shape
        new_layer = _clone_layer(layer)
        new_layer.keys = torch.zeros(B, H, S + extra_seq, D,
                                     dtype=layer.keys.dtype, device=layer.keys.device)
        new_layer.values = torch.zeros(B, H, S + extra_seq, D,
                                       dtype=layer.values.dtype, device=layer.values.device)
        new_cache.layers.append(new_layer)
    return new_cache

@torch.inference_mode()
def forward_jacobi(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    n_token_seq_len: int = 64,
    **kwargs
):
    logits_processors = LogitsProcessorList()

    # Prefill phase
    if past_key_values is None:
        past_key_values = DynamicCache()
        inputs_embeds = self.model.embed_tokens(input_ids)

        past_seen_tokens = past_key_values.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
        position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.model.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()
        scores = logits_processors(input_ids, logits)
        probs = torch.nn.functional.softmax(scores, dim=-1)
        next_token = torch.argmax(probs[:, -1, :], dim=-1, keepdim=True)

        return past_key_values, next_token

    # Generation phase
    #
    # Strategy (from blk32): each iteration re-computes draft KV from
    # immutable prompt KV. We maintain `out` = [accepted... + draft...]
    # at a fixed width, and feed the last (n_token_seq_len+1) tokens
    # so the model sees accepted context.
    #
    # input_ids: [B, n_token_seq_len]  (col 0 = anchor, col 1: = draft)

    B, L = input_ids.shape
    assert past_key_values.layers
    assert L == n_token_seq_len, f"Expected input_ids with seq_len={n_token_seq_len}, got {L}"

    # [anchor, draft_0, ..., draft_{n-1}]
    out = input_ids.clone()
    device = input_ids.device

    total_accepted = torch.zeros(B, dtype=torch.long, device=device)
    unfinished_mask = torch.ones(B, dtype=torch.bool, device=device)
    answer_trajectory_ids = [ input_ids.clone() ]

    next_tokens = torch.zeros(B, 1, dtype=input_ids.dtype, device=device)
    past_kv_fin = empty_like(past_key_values, extra_seq=L)

    while True:
        # 1) Select unfinished samples
        idx_unfin = unfinished_mask.nonzero(as_tuple=False).squeeze(1)
        out_unfin = out[idx_unfin]
        past_kv_unfin = select_batch(past_key_values, idx_unfin.tolist())

        # 2) Forward pass
        inputs_embeds = self.model.embed_tokens(out_unfin)
        attn_mask = torch.ones(
            idx_unfin.shape[0], L, dtype=torch.long, device=device
        )

        past_seen_tokens = past_kv_unfin.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + L, device=device
        )
        position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attn_mask,
                "cache_position": cache_position,
                "past_key_values": past_kv_unfin,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.model.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_kv_unfin,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()

        # 3) Per-sample greedy verification & draft rebuild
        for n, orig_idx in enumerate(idx_unfin.tolist()):
            n_accepted = int(total_accepted[orig_idx])
            remaining_draft = L - n_accepted

            # anchor_pos = L - remaining_draft - 1
            # if anchor_pos < 0:
            #     anchor_pos = 0

            p_scores = logits_processors(out[orig_idx:orig_idx+1], logits[n:n+1])
            p_prob = torch.nn.functional.softmax(p_scores, dim=-1)

            # # Verify draft: compare model predictions starting from anchor_pos
            # # greedy[i] should match feed_tokens[n, anchor_pos+1+i]

            greedy_tokens = torch.argmax(p_prob[:, anchor_pos:-1, :], dim=-1)  # [1, remaining_draft]
            draft_tokens = feed_tokens[n:n+1, anchor_pos+1:]                   # [1, remaining_draft]

            mismatch = (draft_tokens != greedy_tokens)
            n_matched = int((mismatch.cumsum(dim=-1) == 0).sum(dim=-1)[0])
            n_new_accepted = n_matched + 1
            total_accepted[orig_idx] = n_accepted + n_new_accepted

            has_rejected = (n_matched < remaining_draft)

            if has_rejected:
                # Correction token
                nt = torch.argmax(p_prob[:, anchor_pos + n_matched, :], dim=-1, keepdim=True)
                next_tokens[orig_idx] = nt[0]

                # Rebuild draft from remaining logits
                q_start = anchor_pos + n_matched + 1
                q_probs_rem = p_prob[:, q_start:, :]
                if q_probs_rem.shape[1] > 0:
                    new_draft = torch.argmax(q_probs_rem, dim=-1)
                else:
                    new_draft = torch.empty(1, 0, dtype=torch.long, device=device)

                # Update out: keep accepted prefix, append [correction, new_draft...]
                new_remaining = n_token_seq_len - int(total_accepted[orig_idx])
                if new_remaining > 0:
                    # Accepted portion of out (everything before the old draft region)
                    accepted_end = out.shape[1] - remaining_draft + n_matched
                    # Build new out for this sample
                    accepted_part = out[orig_idx, :accepted_end]  # [..., matched_drafts]
                    fill = new_draft.shape[1]
                    # new_out = [accepted_part, correction, new_draft, padding...]
                    new_out = torch.cat([
                        accepted_part,
                        nt[0],
                        new_draft[0, :new_remaining] if fill > 0 else torch.empty(0, dtype=torch.long, device=device),
                    ])
                    # Pad or trim to fixed length
                    target_len = out.shape[1]
                    if new_out.shape[0] < target_len:
                        pad_val = new_out[-1]
                        new_out = torch.cat([new_out, pad_val.expand(target_len - new_out.shape[0])])
                    out[orig_idx] = new_out[:target_len]
            else:
                nt = torch.argmax(p_prob[:, -1, :], dim=-1, keepdim=True)
                next_tokens[orig_idx] = nt[0]

        answer_trajectory_ids.append(out[:, 1:].clone())  # exclude anchor (col 0)

        # 4) Merge finished samples
        finished_n_poss, orig_idxs = [], []
        for n_pos, orig_idx in enumerate(idx_unfin.tolist()):
            if total_accepted[orig_idx] >= n_token_seq_len:
                finished_n_poss.append(n_pos)
                orig_idxs.append(orig_idx)

        if finished_n_poss:
            merge_batch(
                past_kv_fin,
                select_batch(past_kv_unfin, finished_n_poss),
                orig_idxs
            )

        unfinished_mask = total_accepted < n_token_seq_len
        if not unfinished_mask.any():
            break

    return past_kv_fin, next_tokens, answer_trajectory_ids
