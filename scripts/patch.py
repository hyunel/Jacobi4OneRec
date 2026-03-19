import torch
from typing import Optional, List

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


# ---------------------------------------------------------------------------
# KV cache helpers
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

# ---------------------------------------------------------------------------
# Forward function
# ---------------------------------------------------------------------------

@torch.inference_mode()
def forward_jacobi(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    accept_threshold: float = 0.0,
    **kwargs
):
    logits_processors = LogitsProcessorList()
    if temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_k != 0:
        logits_processors.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1))
    if top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1))

    do_sample = (temperature != 1.0 or top_k != 0 or top_p < 1.0)

    def _pick(probs):
        if do_sample:
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(probs, dim=-1, keepdim=True)

    # Stage1: Prefill
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
        scores = logits_processors(input_ids, logits[:, -1, :])
        probs = torch.nn.functional.softmax(scores, dim=-1)
        next_token = _pick(probs)

        return past_key_values, next_token

    # Stage2: Generation
    B, L = input_ids.shape
    assert past_key_values.layers

    device = input_ids.device

    # out: [B, L]
    #   Initially:
    #   [
    #      [anchor, draft_0, ..., draft_{n-1}],
    #      [anchor, draft_0, ..., draft_{n-1}],
    #      ...
    #   ]
    #
    #   As tokens get accepted, out is updated:
    #   [
    #      [accepted_0 , ..., accepted_k, anchor, draft... ],
    #      [accepted_0, ..., anchor, draft...              ],
    #      ...                 └─ [B, n_accepted]
    #   ]
    out = input_ids.clone()

    total_accepted = torch.zeros(B, dtype=torch.long, device=device)
    unfinished_mask = torch.ones(B, dtype=torch.bool, device=device)

    next_tokens = torch.zeros(B, 1, dtype=input_ids.dtype, device=device)
    past_kv_fin = empty_like(past_key_values, extra_seq=L)

    trajectory = [[ input_ids[i] ] for i in range(B)]
    while True:
        # Select unfinished samples
        idx_unfin = unfinished_mask.nonzero(as_tuple=False).squeeze(1)
        out_unfin = out[idx_unfin]
        past_kv_unfin = select_batch(past_key_values, idx_unfin.tolist())

        # Forward pass
        inputs_embeds = self.model.embed_tokens(out_unfin[:, -L:])
        attn_mask = attn_mask = torch.ones(
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

        # Per-sample verification & draft rebuild
        for n, orig_idx in enumerate(idx_unfin.tolist()):
            anchor_idx = total_accepted[orig_idx].item()
            draft_tokens = out_unfin[n:n+1, anchor_idx+1:]

            # TODO: implement non-greedy acceptance here
            # p_scores = logits_processors(out[orig_idx:orig_idx+1], logits[n:n+1])
            p_prob = torch.nn.functional.softmax(logits[n:n+1], dim=-1)

            if do_sample:
                draft_probs = p_prob[:, anchor_idx:-1, :].gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
                mismatch = draft_probs < accept_threshold
            else:
                greedy_tokens = torch.argmax(p_prob[:, anchor_idx:-1, :], dim=-1)
                mismatch = (draft_tokens != greedy_tokens)
            n_matched = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1)[0].item()

            total_accepted[orig_idx] += n_matched + 1

            next_token = _pick(p_prob[:, anchor_idx + n_matched, :])[0]
            next_tokens[orig_idx] = next_token

            # If draft is not fully accepted
            if total_accepted[orig_idx] < L:
                q_probs_rem = p_prob[:, anchor_idx + n_matched + 1:-1, :]
                new_draft = _pick(q_probs_rem.squeeze(0)).squeeze(-1)

                # Update out
                out[orig_idx] = torch.cat([
                    out[orig_idx, :anchor_idx + n_matched + 1], # old anchor + new matched
                    next_token, # next token becomes new anchor
                    new_draft, # draft tokens
                ])

                trajectory[orig_idx].append(out[orig_idx].clone())
            else:
                # Save final KV cache
                merge_batch(
                    past_kv_fin,
                    select_batch(past_kv_unfin, [n]),
                    [orig_idx]
                )

        unfinished_mask = total_accepted < L
        if not unfinished_mask.any():
            break

    return past_kv_fin, next_tokens, trajectory
