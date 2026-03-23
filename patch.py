import torch
from typing import Optional, List

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer


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

def select_batch(cache: Cache, indices, n_rewind: int = 0) -> DynamicCache:
    selector = ... if n_rewind <= 0 else slice(0, -n_rewind)
    indices = indices if isinstance(indices, torch.Tensor) else torch.tensor(indices, dtype=torch.long, device=cache.layers[0].keys.device)
    new_cache = DynamicCache()
    for layer in cache.layers:
        new_layer = _clone_layer(layer)
        new_layer.keys = layer.keys[:, :, selector].index_select(0, indices)
        new_layer.values = layer.values[:, :, selector].index_select(0, indices)
        new_cache.layers.append(new_layer)
    return new_cache

def merge_batch(dst: Cache, src: Cache, indices):
    if not indices:
        return
    for dl, sl in zip(dst.layers, src.layers):
        src_seq = sl.keys.shape[2]
        dl.keys[indices, :, :src_seq, :] = sl.keys
        dl.values[indices, :, :src_seq, :] = sl.values

def empty_like(cache: Cache, extra_seq: int) -> DynamicCache:
    new_cache = DynamicCache()
    for layer in cache.layers:
        B, H, S, D = layer.keys.shape
        new_layer = _clone_layer(layer)
        new_layer.keys = torch.empty(B, H, S + extra_seq, D,
                                     dtype=layer.keys.dtype, device=layer.keys.device)
        new_layer.values = torch.empty(B, H, S + extra_seq, D,
                                       dtype=layer.values.dtype, device=layer.values.device)
        new_cache.layers.append(new_layer)
    return new_cache

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _select_tokens(probs: torch.Tensor, do_sample: bool) -> torch.LongTensor:
    if not do_sample:
        return torch.argmax(probs, dim=-1)

    vocab_size = probs.shape[-1]
    samples = torch.multinomial(probs.reshape(-1, vocab_size), num_samples=1)
    return samples.reshape(*probs.shape[:-1])

def _softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)

def _apply_top_k(probs: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None or top_k <= 0 or top_k >= probs.shape[-1]:
        return probs

    values, indices = torch.topk(probs, k=int(top_k), dim=-1)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, indices, values)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)

def _apply_top_p(probs: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    if top_p is None:
        return probs

    top_p = float(top_p)
    if top_p <= 0.0 or top_p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf <= top_p
    keep[..., 0] = True
    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_indices, filtered)
    return out

def _build_probs_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> torch.Tensor:
    probs = _softmax_with_temperature(logits, temperature)
    probs = _apply_top_k(probs, top_k)
    probs = _apply_top_p(probs, top_p)
    return probs

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
    trajectories: Optional[List[List[torch.LongTensor]]] = None,
    **kwargs
):
    temperature = float(kwargs.get("temperature", 1.0))
    top_k = kwargs.get("top_k", None)
    top_p = kwargs.get("top_p", 1.0)

    do_sample = temperature > 0.0

    # Stage1: Prefill
    if past_key_values is None:
        past_key_values = DynamicCache()
        inputs_embeds = self.model.embed_tokens(input_ids)

        past_seen_tokens = past_key_values.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
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

        return past_key_values

    # Stage2: Generation
    B, L = input_ids.shape
    assert past_key_values.layers

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
    #      ...                 └─ [1, n_accepted]
    #   ]
    out = input_ids.clone()
    device = input_ids.device

    total_accepted = torch.zeros(B, dtype=torch.long, device=device)
    unfinished_mask = torch.ones(B, dtype=torch.bool, device=device)

    next_tokens = torch.zeros(B, 1, dtype=input_ids.dtype, device=device)
    past_kv_fin = empty_like(past_key_values, extra_seq=L-1)

    if trajectories:
        for i in range(B):
            trajectories[i].append(out[i, 1:].clone()) # skip initial anchor token

    while True:
        # Select unfinished samples
        idx_unfin = unfinished_mask.nonzero(as_tuple=False).squeeze(1)
        out_unfin = out[idx_unfin]
        past_kv_unfin = select_batch(past_key_values, idx_unfin)

        # Forward pass
        inputs_embeds = self.model.embed_tokens(out_unfin[:, -L:])

        past_seen_tokens = past_kv_unfin.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + L, device=device
        )

        if position_ids is None:
            position_ids_unfin = cache_position.unsqueeze(0)
        else:
            position_ids_unfin = position_ids[idx_unfin]

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask[idx_unfin],
                "cache_position": cache_position,
                "past_key_values": past_kv_unfin,
                "position_ids": position_ids_unfin,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids_unfin)
        for decoder_layer in self.model.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids_unfin,
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
            p_logits = logits[n:n+1, anchor_idx:, :]
            p_prob = _build_probs_from_logits(p_logits, temperature, top_k, top_p)

            draft_tokens = out_unfin[n:n+1, anchor_idx+1:] # skip first input token (anchor)
            target_tokens = _select_tokens(p_prob[:, :-1, :], do_sample) # skip last token (next token)

            mismatch = (draft_tokens != target_tokens)
            n_matched = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1)[0].item()

            total_accepted[orig_idx] += n_matched + 1

            next_token = _select_tokens(
                p_prob[:, n_matched, :], do_sample
            ).unsqueeze(-1)[0]

            # If draft is not fully accepted
            if total_accepted[orig_idx] < L:
                # Rebuild draft from remaining logits
                q_probs_rem = p_prob[:, n_matched + 1:-1, :]
                new_draft = _select_tokens(q_probs_rem, do_sample).squeeze(0)

                # Update out
                out[orig_idx] = torch.cat([
                    out[orig_idx, :anchor_idx + n_matched + 1], # old anchor + new matched
                    next_token, # next token becomes new anchor
                    new_draft, # draft tokens
                ])

                next_tokens[orig_idx] = next_token
                if trajectories is not None:
                    trajectories[orig_idx].append(out[orig_idx, 1:].clone())
            else:
                # Save final KV cache
                merge_batch(
                    past_kv_fin,
                    select_batch(past_kv_unfin, [n], n_rewind=1),
                    [orig_idx]
                )

        unfinished_mask = total_accepted < L
        if not unfinished_mask.any():
            break

    return past_kv_fin, next_tokens
