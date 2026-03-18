import torch
from typing import Optional, Tuple

# logits processors
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.logits_process import LogitsProcessorList


@torch.inference_mode()
def forward_jacobi(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    n_token_seq_len = 64,
    **kwargs
):
    logits_processors = LogitsProcessorList()

    # Prefill phase, just compute the KV cache on the prompt
    if past_key_values is None:
        past_key_values = DynamicCache()

        # input_ids: prompt
        inputs_embeds = self.model.embed_tokens(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                # "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.model.layers[: self.model.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values, # PATCH1
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()

        scores = logits_processors(input_ids, logits) # PATCH2
        probs = torch.nn.functional.softmax(scores, dim=-1)
        next_token = torch.argmax(probs[:, -1, :], dim=-1, keepdim=True)

        return past_key_values, next_token

    # Generation phase
    #
    # input_ids: [Last Token, Noise, Noise, ...] (length = n_token_seq_len)
    #                └─ Accepted

    assert past_key_values.layers # KV cache must be filled

    answer_trajectory_ids = [ input_ids ]

    total_accepted = 0
    accepted_n_gram = input_ids.clone()
    while total_accepted < n_token_seq_len:
        inputs_embeds = self.model.embed_tokens(input_ids)
        attention_mask = torch.ones_like(input_ids, device=input_ids.device) # No padding token

        past_seen_tokens = past_key_values.get_seq_length()

        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.model.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.model.layers[: self.model.config.num_hidden_layers]:
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

        # Apply logits processor, then softmax
        p_scores = logits_processors(input_ids, logits)
        p_prob = torch.nn.functional.softmax(p_scores, dim=-1)

        # Exclude the last token
        greedy_tokens = torch.argmax(p_prob[:, :-1, :], dim=-1)      # [1, n_token_seq_len-1]

        # Accepted: prefix without change
        # Rejected: first mismatch and after
        mismatch = (input_ids[:, 1:] != greedy_tokens)
        accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1)+1 # TODO: low performance

        n_accepted = int(accepted[0]) # TODO: BATCH SIZE=1???

        # fill accepted tokens
        accepted_n_gram[:, total_accepted:total_accepted+n_accepted] = input_ids[:, :n_accepted]

        total_accepted += n_accepted

        L = input_ids.shape[1]
        has_rejected = (n_accepted < L)

        if has_rejected:
            # Delete the rejected tokens in KV cache
            n_rejected = L - n_accepted
            for layer in past_key_values.layers:
                layer.keys = layer.keys[..., :-n_rejected, :]
                layer.values = layer.values[..., :-n_rejected, :]

            # Append the greedy token at the first mismatch
            next_token = torch.argmax(p_prob[:, n_accepted-1, :], dim=-1, keepdim=True)
            input_ids = next_token

            # Rebuild draft tail greedily from the remaining positions in this pass (after the mismatch slot)
            q_probs_rem = p_prob[:, n_accepted:-1, :]
            if q_probs_rem.shape[1] > 0:
                q_sampled = torch.argmax(q_probs_rem, dim=-1)  # [1, L']
                input_ids = torch.cat((next_token, q_sampled), dim=-1)

                answer_trajectory_ids.append(torch.cat((accepted_n_gram[:, :total_accepted], input_ids), dim=-1))
            continue

        # If we didn't reject anything, append the next greedy token and finish this block
        next_token = torch.argmax(p_prob[:, -1, :], dim=-1, keepdim=True)
        total_accepted += 1
        answer_trajectory_ids.append(accepted_n_gram)

    return past_key_values, next_token, answer_trajectory_ids