        # from torch.nn.attention.flex_attention import create_mask
        # debug_attention_mask = create_mask(
        #     attention_mask.mask_mod,
        #     B=B, H=1, Q_LEN=L, KV_LEN=L,
        #     device=inputs["input_ids"].device,
        # )

        # import matplotlib.pyplot as plt
        # for b in range(B):
        #     plt.figure(figsize=(8, 8))
        #     plt.imshow(debug_attention_mask[b, 0].to(torch.float32).cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        #     plt.xlabel("kv_idx")
        #     plt.ylabel("q_idx")
        #     plt.title(f"debug_attention_mask_b{b}")
        #     plt.tight_layout()
        #     plt.savefig(f"debug_attention_mask_b{b}.png", dpi=200)
        #     plt.close()