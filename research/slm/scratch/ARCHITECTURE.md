# SLM-A architecture and bounded scaling plan

The v1 model has four pre-norm decoder blocks, width 128, four heads of width 32, 4× GELU feed-forward expansion, learned token and position embeddings, RMSNorm, and an untied output projection. Attention explicitly masks future positions. All trainable tensors are initialized by MLX from the configured seed; there is no weight or tokenizer import. Context is 256 tokens, batch size 1, and the train-only byte BPE has exactly 2,000 entries (four special IDs, 256 bytes, 1,740 learned merges).

The actual tensor count, verified from `model.parameters()`, is **1,332,352**. With vocabulary `V`, width `d`, context `C`, blocks `L`, and feed-forward multiplier `m`, this implementation has `2Vd + Cd + L[(4+2m)d² + 2d] + d` trainable scalars. This excludes AdamW state and intermediate activations. For v1: token embedding 256,000, position embedding 32,768, untied head 256,000, blocks 787,456, final norm 128.

The implementation uses bias-free QKV and output projections, bias-free feed-forward projections, two RMSNorm weights per block and one final RMSNorm weight. Computation uses next-token cross-entropy, AdamW, and greedy autoregressive generation. Full attention is `O(C²)` in sequence length; practical memory also includes gradients, AdamW moments, Metal buffers, and other macOS processes.

| Candidate | Layers | Width | Heads | Context | Vocabulary | Approx. parameters | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1–3M | 4 | 128 | 4 | 256 | 2,000 | 1.33M | v1 implementation; bounded test only |
| 5–10M | 6 | 256 | 8 | 512 | 4,000 | 6.90M | design only |
| 15M+ | 8 | 384 | 8 | 512 | 8,000 | 20.50M | design only |

These are candidate parameter counts, not memory feasibility claims or planned runs. A larger model requires new tokenization/data controls, memory profiling, and a fresh guarded preflight. No scale-up should be inferred from synthetic overfitting alone.
