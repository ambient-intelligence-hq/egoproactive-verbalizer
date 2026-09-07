import sys, torch, json, collections
from transformers import AutoModelForImageTextToText, AutoConfig
p = sys.argv[1]
cfg = AutoConfig.from_pretrained(p)
m = AutoModelForImageTextToText.from_pretrained(p, dtype=torch.bfloat16, device_map="cpu")
tot = sum(t.numel() for t in m.parameters())
groups = collections.Counter()
for n, t in m.named_parameters():
    key = ("embed_tokens" if "embed_tokens" in n else
           "lm_head" if "lm_head" in n else
           "vision" if ("visual" in n or "vision" in n) else
           "language_layers")
    groups[key] += t.numel()
print(f"TOTAL params: {tot/1e9:.4f} B")
for k, v in groups.most_common():
    print(f"  {k:16s} {v/1e6:9.1f} M  ({100*v/tot:.1f}%)")
tie = getattr(cfg, "tie_word_embeddings", None)
tc = getattr(cfg, "text_config", None)
if tc is not None and tie is None:
    tie = getattr(tc, "tie_word_embeddings", None)
vocab = getattr(cfg, "vocab_size", None) or getattr(tc, "vocab_size", None)
hidden = getattr(tc, "hidden_size", None) if tc is not None else getattr(cfg, "hidden_size", None)
print(f"vocab_size={vocab} hidden={hidden} tie_word_embeddings={tie}")
if vocab and hidden:
    per_tok = hidden * (2 if not tie else 1)
    print(f"params per vocab token: {per_tok} (embed{'+lm_head' if not tie else ' only, tied'})")
    need = tot - 2_000_000_000
    print(f"need to cut: {need/1e6:.1f} M -> prune {need/per_tok:,.0f} tokens -> target vocab {vocab - need/per_tok:,.0f}")
