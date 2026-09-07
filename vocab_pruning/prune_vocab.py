"""Option F: contiguous vocab truncation for the <=2B small division.

Keep IDs [0, KEEP_LOW) unchanged + relocate the 33 added/special tokens directly after them.
Embedding rows are copied EXACTLY; transformer, vision tower and projector are untouched.

remap(id) = id                              if id < KEEP_LOW          (identity - the common case)
          = KEEP_LOW + (id - ADDED_START)   if id in added/special
          = <UTF-8 byte-token sequence>     otherwise                 (exact content, rare path)

Writes:
  <out>/                    pruned model + processor (tokenizer left UNCHANGED on purpose)
  <out>/vocab_remap.json    {keep_low, added_start, n_new, fallback: {old_id: [new_ids...]}}
"""
import json, os, sys, collections
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig

SRC = sys.argv[1]
OUT = sys.argv[2]
KEEP_LOW = int(sys.argv[3]) if len(sys.argv) > 3 else 143000

proc = AutoProcessor.from_pretrained(SRC)
tok = proc.tokenizer
cfg = AutoConfig.from_pretrained(SRC)
added = sorted(tok.get_added_vocab().values())
ADDED_START = added[0]
assert added == list(range(ADDED_START, ADDED_START + len(added))), "added ids not contiguous"
assert ADDED_START >= KEEP_LOW, "KEEP_LOW overlaps the added block"

# optional: explicitly retained high-ID tokens (observed rare task vocabulary + domain words),
# so the byte-fallback path is not needed for vocabulary we know the task uses.
EXTRAS = []
if len(sys.argv) > 4 and os.path.exists(sys.argv[4]):
    EXTRAS = sorted({int(x) for x in json.load(open(sys.argv[4]))
                     if KEEP_LOW <= int(x) < ADDED_START})
extra_pos = {t: KEEP_LOW + k for k, t in enumerate(EXTRAS)}

keep_ids = list(range(KEEP_LOW)) + EXTRAS + added
n_new = len(keep_ids)
print(f"keep {n_new:,} = {KEEP_LOW:,} low + {len(EXTRAS)} extras + {len(added)} added   "
      f"(old vocab {len(tok):,})", flush=True)

def new_of(i):
    if i < KEEP_LOW:
        return i
    if i in extra_pos:
        return extra_pos[i]
    if ADDED_START <= i < ADDED_START + len(added):
        return KEEP_LOW + len(EXTRAS) + (i - ADDED_START)
    return None

# ---- exhaustive fallback: every prunable id -> its UTF-8 byte tokens (all byte tokens are low-ID) ----
def bytes_to_unicode():
    """GPT2/Qwen byte-level BPE byte->printable-char map (inlined; moved in transformers 5.x)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

byte_tok = {}
V = tok.get_vocab()
for b, ch in bytes_to_unicode().items():
    tid = V.get(ch)
    if isinstance(tid, int) and 0 <= tid < KEEP_LOW:
        byte_tok[b] = tid
print(f"byte tokens resolved: {len(byte_tok)}/256", flush=True)

fallback, unresolved = {}, 0
for i in range(len(tok)):
    if new_of(i) is not None:
        continue
    s = tok.decode([i])
    seq = []
    ok = True
    for b in s.encode("utf-8"):
        t = byte_tok.get(b)
        if t is None:
            ok = False
            break
        seq.append(t)
    if not ok or not seq:                       # last resort: a kept ASCII '?'
        q = tok.encode("?", add_special_tokens=False)
        seq = [t for t in q if t < KEEP_LOW] or [0]
        unresolved += 1
    fallback[i] = seq
print(f"fallback entries: {len(fallback):,}  (byte-exact: {len(fallback)-unresolved:,}, '?'-substituted: {unresolved})", flush=True)

# ---- build the pruned model ----
model = AutoModelForImageTextToText.from_pretrained(SRC, dtype=torch.bfloat16, device_map="cpu")
emb = model.get_input_embeddings()
W = emb.weight.data
print(f"old embedding matrix: {tuple(W.shape)}", flush=True)
idx = torch.tensor(keep_ids, dtype=torch.long)
W_new = W.index_select(0, idx).clone()          # EXACT row copies
print(f"new embedding matrix: {tuple(W_new.shape)}", flush=True)

new_emb = torch.nn.Embedding(n_new, W_new.shape[1], dtype=W_new.dtype)
new_emb.weight.data.copy_(W_new)
model.set_input_embeddings(new_emb)
if getattr(cfg, "tie_word_embeddings", False) or getattr(getattr(cfg, "text_config", None), "tie_word_embeddings", False):
    model.tie_weights()
else:
    head = model.get_output_embeddings()
    if head is not None:
        nh = torch.nn.Linear(head.in_features, n_new, bias=head.bias is not None, dtype=W_new.dtype)
        nh.weight.data.copy_(head.weight.data.index_select(0, idx))
        if head.bias is not None:
            nh.bias.data.copy_(head.bias.data.index_select(0, idx))
        model.set_output_embeddings(nh)

# ---- config: vocab size + relocate every special token id ----
def patch(o):
    if o is None:
        return
    if hasattr(o, "vocab_size") and isinstance(getattr(o, "vocab_size"), int):
        o.vocab_size = n_new
    for f in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
              "eos_token_id", "bos_token_id", "pad_token_id", "image_token_index", "video_token_index"):
        v = getattr(o, f, None)
        if isinstance(v, int):
            nv = new_of(v)
            if nv is not None:
                setattr(o, f, nv)
                print(f"  cfg {f}: {v} -> {nv}", flush=True)
patch(model.config); patch(getattr(model.config, "text_config", None)); patch(getattr(model.config, "vision_config", None))
if hasattr(model, "generation_config") and model.generation_config is not None:
    patch(model.generation_config)

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
proc.save_pretrained(OUT)                       # tokenizer intact: identical BPE splits
json.dump({"keep_low": KEEP_LOW, "added_start": ADDED_START, "n_added": len(added),
           "extras": EXTRAS,
           "n_new": n_new, "old_vocab": len(tok),
           "fallback": {str(k): v for k, v in fallback.items()}},
          open(os.path.join(OUT, "vocab_remap.json"), "w"))

tot = sum(t.numel() for t in model.parameters())
print(f"PRUNED TOTAL PARAMS: {tot/1e9:.4f} B  ({'UNDER' if tot < 2e9 else 'OVER'} 2B)", flush=True)
print("PRUNE_DONE", OUT, flush=True)
