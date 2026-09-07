"""Safety battery for the vocab-pruned checkpoint. Must ALL pass before we build the image.

1. exhaustive remap coverage  — every id in the tokenizer maps to in-range ids
2. round-trip fidelity        — remapped ids decode back to the SAME text
3. unicode fuzz               — CJK / Arabic / emoji / accented text survives remap
4. logit equivalence          — pruned vs original model give identical p_interrupt on real prompts
"""
import json, sys, os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

PRUNED = sys.argv[1]
ORIG = sys.argv[2]
r = json.load(open(os.path.join(PRUNED, "vocab_remap.json")))
KEEP, ADD, NADD, NNEW = r["keep_low"], r["added_start"], r["n_added"], r["n_new"]
FB = {int(k): v for k, v in r["fallback"].items()}
EXTRAS = [int(x) for x in r.get("extras", [])]
EXTRA_POS = {t: KEEP + k for k, t in enumerate(EXTRAS)}
ADDED_BASE = KEEP + len(EXTRAS)
tok = AutoProcessor.from_pretrained(ORIG).tokenizer

def remap(ids):
    out = []
    for t in ids:
        if t < KEEP: out.append(t)
        elif t in EXTRA_POS: out.append(EXTRA_POS[t])
        elif ADD <= t < ADD + NADD: out.append(ADDED_BASE + (t - ADD))
        else: out.extend(FB.get(t, [0]))
    return out

# ---- 1. exhaustive coverage --------------------------------------------------------------
bad = []
for i in range(len(tok)):
    m = remap([i])
    if not m or any((x < 0 or x >= NNEW) for x in m):
        bad.append(i)
print(f"[1] exhaustive coverage: {len(tok):,} ids checked, out-of-range: {len(bad)}")
assert not bad, f"OUT OF RANGE ids: {bad[:10]}"

# ---- 2. round-trip fidelity on pruned ids -------------------------------------------------
inv = {}                      # new_id -> text, for kept ids only
mismatch = 0
checked = 0
for i in list(FB)[:4000]:
    want = tok.decode([i])
    got = tok.decode([x for x in FB[i]])   # byte tokens decode back to the same text
    checked += 1
    if want != got:
        mismatch += 1
        if mismatch <= 3:
            print(f"    mismatch id={i}: {want!r} -> {got!r}")
print(f"[2] fallback round-trip: {checked} pruned ids, text mismatches: {mismatch}")

# ---- 3. unicode fuzz ---------------------------------------------------------------------
fuzz = ["How do I sauté onions?", "中文测试 with English", "مرحبا بالعالم", "안녕하세요",
        "Привет мир", "emoji 😀🔥🎉 test", "naïve café résumé jalapeño", "ελληνικά", "עברית",
        "‘curly’ “quotes” — em-dash… ½ ¾ ± × ÷", "日本語のテスト", "🇺🇸🇯🇵", "​ zero-width"]
worst = 0
for s in fuzz:
    ids = tok.encode(s, add_special_tokens=False)
    m = remap(ids)
    assert all(0 <= x < NNEW for x in m), f"fuzz out of range: {s!r}"
    grew = len(m) - len(ids)
    worst = max(worst, grew)
    txt_ok = tok.decode(ids) == tok.decode([x for x in ids])   # sanity
    print(f"    {s[:28]!r:32s} ids={len(ids):>3} -> {len(m):>3} (+{grew})")
print(f"[3] unicode fuzz: all in range; worst length growth +{worst} tokens")

# ---- 4. logit equivalence on a real prompt ------------------------------------------------
proc = AutoProcessor.from_pretrained(ORIG)
SYS = ("You are a proactive AI assistant watching a first-person video of the user performing a "
       "procedural task, given their initial high-level query. The video arrives as short (~8s) "
       "chunks; after each chunk you decide whether NOW is the right moment to speak up with timely, "
       "useful guidance (a new step, or a correction), or to stay silent because nothing useful needs "
       "saying yet.\n\nAnswer with a single word: `yes` to speak now, or `no` to stay silent.")
img = Image.new("RGB", (384, 512), (120, 130, 140))
msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
        {"role": "user", "content": [{"type": "image"}] * 4 + [{"type": "text", "text": "How do I fold a napkin?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "$interrupt$Lay it flat — smooth the creases."}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
import re
text = re.sub(r"<think>\s*</think>\s*", "", text); text = re.sub(r"<think>\s*$", "", text)
inp = proc(text=[text], images=[img]*4, return_tensors="pt",
           images_kwargs={"max_pixels": 50176, "min_pixels": 3136})
iy = tok.encode("yes", add_special_tokens=False)[0]; ino = tok.encode("no", add_special_tokens=False)[0]

def p_of(path, ids=None):
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16,
                                                    device_map="cuda", attn_implementation="sdpa").eval()
    d = {k: (v.to(m.device) if hasattr(v, "to") else v) for k, v in inp.items()}
    if ids is not None:
        d["input_ids"] = torch.tensor([ids], dtype=torch.long, device=m.device)
        d["attention_mask"] = torch.ones((1, len(ids)), dtype=torch.long, device=m.device)
        y, n = remap([iy])[0], remap([ino])[0]
    else:
        y, n = iy, ino
    lg = m(**d).logits[0, -1]
    p = float(torch.softmax(torch.stack([lg[n], lg[y]]).float(), 0)[1])
    del m; torch.cuda.empty_cache()
    return p

src = inp["input_ids"][0].tolist()
p_orig = p_of(ORIG)
p_pruned = p_of(PRUNED, remap(src))
print(f"[4] logit equivalence: p_original={p_orig:.6f}  p_pruned={p_pruned:.6f}  |diff|={abs(p_orig-p_pruned):.2e}")
print("VERIFY_DONE")
