import os
#!/usr/bin/env python
"""Evaluate the verbalizer from PRE-EXTRACTED frames (convert_train-format samples),
for datasets whose source videos are no longer on disk (e.g. HoloAssist).

Input --samples = per-chunk rows {images, query, history, label, video, chunk}. For each
sample we read logits at the decision position -> p_interrupt (same as eval_verbalizer,
but frames come from disk paths, not on-the-fly extraction). Group by video into per-chunk
answer sequences, sweep tau, score with the OFFICIAL score_proactive.

  VERB_MODEL=Qwen/Qwen3.5-2B python eval_from_frames.py --adapter <ckpt> \
      --samples holoassist_heldout140_samples.jsonl --out eval_ha140.json
"""
import argparse, json, os, re, sys, collections
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

_KIT = os.environ.get("EGOPROACTIVE_STARTER_KIT", "starter_kit")
if not os.path.isdir(_KIT):
    raise SystemExit(
        f"Organizer starter kit not found at {_KIT!r}.\n"
        "Download it from the challenge page and point EGOPROACTIVE_STARTER_KIT at it:\n"
        "  export EGOPROACTIVE_STARTER_KIT=/path/to/starter_kit"
    )
sys.path.insert(0, _KIT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation import score_proactive        # noqa: E402
from train_verbalizer import SYSTEM_PROMPT         # noqa: E402

BASE = os.environ.get("VERB_MODEL", "Qwen/Qwen3.5-4B")
MAX_PIXELS = 50176
TAUS = [0.6, 0.57, 0.55, 0.52, 0.5, 0.48, 0.47, 0.46, 0.45, 0.44, 0.42, 0.4, 0.3]


def _attn():
    try:
        import flash_attn  # noqa
        return "flash_attention_2"
    except Exception:
        return "sdpa"


@torch.no_grad()
def p_interrupt(model, proc, id_yes, id_no, frames, query, history):
    user0 = [{"type": "image"} for _ in frames] + [{"type": "text", "text": query}]
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user0}]
    for t in history:
        msgs.append({"role": t["role"], "content": [{"type": "text", "text": t["text"]}]})
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text = re.sub(r"<think>\s*</think>\s*", "", text)
    text = re.sub(r"<think>\s*$", "", text)
    inp = proc(text=[text], images=frames, return_tensors="pt",
               images_kwargs={"max_pixels": MAX_PIXELS, "min_pixels": 3136}).to(model.device)
    lg = model(**inp).logits[0, -1]
    pair = torch.softmax(torch.stack([lg[id_no], lg[id_yes]]).float(), 0)
    return float(pair[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load_4bit", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.samples)]
    proc = AutoProcessor.from_pretrained(BASE)
    id_yes = proc.tokenizer.encode("yes", add_special_tokens=False)[0]
    id_no = proc.tokenizer.encode("no", add_special_tokens=False)[0]

    is_adapter = os.path.exists(os.path.join(a.adapter, "adapter_config.json"))
    is_full = (not is_adapter) and os.path.exists(os.path.join(a.adapter, "config.json"))
    load_from = a.adapter if is_full else BASE
    model = AutoModelForImageTextToText.from_pretrained(
        load_from, dtype=torch.bfloat16, device_map="cuda", attn_implementation=_attn())
    if is_adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    print(f"[eval-frames] {'FULL' if is_full else 'adapter' if is_adapter else 'BASE'} {a.adapter} | {len(rows)} samples", flush=True)

    # group by video, ordered by chunk
    byvid = collections.defaultdict(list)
    for r in rows:
        byvid[r["video"]].append(r)
    for v in byvid:
        byvid[v].sort(key=lambda r: r["chunk"])

    golden, prob_by_video = [], {}
    for i, (v, chunks) in enumerate(byvid.items()):
        gold_ans, probs = [], []
        for r in chunks:
            frames = [Image.open(p).convert("RGB") for p in r["images"]]
            probs.append(p_interrupt(model, proc, id_yes, id_no, frames, str(r.get("query", "")),
                                     r.get("history", [])))
            gold_ans.append("$interrupt$" if r["label"] == "interrupt" else "$silent$")
        golden.append({"video_path": v, "video_intervals": [[0, 0]] * len(chunks), "answers": gold_ans})
        prob_by_video[v] = probs
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(byvid)}", flush=True)

    best = None
    print("\n=== threshold sweep ===")
    for tau in TAUS:
        preds = [{"video_path": g["video_path"],
                  "answers": ["$interrupt$" if p >= tau else "$silent$" for p in prob_by_video[g["video_path"]]]}
                 for g in golden]
        r = score_proactive(golden, preds)["overall"]
        print(f"  tau={tau:.2f}  Gmean={r['gmean_f1']:.4f} Macro={r['macro_f1']:.4f}  "
              f"int P={r['interrupt_precision']:.3f} R={r['interrupt_recall']:.3f}")
        if best is None or r["gmean_f1"] > best[1]["gmean_f1"]:
            best = (tau, r)
    print(f"BEST tau={best[0]}  G-mean={best[1]['gmean_f1']:.4f}  Macro={best[1]['macro_f1']:.4f}")
    json.dump({"adapter": a.adapter, "best_tau": best[0], "results": best[1],
               "prob_by_video": prob_by_video}, open(a.out, "w"))


if __name__ == "__main__":
    main()
