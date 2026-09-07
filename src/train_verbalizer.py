#!/usr/bin/env python
"""Binary interrupt/silent decision as a single-token verbalizer (yes/no).

Instead of generating `$interrupt$<utterance>` / `$silent$`, the model emits ONE token — `yes`
(speak now) or `no` (stay silent) — and we classify by the renormalized prob of those two tokens.
This DECOUPLES the decision from any utterance, so the model can't dodge interrupting to avoid a
hard-to-predict utterance (the cause of the chunk-0-only collapse). Reuses convert_train.py's
per-decision samples (images/query/history/label); only the target + system prompt change.
"""
import os
import argparse, json, os, re
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = os.environ.get("VERB_MODEL", "Qwen/Qwen3.5-4B")
MAX_PIXELS = 50176

SYSTEM_PROMPT = (
    "You are a proactive AI assistant watching a first-person video of the user performing a "
    "procedural task, given their initial high-level query. The video arrives as short (~8s) "
    "chunks; after each chunk you decide whether NOW is the right moment to speak up with timely, "
    "useful guidance (a new step, or a correction), or to stay silent because nothing useful needs "
    "saying yet.\n\nAnswer with a single word: `yes` to speak now, or `no` to stay silent."
)

processor = AutoProcessor.from_pretrained(MODEL)
tok = processor.tokenizer
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
ASSIST_HDR = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
ID_YES = tok.encode("yes", add_special_tokens=False)
ID_NO = tok.encode("no", add_special_tokens=False)
assert len(ID_YES) == 1 and len(ID_NO) == 1, (ID_YES, ID_NO)


def build_messages(ex, target):
    n_img = len(ex["images"])
    user0 = [{"type": "image"} for _ in range(n_img)] + [{"type": "text", "text": ex["query"]}]
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user0}]
    for t in ex["history"]:                      # real prior conversation, verbatim context
        msgs.append({"role": t["role"], "content": [{"type": "text", "text": t["text"]}]})
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return msgs


def labels_for(input_ids):
    """Supervise only the LAST assistant span (the yes/no token + its <|im_end|>)."""
    ids = input_ids.tolist()
    labels = [-100] * len(ids)
    H = len(ASSIST_HDR)
    starts = [i for i in range(len(ids) - H + 1) if ids[i:i + H] == ASSIST_HDR]
    if not starts:
        return torch.tensor(labels)
    i = starts[-1] + H
    j = i
    while j < len(ids) and ids[j] != IM_END:
        j += 1
    end = min(j + 1, len(ids))
    for k in range(i, end):
        labels[k] = ids[k]
    return torch.tensor(labels)


class Collator:
    def __call__(self, examples):
        ex = examples[0]
        target = "yes" if ex["label"] == "interrupt" else "no"
        msgs = build_messages(ex, target)
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        text = re.sub(r"<think>\s*</think>\s*", "", text)
        images = [Image.open(p).convert("RGB") for p in ex["images"]]
        batch = processor(text=[text], images=images, return_tensors="pt",
                          images_kwargs={"max_pixels": MAX_PIXELS, "min_pixels": 3136})
        batch["labels"] = labels_for(batch["input_ids"][0]).unsqueeze(0)
        return batch


class DecisionTrainer(SFTTrainer):
    """Plain CE on the supervised (yes/no + im_end) tokens; logs yes/no CE separately."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._acc = {"yes": [0.0, 0], "no": [0.0, 0]}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        out = model(**inputs)
        sl = out.logits[:, :-1, :].reshape(-1, out.logits.size(-1))
        lb = labels[:, 1:].reshape(-1).to(sl.device)
        ce = F.cross_entropy(sl.float(), lb, reduction="none", ignore_index=-100)
        n = (lb != -100).sum().clamp(min=1)
        loss = ce.sum() / n
        with torch.no_grad():
            for name, tid in (("yes", ID_YES[0]), ("no", ID_NO[0])):
                m = lb == tid
                if bool(m.any()):
                    self._acc[name][0] += float(ce[m].sum()); self._acc[name][1] += int(m.sum())
        return (loss, out) if return_outputs else loss

    def log(self, logs, *a, **k):
        for name, (s, c) in self._acc.items():
            if c:
                logs[f"loss_{name}"] = s / c
        self._acc = {"yes": [0.0, 0], "no": [0.0, 0]}
        return super().log(logs, *a, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("EP_DATA", "data/train.jsonl"))
    ap.add_argument("--out", default=os.environ.get("EP_OUT", "runs/verbalizer_lora"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--resume", default="")
    a = ap.parse_args()
    os.environ.setdefault("WANDB_PROJECT", "joyai-vl-sft")

    raw = [json.loads(l) for l in open(a.data)]
    raw = [s for s in raw if all(os.path.exists(p) for p in s["images"])]
    print(f"train samples: {len(raw)} | LoRA r={a.rank} alpha={a.alpha}")
    ds = Dataset.from_list(raw)

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.config.use_cache = False
    peft_cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(
        output_dir=a.out, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=a.epochs, learning_rate=1e-4, lr_scheduler_type="cosine",
        warmup_ratio=0.03, logging_steps=5, save_strategy="steps", save_steps=250,
        save_total_limit=3, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to=os.environ.get("VERB_REPORT_TO", "none"),
        run_name=os.path.basename(a.out), dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False, max_length=None)
    trainer = DecisionTrainer(model=model, args=cfg, train_dataset=ds,
                              data_collator=Collator(), peft_config=peft_cfg)
    trainer.train(resume_from_checkpoint=a.resume or None)
    trainer.save_model(a.out); processor.save_pretrained(a.out)
    print("SAVED_ADAPTER", a.out)


if __name__ == "__main__":
    main()
