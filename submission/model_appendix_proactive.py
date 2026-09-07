"""EgoProactive submission model: single-token yes/no VERBALIZER.

The runner (run_generate_proactive.py) calls `generate(frames, messages, max_new_tokens)` once per
~8s chunk with CUMULATIVE frames, and concatenates the returned strings into `answers`.
Scoring (`_score_proactive_session`) only reads `parse_tag()` — whether the string starts with
`$interrupt$` or `$silent$` — so the utterance text is not scored; we emit a short generic cue.

Decision = renormalised P(yes) vs P(no) at the decision position, thresholded at TAU.
This mirrors `eval_from_frames.py::p_interrupt` EXACTLY (same system prompt, same message shape,
same <think> stripping, same max_pixels/min_pixels) so the container reproduces our offline numbers.

Two traps this class handles deliberately:
  1. The runner passes ITS OWN SYSTEM_PROMPT (which asks for `$interrupt$<utterance>`). The
     verbalizer was TRAINED on a different system prompt asking for one yes/no token, so we
     substitute ours — using the runner's would be a train/inference mismatch.
  2. Qwen3.5 emits a `<think>` block from the chat template; training/eval stripped it before the
     forward pass, so we strip it identically.
"""
from __future__ import annotations

import io
import json
import os
import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from model import VideoQAModel

# The model was TRAINED on frames that convert_train.py wrote to disk as JPEGs:
# resized to max side 512, quality 90. The runner hands us raw full-resolution frames,
# which is a train/serve mismatch (measured: container 0.7664 vs direct 0.7958 without this).
# Replicate the training-time preprocessing exactly.
JPEG_MAX_SIDE = 512
JPEG_QUALITY = 90


def _as_training_frame(img: object) -> Image.Image:
    im = img.convert("RGB") if hasattr(img, "convert") else img
    w, h = im.size
    sc = JPEG_MAX_SIDE / max(w, h)
    if sc < 1.0:
        im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# VERBATIM from train_verbalizer.py / eval_from_frames.py
SYSTEM_PROMPT = (
    "You are a proactive AI assistant watching a first-person video of the user performing a "
    "procedural task, given their initial high-level query. The video arrives as short (~8s) "
    "chunks; after each chunk you decide whether NOW is the right moment to speak up with timely, "
    "useful guidance (a new step, or a correction), or to stay silent because nothing useful needs "
    "saying yet.\n\nAnswer with a single word: `yes` to speak now, or `no` to stay silent."
)

MAX_PIXELS = 50176          # must match eval_from_frames.py
MIN_PIXELS = 3136

# The model was TRAINED with at most 32 cumulative frames per decision (convert_train.py
# --max_frames 32). run_evaluation.py defaults --max-frames to 100, so the runner hands us up to
# 100 frames — a train/serve mismatch worth ~3% G-mean (0.763 -> 0.792 on a 544-chunk dev subset).
# Re-stride to 32 here, using convert_train.py's exact stride formula.
TRAIN_MAX_FRAMES = int(os.environ.get("PROACTIVE_MAX_FRAMES", "32"))


def _stride_to(frames: list, cap: int = TRAIN_MAX_FRAMES) -> list:
    if cap > 0 and len(frames) > cap:
        s = len(frames) / cap
        return [frames[int(i * s)] for i in range(cap)]
    return frames
DEFAULT_TAU = float(os.environ.get("PROACTIVE_TAU", "0.55"))
DEFAULT_UTTERANCE = "Here's the next step — keep going."


class ProactiveVerbalizerModel(VideoQAModel):
    """Qwen3.5-VL yes/no verbalizer for the EgoProactive track (single GPU per worker)."""

    def __init__(self, model_path: str, tau: float = DEFAULT_TAU, **kwargs: object) -> None:
        self.model_path = model_path
        self.tau = tau
        attn = "flash_attention_2"
        try:
            import flash_attn  # noqa: F401
        except Exception:
            attn = "sdpa"

        # Layout A (preferred, bit-faithful to eval_from_frames.py): base/ + adapter/ applied at
        # runtime with PEFT. Merging LoRA into bf16 weights rounds W+BA and systematically shifts
        # p_interrupt upward (measured: +58 vs -10 chunk flips on a 544-chunk dev subset).
        base_dir = os.path.join(model_path, "base")
        adapter_dir = os.path.join(model_path, "adapter")
        if os.path.isdir(base_dir) and os.path.isdir(adapter_dir):
            from peft import PeftModel

            self.processor = AutoProcessor.from_pretrained(base_dir)
            self.model = AutoModelForImageTextToText.from_pretrained(
                base_dir, dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn
            )
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
        else:
            # Layout B: pre-merged full weights.
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, dtype=torch.bfloat16, device_map="cuda", attn_implementation=attn
            )
        self.model.eval()
        tok = self.processor.tokenizer
        self.id_yes = tok.encode("yes", add_special_tokens=False)[0]
        self.id_no = tok.encode("no", add_special_tokens=False)[0]

        # ---- vocab-pruned checkpoint support (<=2B small division) ----------------------
        # The tokenizer is UNCHANGED (identical BPE splits); we remap ids onto the pruned
        # embedding table. Identity for ids < keep_low, the 33 specials are relocated, and any
        # pruned id expands to its exact UTF-8 byte tokens (all byte tokens are ids 0..255).
        self.remap = None
        rp = os.path.join(model_path, "vocab_remap.json")
        if os.path.exists(rp):
            r = json.load(open(rp))
            self.keep_low = int(r["keep_low"])
            self.added_start = int(r["added_start"])
            self.n_added = int(r["n_added"])
            self.n_new = int(r["n_new"])
            self.fallback = {int(k): v for k, v in r["fallback"].items()}
            # explicitly retained high-ID tokens (observed task vocabulary + domain words)
            extras = [int(x) for x in r.get("extras", [])]
            self.extra_pos = {t: self.keep_low + k for k, t in enumerate(extras)}
            self.added_base = self.keep_low + len(extras)
            self.remap = True
            self.id_yes = self._remap_one(self.id_yes)
            self.id_no = self._remap_one(self.id_no)
            print(f"[proactive] vocab-pruned model: {self.n_new} tokens, "
                  f"{len(self.fallback)} byte-exact fallbacks", flush=True)

    def _remap_one(self, t: int) -> int:
        if self.remap is None or t < self.keep_low:
            return t
        if t in self.extra_pos:
            return self.extra_pos[t]
        if self.added_start <= t < self.added_start + self.n_added:
            return self.added_base + (t - self.added_start)
        seq = self.fallback.get(t)
        return seq[0] if seq else 0

    def _remap_ids(self, ids: list[int]) -> list[int]:
        """Identity for kept ids; relocate specials; expand pruned ids to exact byte tokens."""
        out: list[int] = []
        for t in ids:
            if t < self.keep_low:
                out.append(t)
            elif t in self.extra_pos:
                out.append(self.extra_pos[t])
            elif self.added_start <= t < self.added_start + self.n_added:
                out.append(self.added_base + (t - self.added_start))
            else:
                out.extend(self.fallback.get(t, [0]))
        return out

    @torch.no_grad()
    def generate(
        self,
        frames: list[object],
        messages: list[dict[str, str]],
        max_new_tokens: int = 16,
        **kwargs: object,
    ) -> str:
        # Never let one bad chunk kill the worker: the organizer runs 8 workers per node and a
        # crash would zero that whole shard. A wrong single chunk is far cheaper.
        try:
            return self._decide(frames, messages)
        except Exception as e:  # noqa: BLE001
            print(f"[proactive] chunk failed ({type(e).__name__}: {e}); defaulting to $silent$", flush=True)
            return "$silent$"

    @torch.no_grad()
    def _decide(self, frames: list[object], messages: list[dict[str, str]]) -> str:
        imgs = [_as_training_frame(f) for f in _stride_to(frames)]

        # The runner supplies: system (ITS prompt — dropped), user(query), then dialog history.
        non_sys = [m for m in messages if m.get("role") != "system"]
        query = non_sys[0]["content"] if non_sys else ""
        history = non_sys[1:]

        user0 = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": str(query)}]
        msgs: list[dict[str, object]] = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user0},
        ]
        for t in history:
            msgs.append(
                {"role": t.get("role", "assistant"),
                 "content": [{"type": "text", "text": str(t.get("content", ""))}]}
            )

        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = re.sub(r"<think>\s*</think>\s*", "", text)
        text = re.sub(r"<think>\s*$", "", text)

        inp = self.processor(
            text=[text], images=imgs, return_tensors="pt",
            images_kwargs={"max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS},
        ).to(self.model.device)

        if self.remap is not None:
            src = inp["input_ids"][0].tolist()
            dst = self._remap_ids(src)
            if dst != src:
                dev = inp["input_ids"].device
                inp["input_ids"] = torch.tensor([dst], dtype=torch.long, device=dev)
                if "attention_mask" in inp:
                    inp["attention_mask"] = torch.ones((1, len(dst)), dtype=torch.long, device=dev)
            bad = max(dst) if dst else 0
            if bad >= self.n_new:                     # must never happen; fail loud in dev
                raise ValueError(f"remapped id {bad} >= vocab {self.n_new}")

        lg = self.model(**inp).logits[0, -1]
        pair = torch.softmax(torch.stack([lg[self.id_no], lg[self.id_yes]]).float(), 0)
        p_interrupt = float(pair[1])
        dbg = os.environ.get("PROACTIVE_DEBUG")
        if dbg:
            with open(dbg, "a") as fh:
                fh.write(json.dumps({
                    "p": p_interrupt, "n_frames": len(imgs),
                    "img_size": list(imgs[0].size) if imgs else None,
                    "n_hist": len(history), "query_head": str(query)[:40],
                    "ntok": int(inp["input_ids"].shape[-1]),
                }) + "\n")
        return f"$interrupt${DEFAULT_UTTERANCE}" if p_interrupt >= self.tau else "$silent$"

    def generate_batch(
        self,
        batch_frames: list[list[object]],
        batch_messages: list[list[dict[str, str]]],
        max_new_tokens: int = 16,
        **kwargs: object,
    ) -> list[str]:
        return [self.generate(f, m, max_new_tokens) for f, m in zip(batch_frames, batch_messages)]
