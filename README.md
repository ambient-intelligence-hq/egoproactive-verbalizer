# Speak or Stay Silent

Winning approach for the **EgoProactive** track of the [Wearable AI Challenge @ ECCV 2026](https://wearable-ai-workshop.github.io/) — 1st in the large division, 2nd in the ≤2B division.

| Division | Model | Params | Macro-F1 | Place |
|---|---|---|---|---|
| Large (2B+) | Qwen3.5-4B + verbalizer LoRA | 4.54 B | **0.7179** | **1st** |
| Small (≤2B) | Qwen3.5-2B + verbalizer LoRA, vocab-pruned | 1.9977 B | 0.6866 | 2nd |

## The idea in one paragraph

The task is: after every ~8s chunk of egocentric video, emit `$interrupt$<utterance>` or `$silent$`.
Trained as text generation this is degenerate — the *whether* to speak gets entangled with *what* to
say, and the model collapses to interrupt precision 1.000 at recall 0.138. Instead we make the model
emit **one token**, `yes` or `no`, and read the decision off those two logits:

```
p_interrupt = softmax([logit_no, logit_yes])[1]      interrupt if p_interrupt >= tau   (tau = 0.55)
```

Loss is cross-entropy on that single token only; everything else is masked. This is worth **+0.249
macro-F1** over the generative formulation — more than every data and scale intervention combined.
Utterances are templated at inference, because the metric never scores their content.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # versions are pinned; see Gotchas
export EGOPROACTIVE_STARTER_KIT=/path/to/organizer/starter_kit
```

## Reproduce

### 1. Data

Download the challenge videos and annotations from
[`facebook/wearable-ai`](https://huggingface.co/datasets/facebook/wearable-ai).

The training mix is the released validation videos **seen twice** plus a synthetic corpus of 234
agent-annotated clips (13,730 rows, 53% interrupt). Both row files are published:

```bash
huggingface-cli download ambient-intelligence-labs/egoproactive-synth-annotations --repo-type dataset \
  --include "val700_train.jsonl" "egoconv_dense_train.jsonl" --local-dir data/
# mix = val700 rows x2 + the full egoconv set, shuffled  (see TRAINING_MIX.md in that repo)
```

To regenerate the synthetic corpus instead of downloading it, see [`annotation/`](annotation/) —
it drives the [Ambient](https://github.com/ambient-intelligence-hq/ambient) video agent with
`annotation_prompt.txt` over a clip manifest, and is resumable (skips clips already annotated).

### 2. Frames

```bash
python src/convert_train.py \
  --rows data/mix_all.jsonl \
  --out  data/train_chunks.jsonl \
  --videos data/videos --frames_root data/frames \
  --max_frames 32
```

`--max_frames 32` is not optional — see Gotchas. Frame extraction over long clips is slow;
shard `--rows` across cores and concatenate the outputs (12-way took ~8 h down to ~40 min).

### 3. Train

```bash
python src/train_verbalizer.py --data data/train_chunks.jsonl --out runs/verbalizer_4b \
  --epochs 1 --rank 32 --alpha 64
```

Defaults match the winning runs: LoRA r=32 α=64 on all seven attention/MLP projections, lr 1e-4
cosine with 0.03 warmup, batch size 1 × 8 accumulation, bf16 + gradient checkpointing.
Swap the base model in the script for the 2B division.

### 4. Evaluate

```bash
python src/eval_from_frames.py --adapter runs/verbalizer_4b --samples data/heldout.jsonl \
  --out results.json
```

**Evaluate out of domain.** In-domain scores on the released set reached 0.99 while the same
checkpoints delivered 0.49 on unseen footage, and the one data lever that won the track looks like a
regression in domain. Hold out a set from a *different* recording domain and rank on that.

### 5. Fit under 2B (small division only)

Qwen3.5-2B is 2.2132 B parameters once the vision tower counts. Contiguous vocabulary pruning brings
it to 1.9977 B with **identical predictions** (544/544 chunks, zero byte-fallback across all 700
validation videos):

```bash
python vocab_pruning/count_rare.py    merged_2b               # observed rare token IDs
python vocab_pruning/build_extras.py  600                     # -> extra_ids.json (tokens to keep above the cut)
python vocab_pruning/prune_vocab.py   merged_2b pruned_2b 143000 extra_ids.json
python vocab_pruning/verify_prune.py  pruned_2b merged_2b     # must pass 4/4
python vocab_pruning/count_params.py  pruned_2b               # must read < 2.0000 B
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `EGOPROACTIVE_STARTER_KIT` | `starter_kit` | Path to the organizer starter kit (required) |
| `VERB_MODEL` | `Qwen/Qwen3.5-4B` | Base model to fine-tune |
| `VERB_REPORT_TO` | `none` | Set to `wandb` to log to Weights & Biases |
| `EP_DATA` / `EP_OUT` | `data/train.jsonl` / `runs/verbalizer_lora` | Training defaults |
| `EP_VIDEOS` / `EP_FRAMES` | `data/videos` / `data/frames` | Frame-extraction defaults |

## Weights

LoRA adapters (private; request access):

| Submission | Repo |
|---|---|
| 4B large (1st place) | `ambient-intelligence-labs/egoproactive-4b-lora` |
| 2B small (2nd place) | `ambient-intelligence-labs/egoproactive-2b-lora` |

## Citation

The full method, ablations and negative results are in the tech report. If you use this work:

```bibtex
@techreport{umapathi2026speak,
  title  = {Ambient @ EgoProactive 2026 : Proactive Egocentric Assistance with Visually Grounded Supervision},
  author = {Umapathi, Logesh Kumar},
  year   = {2026},
  institution = {Team Ambient},
  note   = {Wearable AI Challenge @ ECCV 2026, EgoProactive track}
}
```
