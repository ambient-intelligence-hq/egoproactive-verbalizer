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

## Repository layout

```
src/                     training and evaluation
  convert_train.py         videos + annotations -> per-chunk training rows (+ frames)
  train_verbalizer.py      LoRA fine-tune of the yes/no verbalizer
  eval_from_frames.py      score an adapter, sweeping tau
annotation/              synthetic supervision (see "Data" below)
  annotate_full.py         resumable driver: runs the agent over a clip manifest
  annotation_prompt.txt    the tuned "t3" placement policy
submission/              the container submitted to the organizers
  Containerfile            4B large-division image
  Containerfile.2b         2B small-division image (expects a pruned model)
  model_appendix_proactive.py   the submission model class
  register_proactive.py    registers it under the allowed `qwen` model type
  patch_maxframes.py       fixes the harness frame-count default (see Gotchas)
  merge_in_container.py    merges the LoRA into the base model
vocab_pruning/           getting a 2.2132B model under the 2B division limit
```

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
huggingface-cli download infinitylogesh/egoproactive-synth-annotations --repo-type dataset \
  --include "val700_train.jsonl" "egoconv_dense_train.jsonl" --local-dir data/
# mix = val700 rows x2 + the full egoconv set, shuffled  (see TRAINING_MIX.md in that repo)
```

To regenerate the synthetic corpus instead of downloading it, see [`annotation/`](annotation/) —
it drives the [Ambient](https://github.com/ambient-intelligence-hq/ambient) video agent with
`annotation_prompt.txt` over a clip manifest, and is resumable (skips clips already annotated).

### 2. Frames

```bash
python src/convert_train.py --videos data/videos --frames_root data/frames --max_frames 32
```

`--max_frames 32` is not optional — see Gotchas.

### 3. Train

```bash
python src/train_verbalizer.py --data data/mix_all.jsonl --out runs/verbalizer_4b \
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

### 6. Build the submission image

```bash
python submission/merge_in_container.py Qwen/Qwen3.5-4B runs/verbalizer_4b merged_4b
sudo chown -R "$USER" merged_4b          # not optional - see Gotchas
docker build -f submission/Containerfile -t proactive:v1 .
bash starter_kit/validate_image.sh proactive:v1        # must be 12/12
```

## Gotchas

These each cost us real accuracy or a wasted training run.

1. **Frame count must match between training and serving.** The evaluation harness defaults to 100
   frames per decision; the models are trained on 32. The mismatch costs 0.027 macro-F1.
   `patch_maxframes.py` fixes the default at image build time. Do **not** additionally re-stride
   inside the model class — double-striding scored 0.61 against 0.79.
2. **`chown` after merging.** Merging as root leaves weights unreadable to the build step, which then
   bakes **zero-byte tensors** into the image — and the organizers' 12-check validation still passes,
   because it only asserts the model directory is non-empty.
3. **Left-pad batched Qwen-VL training.** Right-padding corrupts M-RoPE and shifts the loss by ~0.09.
   Batch size 1 sidesteps it.
4. **Pinned versions matter.** `trl` 1.9.2 dropped `warmup_ratio`. Use the pins in `requirements.txt`.
5. **Dialogue history is load-bearing.** Remove it and the model fires on every chunk — it has no way
   to know it already spoke.
6. **More synthetic data is not better.** A second corpus added on top displaced the good data at
   equal compute and cost 0.042. Hold compute fixed when ablating.

## Weights

LoRA adapters (private; request access):

| Submission | Repo |
|---|---|
| 4B large (1st place) | `infinitylogesh/egoproactive-verbalizer-4b-val700-egoconv` |
| 2B small (2nd place) | `infinitylogesh/egoproactive-verbalizer-2b-val700-egoconv` |

## Citation

The full method, ablations and negative results are in the tech report. If you use this work:

```bibtex
@techreport{umapathi2026speak,
  title  = {Speak or Stay Silent: A Single-Token Verbalizer and Agent-Generated
            Supervision for Proactive Egocentric Assistance},
  author = {Umapathi, Logesh Kumar},
  year   = {2026},
  institution = {Team Ambient},
  note   = {Wearable AI Challenge @ ECCV 2026, EgoProactive track}
}
```
