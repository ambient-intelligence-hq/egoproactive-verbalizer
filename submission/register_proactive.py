"""Append the proactive verbalizer registration to the starter kit's model.py.

argparse restricts --model-type to {llama4, qwen}, so we OVERRIDE the `qwen` key.
`qwen` already carries DEFAULT_GPU_COUNTS=1 / TP=1, which is what we want for a 4B/2B:
the organizer's runner then computes num_workers = 8 // 1 = 8 -> eight independent
single-GPU worker copies per node (8x throughput, identical per-query output).
"""
import os
import sys

MODEL_PY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("STARTER_MODEL_PY", "/app/model.py")
MODEL_DIR = sys.argv[2] if len(sys.argv) > 2 else "/models/proactive"

APPEND = f'''

# --- EgoProactive verbalizer registration (participant extension) -----------
from model_appendix_proactive import ProactiveVerbalizerModel  # noqa: E402

MODEL_REGISTRY["qwen"] = ProactiveVerbalizerModel
MODEL_REGISTRY["proactive"] = ProactiveVerbalizerModel      # alias (unused by argparse)
DEFAULT_MODEL_IDS["qwen"] = "{MODEL_DIR}"
DEFAULT_MODEL_IDS["proactive"] = "{MODEL_DIR}"
DEFAULT_BATCH_SIZES["qwen"] = 1                              # generate() is single-sample
DEFAULT_BATCH_SIZES["proactive"] = 1
DEFAULT_GPU_COUNTS["qwen"] = 1                               # 8 workers on an 8-GPU node
DEFAULT_GPU_COUNTS["proactive"] = 1
DEFAULT_TP_SIZES["qwen"] = 1
DEFAULT_TP_SIZES["proactive"] = 1
'''

src = open(MODEL_PY).read()
if "ProactiveVerbalizerModel" in src:
    print("already registered")
else:
    open(MODEL_PY, "a").write(APPEND)
    print("registered ProactiveVerbalizerModel under 'qwen' ->", MODEL_DIR)
