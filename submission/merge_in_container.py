import os, sys, torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
BASE = sys.argv[1]; ADP = sys.argv[2]; OUT = sys.argv[3]
print("loading base:", BASE, flush=True)
proc = AutoProcessor.from_pretrained(BASE)
base = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
print("merging adapter:", ADP, flush=True)
m = PeftModel.from_pretrained(base, ADP)
m = m.merge_and_unload()
os.makedirs(OUT, exist_ok=True)
m.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
proc.save_pretrained(OUT)
print("MERGED_DONE", OUT, flush=True)
