"""How many UNIQUE high-ID tokens does real task text actually use? If it fits the spare
budget we can keep them explicitly and remove the fallback path for observed vocabulary."""
import json, collections
from transformers import AutoProcessor
tok = AutoProcessor.from_pretrained("/models/proactive").tokenizer
KEEP = 143000; ADD = 248044
rare = collections.Counter()
for path in ["/data/val700.jsonl"]:
    for line in open(path):
        r = json.loads(line)
        texts = [str(r.get("query","")), str(r.get("task",""))] + [str(a) for a in r.get("answers",[])]
        for turns in r.get("dialog", []):
            texts += [str(t.get("text","")) for t in turns]
        for s in texts:
            for t in tok.encode(s, add_special_tokens=False):
                if KEEP <= t < ADD:
                    rare[t] += 1
print(f"unique high-ID tokens used by val700: {len(rare):,}")
print(f"total occurrences: {sum(rare.values()):,}")
print(f"budget headroom (to stay <2B): ~1,186 tokens -> {'FITS' if len(rare)<=1186 else 'DOES NOT FIT'}")
print("top:", [(tok.decode([t]), c) for t,c in rare.most_common(6)])
json.dump(sorted(rare), open("/work/rare_ids.json","w"))
