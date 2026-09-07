"""Does ANY chunk across ALL 700 val videos still need the byte-fallback under v7's keep-set?
Text-only check (no GPU): if zero, the pruned model is provably identical on the whole val set."""
import json, collections
from transformers import AutoProcessor
r = json.load(open("/models/proactive/vocab_remap.json"))
KEEP, ADD, NADD = r["keep_low"], r["added_start"], r["n_added"]
extras = set(int(x) for x in r.get("extras", []))
FB = {int(k) for k in r["fallback"]}
tok = AutoProcessor.from_pretrained("/models/proactive").tokenizer
print(f"keep_low={KEEP:,} extras={len(extras)} added={NADD} -> vocab {r['n_new']:,}")

SYS=("You are a proactive AI assistant watching a first-person video of the user performing a "
     "procedural task, given their initial high-level query. The video arrives as short (~8s) "
     "chunks; after each chunk you decide whether NOW is the right moment to speak up with timely, "
     "useful guidance (a new step, or a correction), or to stay silent because nothing useful needs "
     "saying yet.\n\nAnswer with a single word: `yes` to speak now, or `no` to stay silent.")
def hits(ids): return [t for t in ids if t in FB]
sys_hits = hits(tok.encode(SYS, add_special_tokens=False))

vids=chunks=chunk_hits=tok_hits=tok_tot=0
remaining=collections.Counter()
for line in open("/data/val700.jsonl"):
    row=json.loads(line); vids+=1
    q=str(row.get("query","")); dialog=row.get("dialog",[])
    for j in range(len(row.get("answers",[]))):
        turns = dialog[j][1:] if j < len(dialog) else []
        ids=[t for s in [q]+[str(t.get("text","")) for t in turns[-4:]]
               for t in tok.encode(s, add_special_tokens=False)]
        h=hits(ids); chunks+=1; tok_tot+=len(ids); tok_hits+=len(h)
        if h:
            chunk_hits+=1
            for t in h: remaining[t]+=1
print(f"videos={vids}  chunks={chunks:,}")
print(f"system prompt fallback hits: {len(sys_hits)}")
print(f"CHUNKS still needing fallback: {chunk_hits:,}/{chunks:,} = {100*chunk_hits/chunks:.4f}%")
print(f"TOKEN occurrences needing fallback: {tok_hits:,}/{tok_tot:,} = {100*tok_hits/max(tok_tot,1):.5f}%")
if remaining:
    print("still-falling-back tokens:", [(tok.decode([t]),c) for t,c in remaining.most_common(10)])
else:
    print("RESULT: ZERO fallback across all 700 val videos -> pruned model is provably identical on the whole val set")
