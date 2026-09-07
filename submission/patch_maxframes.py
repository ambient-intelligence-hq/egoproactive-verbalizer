"""Set --max-frames default to 32 in run_evaluation.py (and run_generate_*).
Our verbalizer was trained with 32 cumulative frames per decision (convert_train --max_frames 32);
the stock default of 100 is a train/serve mismatch. Patching the default keeps the SINGLE stride
(16*(j+1) -> 32) that training used — re-striding 100 -> 32 in the model class picks a different
subset and is worse."""
import re, sys
p = sys.argv[1] if len(sys.argv) > 1 else "/app/run_evaluation.py"
s = open(p).read()
new, n = re.subn(r'("--max-frames"[^)]*?default=)100\b', r"\g<1>32", s, flags=re.S)
open(p, "w").write(new)
print(f"patched {p}: {n} occurrence(s) 100 -> 32")
