"""Full-run driver: annotate all videos in MANIFEST_FILE with the ambient agent (t3 policy).
Hardened for a multi-hour run:
  - concurrency capped by CONCURRENCY (semaphore)
  - incremental: writes full_ann/ann_<stem>.json the moment each clip finishes
  - resumable: skips any clip whose output already exists
Env: AGENT_MODEL, MANIFEST_FILE, VIDEO_FOLDER, CONCURRENCY (default 8)."""
import asyncio, json, os, re, subprocess, sys, traceback, glob
sys.path.insert(0, ".")
import dotenv; dotenv.load_dotenv()
from pydantic import BaseModel, Field
from ambient.agent import run_agent

VF = os.environ["VIDEO_FOLDER"]
PROMPT = open("annotation_prompt.txt").read()
MANIFEST = json.load(open(os.environ.get("MANIFEST_FILE", "egoconv235_manifest.json")))
CONC = int(os.environ.get("CONCURRENCY", "8"))
DENSE = bool(os.environ.get("DENSE"))
OUT = os.environ.get("OUT_DIR") or ("full_ann_dense" if DENSE else "full_ann")
os.makedirs(OUT, exist_ok=True)


class WalkEvent(BaseModel):
    timestamp: float = Field(description="seconds into the video when the assistant speaks")
    decision: str = Field(description="one of: intervene, correct, confirm, redirect, insist, observe")
    interaction_type: str = Field(default="")
    visible_context: str = Field(default="")
    utterance: str = Field(description="the exact short sentence the proactive assistant speaks")
    timing_rationale: str = Field(default="")
    confidence: float = Field(default=0.0)


class SilentInterval(BaseModel):
    start_time: float
    end_time: float
    reason: str = ""


class ProactiveAnnotation(BaseModel):
    task_name: str = Field(default="")
    goal: str = Field(default="")
    walkthrough: list[WalkEvent] = Field(description="timestamped proactive assistant events")
    silent_intervals: list[SilentInterval] = Field(default_factory=list)


def POLICY_DENSE(dur):
    target = max(4, round(dur / 14))   # ~one cue per 14s -> ~50% of 8s chunks
    return f"""
# INTERVENTION POLICY (dense proactive step-by-step coaching)
Follow these rules exactly — they define WHEN the assistant speaks:
1. COVER THE SETUP PHASE FROM THE START. Your FIRST event must be at ~0-2s (gather/prepare the
   materials or move to the work location), then a cue for each preparation step before the main
   manipulation begins.
2. FINE-GRAINED STEPS — AIM FOR ABOUT {target} CUES, SPREAD EVENLY START TO END. Decompose the task
   into fine sub-steps across the WHOLE video and emit roughly {target} interventions (about one
   every 12-15 seconds of active work). Do NOT summarize the video into a few major steps — annotate
   each meaningful sub-action as it comes. Space events across the full 0..{dur:.0f}s timeline, not
   clustered at the start.
3. FIRE AT ONSET, SLIGHTLY EARLY — NEVER AFTER. Time each utterance to the moment the step is ABOUT
   to begin; when unsure place it EARLIER, at the start of the step's window.
4. COLLAPSE ONLY TIGHT REPETITION. If the SAME motion repeats rapidly within a few seconds, give it
   ONE cue. But distinct sub-steps of an ongoing activity EACH get their own cue — do not silence a
   long working stretch; guide it step by step.
5. GROUND EVERY CUE in what you actually inspected. Inspect the entire timeline (see the tool budget)
   before annotating; do not invent steps for parts you did not view. One short verify at the true end.
"""


def POLICY(dur):
    cap = max(3, round(dur / 8))
    return f"""
# INTERVENTION POLICY (annotate like a real-time proactive step-by-step coach)
Follow these rules exactly — they define WHEN the assistant speaks:
1. COVER THE SETUP PHASE FROM THE START. Your FIRST event must be at ~0-2s and tell the user to
   gather/prepare the materials or move to the work location — emit it even before the main action
   is visible. Then give a cue for each key preparation step (locate/position the tool, ready the
   surface, open/align things) BEFORE the main manipulation begins. Never wait for the main action
   to appear before starting.
2. ONE CUE PER DISTINCT STEP — NO TARGET COUNT. Emit one short cue only when a genuinely NEW step or
   sub-action begins. Do NOT aim for any particular number of events — let the task's real distinct
   steps decide it. Do NOT add follow-ups, refinements, encouragement, or "looks good / all set"
   remarks to a step already underway or just finished.
3. COLLAPSE REPETITION HARD. For ANY repeated or continuous motion — drawing/coloring strokes,
   rolling, wiping, sanding, folding passes, stirring, scrubbing, brushing — emit exactly ONE cue at
   the very START of that motion, then STAY SILENT for its entire duration and record it as a single
   silent_interval. NEVER emit a cue per stroke / pass / repetition. As a hard ceiling, never place
   more than one event per 8 seconds anywhere (for this {dur:.0f}s video, at most ~{cap} events).
4. FIRE AT ONSET, SLIGHTLY EARLY — NEVER AFTER. Time each utterance to the moment the step is ABOUT
   to begin, just before or exactly as the action starts — a proactive coach speaks a beat AHEAD of
   the action. When unsure, place the timestamp EARLIER, at the start of the step's window, not in
   the middle of the motion.
5. GO SILENT AFTER COMPLETION. Once the task is essentially finished, stay silent. At most ONE short
   verify cue at the true end, and only if a check is genuinely warranted.
"""


def probe(stem):
    vids = [p for p in glob.glob(os.path.join(VF, f"{stem}*"))
            if p.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi"))]
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", vids[0]], capture_output=True, text=True).stdout.strip()
    return float(out)


_DEC = json.JSONDecoder()


def _extract_ann(txt):
    """Use a real JSON parser (raw_decode) at each '{' start — handles braces inside strings and
    ignores trailing prose, unlike naive regex/brace-counting."""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    txt = re.sub(r"```(?:json)?|```", "", txt)
    fallback = None
    for m in re.finditer(r"\{", txt):
        try:
            d, _ = _DEC.raw_decode(txt[m.start():])
        except Exception:
            continue
        if isinstance(d, dict) and ("walkthrough" in d or "task_name" in d):
            return d
        if fallback is None and isinstance(d, dict):
            fallback = d
    return fallback


def answer_from_trajectory(stem):
    # newest-first, but fall back across files: a newer trajectory may be a dud (no walkthrough)
    tfs = sorted(glob.glob(os.path.join(".", "ambient/trajectories", f"*{stem}*.json")),
                 key=os.path.getmtime, reverse=True)
    model = None
    for tf in tfs:
        t = json.load(open(tf))
        model = model or t.get("model")
        for turn in reversed(t.get("turns", [])):
            d = _extract_ann(turn.get("text") or "")
            if d is not None and d.get("walkthrough"):
                return d, t.get("model") or model
    return None, model


async def do_one(sem, stem, query, idx, total):
    outp = f"{OUT}/ann_{stem}.json"
    if os.path.exists(outp):
        return
    async with sem:
        try:
            dur = probe(stem)
            if DENSE:
                ncalls = min(12, max(5, round(dur / 45)))   # cover the whole long timeline
                seg = dur / ncalls
                pol = POLICY_DENSE(dur)
                budget = (f"TOOL BUDGET: inspect the ENTIRE timeline with up to {ncalls} focus_clip "
                          f"calls, one per ~{seg:.0f}s window, walking start->end so you SEE every part "
                          f"before annotating (first window is the 0-{seg:.0f}s setup). Request clips "
                          f"only with a POSITIVE duration within 0..{dur:.0f}s.")
                maxturns = ncalls + 6
            else:
                pol = POLICY(dur)
                budget = (f"TOOL BUDGET: inspect the video with AT MOST 5 focus_clip/search_clip calls "
                          f"total. Spend your FIRST inspection on the opening 0-{max(6,int(dur*0.15))}s "
                          f"to capture the setup/prep steps, then cover key middle actions and the end. "
                          f"Request clips only with a POSITIVE duration within 0..{dur:.0f}s.")
                maxturns = 12
            q = (PROMPT + pol + f"\n\n# This input\nVideo id: {stem}. The user asked: \"{query}\"\n"
                 f"Annotate WHEN a proactive assistant should speak to guide this exact task, and when "
                 f"it should stay silent, following the INTERVENTION POLICY above. The video is "
                 f"approximately {dur:.0f} seconds long (first-person egocentric).\n\n"
                 f"{budget} After that you MUST stop calling tools and output ONLY the final JSON "
                 f"walkthrough per the schema as your final message — no further tool calls, no prose "
                 f"outside the JSON. Keep all timestamps within 0..{dur:.0f}s.")
            msgs = await run_agent(video_id=stem, question=q, max_turns=maxturns,
                                   output_structure=ProactiveAnnotation)
            ann, model = answer_from_trajectory(stem)
            json.dump({"stem": stem, "query": query, "dur": dur, "model_used": model, "annotation": ann},
                      open(outp, "w"), indent=2)
            nw = len(ann.get("walkthrough", [])) if ann else "PARSE_FAIL"
            print(f">>> [{idx}/{total}] {stem}: events={nw} model={model}", flush=True)
        except Exception as e:
            json.dump({"stem": stem, "query": query, "annotation": None, "error": str(e)},
                      open(outp, "w"), indent=2)
            print(f">>> [{idx}/{total}] {stem} ERROR: {str(e)[:80]}", flush=True)


async def main():
    sem = asyncio.Semaphore(CONC)
    items = list(MANIFEST.items())
    total = len(items)
    done = sum(1 for s, _ in items if os.path.exists(f"{OUT}/ann_{s}.json"))
    print(f"START full run: {total} clips, {done} already done, concurrency={CONC}", flush=True)
    await asyncio.gather(*[do_one(sem, s, q, i + 1, total) for i, (s, q) in enumerate(items)])
    print("FULL_RUN_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
