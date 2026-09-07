#!/usr/bin/env python
"""Turn split rows into per-decision training samples that match the official harness prompt.

For every chunk j of every training video we build ONE sample whose visual context is exactly
what `run_generate_proactive.py` feeds the model at that decision:
  - 16 frames/interval (via the harness's own extract_frames), cumulative over intervals [0..j],
    strided down to --max-frames (32) with the identical stride math,
  - context = system prompt + query + last --max-history-turns dialog turns,
  - target  = answers[j]  ($interrupt$<utterance> | $silent$),  supervised (only this turn).
Frames are cached once per (video, interval) as jpgs (downscaled; the processor downsamples to
~50k px anyway) and referenced by path, so we don't re-decode per decision.

  python convert_train.py --rows train_rows.jsonl --videos <dir> --frames_root <dir> --out train.jsonl
"""
import argparse, json, os, sys

sys.path.insert(0, os.environ.get("EGOPROACTIVE_STARTER_KIT", "starter_kit"))
from model import extract_frames                       # noqa: E402  (harness's own extractor)
from run_generate_proactive import SYSTEM_PROMPT       # noqa: E402


def stride_to(frames, cap):
    if cap > 0 and len(frames) > cap:
        s = len(frames) / cap
        return [frames[int(i * s)] for i in range(cap)]
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--videos", default=os.environ.get("EP_VIDEOS", "data/videos"))
    ap.add_argument("--frames_root", default=os.environ.get("EP_FRAMES", "data/frames"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames_per_interval", type=int, default=16)
    ap.add_argument("--max_frames", type=int, default=32)
    ap.add_argument("--max_history_turns", type=int, default=4)
    ap.add_argument("--jpeg_max_side", type=int, default=512)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.rows)]
    os.makedirs(a.frames_root, exist_ok=True)
    n_samp = n_int = n_sil = miss = 0
    fout = open(a.out, "w")

    for vi, r in enumerate(rows):
        vp = os.path.join(a.videos, str(r["video_path"]))
        if not os.path.exists(vp):
            miss += 1; continue
        stem = os.path.splitext(os.path.basename(r["video_path"]))[0]
        intervals = [(float(s), float(e)) for s, e in r["video_intervals"]]
        query = str(r.get("query", ""))
        dialog = r.get("dialog", [])
        answers = r["answers"]

        # extract + cache 16 frames per interval once
        per_iv_paths = []
        for k, iv in enumerate(intervals):
            outdir = os.path.join(a.frames_root, stem, f"iv{k:03d}")
            os.makedirs(outdir, exist_ok=True)
            paths = [os.path.join(outdir, f"f{m:02d}.jpg") for m in range(a.frames_per_interval)]
            if not all(os.path.exists(p) for p in paths):
                imgs = extract_frames(vp, intervals=[iv], frames_per_interval=a.frames_per_interval)
                paths = []
                for m, im in enumerate(imgs):
                    im = im.convert("RGB")
                    w, h = im.size; sc = a.jpeg_max_side / max(w, h)
                    if sc < 1.0:
                        im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))))
                    p = os.path.join(outdir, f"f{m:02d}.jpg"); im.save(p, "JPEG", quality=90)
                    paths.append(p)
            per_iv_paths.append(paths)

        for j in range(len(intervals)):
            cum = [p for k in range(j + 1) for p in per_iv_paths[k]]
            cum = stride_to(cum, a.max_frames)
            if not cum:
                continue
            turns_after = dialog[j][1:] if j < len(dialog) and len(dialog[j]) >= 1 else []
            if a.max_history_turns >= 0:
                turns_after = turns_after[-a.max_history_turns:] if a.max_history_turns else []
            history = [{"role": ("assistant" if str(t.get("role", "user")).lower() == "assistant" else "user"),
                        "text": str(t.get("text") or "")} for t in turns_after if (t.get("text") or "")]
            target = answers[j]
            label = "interrupt" if target.startswith("$interrupt$") else "silent"
            fout.write(json.dumps({
                "images": cum, "query": query, "history": history, "target": target,
                "label": label, "video": r["video_path"], "domain": r.get("domain"),
                "chunk": j, "system": SYSTEM_PROMPT,
            }, ensure_ascii=False) + "\n")
            n_samp += 1; n_int += (label == "interrupt"); n_sil += (label == "silent")
        if (vi + 1) % 50 == 0:
            print(f"  {vi+1}/{len(rows)} videos | {n_samp} samples", flush=True)

    fout.close()
    print(f"DONE samples={n_samp} interrupt={n_int} silent={n_sil} missing_videos={miss}")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
