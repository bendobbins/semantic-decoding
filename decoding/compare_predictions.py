"""Build a readable side-by-side comparison of each decoder run's predicted
words against the actual story transcript.

For every result file under results/<subject>/<experiment>/<run>.npz this loads
the predicted (words, times) and the matching reference TextGrid, splits both
into fixed time chunks, and writes the actual text next to each run's decoded
text so you can read down the story and see what each model produced.

Outputs (default into the results dir):
    comparison.md   chunks grouped by time window; actual text then each run.
    comparison.csv  same data in long format (chunk x run) for spreadsheets.

Usage:
    python compare_predictions.py --subject S1 --experiment perceived_speech
    python compare_predictions.py --chunk 15        # 15s chunks instead of 20s
"""

import os
import csv
import glob
import argparse
import numpy as np

import config
from utils_ridge.textgrid import TextGrid

BAD_WORDS_PERCEIVED_SPEECH = frozenset(
    ["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])
BAD_WORDS_OTHER_TASKS = frozenset(["", "sp", "uh"])


def load_reference(experiment, task):
    """Word-level reference transcript for a story (mirrors utils_eval)."""
    skip = BAD_WORDS_PERCEIVED_SPEECH if experiment in (
        "perceived_speech", "perceived_multispeaker") else BAD_WORDS_OTHER_TASKS
    grid_path = os.path.join(config.DATA_TEST_DIR, "test_stimulus",
                             experiment, task.split("_")[0] + ".TextGrid")
    with open(grid_path) as f:
        grid = TextGrid(f.read())
        tier = 1 if experiment == "perceived_speech" else 0
        transcript = grid.tiers[tier].make_simple_transcript()
        transcript = [(float(s), float(e), w.lower()) for s, e, w in transcript
                      if w.lower().strip("{}").strip() not in skip]
    words = np.array([x[2] for x in transcript])
    times = np.array([(x[0] + x[1]) / 2 for x in transcript])
    return words, times


def chunk_text(words, times, start, end):
    """Join the words whose timestamp falls in [start, end)."""
    return " ".join(w for w, t in zip(words, times) if start <= t < end)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="S1")
    parser.add_argument("--experiment", type=str, default="perceived_speech")
    parser.add_argument("--chunk", type=float, default=20.0,
                        help="chunk size in seconds")
    parser.add_argument("--results-dir", type=str, default=config.RESULT_DIR)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = os.path.join(args.results_dir, args.subject, args.experiment)
    npz_files = sorted(glob.glob(os.path.join(run_dir, "*.npz")))
    if not npz_files:
        raise SystemExit("No result .npz files found under %s" % run_dir)

    out_dir = args.out_dir or run_dir
    os.makedirs(out_dir, exist_ok=True)

    # load every run's prediction, keyed by run name, grouped by story
    runs = {}            # run_name -> (words, times)
    stories = {}         # story -> reference (words, times)
    run_story = {}       # run_name -> story
    for npz_path in npz_files:
        run = os.path.splitext(os.path.basename(npz_path))[0]
        story = run.split("_")[0]
        data = np.load(npz_path, allow_pickle=True)
        runs[run] = (data["words"], data["times"])
        run_story[run] = story
        if story not in stories:
            stories[story] = load_reference(args.experiment, story)

    md_path = os.path.join(out_dir, "comparison.md")
    csv_path = os.path.join(out_dir, "comparison.csv")
    csv_rows = []

    with open(md_path, "w") as md:
        md.write("# Decoded vs. actual story\n\n")
        md.write("Subject `%s`, experiment `%s`, %g-second chunks.\n\n"
                 % (args.subject, args.experiment, args.chunk))

        for story, (ref_words, ref_times) in stories.items():
            story_runs = [r for r in runs if run_story[r] == story]
            # span the chunks across the union of reference + prediction times
            all_times = [ref_times] + [runs[r][1] for r in story_runs]
            t_min = min(float(t.min()) for t in all_times)
            t_max = max(float(t.max()) for t in all_times)
            start = int(t_min // args.chunk * args.chunk)

            md.write("## Story: `%s`\n\n" % story)
            edge = start
            while edge < t_max:
                lo, hi = edge, edge + args.chunk
                actual = chunk_text(ref_words, ref_times, lo, hi)
                md.write("### %d-%ds\n\n" % (lo, hi))
                md.write("**actual:** %s\n\n" % (actual or "_(silence)_"))
                for run in story_runs:
                    w, t = runs[run]
                    pred = chunk_text(w, t, lo, hi)
                    md.write("- **%s:** %s\n" % (run, pred or "_(none)_"))
                    csv_rows.append({
                        "story": story, "chunk_start": lo, "chunk_end": hi,
                        "source": run, "text": pred,
                    })
                csv_rows.append({
                    "story": story, "chunk_start": lo, "chunk_end": hi,
                    "source": "ACTUAL", "text": actual,
                })
                md.write("\n")
                edge += args.chunk

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["story", "chunk_start", "chunk_end", "source", "text"])
        writer.writeheader()
        # keep ACTUAL row first within each chunk for readability
        csv_rows.sort(key=lambda r: (r["story"], r["chunk_start"],
                                     r["source"] != "ACTUAL", r["source"]))
        writer.writerows(csv_rows)

    print("Compared %d run(s) across %d story(ies)." % (len(runs), len(stories)))
    print("Wrote %s" % md_path)
    print("Wrote %s" % csv_path)
