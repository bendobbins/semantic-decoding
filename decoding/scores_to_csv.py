"""Convert decoder score .npz files into CSV files for comparison across runs.

Each score file (produced by evaluate_predictions.py) contains four dicts keyed
by (story, metric):
    window_scores   raw score per evaluation window
    window_zscores  null-normalized z-score per window
    story_scores    raw score per window (same granularity; averaged for story)
    story_zscores    a single story-level z-score (the headline number)

This script walks the scores directory and writes two CSVs:

    summary.csv   one row per run/story/metric with the story-level z-score and
                  the mean raw score. Small and ideal for comparing runs.

    windows.csv   long format, one row per run/story/metric/window with the raw
                  score and z-score. Larger but still CSV-readable.

Usage:
    python scores_to_csv.py [--scores-dir DIR] [--out-dir DIR]

Both default to config.SCORE_DIR (<repo>/scores), so the script works from any
working directory.
"""

import os
import csv
import glob
import argparse
import numpy as np

import config


def parse_path(npz_path, scores_dir):
    """Pull subject / experiment / run name out of <scores>/<subj>/<exp>/<run>.npz"""
    rel = os.path.relpath(npz_path, scores_dir)
    parts = rel.split(os.sep)
    subject = parts[0] if len(parts) > 2 else ""
    experiment = parts[1] if len(parts) > 2 else ""
    run = os.path.splitext(parts[-1])[0]
    return subject, experiment, run


def load_dicts(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {key: d[key].item() for key in d.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=str, default=config.SCORE_DIR)
    parser.add_argument("--out-dir", type=str, default=config.SCORE_DIR)
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.scores_dir, "**", "*.npz"), recursive=True))
    if not npz_files:
        raise SystemExit("No .npz score files found under %s" % args.scores_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    windows_path = os.path.join(args.out_dir, "windows.csv")

    summary_rows = []
    window_rows = []

    for npz_path in npz_files:
        subject, experiment, run = parse_path(npz_path, args.scores_dir)
        data = load_dicts(npz_path)

        window_scores = data.get("window_scores", {})
        window_zscores = data.get("window_zscores", {})
        story_scores = data.get("story_scores", {})
        story_zscores = data.get("story_zscores", {})

        for (story, metric), raw in story_scores.items():
            summary_rows.append({
                "subject": subject,
                "experiment": experiment,
                "run": run,
                "story": story,
                "metric": metric,
                "mean_raw_score": float(np.mean(raw)),
                "story_zscore": float(story_zscores[(story, metric)]),
            })

        for (story, metric), raw in window_scores.items():
            zs = window_zscores.get((story, metric))
            for i, val in enumerate(np.atleast_1d(raw)):
                window_rows.append({
                    "subject": subject,
                    "experiment": experiment,
                    "run": run,
                    "story": story,
                    "metric": metric,
                    "window": i,
                    "raw_score": float(val),
                    "zscore": float(zs[i]) if zs is not None else "",
                })

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject", "experiment", "run", "story", "metric",
            "mean_raw_score", "story_zscore"])
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(windows_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject", "experiment", "run", "story", "metric",
            "window", "raw_score", "zscore"])
        writer.writeheader()
        writer.writerows(window_rows)

    print("Wrote %d summary rows -> %s" % (len(summary_rows), summary_path))
    print("Wrote %d window rows  -> %s" % (len(window_rows), windows_path))
