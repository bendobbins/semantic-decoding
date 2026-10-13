#!/usr/bin/env python3
"""Check the story .mat files made by slice_neural_by_story.py against what they were made from.

For every story in the story list that has a TextGrid, this re-derives from the inputs (TextGrid xmin/xmax,
the raw recording's sampling rate, the FIR order) which run samples the story file should hold, and checks:
  - the file is a MATLAB 7.3 file laid out like the run files: 512-byte user block with the MATLAB header,
    all_data a 1x1 cell whose one element is a double matrix, MATLAB_class attributes, same chunking and
    compression as the run file
  - the matrix is exactly the run file's samples over that span (read separately and compared value for value),
    with the right number of channels and no NaN
  - the index (<patient>_<task>_neural_index.json) and stats file agree with the file
It also reports story files in the folder that are not in the story list.

Usage, from the repo root with the repo's .venv:
  python preprocessing/validate_neural_by_story.py SE
"""
import argparse
import hashlib
import json
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_story_textgrids import load_story_list, recording_info                              # noqa: E402
from slice_neural_by_story import USERBLOCK, Layout, add_arguments, fir_delay, open_band_file, read_textgrid   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Check the story .mat files made by slice_neural_by_story.py.")
    add_arguments(ap)
    args = ap.parse_args()
    layout = Layout(args)
    failures = []

    def check(ok, message):
        if not ok:
            failures.append(message)
        return ok

    runs, stories = load_story_list(layout.story_list)
    with_grid = [s for s in stories if os.path.exists(layout.textgrid(s["name"]))]
    index = {}
    if check(os.path.exists(layout.index), f"no index at {layout.index}"):
        with open(layout.index) as f:
            index = {e["file"]: e for e in json.load(f).get("stories", [])}
    stats = ""
    if check(os.path.exists(layout.stats), f"no stats file at {layout.stats}"):
        with open(layout.stats) as f:
            stats = f.read()
    extra = sorted(f for f in os.listdir(layout.out_dir) if f.endswith(".mat") and f[:-4] not in {s["name"] for s in stories})
    if extra:
        print(f"note: .mat files in {layout.out_dir} that are not in the story list: {extra}")

    run_cache = {}
    checked = 0
    for s in with_grid:
        name, run = s["name"], s["run"]
        path = layout.story_file(name)
        if not check(os.path.exists(path), f"{name}: no {path}"):
            continue
        if run not in run_cache:
            fs, n_raw = recording_info(layout.paths, run)
            f, ds = open_band_file(layout.band_file(run))
            check(ds.shape[1] == n_raw, f"run {run}: band file has {ds.shape[1]} samples, raw recording {n_raw}")
            delay, order = fir_delay(fs)
            if args.delay_samples is not None:
                delay = args.delay_samples
            check(delay is not None, f"run {run}: FIR order {order} has a half-sample delay; pass --delay-samples")
            run_cache[run] = (fs, f, ds, delay)
        fs, _, src, delay = run_cache[run]
        if delay is None:
            continue

        # what the file should hold
        xmin, xmax, words = read_textgrid(layout.textgrid(name))
        first, last = int(round(xmin * fs)), int(round(xmax * fs))
        a, b = first + delay, min(last + delay, src.shape[1])
        check(0 <= a < b, f"{name}: expected span {a}-{b} is empty")

        # the file's layout
        with open(path, "rb") as f:
            head = f.read(128)
        check(head.startswith(b"MATLAB 7.3 MAT-file") and head[124:128] == b"\x00\x02IM", f"{name}: MATLAB 7.3 header")
        with h5py.File(path, "r") as f:
            check(f.userblock_size == USERBLOCK and set(f.keys()) == {"#refs#", "all_data"}, f"{name}: file structure")
            cell = f["all_data"]
            check(cell.shape == (1, 1) and cell.dtype == h5py.ref_dtype and cell.attrs.get("MATLAB_class") == b"cell",
                  f"{name}: all_data is not a 1x1 cell")
            ds = f[cell[0, 0]]
            check(ds.dtype == np.float64 and ds.ndim == 2 and ds.attrs.get("MATLAB_class") == b"double",
                  f"{name}: all_data{{1,1}} is not a 2-D double matrix")
            check(ds.chunks == (tuple(min(c, n) for c, n in zip(src.chunks, ds.shape)) if src.chunks else None)
                  and ds.compression == src.compression and ds.compression_opts == src.compression_opts,
                  f"{name}: chunking or compression differs from the run file")
            empty = f["#refs#/a"]
            check(empty.attrs.get("MATLAB_class") == b"canonical empty" and empty.attrs.get("MATLAB_empty") == 1,
                  f"{name}: #refs#/a is not MATLAB's canonical empty")
            data = np.asarray(ds)

        # the data
        expected = np.asarray(src[:, a:b])
        check(data.shape == expected.shape, f"{name}: shape {data.shape}, expected {expected.shape} (channels x samples)")
        same = data.shape == expected.shape and np.array_equal(data, expected)
        check(same, f"{name}: data differs from run samples {a}-{b}")
        check(not np.isnan(data).any(), f"{name}: NaN values")
        md5 = hashlib.md5(np.ascontiguousarray(data)).hexdigest()

        # the index and stats
        e = index.get(name + ".mat")
        if check(e is not None, f"{name}: not in the index"):
            check(e["run"] == run and e["source_first_sample"] == a and e["source_end_sample"] == b
                  and e["n_samples"] == b - a and e["n_channels"] == data.shape[0]
                  and abs(e["start_time"] - first / fs) < 1e-9 and abs(e["end_time"] - (b - delay) / fs) < 1e-9
                  and e["delay_samples"] == delay and e["fs"] == fs and e["md5"] == md5,
                  f"{name}: index entry disagrees with the file")
        row = re.search(rf"^ +\d+ +{re.escape(name + '.mat')} +(\S+) +([\d.]+)-([\d.]+) +(\d+)-(\d+) +[\d:.]+ +([\d,]+) +(\d+)$",
                        stats, flags=re.M)
        if check(row is not None, f"{name}: no row in the stats table"):
            check(row.group(1) == run and int(row.group(4)) == a and int(row.group(5)) == b
                  and int(row.group(6).replace(",", "")) == b - a and int(row.group(7)) == len(words)
                  and abs(float(row.group(2)) - first / fs) < 1e-6 and abs(float(row.group(3)) - (b - delay) / fs) < 1e-6,
                  f"{name}: stats row disagrees with the file")
        checked += 1
        print(f"{name + '.mat':<40} run {run:<3} run samples {a:>9,}-{b:<9,} {data.shape[1]:>9,} x {data.shape[0]}  "
              f"{'matches the run file' if same else 'DIFFERS'}")
        del data, expected
    for _, f, _, _ in run_cache.values():
        f.close()

    missing = [s["name"] for s in stories if s not in with_grid]
    print(f"checked {checked} story files; stories without a TextGrid (so no file expected): {missing or 'none'}")
    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
