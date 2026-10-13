#!/usr/bin/env python3
"""Cut a patient's continuous band-power recordings into one .mat file per story, lined up with the
story TextGrids.

For patient initials P and task T (e.g. SE and emptyTask) this reads
  preprocessing/story_lists/P_T_stories.toml         which run each story was played in
  data/data_train/train_stimulus/<story>.TextGrid    the story's span in recording time (its xmin and xmax)
  data/original_data/P_T_<run>.mat                   the raw recording: sampling rate and length (data/ is
                                                     also searched)
  data/P_Processed/FIR/P_T_<run>_<band>_all_data.mat the continuous band power (default band HG), where
                                                     all_data{1,1} is a samples x channels double matrix
and writes, to data/data_train/train_response/P/ unless --out-dir is given,
  <story>.mat               same layout as the run file: a MATLAB 7.3 file whose all_data{1,1} is the
                            story's samples x channels double matrix
  P_T_neural_stats.txt      what every file covers, in recording time, samples and audio time
  P_T_neural_index.json     the same, machine-readable, with a checksum of every file's data

Alignment. save_data_by_freq.m makes the band power with a causal FIR (filter(), not filtfilt(), and the
line that would trim the delay is commented out), so the stored signal lags the raw recording by the
filter's group delay. fir1 of order N = floor(0.4 * fs) has N + 1 symmetric taps, so that delay is exactly
N / 2 samples: 400 samples = 0.2 s at 2 kHz. This script removes it: sample k of <story>.mat is the band
power for recording time start + k / fs, where start is the TextGrid's xmin rounded to a sample, taken
from run sample round(start * fs) + delay + k. So TextGrid and labels/ times can be used as they are with
these files. (data_viz.ipynb adds FIR_DELAY_S to word times because it reads the run files; that must not
be done with these files. It also uses floor((N - 1) / 2) = 399 samples; the 0.5 ms difference is below
anything the word timings resolve. Pass --delay-samples 399 to match it exactly.)

  - a story that ends within `delay` of the end of its recording is cut short by up to that much, because
    the band power for its last moments was never computed; the stats file says by how much
  - a story with no TextGrid (it was outside the recording) is skipped and listed
  - existing files are only replaced if this patient's index lists them, or with --overwrite

Usage, from the repo root with the repo's .venv:
  python preprocessing/slice_neural_by_story.py SE
  python preprocessing/slice_neural_by_story.py SE --runs 20              # only the stories in run 20
Check the result with preprocessing/validate_neural_by_story.py.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import re
import sys
import textwrap
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_story_textgrids import REPO_DIR, Paths, clock, load_story_list, recording_info   # noqa: E402

USERBLOCK = 512              # MATLAB 7.3 files start with a 512-byte user block holding a 128-byte header
INDEX_VERSION = 1


# ---- command line ----------------------------------------------------------------------------------

def add_arguments(ap):
    """Options shared with validate_neural_by_story.py."""
    ap.add_argument("patient", help="patient initials, as in the file names (e.g. SE)")
    ap.add_argument("--task", default="emptyTask", help="task name in the file names (default: emptyTask)")
    ap.add_argument("--band", default="HG", help="band in the run file names, <patient>_<task>_<run>_<band>_all_data.mat "
                                                  "(default: HG)")
    ap.add_argument("--story-list", help="default: preprocessing/story_lists/<patient>_<task>_stories.toml")
    ap.add_argument("--data-dir", default=os.path.join(REPO_DIR, "data"), help="default: data/")
    ap.add_argument("--textgrid-dir", default=os.path.join(REPO_DIR, "data", "data_train", "train_stimulus"),
                    help="where the story TextGrids are (default: data/data_train/train_stimulus/)")
    ap.add_argument("--out-dir", help="where the story .mat files go (default: data/data_train/train_response/<patient>/, "
                                      "or train_response_<band>/<patient>/ for a band other than HG)")
    ap.add_argument("--delay-samples", type=int, help="samples to shift by instead of the FIR group delay N/2")


class Layout:
    """Where one patient's files are."""

    def __init__(self, args):
        self.patient, self.task, self.band = args.patient, args.task, args.band
        self.prefix = f"{args.patient}_{args.task}"
        self.data_dir = os.path.abspath(args.data_dir)
        self.textgrid_dir = os.path.abspath(args.textgrid_dir)
        folder = "train_response" if args.band == "HG" else f"train_response_{args.band}"
        self.out_dir = os.path.abspath(args.out_dir or os.path.join(REPO_DIR, "data", "data_train", folder, args.patient))
        self.paths = Paths(args.patient, args.task, self.data_dir, self.textgrid_dir,
                           args.story_list and os.path.abspath(args.story_list))
        self.story_list = self.paths.story_list
        self.stats = os.path.join(self.out_dir, f"{self.prefix}_neural_stats.txt")
        self.index = os.path.join(self.out_dir, f"{self.prefix}_neural_index.json")

    def raw(self, run):
        return self.paths.raw(run)

    def band_file(self, run):
        return os.path.join(self.data_dir, f"{self.patient}_Processed", "FIR", f"{self.prefix}_{run}_{self.band}_all_data.mat")

    def textgrid(self, name):
        return os.path.join(self.textgrid_dir, name + ".TextGrid")

    def story_file(self, name):
        return os.path.join(self.out_dir, name + ".mat")


# ---- inputs ----------------------------------------------------------------------------------------

def read_textgrid(path):
    """(xmin, xmax, words) of a long-format TextGrid: its span, and the [(start, end, text)] of the tier
    named "word" (the last tier if none is), leaving out "sp" and empty intervals."""
    with open(path, encoding="latin-1") as f:
        text = f.read()
    lines = text.split("\n")
    head = (re.fullmatch(r"xmin = (\S+) ?", lines[3]) if len(lines) > 4 else None,
            re.fullmatch(r"xmax = (\S+) ?", lines[4]) if len(lines) > 4 else None)
    if not lines[0].startswith('File type = "ooTextFile"') or not all(head):
        raise SystemExit(f"{path}: not a long-format TextGrid (no xmin/xmax on lines 4 and 5)")
    tiers = re.split(r"\n +item \[\d+\]:\n", text)[1:]
    word_tiers = [t for t in tiers if re.search(r'name = "word" ?\n', t)] or tiers[-1:]
    if not word_tiers:
        raise SystemExit(f"{path}: no tiers")
    intervals = re.findall(r'xmin = (\S+) ?\n +xmax = (\S+) ?\n +text = "(.*)" ?(?:\n|$)', word_tiers[0])
    words = [(float(a), float(b), w) for a, b, w in intervals if w not in ("", "sp")]
    return float(head[0].group(1)), float(head[1].group(1)), words


def fir_delay(fs):
    """(delay in samples, filter order) of the causal fir1 filter in save_data_by_freq.m, or (None, order)
    if the order is odd and the delay is not a whole number of samples."""
    order = math.floor(0.4 * fs)
    return (order // 2 if order % 2 == 0 else None), order


def open_band_file(path):
    """(HDF5 file, dataset) of a run's continuous band power; the dataset is (n_channels, n_samples)."""
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    f = h5py.File(path, "r")
    if "all_data" not in f or f["all_data"].shape != (1, 1) or f["all_data"].dtype != h5py.ref_dtype:
        raise SystemExit(f"{path}: expected all_data to be a 1x1 cell (the continuous format)")
    ds = f[f["all_data"][0, 0]]
    if ds.ndim != 2 or ds.dtype != np.float64:
        raise SystemExit(f"{path}: expected all_data{{1,1}} to be a 2-D double matrix, got {ds.shape} {ds.dtype}")
    return f, ds


# ---- output ----------------------------------------------------------------------------------------

def matlab_string_attr(obj, name, text):
    """A string attribute the way MATLAB writes MATLAB_class etc.: fixed-length ASCII of exactly len(text)
    bytes, typed null-terminated but stored without the terminator. Written with the same memory type as
    the file type, so HDF5 copies the bytes as they are instead of truncating them to fit a terminator."""
    tid = h5py.h5t.C_S1.copy()
    tid.set_size(len(text))
    tid.set_strpad(h5py.h5t.STR_NULLTERM)
    attr = h5py.h5a.create(obj.id, name.encode(), tid, h5py.h5s.create(h5py.h5s.SCALAR))
    attr.write(np.array(text.encode(), dtype=f"S{len(text)}"), mtype=tid)


def write_mat(path, data, chunks, compression_opts):
    """Write `data`, a (n_channels, n_samples) float64 array, as a MATLAB 7.3 file whose all_data{1,1} is
    the samples x channels matrix, laid out like the run files save_data_by_freq.m writes."""
    chunks = None if chunks is None else tuple(min(c, n) for c, n in zip(chunks, data.shape))
    with h5py.File(path, "w", userblock_size=USERBLOCK) as f:
        refs = f.create_group("#refs#")
        empty = refs.create_dataset("a", data=np.zeros(2, dtype=np.uint64))
        matlab_string_attr(empty, "MATLAB_class", "canonical empty")
        empty.attrs.create("MATLAB_empty", np.uint8(1))
        if compression_opts is None:
            matrix = refs.create_dataset("b", data=data, chunks=chunks)
        else:
            matrix = refs.create_dataset("b", data=data, chunks=chunks, compression="gzip", compression_opts=compression_opts)
        matlab_string_attr(matrix, "H5PATH", "/#refs#/b")
        matlab_string_attr(matrix, "MATLAB_class", "double")
        cell = f.create_dataset("all_data", shape=(1, 1), dtype=h5py.ref_dtype)
        cell[0, 0] = matrix.ref
        matlab_string_attr(cell, "MATLAB_class", "cell")
    header = (f"MATLAB 7.3 MAT-file, Platform: {sys.platform}, Created on: {time.ctime()} HDF5 schema 1.00 ."
              .encode().ljust(116) + bytes(8) + b"\x00\x02IM")          # version 0x0200, little-endian
    with open(path, "r+b") as f:
        f.write(header)


def read_back(path):
    """The (n_channels, n_samples) array in a story file, read the way data_viz.ipynb reads run files."""
    with h5py.File(path, "r") as f:
        return np.asarray(f[f["all_data"][0, 0]])


# ---- stats -----------------------------------------------------------------------------------------

def uncovered(spans, total):
    """Parts of [0, total) not inside any (start, end) span."""
    gaps, t = [], 0.0
    for a, b in sorted(spans):
        if a > t + 1e-9:
            gaps.append((t, a))
        t = max(t, b)
    if total > t + 1e-9:
        gaps.append((t, total))
    return gaps


def write_stats(layout, runs, offsets, run_info, entries, skipped, unknown, delay_note):
    rel = lambda path: os.path.relpath(path, REPO_DIR)
    wrap = lambda text, first, rest: textwrap.fill(text, width=110, initial_indent=first, subsequent_indent=rest,
                                                  break_on_hyphens=False, break_long_words=False)
    L = [f"STORY NEURAL DATA: {layout.prefix}, band {layout.band}",
         f"Made {datetime.date.today()} by preprocessing/slice_neural_by_story.py from {rel(layout.story_list)},",
         f"the TextGrids in {rel(layout.textgrid_dir)}/ and {rel(layout.band_file('<run>'))}.",
         "",
         "WHAT EACH FILE IS",
         f"  <story>.mat has the same layout as the run files: all_data{{1,1}} is a samples x channels double matrix",
         f"  ({layout.band} power, z-scored over the whole run by save_data_by_freq.m; the channels are the run file's",
         "  columns, in the same order). Sample k (0-based; row k+1 in MATLAB) is the band power for",
         "      recording time  start + k / fs",
         "  where start is the TextGrid's xmin rounded to a sample. Times in the TextGrids and in labels/ can be",
         "  used with these files directly; no delay has to be added.",
         "",
         "TIME BASES",
         "  recording time  seconds from the start of that run's ECoG recording: the TextGrid and labels/ times.",
         "  run sample      0-based index into the run file's all_data{1,1} (row index - 1 in MATLAB). Recording",
         "                  time t is run sample round(t * fs) + delay, because the band power was made with a",
         "                  causal FIR filter whose group delay was not trimmed (see below).",
         "  audio time      position in the run's audio file, as on the story list: recording time + offset.",
         "",
         "FIR DELAY",
         wrap(delay_note, "  ", "  "),
         "",
         f"  {'run':<4} {'fs (Hz)':>8} {'delay':>14}  {'recording length':<34}  {'offset (s)':>13}  band-power file",
         ]
    for run, info in run_info.items():
        length = f"{info['n_samples'] / info['fs']:.3f} s ({info['n_samples']:,} samples)"
        L.append(f"  {run:<4} {info['fs']:>8g} {info['delay_samples']:>6} samples  {length:<34}  "
                 f"{offsets.get(run, float('nan')):>+13.6f}  {rel(layout.band_file(run))}")
    L.append("  The band-power file and the raw recording of every run have the same number of samples.")

    head = ("#", "file", "run", "recording time (s)", "run samples (0-based)", "length", "samples", "words")
    rows = [(str(k), e["file"], e["run"], f"{e['start_time']:.6f}-{e['end_time']:.6f}",
             f"{e['source_first_sample']}-{e['source_end_sample']}", clock(e["n_samples"] / e["fs"]),
             f"{e['n_samples']:,}", str(e["n_words"])) for k, e in enumerate(entries, 1)]
    widths = [max(len(r[i]) for r in rows + [head]) for i in range(len(head))]
    row_text = lambda r: "  " + "  ".join([r[0].rjust(widths[0])] + [r[i].ljust(widths[i]) for i in range(1, 5)]
                                          + [r[i].rjust(widths[i]) for i in range(5, 8)])
    L += ["", "FILES", row_text(head)] + [row_text(r) for r in rows]

    for k, e in enumerate(entries, 1):
        off = offsets.get(e["run"])
        audio = (f"   audio time      {clock(e['start_time'] + off)} - {clock(e['end_time'] + off)}" if off is not None
                 else "   audio time      (no offset in the story list)")
        L += ["", f"{k}. {e['file']}",
              f"   run             {e['run']}",
              f"   TextGrid        {rel(e['textgrid'])}: xmin {e['xmin']:.6f}, xmax {e['xmax']:.6f}, {e['n_words']} words "
              f"({e['first_word_time']:.6f} - {e['last_word_end']:.6f} s)",
              f"   recording time  {e['start_time']:.6f} - {e['end_time']:.6f} s  (sample k is time {e['start_time']:.6f} + k/{e['fs']:g})",
              f"   run samples     {e['source_first_sample']:,} - {e['source_end_sample']:,}  (0-based, end excluded; "
              f"MATLAB rows {e['source_first_sample'] + 1:,}:{e['source_end_sample']:,} of all_data{{1,1}})",
              audio,
              f"   size            {e['n_samples']:,} samples x {e['n_channels']} channels = {e['n_samples'] / e['fs']:.3f} s "
              f"({clock(e['n_samples'] / e['fs'])}); {e['bytes'] / 1e6:.0f} MB on disk",
              f"   values          mean {e['mean']:+.4f}, sd {e['std']:.4f}, min {e['min']:+.2f}, max {e['max']:+.2f}, "
              f"NaN {e['nan']}   (z-score units of the whole run)",
              f"   md5 of data     {e['md5']}"]
        for i, note in enumerate(e["notes"]):
            L.append(wrap(note, f"   {'notes' if i == 0 else '':<15} - ", " " * 21))

    L += ["", "NOT IN ANY FILE"]
    for s in skipped:
        L.append(wrap(f"{s['name']} (run {s['run']}): no TextGrid at {rel(layout.textgrid(s['name']))}, so no neural file. "
                      "(A story entirely outside its recording has no TextGrid.)", "  - ", "    "))
    for name in unknown:
        L.append(wrap(f"{name}.TextGrid is in {rel(layout.textgrid_dir)}/ but not in the story list, so its run is unknown "
                      "and it was not sliced. Add it to the story list (or rename it there) and run again.", "  - ", "    "))
    for run, info in run_info.items():
        spans = [(e["start_time"], e["end_time"]) for e in entries if e["run"] == run]
        gaps = uncovered(spans, info["n_samples"] / info["fs"])
        covered = sum(b - a for a, b in spans)
        text = (f"Run {run}: {covered:.1f} of {info['n_samples'] / info['fs']:.1f} s are in a story file. Not in any: "
                + "; ".join(f"{a:.3f}-{b:.3f} s ({b - a:.1f} s)" for a, b in gaps) + ".")
        L.append(wrap(text, "  - ", "    "))
    L.append("")
    with open(layout.stats, "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")


# ---- main ------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Cut continuous band-power recordings into one .mat file per story TextGrid.")
    add_arguments(ap)
    ap.add_argument("--runs", help="only stories from these runs, comma-separated (default: all)")
    ap.add_argument("--stories", help="only these stories, comma-separated names (default: all)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing files even if the index doesn't list them")
    args = ap.parse_args()
    layout = Layout(args)

    runs, all_stories = load_story_list(layout.story_list)
    order = {s["name"]: i for i, s in enumerate(all_stories)}
    offsets = {run: float(cfg["offset"]) for run, cfg in runs.items() if "offset" in cfg}
    wanted_runs = set(args.runs.split(",")) if args.runs else None
    wanted = set(args.stories.split(",")) if args.stories else None
    if wanted and (missing := wanted - set(order)):
        raise SystemExit(f"not in the story list: {sorted(missing)}")
    stories = [s for s in all_stories
               if (wanted_runs is None or s["run"] in wanted_runs) and (wanted is None or s["name"] in wanted)]
    if not stories:
        raise SystemExit("no stories selected")

    todo, skipped = [], []
    for s in stories:
        if os.path.exists(layout.textgrid(s["name"])):
            todo.append(s)
        else:
            skipped.append(s)
    if not todo:
        raise SystemExit(f"none of the selected stories has a TextGrid in {layout.textgrid_dir}")
    unknown = sorted(f[:-len(".TextGrid")] for f in os.listdir(layout.textgrid_dir)
                     if f.endswith(".TextGrid") and f[:-len(".TextGrid")] not in order)

    # the runs: sampling rate, length, delay, and the band-power file (checked to be the raw recording's length)
    run_info, band_files = {}, {}
    for run in dict.fromkeys(s["run"] for s in todo):
        fs, n_raw = recording_info(layout.paths, run)
        f, ds = open_band_file(layout.band_file(run))
        if ds.shape[1] != n_raw:
            raise SystemExit(f"run {run}: {layout.band_file(run)} has {ds.shape[1]} samples but the raw recording has {n_raw}")
        delay, fir_order = fir_delay(fs)
        if args.delay_samples is not None:
            delay = args.delay_samples
        elif delay is None:
            raise SystemExit(f"run {run}: fs = {fs:g} gives FIR order {fir_order}, whose group delay {fir_order / 2} is not a "
                             "whole number of samples; pass --delay-samples")
        run_info[run] = dict(fs=fs, n_samples=n_raw, n_channels=ds.shape[0], delay_samples=delay, fir_order=fir_order,
                             raw_file=layout.raw(run), band_file=layout.band_file(run),
                             chunks=ds.chunks, compression_opts=ds.compression_opts if ds.compression == "gzip" else None)
        band_files[run] = (f, ds)
    delays = sorted({info["delay_samples"] for info in run_info.values()})
    orders = sorted({info["fir_order"] for info in run_info.values()})
    seconds = sorted({f"{info['delay_samples'] / info['fs']:.4f} s at {info['fs']:g} Hz" for info in run_info.values()})
    if args.delay_samples is not None:
        delay_note = (f"The stories were shifted by {args.delay_samples} samples ({', '.join(seconds)}; --delay-samples) "
                      f"instead of the FIR group delay of order/2 = {sorted(o // 2 for o in orders)} samples.")
    else:
        delay_note = (f"save_data_by_freq.m filters with fir1 of order floor(0.4 * fs) = {orders}, applied causally with "
                      f"filter(); a symmetric FIR of that order delays everything by exactly order/2 = {delays} samples "
                      f"({', '.join(seconds)}), and save_data_by_freq.m does not trim it. Every story below was therefore "
                      "taken from run sample round(start * fs) + delay onwards, so that its first sample is the band power "
                      "for recording time start. data_viz.ipynb uses floor((order - 1) / 2) (399 samples at 2 kHz) for the "
                      "same delay; the half-millisecond difference is below the precision of the word timings.")

    # which files would be replaced
    os.makedirs(layout.out_dir, exist_ok=True)
    previous = {}
    if os.path.exists(layout.index):
        with open(layout.index) as f:
            previous = {e["file"]: e for e in json.load(f).get("stories", [])}
    blocked = [s["name"] + ".mat" for s in todo
               if os.path.exists(layout.story_file(s["name"])) and s["name"] + ".mat" not in previous]
    if blocked and not args.overwrite:
        raise SystemExit(f"{layout.out_dir} already has these files, and {os.path.basename(layout.index)} doesn't list them "
                         "as made for this patient:\n  " + "\n  ".join(blocked)
                         + "\nUse --out-dir to write somewhere else, or --overwrite to replace them.")

    entries = []
    for s in todo:
        run, name = s["run"], s["name"]
        info, (_, ds) = run_info[run], band_files[run]
        fs, delay, n_run = info["fs"], info["delay_samples"], info["n_samples"]
        xmin, xmax, words = read_textgrid(layout.textgrid(name))
        if not 0 <= xmin < xmax <= n_run / fs + 1e-6:
            raise SystemExit(f"{name}: TextGrid span {xmin}-{xmax} s is outside the run {run} recording (0-{n_run / fs:.3f} s)")
        if words and (words[0][0] < xmin - 1e-6 or words[-1][1] > xmax + 1e-6):
            raise SystemExit(f"{name}: words outside the TextGrid's own span")
        first, last = int(round(xmin * fs)), int(round(xmax * fs))          # run samples in recording time
        source_first, source_end = first + delay, min(last + delay, n_run)  # after the delay, clipped to the run
        notes = list(s.get("notes", []))
        if last + delay > n_run:
            lost = (last + delay - n_run) / fs
            notes.insert(0, f"Cut short by {lost:.4f} s: the story runs to within {lost:.4f} s of the end of the recording, and "
                            "the band power for the last `delay` samples of a recording was never computed (causal filter). "
                            f"The file ends at recording time {(source_end - delay) / fs:.6f} s instead of {last / fs:.6f} s.")
        if source_end - source_first < 1:
            raise SystemExit(f"{name}: nothing left after the delay shift")
        if words and words[-1][1] * fs + delay > n_run:
            notes.append("The last word ends after the file does.")

        t0 = time.time()
        data = np.asarray(ds[:, source_first:source_end])                 # (n_channels, n_samples)
        path = layout.story_file(name)
        write_mat(path, data, info["chunks"], info["compression_opts"])
        back = read_back(path)
        if back.shape != data.shape or not np.array_equal(back, data):
            raise SystemExit(f"{name}: the file read back differs from the slice that was written")
        del back
        entries.append(dict(
            file=name + ".mat", path=path, run=run, textgrid=layout.textgrid(name), xmin=xmin, xmax=xmax, n_words=len(words),
            first_word_time=words[0][0] if words else float("nan"), last_word_end=words[-1][1] if words else float("nan"),
            fs=fs, delay_samples=delay, start_time=first / fs, end_time=(source_end - delay) / fs,
            source_first_sample=source_first, source_end_sample=source_end, n_samples=int(data.shape[1]),
            n_channels=int(data.shape[0]), bytes=os.path.getsize(path), mean=float(data.mean()), std=float(data.std()),
            min=float(data.min()), max=float(data.max()), nan=int(np.isnan(data).sum()),
            md5=hashlib.md5(np.ascontiguousarray(data)).hexdigest(), notes=notes))
        print(f"{name + '.mat':<40} run {run:<3} recording {first / fs:11.6f}-{(source_end - delay) / fs:11.6f} s  "
              f"run samples {source_first:>9,}-{source_end:<9,} {data.shape[1]:>9,} x {data.shape[0]}  ({time.time() - t0:.0f} s)")
        del data
    for f, _ in band_files.values():
        f.close()

    # keep index entries of stories that were not selected this time but still exist
    kept = [e for name, e in previous.items() if name not in {e["file"] for e in entries}
            and os.path.exists(os.path.join(layout.out_dir, name)) and name[:-4] in order]
    all_entries = sorted(entries + kept, key=lambda e: order[e["file"][:-4]])
    index = dict(version=INDEX_VERSION, patient=layout.patient, task=layout.task, band=layout.band,
                 made=datetime.datetime.now().isoformat(timespec="seconds"), story_list=layout.story_list,
                 textgrid_dir=layout.textgrid_dir,
                 sample_time="recording time of sample k (0-based) = start_time + k / fs; source run sample = source_first_sample + k",
                 runs={run: {k: v for k, v in info.items() if k != "chunks"} | {"chunks": list(info["chunks"]) if info["chunks"] else None}
                       for run, info in run_info.items()},
                 stories=[{k: v for k, v in e.items()} for e in all_entries])
    with open(layout.index, "w") as f:
        json.dump(index, f, indent=1)
    write_stats(layout, runs, offsets, run_info, all_entries, skipped, unknown, delay_note)

    total = sum(e["bytes"] for e in entries)
    print(f"wrote {len(entries)} story files ({total / 1e9:.2f} GB), {os.path.basename(layout.stats)} and "
          f"{os.path.basename(layout.index)} to {layout.out_dir}")
    for s in skipped:
        print(f"skipped {s['name']!r} (run {s['run']}): no TextGrid")
    for name in unknown:
        print(f"WARNING: {name}.TextGrid is not in the story list, so it was not sliced")


if __name__ == "__main__":
    main()
