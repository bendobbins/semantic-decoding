#!/usr/bin/env python3
"""Check the TextGrids made by make_story_textgrids.py against everything they were made from.

Works out what every file should contain from labels/, the recordings, the story list and the audio
transcripts it names, rather than trusting the generator (it only borrows the story list reader and the
TextGrid writer), and checks that:
  - the writer reproduces the fMRI reference TextGrids byte for byte, and the new files only use line
    layouts found in them
  - every labels/ word sits exactly one offset away from a word in the run's audio transcript
  - each file spans what the story list and the clipping / snapping rules say, holds exactly the labels/
    words inside that span (times unchanged, none cut), lies inside its recording, has phones that
    tile every word, and reads back through the repo's load_textgrids + make_word_ds as those words
  - no labels/ word is in two files, and the stats file agrees with the TextGrids

Usage, from the repo root with the repo's .venv:
  python preprocessing/validate_story_textgrids.py SE
"""
import argparse
import collections
import glob
import os
import re
import sys
import warnings

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_story_textgrids as gen                                            # noqa: E402

sys.path.insert(0, os.path.join(gen.REPO_DIR, "decoding"))
warnings.filterwarnings("ignore", category=SyntaxWarning)                    # old regex strings in textgrid.py
from utils_ridge.textgrid import TextGrid                                     # noqa: E402
from utils_ridge.dsutils import make_word_ds, DEFAULT_BAD_WORDS               # noqa: E402
from utils_ridge.stimulus_utils import load_textgrids, load_simulated_trfiles  # noqa: E402

REFERENCE_DIR = os.path.join(os.path.dirname(gen.REPO_DIR), "semantic-decoding", "data", "data_train", "train_stimulus")


def template(line):
    """A line with its quoted strings and numbers blanked out."""
    return re.sub(r"-?\d+(?:\.\d+)?", "N", re.sub(r'"[^"]*"', "S", line))


def parse_long(text):
    """Strict line-by-line parse of a long-format TextGrid, keeping numbers as written."""
    lines = text.split("\n")

    def field(i, pattern):
        m = re.fullmatch(pattern, lines[i]) if i < len(lines) else None
        if not m:
            raise ValueError(f"line {i + 1}: {lines[i] if i < len(lines) else 'missing'!r}")
        return m.group(1)

    xmin, xmax = field(3, r"xmin = (\S+) "), field(4, r"xmax = (\S+) ")
    size = int(field(6, r"size = (\d+) "))
    i, tiers = 8, []
    for k in range(size):
        field(i, rf"    item \[({k + 1})\]:")
        name = field(i + 2, r'        name = "(.*)" ')
        txmin, txmax = field(i + 3, r"        xmin = (\S+) "), field(i + 4, r"        xmax = (\S+) ")
        n = int(field(i + 5, r"        intervals: size = (\d+) "))
        i += 6
        intervals = []
        for j in range(n):
            field(i, rf"        intervals \[({j + 1})\]:")
            intervals.append((field(i + 1, r"            xmin = (\S+) "), field(i + 2, r"            xmax = (\S+) "),
                              field(i + 3, r'            text = "(.*)" ')))
            i += 4
        tiers.append((name, txmin, txmax, intervals))
    if lines[:3] != ['File type = "ooTextFile"', 'Object class = "TextGrid"', ""] or lines[i:] != [""]:
        raise ValueError("header or trailing lines")
    return xmin, xmax, tiers


def expected_edge(words, t, recording_end):
    """Where a story edge at recording time t should end up: clipped to the recording, then moved out of
    any word it falls inside (to the middle of the silence on the side away from the word's midpoint)."""
    t = min(max(t, 0.0), recording_end)
    for i, (s, e, _) in enumerate(words):
        if s < t < e:
            if (s + e) / 2 < t:
                return round((e + words[i + 1][0]) / 2, 6) if i + 1 < len(words) else e
            return round((words[i - 1][1] + s) / 2, 6) if i else s
    return t


def clock_seconds(text):
    minutes, seconds = text.split(":")
    return int(minutes) * 60 + float(seconds)


def main():
    ap = argparse.ArgumentParser(description="Check the TextGrids made by make_story_textgrids.py.")
    gen.add_path_arguments(ap)
    ap.add_argument("--reference-dir", default=REFERENCE_DIR,
                    help="fMRI train_stimulus TextGrids to compare the layout with "
                         "(default: ../semantic-decoding/data/data_train/train_stimulus)")
    args = ap.parse_args()
    paths = gen.paths_from(args)
    runs, stories = gen.load_story_list(paths.story_list)
    failures = []

    def check(ok, message):
        if not ok:
            failures.append(message)
        return ok

    # ---- 1. the reference layout -----------------------------------------------------------------
    ref_templates = set()
    reference = sorted(glob.glob(os.path.join(args.reference_dir, "*.TextGrid")))
    long_format = reproduced = 0
    for path in reference:
        text = open(path, "rb").read().decode("latin-1")
        lines = text.split("\n")
        if not text.startswith("File type") or "\r" in text or len(lines) < 4 or "xmin" not in lines[3]:
            continue                                                  # the chronological and short formats
        long_format += 1
        ref_templates |= {template(line) for line in lines}
        try:
            xmin, xmax, tiers = parse_long(text)
        except ValueError:
            continue
        if all((t[1], t[2]) == (xmin, xmax) for t in tiers):
            reproduced += gen.textgrid_text(xmin, xmax, [(t[0], t[3]) for t in tiers], num=str) == text
    if long_format:
        print(f"layout: the writer reproduces {reproduced}/{long_format} long-format reference TextGrids byte for byte")
        check(reproduced == long_format, "the TextGrid writer doesn't reproduce every reference TextGrid")
    else:
        print(f"layout: no long-format TextGrids in {args.reference_dir}, so the layout isn't compared")

    # ---- 2. inputs, read here rather than through the generator ------------------------------------
    run_order = list(dict.fromkeys(s["run"] for s in stories))
    labels, recording = {}, {}
    for run in run_order:
        with open(paths.labels(run)) as f:
            labels[run] = [(float(a), float(b), w.strip()) for a, b, w in
                           (line.rstrip("\n").split("\t") for line in f if line.strip())]
        with h5py.File(paths.raw(run), "r") as f:
            fs = float(np.array(f["parameters"]["SamplingRate"]["NumericValue"]).flatten()[0])
            recording[run] = (fs, f["signal"].shape[1])
        if os.path.exists(paths.hg(run)):
            with h5py.File(paths.hg(run), "r") as f:
                check(f[f["all_data"][0, 0]].shape[1] == recording[run][1], f"run {run}: HG and raw sample counts differ")
    if not check(os.path.exists(paths.stats), f"no stats file at {paths.stats}"):
        return report(failures)
    with open(paths.stats) as f:
        stats = f.read()

    offsets = {}
    run_rows = re.findall(r"^  (\S+) +([+-]\d+\.\d+)  [\d.]+ s \(([\d,]+) samples at ([\d.]+) Hz\)", stats, flags=re.M)
    for run, offset, n, fs in run_rows:
        offsets[run] = float(offset)
        check(run in recording and int(n.replace(",", "")) == recording[run][1] and float(fs) == recording[run][0],
              f"stats: run {run}'s recording length or sampling rate")
    check(sorted(offsets) == sorted(run_order), f"stats: offsets for runs {sorted(offsets)}, story list uses {sorted(run_order)}")
    for run in run_order:
        if run not in offsets:
            continue
        cfg, offset = runs[run], offsets[run]
        if "offset" in cfg:
            check(abs(float(cfg["offset"]) - offset) < 1e-9, f"run {run}: stats offset {offset}, story list {cfg['offset']}")
        if "audio_transcript" in cfg:
            spots = set()
            with open(os.path.join(paths.data_dir, cfg["audio_transcript"])) as f:
                for line in f:
                    if line.strip():
                        a, b, _ = line.rstrip("\n").split("\t")
                        spots.add((round(float(a), 6), round(float(b), 6)))
            off_by = [x for x in labels[run] if (round(x[0] + offset, 6), round(x[1] + offset, 6)) not in spots]
            print(f"offset: run {run}: {len(labels[run]) - len(off_by)}/{len(labels[run])} labels/ words sit exactly "
                  f"{offset:+.6f} s from a word in {cfg['audio_transcript']}")
            check(not off_by, f"run {run}: {len(off_by)} labels/ words don't line up with {cfg['audio_transcript']}: {off_by[:3]}")

    # ---- 3. which files there should be, and their spans -------------------------------------------
    expected = []
    for story in stories:
        run = story["run"]
        if run not in offsets:
            continue
        fs, n = recording[run]
        start, end = (round(story["parts"][0][1] - offsets[run], 6), round(story["parts"][-1][2] - offsets[run], 6))
        if end > 0 and start < n / fs:
            expected.append((story["name"], run, expected_edge(labels[run], start, n / fs),
                             expected_edge(labels[run], end, n / fs)))
    rows = re.findall(r"^ +(\d+)  (.+?)\.TextGrid +(\S+) +([\d.]+)-([\d.]+) +([\d:.]+)-([\d:.]+) +([\d:.]+) +(\d+)$",
                      stats, flags=re.M)
    check([r[1] for r in rows] == [e[0] for e in expected],
          f"stats lists {[r[1] for r in rows]}, but the story list gives {[e[0] for e in expected]}")

    # ---- 4. every TextGrid -----------------------------------------------------------------------
    used = {run: collections.Counter() for run in run_order}
    facts, word_phones = {}, collections.defaultdict(set)
    for name, run, want_start, want_end in expected:
        path = os.path.join(paths.out_dir, name + ".TextGrid")
        if not check(os.path.exists(path), f"missing {path}"):
            continue
        raw = open(path, "rb").read()
        check(b"\r" not in raw and raw.endswith(b"\n") and not raw.endswith(b"\n\n"), f"{name}: line endings")
        try:
            text = raw.decode("ascii")
            xmin_s, xmax_s, tiers = parse_long(text)
        except (UnicodeDecodeError, ValueError) as err:
            check(False, f"{name}: not a well-formed long-format TextGrid ({err})")
            continue
        if ref_templates:
            extra = {template(line) for line in text.split("\n")} - ref_templates
            check(not extra, f"{name}: line layouts not found in the reference TextGrids: {extra}")
        numbers = re.findall(r"x(?:min|max) = (\S+) $", text, flags=re.M)
        check(all(re.fullmatch(r"\d+(?:\.\d+)?", x) and x == "%.12g" % float(x) for x in numbers), f"{name}: number format")
        check([t[0] for t in tiers] == ["phone", "word"], f"{name}: tiers are {[t[0] for t in tiers]}")
        xmin, xmax = float(xmin_s), float(xmax_s)
        fs, n = recording[run]
        check(abs(xmin - want_start) < 1e-9 and abs(xmax - want_end) < 1e-9,
              f"{name}: spans {xmin}-{xmax}, but the story list and rules give {want_start}-{want_end}")
        check(0 <= xmin < xmax <= n / fs, f"{name}: outside the run {run} recording (0-{n / fs:.3f} s)")

        grid = TextGrid(text)
        check(grid.text_type == "ooTextFile" and grid.size == 2 and (grid.xmin, grid.xmax) == (xmin, xmax),
              f"{name}: the repo's TextGrid reader sees a different header")
        for tier, (tier_name, txmin, txmax, intervals) in zip(grid.tiers, tiers):
            check(tier.nameid == tier_name and tier.simple_transcript == intervals and tier.size == len(intervals),
                  f"{name}/{tier_name}: the repo's TextGrid reader disagrees with a strict parse")
            check((txmin, txmax) == (xmin_s, xmax_s) and intervals and intervals[0][0] == xmin_s
                  and intervals[-1][1] == xmax_s, f"{name}/{tier_name}: tier span")
            check(all(p[1] == q[0] for p, q in zip(intervals, intervals[1:])), f"{name}/{tier_name}: gaps or overlaps")
            check(all(float(a) < float(b) for a, b, _ in intervals), f"{name}/{tier_name}: empty interval")
        phone_intervals, word_intervals = tiers[0][3], tiers[1][3]

        words = [(float(a), float(b), w) for a, b, w in word_intervals if w != gen.SILENCE]
        inside = [i for i, (s, e, _) in enumerate(labels[run]) if xmin <= (s + e) / 2 < xmax]
        check(words == [labels[run][i] for i in inside], f"{name}: words differ from the labels/ words inside it")
        check(all(xmin <= s and e <= xmax for s, e, _ in words), f"{name}: a word is cut by the file's edge")
        check(all(not p[2] == q[2] == gen.SILENCE for p, q in zip(word_intervals, word_intervals[1:])),
              f"{name}: two sp intervals in a row")
        used[run].update(inside)

        k = 0
        for a, b, w in word_intervals:
            group = []
            while k < len(phone_intervals) and (not group or group[-1][1] != b):
                group.append(phone_intervals[k])
                k += 1
            ok = bool(group) and group[0][0] == a and group[-1][1] == b
            if w == gen.SILENCE:
                ok = ok and len(group) == 1 and group[0][2] == gen.SILENCE
            elif ok:
                durations = [float(y) - float(x) for x, y, _ in group]
                ok = all(gen.ARPABET.fullmatch(p) for _, _, p in group) and max(durations) - min(durations) < 1e-6
                word_phones[w].add(tuple(p for _, _, p in group))
            if not check(ok, f"{name}: phones don't line up with {w!r} at {a}"):
                break
        check(k == len(phone_intervals), f"{name}: phone intervals left over after the last word")

        kept = [(float(a), float(b), w) for a, b, w in grid.tiers[1].make_simple_transcript()
                if w.lower().strip("{}").strip() not in DEFAULT_BAD_WORDS]
        check(kept == words, f"{name}: make_word_ds's bad-word filter keeps different intervals")
        if os.path.basename(paths.out_dir) == "train_stimulus":
            grids = load_textgrids([name], os.path.dirname(paths.out_dir))
        else:
            grids = {name: TextGrid(text)}
        sequence = make_word_ds(grids, load_simulated_trfiles({name: int(xmax / 2) + 20}))[name]
        check(list(sequence.data) == [w.lower() for *_, w in words]
              and np.allclose(sequence.data_times, [(s + e) / 2 for s, e, _ in words]), f"{name}: make_word_ds output differs")
        facts[name] = dict(run=run, xmin=xmin, xmax=xmax, words=words,
                           phones=sum(p[2] != gen.SILENCE for p in phone_intervals))
    check(all(len(v) == 1 for v in word_phones.values()), "a word has different phones in different places")

    # ---- 5. labels/ coverage ---------------------------------------------------------------------
    for run in run_order:
        check(all(c == 1 for c in used[run].values()), f"run {run}: a labels/ word is in more than one TextGrid")
        unused = [labels[run][i] for i in range(len(labels[run])) if i not in used[run]]
        print(f"coverage: run {run}: {len(used[run])}/{len(labels[run])} labels/ words are in a TextGrid"
              + (f"; not in any: {' '.join(w for *_, w in unused[:30])}{' ...' if len(unused) > 30 else ''}" if unused else ""))

    # ---- 6. the stats file agrees with the TextGrids ---------------------------------------------
    for _, name, run, a, b, audio_a, audio_b, length, count in rows:
        f = facts.get(name)
        if not f:
            continue
        check(run == f["run"] and abs(float(a) - f["xmin"]) < 1e-6 and abs(float(b) - f["xmax"]) < 1e-6
              and int(count) == len(f["words"]), f"stats: FILES row for {name}")
        check(abs(clock_seconds(audio_a) - (f["xmin"] + offsets[run])) < 0.006
              and abs(clock_seconds(audio_b) - (f["xmax"] + offsets[run])) < 0.006
              and abs(clock_seconds(length) - (f["xmax"] - f["xmin"])) < 0.006, f"stats: audio time or length of {name}")
    for k, (name, run, _, _) in enumerate(expected, 1):
        block = re.search(rf"^{k}\. {re.escape(name)}\.TextGrid\n(.*?)(?:\n\n|\Z)", stats, flags=re.M | re.S)
        f = facts.get(name)
        if not check(block is not None, f"stats: no details for {name}") or not f:
            continue
        body = block.group(1)
        fs, n = recording[run]
        m = re.search(r"samples\s+([\d,]+) - ([\d,]+)\s+\(of ([\d,]+), at ([\d.]+) Hz\)", body)
        check(bool(m) and [int(m.group(i).replace(",", "")) for i in (1, 2, 3)] == [round(f["xmin"] * fs), round(f["xmax"] * fs), n]
              and float(m.group(4)) == fs, f"stats: samples of {name}")
        m = re.search(r"words\s+(\d+) ", body)
        check(bool(m) and int(m.group(1)) == len(f["words"]), f"stats: word count of {name}")
        m = re.search(r"phones\s+(\d+)", body)
        check(bool(m) and int(m.group(1)) == f["phones"], f"stats: phone count of {name}")
        m = re.search(r"first word\s+(\S+) at ([\d.]+) s", body)
        check(bool(m) and (m.group(1), float(m.group(2))) == (f["words"][0][2], f["words"][0][0]), f"stats: first word of {name}")
        m = re.search(r"last word\s+(\S+), ending at ([\d.]+) s", body)
        check(bool(m) and (m.group(1), float(m.group(2))) == (f["words"][-1][2], f["words"][-1][1]), f"stats: last word of {name}")
        parts = re.findall(r"recording ([\d.]+)-([\d.]+) s\s+(\d+) words", body)
        check(bool(parts) and abs(float(parts[0][0]) - f["xmin"]) < 5e-4 and abs(float(parts[-1][1]) - f["xmax"]) < 5e-4
              and sum(int(p[2]) for p in parts) == len(f["words"]), f"stats: parts of {name}")

    print(f"checked {len(facts)} TextGrids: {sum(len(f['words']) for f in facts.values())} words, "
          f"{sum(f['phones'] for f in facts.values())} phones")
    return report(failures)


def report(failures):
    print("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED:\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
