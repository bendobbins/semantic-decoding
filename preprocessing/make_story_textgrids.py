#!/usr/bin/env python3
"""Make one Praat TextGrid per story from a patient's labels/ files and story list.

For patient initials P and task T (e.g. SE and emptyTask) this reads
  preprocessing/story_lists/P_T_stories.toml      each run's offset and each story's parts, in audio time
                                                  (SE_emptyTask_stories.toml documents the format)
  data/labels/P_T_<run>_story_words.txt           start<TAB>end<TAB>word, in recording time
  data/original_data/P_T_<run>.mat                raw BCI2000 recording: sampling rate and length (data/ is
                                                  also searched)
  data/P_Processed/FIR/P_T_<run>_HG_all_data.mat  continuous HG data: its length is checked if it exists
and writes, to data/data_train/train_stimulus/ unless --out-dir is given,
  <story name>.TextGrid                           Praat long text format, tiers "phone" then "word", laid
                                                  out line for line like the fMRI train_stimulus TextGrids
  P_T_textgrid_stats.txt                          each file's length, recording / sample / audio range and
                                                  contents, and everything left out

Recording time is seconds from the start of a run's ECoG recording: the labels/ times, and every time in
the TextGrids (they are not shifted to start at 0). Audio time is the position in data/P_T_<run>_in.wav,
which is what the story list uses. audio time = recording time + the run's offset.

  - A story's words are the labels/ words whose midpoint is inside it, copied unchanged. Every gap
    between words is "sp".
  - A story edge that falls inside a word moves to the middle of the silence next to that word.
  - A story that runs past either end of its recording is clipped; one entirely outside it gets no
    TextGrid. Both are reported in the stats file.
  - Phones are approximate: g2p_en (CMUdict, plus its guesser for other words), with each word's time
    split evenly across its phones.

An existing TextGrid is only replaced if this patient's stats file lists it, or with --overwrite.

Usage, from the repo root with the repo's .venv:
  python preprocessing/make_story_textgrids.py SE
  python preprocessing/make_story_textgrids.py XY --out-dir data/data_train_XY/train_stimulus
Check the result with preprocessing/validate_story_textgrids.py.
"""
import argparse
import collections
import datetime
import difflib
import math
import os
import re
import textwrap
import tomllib

import h5py
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILENCE = "sp"
ARPABET = re.compile(r"(?:AA|AE|AH|AO|AW|AY|EH|ER|EY|IH|IY|OW|OY|UH|UW)[012]"
                     r"|B|CH|D|DH|F|G|HH|JH|K|L|M|N|NG|P|R|S|SH|T|TH|V|W|Y|Z|ZH")
LIST_TIME = re.compile(r"(\d+):([0-5]\d(?:\.\d+)?)")


# ---- where things are ----------------------------------------------------------------------------

class Paths:
    """One patient's inputs and outputs."""

    def __init__(self, patient, task, data_dir, out_dir, story_list=None):
        self.patient, self.prefix = patient, f"{patient}_{task}"
        self.data_dir, self.out_dir = data_dir, out_dir
        self.story_list = story_list or os.path.join(REPO_DIR, "preprocessing", "story_lists",
                                                     f"{self.prefix}_stories.toml")
        self.stats = os.path.join(out_dir, f"{self.prefix}_textgrid_stats.txt")

    def labels(self, run):
        return os.path.join(self.data_dir, "labels", f"{self.prefix}_{run}_story_words.txt")

    def raw(self, run):
        """The raw BCI2000 recording: data/original_data/ first, then data/ itself."""
        for folder in (os.path.join(self.data_dir, "original_data"), self.data_dir):
            path = os.path.join(folder, f"{self.prefix}_{run}.mat")
            if os.path.exists(path):
                return path
        return os.path.join(self.data_dir, "original_data", f"{self.prefix}_{run}.mat")

    def hg(self, run):
        return os.path.join(self.data_dir, f"{self.patient}_Processed", "FIR", f"{self.prefix}_{run}_HG_all_data.mat")

    def audio(self, run):
        return os.path.join(self.data_dir, f"{self.prefix}_{run}_in.wav")


def add_path_arguments(ap):
    """Command-line options shared with validate_story_textgrids.py."""
    ap.add_argument("patient", help="patient initials, as in the file names (e.g. SE)")
    ap.add_argument("--task", default="emptyTask", help="task name in the file names (default: emptyTask)")
    ap.add_argument("--story-list", help="default: preprocessing/story_lists/<patient>_<task>_stories.toml")
    ap.add_argument("--data-dir", default=os.path.join(REPO_DIR, "data"), help="default: data/")
    ap.add_argument("--out-dir", default=os.path.join(REPO_DIR, "data", "data_train", "train_stimulus"),
                    help="where the TextGrids and stats file go (default: data/data_train/train_stimulus/)")


def paths_from(args):
    return Paths(args.patient, args.task, os.path.abspath(args.data_dir), os.path.abspath(args.out_dir),
                 args.story_list and os.path.abspath(args.story_list))


# ---- inputs --------------------------------------------------------------------------------------

def read_words(path, labels=True):
    """[(start, end, word)] from a start<TAB>end<TAB>word file. For labels/ files, also checks that every
    word can go in a TextGrid: in order, not overlapping, and not empty, "sp" or containing a quote."""
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    words = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            try:
                start, end, word = float(fields[0]), float(fields[1]), fields[2].strip()
            except (ValueError, IndexError):
                raise SystemExit(f"{path}, line {n}: expected start<TAB>end<TAB>word, got {line!r}")
            if labels and not (start < end and word and word.lower() != SILENCE and '"' not in word):
                raise SystemExit(f"{path}, line {n}: can't use {line!r} as a word")
            words.append((start, end, word))
    if labels and not all(a[1] <= b[0] for a, b in zip(words, words[1:])):
        raise SystemExit(f"{path}: words are out of order or overlap")
    return words


def recording_info(paths, run):
    """(sampling rate, number of samples) of a run's ECoG recording."""
    if not os.path.exists(paths.raw(run)):
        raise SystemExit(f"missing {paths.raw(run)}, needed for run {run}'s sampling rate and length")
    with h5py.File(paths.raw(run), "r") as f:
        fs = float(np.array(f["parameters"]["SamplingRate"]["NumericValue"]).flatten()[0])
        n_samples = f["signal"].shape[1]                 # stored transposed: (n_channels, n_samples)
    if os.path.exists(paths.hg(run)):
        with h5py.File(paths.hg(run), "r") as f:
            n_hg = f[f["all_data"][0, 0]].shape[1]        # (n_channels, n_samples) too
        if n_hg != n_samples:
            raise SystemExit(f"run {run}: the HG data has {n_hg} samples but the raw recording has {n_samples}")
    return fs, n_samples


def list_seconds(value, where):
    """Audio time from the story list: "m:ss" or "m:ss.ss" (minutes can pass 59), or a number of seconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    m = LIST_TIME.fullmatch(str(value))
    if not m:
        raise SystemExit(f"{where}: can't read the time {value!r} (write m:ss, m:ss.ss or seconds)")
    return int(m.group(1)) * 60 + float(m.group(2))


def clock(seconds):
    """m:ss.ss, with minutes not wrapped at 60 (like the story list)."""
    cs = int(round(seconds * 100))
    sign, cs = ("-", -cs) if cs < 0 else ("", cs)
    return f"{sign}{cs // 6000}:{cs % 6000 / 100:05.2f}"


def load_story_list(path):
    """({run: {"offset", "audio_transcript"}}, stories) from the TOML story list, with every story's parts
    as (label, start, end) in audio seconds. Stops with a message on anything inconsistent."""
    if not os.path.exists(path):
        raise SystemExit(f"no story list at {path}")
    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as err:
            raise SystemExit(f"{path}: {err}")
    file = os.path.basename(path)
    if set(data) - {"runs", "stories"}:
        raise SystemExit(f"{file}: unknown sections {sorted(set(data) - {'runs', 'stories'})}")
    runs = {str(run): cfg for run, cfg in data.get("runs", {}).items()}
    for run, cfg in runs.items():
        if not isinstance(cfg, dict) or set(cfg) - {"offset", "audio_transcript"} or not set(cfg):
            raise SystemExit(f"{file}: [runs.{run}] needs an offset, an audio_transcript or both, and nothing else")

    stories = []
    for n, entry in enumerate(data.get("stories", []), 1):
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip() or name != name.strip() or re.search(r'[/\\:"\x00-\x1f]', name):
            raise SystemExit(f"{file}, story {n}: bad name {name!r} (it is the file name, so no / \\ : or \")")
        where = f"{file}, {name!r}"
        if set(entry) - {"name", "run", "parts", "notes"}:
            raise SystemExit(f"{where}: unknown keys {sorted(set(entry) - {'name', 'run', 'parts', 'notes'})}")
        run = str(entry.get("run", ""))
        if run not in runs:
            raise SystemExit(f"{where}: run {run!r} has no [runs.{run}] section")
        parts = entry.get("parts")
        if not isinstance(parts, list) or not parts or not all(isinstance(p, list) and len(p) == 3 for p in parts):
            raise SystemExit(f"{where}: parts must be a list of [label, start, end]")
        parts = [(str(label), list_seconds(a, where), list_seconds(b, where)) for label, a, b in parts]
        for label, a, b in parts:
            if not a < b:
                raise SystemExit(f"{where}: part {label!r} doesn't end after it starts")
        for p, q in zip(parts, parts[1:]):
            if p[2] != q[1]:
                raise SystemExit(f"{where}: {p[0]!r} ends at {clock(p[2])} but {q[0]!r} starts at {clock(q[1])}. A story's "
                                 "parts can't have gaps or overlaps; to leave something out, make separate stories.")
        notes = entry.get("notes", [])
        if not isinstance(notes, list) or not all(isinstance(x, str) for x in notes):
            raise SystemExit(f"{where}: notes must be a list of strings")
        stories.append(dict(name=name, run=run, parts=parts, notes=notes))

    if not stories:
        raise SystemExit(f"{file}: no [[stories]]")
    repeated = sorted(name for name, count in collections.Counter(s["name"] for s in stories).items() if count > 1)
    if repeated:
        raise SystemExit(f"{file}: story names used more than once: {repeated}")
    for run in runs:
        in_run = sorted((s for s in stories if s["run"] == run), key=lambda s: s["parts"][0][1])
        for s, t in zip(in_run, in_run[1:]):
            if t["parts"][0][1] < s["parts"][-1][2]:
                raise SystemExit(f"{file}: {s['name']!r} and {t['name']!r} overlap in run {run}")
    return runs, stories


# ---- offsets -------------------------------------------------------------------------------------

def measure_offset(labels, transcript):
    """(offset, count): the audio - recording time difference shared by the most labels/ words and their
    matches in an audio-time transcript, and how many words share it."""
    norm = lambda w: re.sub(r"[^a-z0-9']", "", w.lower())
    matcher = difflib.SequenceMatcher(None, [norm(w) for *_, w in transcript], [norm(w) for *_, w in labels],
                                      autojunk=False)
    offsets = collections.Counter()
    for i, j, n in matcher.get_matching_blocks():
        for k in range(n):
            offsets[round(transcript[i + k][0] - labels[j + k][0], 6)] += 1
    return offsets.most_common(1)[0] if offsets else (None, 0)


def run_offsets(paths, runs, labels):
    """{run: offset} and {run: where the offset came from} for every run in `labels`."""
    offsets, sources = {}, {}
    for run in labels:
        given = runs[run].get("offset")
        if "audio_transcript" not in runs[run]:
            offsets[run], sources[run] = float(given), "given in the story list"
            continue
        transcript = os.path.join(paths.data_dir, runs[run]["audio_transcript"])
        measured, count = measure_offset(labels[run], read_words(transcript, labels=False))
        name = os.path.relpath(transcript, REPO_DIR)
        if count < 0.95 * len(labels[run]):
            raise SystemExit(f"run {run}: only {count} of {len(labels[run])} labels/ words line up with {name} at one offset")
        if given is not None and abs(measured - float(given)) > 1e-6:
            raise SystemExit(f"run {run}: the story list gives offset {given}, but {name} puts it at {measured}")
        offsets[run] = measured if given is None else float(given)
        sources[run] = (f"measured from {name}, where {count} of the {len(labels[run])} labels/ words are exactly this much later"
                        + (" (matches the story list)" if given is not None else ""))
    return offsets, sources


# ---- placing stories -----------------------------------------------------------------------------

def snap(words, t):
    """(t, None, None), or if t falls inside a word: (the middle of the silence after the word if the word's
    midpoint is before t, otherwise the middle of the silence before it; the word; "after" or "before")."""
    for i, (s, e, word) in enumerate(words):
        if s < t < e:
            if (s + e) / 2 < t:
                return (round((e + words[i + 1][0]) / 2, 6) if i + 1 < len(words) else e), word, "after"
            return (round((words[i - 1][1] + s) / 2, 6) if i else s), word, "before"
    return t, None, None


def place_story(story, words, offset, fs, n_samples):
    """The story in recording time: its span after clipping it to the recording and moving its edges out
    of words, its parts, and notes about both. None if it is entirely outside the recording."""
    run, recording_end = story["run"], n_samples / fs
    parts = [(label, round(a - offset, 6), round(b - offset, 6)) for label, a, b in story["parts"]]
    listed_start, listed_end = parts[0][1], parts[-1][2]
    if listed_end <= 0 or listed_start >= recording_end:
        return None

    start, end = max(listed_start, 0.0), min(listed_end, recording_end)
    notes, clipped = [], []
    if start > listed_start:
        clipped.append((listed_start, start))
        notes.append(f"The story list starts it at {clock(listed_start + offset)}, but the run {run} recording starts at "
                     f"audio {clock(offset)}, so the first {start - listed_start:.2f} s have no neural data and are not in "
                     "the file.")
    if end < listed_end:
        clipped.append((end, listed_end))
        notes.append(f"The story list ends it at {clock(listed_end + offset)}, but the run {run} recording ends at audio "
                     f"{clock(recording_end + offset)}, so the last {listed_end - end:.2f} s have no neural data and are "
                     "not in the file.")
    moved, word, side = snap(words, start)
    if word is not None:
        notes.append(f"Starts at {clock(moved + offset)} instead of {clock(start + offset)}, in the silence {side} "
                     f"\"{word}\", so that word isn't cut in half.")
        start = moved
    moved, word, side = snap(words, end)
    if word is not None:
        notes.append(f"Ends at {clock(moved + offset)} instead of {clock(end + offset)}, in the silence {side} "
                     f"\"{word}\", so that word isn't cut in half.")
        end = moved
    if not 0 <= start < end <= recording_end:
        raise SystemExit(f"{story['name']!r}: moving its edges out of words takes it outside the run {run} recording")

    placed = []
    for label, a, b in parts:
        a, b = min(max(a, start), end), min(max(b, start), end)
        if a < b:
            placed.append([label, a, b])
    placed[0][1], placed[-1][2] = start, end
    return dict(name=story["name"], run=run, start=start, end=end, clipped=clipped,
                parts=[tuple(p) for p in placed], notes=notes + story["notes"])


# ---- TextGrid ------------------------------------------------------------------------------------

def make_tiers(words, start, end, phones_of):
    """Phone and word intervals covering [start, end]; every gap between words is "sp"."""
    phone_tier, word_tier = [], []
    t = start
    for s, e, word in words:
        assert s >= t, f"word {word!r} at {s} starts before {t}"
        if s > t:
            phone_tier.append((t, s, SILENCE))
            word_tier.append((t, s, SILENCE))
        phones = phones_of(word)
        cuts = [s] + [s + (e - s) * k / len(phones) for k in range(1, len(phones))] + [e]
        phone_tier += [(cuts[k], cuts[k + 1], phone) for k, phone in enumerate(phones)]
        word_tier.append((s, e, word))
        t = e
    if end > t:
        phone_tier.append((t, end, SILENCE))
        word_tier.append((t, end, SILENCE))
    return phone_tier, word_tier


def number(x):
    """Like Python 2's str(float), which is how most numbers in the reference TextGrids are written."""
    text = "%.12g" % x
    assert "e" not in text and "n" not in text, x
    return text


def textgrid_text(xmin, xmax, tiers, num=number):
    """Praat long text format, line for line like the fMRI train_stimulus TextGrids (including their
    trailing spaces). `tiers` is [(name, [(xmin, xmax, text), ...]), ...]."""
    out = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
           f"xmin = {num(xmin)} ", f"xmax = {num(xmax)} ", "tiers? <exists> ",
           f"size = {len(tiers)} ", "item []: "]
    for i, (name, intervals) in enumerate(tiers, 1):
        out += [f"    item [{i}]:", '        class = "IntervalTier" ', f'        name = "{name}" ',
                f"        xmin = {num(xmin)} ", f"        xmax = {num(xmax)} ",
                f"        intervals: size = {len(intervals)} "]
        for j, (a, b, text) in enumerate(intervals, 1):
            out += [f"        intervals [{j}]:", f"            xmin = {num(a)} ",
                    f"            xmax = {num(b)} ", f'            text = "{text}" ']
    return "\n".join(out) + "\n"


# ---- stats file ----------------------------------------------------------------------------------

def phrases(words, gap=1.0):
    """Consecutive words grouped into phrases, split at silences longer than `gap` seconds."""
    groups = []
    for w in words:
        if groups and w[0] - groups[-1][-1][1] <= gap:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def fir_delay(fs):
    """The FIR group delay that data_viz.ipynb adds to word times when it indexes the HG data."""
    return math.floor((math.floor(0.4 * fs) - 1) / 2) / fs


def listed_in_stats(path):
    """TextGrid file names in the FILES table of an existing stats file."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(re.findall(r"^ +\d+  (.+?\.TextGrid)  ", f.read(), flags=re.M))


def write_stats(paths, recordings, offsets, sources, placed, skipped, leftovers):
    rel = lambda path: os.path.relpath(path, REPO_DIR)
    wrap = lambda text, first, rest: textwrap.fill(text, width=110, initial_indent=first, subsequent_indent=rest,
                                                  break_on_hyphens=False, break_long_words=False)
    rates = sorted({fs for fs, _ in recordings.values()})
    L = [f"STORY TEXTGRIDS: {paths.prefix}",
         f"Made {datetime.date.today()} by preprocessing/make_story_textgrids.py from {rel(paths.story_list)}",
         f"and {rel(paths.labels('<run>'))}.",
         "",
         "TIME BASES",
         "  recording time  seconds from the start of that run's ECoG recording. Every time inside the TextGrids",
         "                  is recording time (the same times as the labels/ files).",
         f"  audio time      position in {rel(paths.audio('<run>'))}: the times in the story list.",
         "                  audio time = recording time + offset (table below).",
         "  samples         0-based sample index = recording time x sampling rate. The continuous HG data",
         f"                  ({rel(paths.hg('<run>'))}) has the same samples.",
         "                  No filter-delay shift is included; data_viz.ipynb adds FIR_DELAY_S ("
         + ", ".join(f"{fir_delay(fs):g} s at {fs:g} Hz" for fs in rates) + ")",
         "                  to word times when it indexes the HG data.",
         ""]
    lengths = {run: f"{n / fs:.3f} s ({n:,} samples at {fs:g} Hz)" for run, (fs, n) in recordings.items()}
    width = max(map(len, lengths.values()))
    L.append(f"  {'run':<4} {'offset (s)':>13}  {'recording length':<{width}}  audio covered by the recording")
    for run, (fs, n) in recordings.items():
        L.append(f"  {run:<4} {offsets[run]:>+13.6f}  {lengths[run]:<{width}}  "
                 f"{clock(offsets[run])} - {clock(offsets[run] + n / fs)}")
    L += [wrap(f"Run {run} offset: {sources[run]}.", "  ", "    ") for run in recordings]

    head = ("#", "file", "run", "recording time (s)", "audio time", "length", "words")
    rows = [(str(k), s["name"] + ".TextGrid", s["run"], f"{s['start']:.6f}-{s['end']:.6f}",
             f"{clock(s['start'] + offsets[s['run']])}-{clock(s['end'] + offsets[s['run']])}",
             clock(s["end"] - s["start"]), str(len(s["words"]))) for k, s in enumerate(placed, 1)]
    widths = [max(len(row[i]) for row in rows + [head]) for i in range(len(head))]
    row_text = lambda row: "  " + "  ".join([row[0].rjust(widths[0])]
                                            + [row[i].ljust(widths[i]) for i in range(1, 5)]
                                            + [row[5].rjust(widths[5]), row[6].rjust(widths[6])])
    L += ["", "FILES", row_text(head)] + [row_text(row) for row in rows]

    label_width = max(len(p[0]) for s in placed for p in s["parts"])
    for k, s in enumerate(placed, 1):
        run, offset = s["run"], offsets[s["run"]]
        fs, n = recordings[run]
        words = s["words"]
        silences = [x for x in s["word_tier"] if x[2] == SILENCE]
        L += ["", f"{k}. {s['name']}.TextGrid",
              f"   run             {run}",
              f"   length          {s['end'] - s['start']:.3f} s ({clock(s['end'] - s['start'])})",
              f"   recording time  {s['start']:.6f} - {s['end']:.6f} s   (the run {run} recording is {n / fs:.3f} s long)",
              f"   samples         {round(s['start'] * fs):,} - {round(s['end'] * fs):,}   (of {n:,}, at {fs:g} Hz)",
              f"   audio time      {clock(s['start'] + offset)} - {clock(s['end'] + offset)}   "
              f"({s['start'] + offset:.3f} - {s['end'] + offset:.3f} s in {rel(paths.audio(run))})"]
        for i, (label, a, b, count) in enumerate(s["parts"]):
            L.append(f"   {'contents' if i == 0 else '':<15} {label:<{label_width}}  audio {clock(a + offset)}-"
                     f"{clock(b + offset)}   recording {a:.3f}-{b:.3f} s   {count:>5} words")
        L += [f"   words           {len(words)}  (speech {sum(e0 - s0 for s0, e0, _ in words):.1f} s; \"{SILENCE}\" "
              f"{sum(e0 - s0 for s0, e0, _ in silences):.1f} s in {len(silences)} intervals)",
              f"   phones          {sum(1 for x in s['phone_tier'] if x[2] != SILENCE)}",
              f"   first word      {words[0][2]} at {words[0][0]:.6f} s (audio {clock(words[0][0] + offset)})",
              f"   last word       {words[-1][2]}, ending at {words[-1][1]:.6f} s (audio {clock(words[-1][1] + offset)})"]
        for i, note in enumerate(s["notes"]):
            L.append(wrap(note, f"   {'notes' if i == 0 else '':<15} - ", " " * 21))

    left_out = []
    for s in skipped:
        run, offset = s["run"], offsets[s["run"]]
        fs, n = recordings[run]
        left_out.append(f"{s['name']} (run {run}, audio {clock(s['parts'][0][1])}-{clock(s['parts'][-1][2])}): entirely "
                        f"outside the run {run} recording, which covers audio {clock(offset)}-{clock(offset + n / fs)}, "
                        "so there is no neural data and no TextGrid.")
    for s in placed:
        offset = offsets[s["run"]]
        for a, b in s["clipped"]:
            left_out.append(f"{b - a:.2f} s of {s['name']} (audio {clock(a + offset)}-{clock(b + offset)}): outside the "
                            f"run {s['run']} recording, so there is no neural data.")
    for run in recordings:
        in_run = sorted((s for s in placed if s["run"] == run), key=lambda s: s["start"])
        for a, b in zip(in_run, in_run[1:]):
            if b["start"] > a["end"]:
                left_out.append(f"Run {run}, recording {a['end']:.6f} - {b['start']:.6f} s (audio "
                                f"{clock(a['end'] + offsets[run])} - {clock(b['start'] + offsets[run])}, "
                                f"{b['start'] - a['end']:.2f} s), between {a['name']} and {b['name']}: in the recording "
                                "but not in any story on the list.")
        if leftovers[run]:
            groups = phrases(leftovers[run])
            listed = " and ".join(f"\"{' '.join(w for *_, w in g)}\" (audio {clock(g[0][0] + offsets[run])}-"
                                  f"{clock(g[-1][1] + offsets[run])})" for g in groups[:10])
            more = f" and {len(groups) - 10} more phrases" if len(groups) > 10 else ""
            left_out.append(f"Run {run}: {len(leftovers[run])} labels/ words outside every story: {listed}{more}.")
    L += ["", "NOT IN ANY TEXTGRID"] + ([wrap(x, "  - ", "    ") for x in left_out] or ["  (nothing)"])

    L += ["", "FORMAT",
          wrap("Praat long text format with two IntervalTiers: item [1] \"phone\", item [2] \"word\". Same header, "
               "indentation, trailing spaces and number style as the fMRI train_stimulus TextGrids.", "  - ", "    "),
          wrap("Each file covers only its story: xmin/xmax are the story's start and end in recording time, not 0.",
               "  - ", "    "),
          wrap("word tier: the labels/ words and times, unchanged. Every gap between words, and before the first "
               "or after the last word, is \"sp\".", "  - ", "    "),
          wrap("phone tier: approximate. ARPAbet phones from g2p_en (CMUdict, plus its guesser for words not in "
               "CMUdict), with each word's time split evenly across its phones; \"sp\" wherever the word tier has "
               "\"sp\".", "  - ", "    ")]
    with open(paths.stats, "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")


# ---- main ----------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Make one TextGrid per story from a patient's labels/ files and story list.")
    add_path_arguments(ap)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing TextGrids even if this patient's stats file doesn't list them")
    args = ap.parse_args()
    paths = paths_from(args)

    runs, stories = load_story_list(paths.story_list)
    run_order = list(dict.fromkeys(s["run"] for s in stories))
    labels = {run: read_words(paths.labels(run)) for run in run_order}
    offsets, sources = run_offsets(paths, runs, labels)
    recordings = {run: recording_info(paths, run) for run in run_order}

    placed, skipped = [], []
    for story in stories:
        spot = place_story(story, labels[story["run"]], offsets[story["run"]], *recordings[story["run"]])
        if spot is None:
            skipped.append(story)
            print(f"no TextGrid for {story['name']!r}: it is entirely outside the run {story['run']} recording")
        else:
            placed.append(spot)
    if not placed:
        raise SystemExit("no story is inside its recording")

    used = {run: set() for run in run_order}
    for run in run_order:
        in_run = sorted((s for s in placed if s["run"] == run), key=lambda s: s["start"])
        for a, b in zip(in_run, in_run[1:]):
            if b["start"] < a["end"]:
                raise SystemExit(f"{a['name']!r} and {b['name']!r} overlap once their edges are moved out of words")
    for story in placed:
        run = story["run"]
        idx = [i for i, (s, e, _) in enumerate(labels[run]) if story["start"] <= (s + e) / 2 < story["end"]]
        if not idx:
            raise SystemExit(f"{story['name']!r} has no labels/ words in it (run {run}, recording "
                             f"{story['start']:.3f}-{story['end']:.3f} s)")
        used[run] |= set(idx)
        story["words"] = [labels[run][i] for i in idx]
        assert all(story["start"] <= s and e <= story["end"] for s, e, _ in story["words"])
        story["parts"] = [(label, a, b, sum(a <= (s + e) / 2 < b for s, e, _ in story["words"]))
                          for label, a, b in story["parts"]]
    leftovers = {run: [w for i, w in enumerate(labels[run]) if i not in used[run]] for run in run_order}

    os.makedirs(paths.out_dir, exist_ok=True)
    targets = [s["name"] + ".TextGrid" for s in placed]
    previous = listed_in_stats(paths.stats)
    blocked = [t for t in targets if os.path.exists(os.path.join(paths.out_dir, t)) and t not in previous]
    if blocked and not args.overwrite:
        raise SystemExit(f"{paths.out_dir} already has these files, and {os.path.basename(paths.stats)} doesn't list "
                         "them as made for this patient:\n  " + "\n  ".join(blocked)
                         + "\nUse --out-dir to write somewhere else, or --overwrite to replace them.")

    from g2p_en import G2p                 # slow to import, so only once everything else checks out
    g2p, cache = G2p(), {}

    def phones_of(word):
        if word not in cache:
            cache[word] = [p for p in g2p(word) if ARPABET.fullmatch(p)]
            if not cache[word]:
                raise SystemExit(f"g2p_en gives no phones for {word!r}")
        return cache[word]

    for story in placed:
        story["phone_tier"], story["word_tier"] = make_tiers(story["words"], story["start"], story["end"], phones_of)
        with open(os.path.join(paths.out_dir, story["name"] + ".TextGrid"), "w", newline="\n") as f:
            f.write(textgrid_text(story["start"], story["end"],
                                  [("phone", story["phone_tier"]), ("word", story["word_tier"])]))
        print(f"{story['name'] + '.TextGrid':<45} run {story['run']:<3} "
              f"{story['start']:12.6f}-{story['end']:12.6f} s  {len(story['words']):5} words")
    write_stats(paths, recordings, offsets, sources, placed, skipped, leftovers)

    print(f"wrote {len(placed)} TextGrids and {os.path.basename(paths.stats)} to {paths.out_dir}")
    print(f"labels/ words in no TextGrid: { {run: len(words) for run, words in leftovers.items()} }")
    stale = sorted(t for t in previous - set(targets) if os.path.exists(os.path.join(paths.out_dir, t)))
    if stale:
        print("no longer made from the story list, left in place:\n  " + "\n  ".join(stale))


if __name__ == "__main__":
    main()
