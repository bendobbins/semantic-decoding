"""Bar charts comparing decoder runs on each language-similarity metric.

Reads the score .npz files written by evaluate_predictions.py and produces one
2x2 figure (BLEU / METEOR / WER / BERT) per subject per test condition,
comparing every run of that subject against the others.

Subjects are auto-discovered and always plotted separately -- see subject_of().
Runs compared within each subject <S> (auto-discovered):
    <S>              16 h of training data, no added training noise
    <S>_small         2 h of training data, no added training noise
    <S>_noise_s0.25   2 h + gaussian training noise, std = 0.25 * voxel std
    <S>_noise_s0.5    2 h + gaussian training noise, std = 0.50 * voxel std
    <S>_noise_s0.7    2 h + gaussian training noise, std = 0.70 * voxel std

A subject missing some runs still plots: absent bars are left empty (the axis
label stays, so a gap reads as "not evaluated yet") and the run is reported as
incomplete on stdout.

Test conditions (one set of figures each):
    wheretheressmoke                 no noise added to the test responses
    wheretheressmoke_noise100_s0.5   0.5 * voxel std gaussian noise added

Two values are plotted for each condition:
    zscore   story-level z-score against the null distribution (the headline
             number; 0 means "no better than a random decode")
    raw      mean raw metric score across evaluation windows, +/- 1 SEM

Note that all four metrics are oriented so that HIGHER IS BETTER: WER is stored
by utils_eval.WER(use_score = True) as 1 - WER, not as the error rate.

Usage:
    python plot_scores.py                    # every subject, both conditions
    python plot_scores.py --subject S1 S3    # only these subjects
    python plot_scores.py --value zscore     # just the z-score figures
    python plot_scores.py --separate         # also write one PNG per metric
    python plot_scores.py --zoom             # raw panels: crop y to data range
    python plot_scores.py --nulls 25         # name the null count in the caption

Pass --nulls only when every subject was evaluated at that count: it is not
recorded in the .npz, so the script cannot verify it, and z-scores from
different null counts are not comparable.
"""

import os
import re
import glob
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

import config

# ----------------------------------------------------------------------------
# palette
#
# The four small-dataset runs form an ordered sweep (training noise 0 -> 0.7),
# so they take an ordinal ramp: one hue, monotone lightness, so the reader sees
# the sweep order in the color. S1 is not a point on that sweep -- it is the
# full-data reference -- so it takes a separate categorical hue.
#
# Validated with the dataviz palette validator against the #fcfcfb surface:
# the 4-step ramp passes the ordinal checks (monotone L, adjacent dL >= 0.06,
# light end 2.06:1, single hue, 3 deg spread), and the reference-vs-ramp
# boundary clears CVD separation (dE 29.7 deutan) and the normal-vision floor
# (dE 33.1). The ramp's light end sits below 3:1 on the surface, which requires
# a relief channel -- that is why every bar carries a visible value label.
# ----------------------------------------------------------------------------

REFERENCE_COLOR = "#008300"          # categorical slot 2 -- the full-data run
RAMP_4 = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]   # the validated 4 steps
RAMP_STEPS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_RULE = "#c3c2b7"

# ----------------------------------------------------------------------------
# what to plot
# ----------------------------------------------------------------------------

METRICS = ["BLEU", "METEOR", "WER", "BERT"]

# evaluate_predictions.py calls utils_eval.windows(..., config.WINDOW) and takes
# that function's default step, so the windows advance one second at a time.
WINDOW_STEP = 1

METRIC_TITLES = {
    "BLEU": "BLEU-1",
    "METEOR": "METEOR",
    "WER": "WER score  (1 − WER)",
    "BERT": "BERTScore",
}

TASK_TITLES = {
    "wheretheressmoke": "clean test data",
    "wheretheressmoke_noise100_s0.5": "test data + 0.5σ voxel noise",
}


def task_title(task):
    return TASK_TITLES.get(task, task)


def subject_of(run):
    """Subject a run directory belongs to: S1, S1_small, S1_noise_s0.25 -> S1.

    Figures are built one subject at a time. Pooling subjects into a single
    chart would interleave the noise sweeps, push the bar count past the four
    ramp steps the palette is validated for, and give two runs at the same
    noise level different colors.
    """
    return run.split("_")[0]


def parse_run(run):
    """Sort key + axis label for a run directory name.

    Returns (noise_level, label) where noise_level is None for the full-data
    reference run so it can be held out of the ordinal ramp.
    """
    m = re.search(r"_noise_s([0-9]*\.?[0-9]+)$", run)
    if m:
        level = float(m.group(1))
        return level, "2 h\n%gσ" % level
    if run.endswith("_small"):
        return 0.0, "2 h\nno noise"
    return None, "16 h\nno noise"


def order_runs(runs):
    """Reference run first, then the noise sweep in increasing order."""
    reference = [r for r in runs if parse_run(r)[0] is None]
    sweep = sorted([r for r in runs if parse_run(r)[0] is not None],
                   key=lambda r: parse_run(r)[0])
    return sorted(reference) + sweep


def run_colors(runs):
    """Reference hue for the full-data run, ordinal ramp across the sweep."""
    sweep = [r for r in runs if parse_run(r)[0] is not None]
    if len(sweep) == 4:
        ramp = RAMP_4
    else:
        # keep the ramp's documented steps; spread them over however many runs
        idx = np.linspace(0, len(RAMP_STEPS) - 1, max(len(sweep), 1))
        ramp = [RAMP_STEPS[int(round(i))] for i in idx]
    colors, step = {}, 0
    for r in runs:
        if parse_run(r)[0] is None:
            colors[r] = REFERENCE_COLOR
        else:
            colors[r] = ramp[step]
            step += 1
    return colors


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------

def load_scores(scores_dir, experiment):
    """{run: {task: {metric: (per_window_scores, story_zscore)}}}"""
    data = {}
    pattern = os.path.join(scores_dir, "*", experiment, "*.npz")
    for path in sorted(glob.glob(pattern)):
        run = os.path.relpath(path, scores_dir).split(os.sep)[0]
        task = os.path.splitext(os.path.basename(path))[0]
        npz = np.load(path, allow_pickle=True)
        story_scores = npz["story_scores"].item()
        story_zscores = npz["story_zscores"].item()
        entry = {}
        for (_story, metric), windows in story_scores.items():
            entry[metric] = (np.asarray(windows, dtype=float),
                             float(story_zscores[(_story, metric)]))
        data.setdefault(run, {})[task] = entry
    return data


def effective_n(n_windows):
    """Independent-sample count behind n overlapping evaluation windows.

    utils_eval.windows() slides a config.WINDOW-second window by WINDOW_STEP
    seconds, so consecutive windows share all but WINDOW_STEP of their content
    and the n windows are worth only n * step / duration independent samples.
    Dividing the SD by sqrt(n) instead understates the SEM by sqrt(20) ~ 4.5x
    here -- confirmed against the SEM of a decimated, non-overlapping subsample.
    """
    return max(n_windows * WINDOW_STEP / float(config.WINDOW), 1.0)


def panel_values(data, runs, task, metric, value):
    """Per-run (height, error) for one panel. Missing runs come back as nan."""
    heights, errors = [], []
    for run in runs:
        entry = data.get(run, {}).get(task, {}).get(metric)
        if entry is None:
            heights.append(np.nan)
            errors.append(0.0)
            continue
        windows, zscore = entry
        if value == "zscore":
            heights.append(zscore)
            errors.append(0.0)          # one z-score per story, nothing to pool
        else:
            heights.append(float(windows.mean()))
            errors.append(float(windows.std(ddof=1)
                                / np.sqrt(effective_n(len(windows)))))
    return np.array(heights), np.array(errors)


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------

def rounded_bar(ax, x, height, width, color, radius_px=4.0):
    """A column with a rounded cap and a square baseline.

    The radius is specified in points on the page and converted separately for
    each axis, so the corner stays circular whatever the data range is.
    """
    fig = ax.figure
    pos = ax.get_position()
    ax_w_px = fig.get_figwidth() * fig.dpi * pos.width
    ax_h_px = fig.get_figheight() * fig.dpi * pos.height
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    scale = fig.dpi / 100.0                      # 4 "css px" at the output dpi
    rx = radius_px * scale * (x1 - x0) / ax_w_px
    ry = radius_px * scale * (y1 - y0) / ax_h_px
    rx = min(rx, width / 2.0)
    ry = min(ry, abs(height))

    left, right, top = x - width / 2.0, x + width / 2.0, height
    verts = [
        (left, 0.0),
        (left, top - ry),
        (left, top), (left + rx, top),           # quadratic corner
        (right - rx, top),
        (right, top), (right, top - ry),         # quadratic corner
        (right, 0.0),
        (left, 0.0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color,
                           edgecolor="none", zorder=3))


def label_decimals(values, value):
    """Fewest decimals that still give every run a distinct label."""
    if value == "zscore":
        return 1
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return 3
    for decimals in range(3, 6):
        labels = ["%.*f" % (decimals, v) for v in finite]
        if len(set(labels)) == len(labels):
            return decimals
    return 5


def draw_panel(ax, data, runs, colors, task, metric, value, zoom):
    heights, errors = panel_values(data, runs, task, metric, value)
    positions = np.arange(len(runs))
    width = 0.55                                  # never fill the slot

    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, linestyle="-")
    ax.xaxis.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_RULE)
        ax.spines[side].set_linewidth(0.8)

    ax.set_xlim(-0.6, len(runs) - 0.4)
    top = np.nanmax(heights + errors)
    if zoom and value == "raw":
        low = np.nanmin(heights - errors)
        pad = (top - low) * 0.35 or abs(top) * 0.05
        ax.set_ylim(low - pad, top + pad)
    else:
        ax.set_ylim(0, top * 1.20)

    for pos, height, err, run in zip(positions, heights, errors, runs):
        if not np.isfinite(height):
            continue
        rounded_bar(ax, pos, height, width, colors[run])
        if err > 0:
            ax.errorbar(pos, height, yerr=err, fmt="none", ecolor=INK_MUTED,
                        elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)

    # every bar is labeled: this is the relief channel for the ramp's light end,
    # and it is the only way the sub-0.01 BERT differences are readable at all
    decimals = label_decimals(heights, value)
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for pos, height, err in zip(positions, heights, errors):
        if not np.isfinite(height):
            continue
        ax.annotate("%.*f" % (decimals, height),
                    xy=(pos, height + err + span * 0.025),
                    ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(positions)
    ax.set_xticklabels([parse_run(r)[1] for r in runs], fontsize=8.5,
                       color=INK_SECONDARY)
    ax.tick_params(axis="both", length=0, labelsize=8, colors=INK_MUTED)
    for tick in ax.get_xticklabels():
        tick.set_color(INK_SECONDARY)
    ax.set_title(METRIC_TITLES.get(metric, metric), fontsize=10.5,
                 color=INK_PRIMARY, pad=8, loc="left")
    ax.set_ylabel("z-score vs. null" if value == "zscore" else "mean score",
                  fontsize=8.5, color=INK_MUTED)


def figure_notes(value, zoom, n_windows, metric=None, nulls=None):
    """Footer notes. `metric` limits them to a single-metric figure.

    `nulls` is the --null count the scores were produced with. It is not
    recorded in the .npz, so it has to be passed in; without it the caption
    stays generic rather than asserting a count that may be wrong.
    """
    if metric is None:
        notes = ["All metrics oriented so higher is better — WER is stored as "
                 "1 − WER, not the error rate."]
    elif metric == "WER":
        notes = ["Higher is better: this is 1 − WER, not the error rate."]
    else:
        notes = ["Higher is better."]
    if value == "raw":
        notes.append("Error bars are ±1 SEM. The %d evaluation windows are "
                     "%g s long at a %g s step, so they overlap heavily; the "
                     "SEM uses the ~%.0f independent windows they are worth, "
                     "not %d." % (n_windows, config.WINDOW, WINDOW_STEP,
                                  effective_n(n_windows), n_windows))
        if zoom:
            notes.append("y-axis is cropped to the data range and does NOT "
                         "start at zero — bar lengths are not proportional.")
    elif nulls:
        notes.append("z-scores are against a %d-sequence null (evaluate_"
                     "predictions.py --null %d); 0 means no better than a "
                     "random decode. The denominator carries ~%.0f%% relative "
                     "error at this count." % (nulls, nulls,
                                               100 / np.sqrt(2 * (nulls - 1))))
    else:
        notes.append("z-scores are against the null distribution; 0 means no "
                     "better than a random decode.")
    return "\n".join(notes)


def make_figure(data, runs, colors, subject, task, value, zoom, n_windows,
                nulls):
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.2))
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.855, bottom=0.135,
                        wspace=0.22, hspace=0.42)

    subtitle = ("Story-level z-score against the null distribution"
                if value == "zscore" else
                "Mean raw score across evaluation windows")
    fig.text(0.075, 0.955, "%s — training-noise sweep, %s"
             % (subject, task_title(task)),
             fontsize=15, color=INK_PRIMARY, ha="left", va="center")
    fig.text(0.075, 0.915, subtitle, fontsize=10, color=INK_SECONDARY,
             ha="left", va="center")

    colors_map = colors
    for ax, metric in zip(axes.ravel(), METRICS):
        draw_panel(ax, data, runs, colors_map, task, metric, value, zoom)

    fig.text(0.075, 0.028, figure_notes(value, zoom, n_windows, nulls=nulls),
             fontsize=7.5, color=INK_MUTED, ha="left", va="bottom",
             linespacing=1.5)
    return fig


def make_single(data, runs, colors, subject, task, metric, value, zoom,
                n_windows, nulls):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.80, bottom=0.24)

    fig.text(0.14, 0.94, "%s — %s, %s" % (METRIC_TITLES.get(metric, metric),
                                          subject, task_title(task)),
             fontsize=12, color=INK_PRIMARY, ha="left", va="center")
    fig.text(0.14, 0.885,
             "Story-level z-score against the null distribution"
             if value == "zscore" else
             "Mean raw score across evaluation windows",
             fontsize=8.5, color=INK_SECONDARY, ha="left", va="center")

    draw_panel(ax, data, runs, colors, task, metric, value, zoom)
    ax.set_title("", loc="left")        # the header above already names it
    fig.text(0.14, 0.03, figure_notes(value, zoom, n_windows, metric, nulls),
             fontsize=6.5, color=INK_MUTED, ha="left", va="bottom",
             linespacing=1.5)
    return fig


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bar charts comparing decoder runs on each metric.")
    parser.add_argument("--scores-dir", type=str, default=config.SCORE_DIR)
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(config.REPO_DIR, "figures"))
    parser.add_argument("--experiment", type=str, default="perceived_speech")
    parser.add_argument("--value", type=str, default="both",
                        choices=["raw", "zscore", "both"],
                        help="raw metric score, null-normalized z-score, or both")
    parser.add_argument("--separate", action="store_true",
                        help="also write one PNG per metric")
    parser.add_argument("--zoom", action="store_true",
                        help="crop the raw y-axis to the data range instead of "
                             "starting at zero (clearly annotated when used)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--subject", nargs="+", type=str, default=[],
                        help="only plot these subjects (default: all found)")
    parser.add_argument("--nulls", type=int, default=None,
                        help="--null count the scores were evaluated with; "
                             "used for the caption only (not stored in the npz)")
    args = parser.parse_args()

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial",
                                       "DejaVu Sans"]

    data = load_scores(args.scores_dir, args.experiment)
    if not data:
        raise SystemExit("No .npz score files found under %s/*/%s"
                         % (args.scores_dir, args.experiment))

    values = ["raw", "zscore"] if args.value == "both" else [args.value]

    subjects = sorted({subject_of(r) for r in data})
    if args.subject:
        missing = [s for s in args.subject if s not in subjects]
        if missing:
            raise SystemExit("No scores for subject(s): %s (found: %s)"
                             % (", ".join(missing), ", ".join(subjects)))
        subjects = [s for s in subjects if s in args.subject]

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for subject in subjects:
        # one subject per figure -- see subject_of()
        sub_data = {r: v for r, v in data.items() if subject_of(r) == subject}
        runs = order_runs(list(sub_data))
        colors = run_colors(runs)
        # tasks and window counts are read per subject rather than once
        # globally, so a subject part-way through evaluation still plots
        tasks = sorted({t for run in sub_data.values() for t in run})

        for task in tasks:
            lengths = {len(e[0]) for r in sub_data.values()
                       for m, e in r.get(task, {}).items()}
            n_windows = max(lengths) if lengths else 0

            for value in values:
                fig = make_figure(sub_data, runs, colors, subject, task, value,
                                  args.zoom, n_windows, args.nulls)
                path = os.path.join(args.out_dir, "%s_%s_%s.png"
                                    % (subject, task, value))
                fig.savefig(path, dpi=args.dpi, facecolor=SURFACE)
                plt.close(fig)
                written.append(path)

                if args.separate:
                    sub = os.path.join(args.out_dir, "individual")
                    os.makedirs(sub, exist_ok=True)
                    for metric in METRICS:
                        fig = make_single(sub_data, runs, colors, subject, task,
                                          metric, value, args.zoom, n_windows,
                                          args.nulls)
                        path = os.path.join(sub, "%s_%s_%s_%s.png"
                                            % (subject, task, value, metric))
                        fig.savefig(path, dpi=args.dpi, facecolor=SURFACE)
                        plt.close(fig)
                        written.append(path)

        absent = [r for r in runs if any(task not in sub_data.get(r, {})
                                         for task in tasks)]
        print("%s: %d runs (%s)%s" % (subject, len(runs), ", ".join(runs),
              "  [incomplete: %s]" % ", ".join(absent) if absent else ""))

    for path in written:
        print("wrote %s" % os.path.relpath(path, config.REPO_DIR))
