"""Scores the word rate model's predicted word times against the actual onsets of a test story, before decoding.

Predicted times come from the calls run_decoder.py makes (predict_word_rate, predict_word_times). Two references
use no timing information from the response: words evenly spaced at the training mean rate (the model with its
weights zeroed), and the predicted times circularly shifted by at least 10 s (chance; p = (1 + shifts scoring at
least as high) / (1 + shifts)). Words are compared by the bin they fall in, as the decoder's get_onset_mat assigns them.

Usage:
    python evaluate_word_times.py --subject SE --experiment perceived_speech --task allatsea
"""
import os
import numpy as np
import argparse
import h5py

import config
from utils_stim import predict_word_rate, predict_word_times
from utils_resp import bin_resp_decoding
from utils_ridge import stimulus_utils
from utils_ridge.dsutils import DEFAULT_BAD_WORDS
from utils_ridge.interpdata import onsetinterp2D

def word_bins(times, bin_times):
    """sorted bin index of each word"""
    counts = onsetinterp2D(np.ones((len(times), 1)), times, bin_times)[:, 0].astype(int)
    return np.repeat(np.arange(len(bin_times)), counts)

def onset_f1(pred, true, tol):
    """F1 of predicted vs actual word bins ([true] sorted): each word is matched at most once, closest pairs
    first, and only to a word at most [tol] bins away"""
    pairs = []
    for i, p in enumerate(pred):
        for j in range(np.searchsorted(true, p - tol), np.searchsorted(true, p + tol, side = "right")):
            pairs.append((abs(true[j] - p), i, j))
    matched_pred, matched_true = set(), set()
    for _, i, j in sorted(pairs):
        if i not in matched_pred and j not in matched_true:
            matched_pred.add(i)
            matched_true.add(j)
    return 2 * len(matched_pred) / (len(pred) + len(true))

def window_counts(bins, n_bins, width):
    """number of words in each full window of [width] bins"""
    counts = np.bincount(bins, minlength = n_bins)
    return counts[:n_bins // width * width].reshape(-1, width).sum(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--experiment", type = str, required = True)
    parser.add_argument("--task", type = str, required = True)
    parser.add_argument("--model_dir", type = str, default = "models")
    parser.add_argument("--shifts", type = int, default = 100, help = "circular shifts for the chance level")
    args = parser.parse_args()

    # load response and bin grid of the test story, as in run_decoder.py
    subject = args.subject.split("_")[0]
    with h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", subject, args.experiment, args.task + ".mat"), "r") as hf:
        resp = np.nan_to_num(hf[hf["all_data"][0, 0]][()].T)
    resp = bin_resp_decoding(resp, args.subject, args.task)
    grid = stimulus_utils.load_test_textgrid(args.task, args.subject, config.DATA_TEST_DIR)
    bin_times = np.arange(resp.shape[0]) * config.BIN_SIZE + grid.xmin + config.BIN_SIZE / 2.0
    n_bins = len(bin_times)

    # load word rate model, with run_decoder.py's checks
    word_rate_model = np.load(os.path.join(config.REPO_DIR, args.model_dir, args.subject, "word_rate_model.npz"))
    assert args.task not in word_rate_model["stories"]
    assert word_rate_model["bin_size"] == config.BIN_SIZE, "word rate model was trained with another BIN_SIZE"
    assert list(word_rate_model["resp_delays"]) == config.RESP_DELAYS, "word rate model was trained with other RESP_DELAYS"
    wt, mean_rate = word_rate_model["weights"], word_rate_model["mean_rate"]

    # word bins: actual onsets, predicted times, evenly spaced times, shifted predicted times
    transcript = grid.tiers[1].make_simple_transcript()
    onsets = [float(x[0]) for x in transcript if x[2].lower().strip("{}").strip() not in DEFAULT_BAD_WORDS]
    true = word_bins(onsets, bin_times)
    pred = word_bins(predict_word_times(predict_word_rate(resp, wt, mean_rate), bin_times)[0], bin_times)
    even = word_bins(predict_word_times(predict_word_rate(resp, np.zeros_like(wt), mean_rate), bin_times)[0], bin_times)
    min_shift = int(10 / config.BIN_SIZE)
    rng = np.random.default_rng(0)
    shifted = [(pred + s) % n_bins for s in rng.integers(min_shift, n_bins - min_shift + 1, args.shifts)]

    print("%s / %s: %d bins (%.0f s); word rate model trained on %d stories, alpha %g" % (args.subject, args.task,
        n_bins, n_bins * config.BIN_SIZE, len(word_rate_model["stories"]), word_rate_model["alphas"][0]))
    print("words: actual %d, predicted %d, evenly spaced %d" % (len(true), len(pred), len(even)))
    count_r = {w: np.corrcoef(window_counts(pred, n_bins, round(w / config.BIN_SIZE)),
                              window_counts(true, n_bins, round(w / config.BIN_SIZE)))[0, 1] for w in [0.5, 1, 2]}
    print("word-count r, predicted vs actual: " + ", ".join("%g s windows %.2f" % (w, r) for w, r in count_r.items()))
    print("onset F1 within      predicted   evenly spaced   shifted (chance, %d shifts)" % args.shifts)
    for tol in [1, 2, 4]: # bins
        f1, null = onset_f1(pred, true, tol), np.array([onset_f1(s, true, tol) for s in shifted])
        label = "+-%.0f ms (%d bin%s)" % (tol * config.BIN_SIZE * 1000, tol, "" if tol == 1 else "s")
        print("  %-18s %.3f       %.3f           %.3f, p = %.3f" % (
            label, f1, onset_f1(even, true, tol), null.mean(), (1 + (null >= f1).sum()) / (1 + len(null))))
