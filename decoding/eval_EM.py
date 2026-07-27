"""Fast encoding-level comparison for the augmentation bake-off.

Predicts a held-out real test story (default: perceived_speech / wheretheressmoke)
from each condition's *actual* trained encoding model and reports the voxelwise
prediction correlation. This is the quick triage metric to run before the far
slower decode -> score pipeline: it evaluates the real saved weights on data no
condition trained on, so it isolates whether augmentation improved the encoding
model. Higher = better.

    python decoding/eval_EM.py --conditions S1_base S1_aug_wordnet S1_aug_embedding S1_aug_llm S1_aug_random

The stimulus is built from the true test transcript with the same TR alignment the
training pipeline uses (validated: the baseline model scores ~0.31 here). Each
condition is scored on its own selected voxels + tr_stats, exactly as the decoder
would use it.
"""

import os
import json
import argparse

import numpy as np
import h5py

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_ridge.textgrid import TextGrid
from utils_ridge.stimulus_utils import TRFile
from utils_ridge.dsutils import DEFAULT_BAD_WORDS
from utils_ridge.interpdata import lanczosinterp2D
from utils_ridge.util import make_delayed


def voxelwise_corr(pred, actual):
    """Pearson correlation per column between pred and actual (T x V each)."""
    p = pred - pred.mean(0)
    a = actual - actual.mean(0)
    den = np.sqrt((p ** 2).sum(0) * (a ** 2).sum(0))
    den[den == 0] = np.nan
    return (p * a).sum(0) / den


def test_transcript(experiment, task):
    tier = 1 if experiment == "perceived_speech" else 0
    grid = TextGrid(open(os.path.join(config.DATA_TEST_DIR, "test_stimulus",
                                      experiment, task.split("_")[0] + ".TextGrid")).read())
    words, starts, ends = [], [], []
    for s, e, w in grid.tiers[tier].make_simple_transcript():
        if w.lower().strip("{}").strip() in DEFAULT_BAD_WORDS:
            continue
        words.append(w.lower()); starts.append(float(s)); ends.append(float(e))
    return words, (np.array(starts) + np.array(ends)) / 2.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", required=True,
                        help="condition/subject names, e.g. S1_base S1_aug_wordnet ...")
    parser.add_argument("--experiment", type=str, default="perceived_speech")
    parser.add_argument("--task", type=str, default="wheretheressmoke")
    parser.add_argument("--gpt", type=str, default="perceived")
    parser.add_argument("--topk", type=int, default=1000,
                        help="report the mean over each model's top-k best-predicted voxels")
    args = parser.parse_args()

    # true test transcript -> GPT word features (once)
    words, data_times = test_transcript(args.experiment, args.task)
    with open(os.path.join(config.DATA_LM_DIR, args.gpt, "vocab.json")) as f:
        vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, args.gpt, "model"), vocab=vocab, device=config.GPT_DEVICE)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    word_vecs = features.make_stim(words)  # [n_words, n_feat]

    print("%-24s  %8s  %8s  %8s" % ("condition", "mean", "top%d" % args.topk, "median"))
    print("-" * 54)
    for cond in args.conditions:
        resp_path = os.path.join(config.DATA_TEST_DIR, "test_response", cond,
                                 args.experiment, args.task + ".hf5")
        model_path = os.path.join(config.MODEL_DIR, cond, "encoding_model_%s.npz" % args.gpt)
        if not (os.path.exists(resp_path) and os.path.exists(model_path)):
            print("%-24s  (missing response or model)" % cond)
            continue

        with h5py.File(resp_path, "r") as hf:
            resp = np.nan_to_num(hf["data"][:])
        R = resp.shape[0]

        em = np.load(model_path)
        weights, vox, tr_stats = em["weights"], em["voxels"], em["tr_stats"]
        r_mean, r_std = tr_stats[0], tr_stats[1]

        # downsample word features to the response's TR grid (training-style alignment)
        trf = TRFile(None, 2.0)
        trf.soundstarttime = 10.0
        trf.simulate((R + 20) - 5)
        trtimes = trf.get_reltriggertimes() + 1.0
        ds = lanczosinterp2D(word_vecs, data_times, trtimes)[5 + config.TRIM:-config.TRIM]
        ds = np.nan_to_num(np.dot((ds - r_mean), np.linalg.inv(np.diag(r_std))))
        stim = make_delayed(ds, config.STIM_DELAYS)

        n = min(stim.shape[0], R)
        corrs = voxelwise_corr(stim[:n].dot(weights[:, vox]), resp[:n][:, vox])
        corrs = corrs[~np.isnan(corrs)]
        topk = np.sort(corrs)[-args.topk:]
        print("%-24s  %8.4f  %8.4f  %8.4f"
              % (cond, corrs.mean(), topk.mean(), np.median(corrs)))
