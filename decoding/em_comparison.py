"""Extras for comparing encoding models across datasets (SE vs podcast); not part of the decoding pipeline.
Used only when train_EM.py gets --features onset or --null_shifts N. To remove: delete this file and the
lines ending in "# comparison" in train_EM.py.
"""
import numpy as np

from utils_ridge.ridge import zs

class OnsetFeatures():
    """stand-in for LMFeatures: one constant feature per word, so get_stim encodes only when words start
    """
    def make_stim(self, words):
        return np.ones((len(words), 1))

def shifted_null_r(preds, resps, n_shifts, min_shift, seed = 0):
    """chance distribution (n_shifts x channels) of the pooled held-out r: each story's responses are
    circularly shifted by its own random offset of at least [min_shift] bins either way, which keeps their
    autocorrelation and never crosses a story boundary; [preds] and [resps] are per-story lists in one order
    """
    short = [len(r) for r in resps if len(r) <= 2 * min_shift]
    if short: raise ValueError("stories of %s bins are too short for shifts of at least %d bins" % (short, min_shift))
    rng = np.random.default_rng(seed)
    zpred = zs(np.vstack(preds))
    # shifting rows within stories leaves the pooled mean and sd unchanged, so z-score once and shift the z-scores
    zresps = np.split(zs(np.vstack(resps)), np.cumsum([len(r) for r in resps])[:-1])
    null_r = np.zeros([n_shifts, zpred.shape[1]])
    for i in range(n_shifts):
        shifted = [np.roll(z, rng.integers(min_shift, len(z) - min_shift + 1), axis = 0) for z in zresps]
        null_r[i] = (zpred * np.vstack(shifted)).mean(0)
    return null_r
