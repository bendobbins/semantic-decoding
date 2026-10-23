import os
import numpy as np
import json

import config
from utils_ridge.stimulus_utils import TRFile, load_textgrids, load_simulated_trfiles
from utils_ridge.dsutils import make_word_ds
from utils_ridge.interpdata import lanczosinterp2D, onsetinterp2D
from utils_ridge.util import make_delayed

def get_story_wordseqs(stories, subject):
    """loads words and word times of stimulus stories
    """
    grids = load_textgrids(stories, subject, config.DATA_TRAIN_DIR)
    wordseqs = make_word_ds(grids, subject)
    return wordseqs
    
def get_stim(stories, subject, features, bin_stats = None):
    """extract quantitative features of stimulus stories
    """
    word_seqs = get_story_wordseqs(stories, subject)
    word_vecs = {story : features.make_stim(word_seqs[story].data) for story in stories}
    word_mat = np.vstack([word_vecs[story] for story in stories])
    word_mean, word_std = word_mat.mean(0), word_mat.std(0)
    
# 2-300 ms for short bin size, up to maybe 1s, be able to adjust it
    # ds_vecs = {story : lanczosinterp2D(word_vecs[story], word_seqs[story].data_times, word_seqs[story].bin_times) 
    #            for story in stories}

    # change amt of stim preceding bin using STIM_DELAYS, list(range(0, 5)) is 200ms, list(range(0, 7)) is 300ms
    ds_vecs = {story: onsetinterp2D(word_vecs[story], word_seqs[story].data_starts, word_seqs[story].bin_times)
               for story in stories}
    ds_mat = np.vstack([ds_vecs[story][config.TRIM:] for story in stories])
    if bin_stats is None: 
        r_mean, r_std = ds_mat.mean(0), ds_mat.std(0)
        r_std[r_std == 0] = 1
    else: 
        r_mean, r_std = bin_stats
    ds_mat = np.nan_to_num(np.dot((ds_mat - r_mean), np.linalg.inv(np.diag(r_std))))
    del_mat = make_delayed(ds_mat, config.STIM_DELAYS)
    if bin_stats is None: return del_mat, (r_mean, r_std), (word_mean, word_std)
    else: return del_mat

def predict_word_rate(resp, wt, mean_rate):
    """predict word rate at each acquisition time
    """
    delresp = make_delayed(resp, config.RESP_DELAYS)
    rate = ((delresp.dot(wt) + mean_rate)).reshape(-1).clip(min = 0)
    return np.round(rate).astype(int)

def predict_word_times(word_rate, resp, starttime = 0, tr = 2):
    """predict evenly spaced word times from word rate
    """
    half = tr / 2
    trf = TRFile(None, tr)
    trf.soundstarttime = starttime
    trf.simulate(resp.shape[0])
    tr_times = trf.get_reltriggertimes() + half

    word_times = []
    for mid, num in zip(tr_times, word_rate):  
        if num < 1: continue
        word_times.extend(np.linspace(mid - half, mid + half, num, endpoint = False) + half / num)
    return np.array(word_times), tr_times