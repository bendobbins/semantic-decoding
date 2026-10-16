import os
import numpy as np
import json
import argparse

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge, zs
from em_comparison import OnsetFeatures, shifted_null_r # comparison
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--sessions", nargs = "+", type = int, required = True)
    parser.add_argument("--model_dir", type = lambda p: os.path.join(config.REPO_DIR, p), default = "models")
    parser.add_argument("--features", type = str, default = "gpt", choices = ["gpt", "onset"]) # comparison
    parser.add_argument("--null_shifts", type = int, default = 0) # comparison: shifts for the chance level; 0 = off
    args = parser.parse_args()

    # training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f) 
    for sess in args.sessions:
        stories.extend(sess_to_story[str(args.subject) + str(sess)])

    # load gpt
    # TODO: take out imagined?
    with open(os.path.join(config.DATA_LM_DIR, args.gpt, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    gpt = GPT(path = os.path.join(config.DATA_LM_DIR, args.gpt, "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    if args.features == "onset": features, args.gpt = OnsetFeatures(), "onset" # comparison: onsets only, saved as encoding_model_onset
    
    # estimate encoding model
    rresp = get_resp(args.subject, stories, stack = True)
    n_channels = rresp.shape[1]
    rstim, bin_stats, word_stats = get_stim(stories, args.subject, features)
    nchunks = int(np.ceil(rresp.shape[0] / 5 / config.CHUNKLEN))
    weights, alphas, bscorrs = bootstrap_ridge(rstim, rresp, use_corr = False, alphas = config.ALPHAS,
        nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)        
    bs_scores = bscorrs # alphas x channels x bootstraps; R^2 floored at 0 (use_corr = False), not r
    bscorrs = bscorrs.mean(2).max(0)
    del rstim, rresp
    
    # estimate noise model
    stim_dict = {story : get_stim([story], args.subject, features, bin_stats = bin_stats) for story in stories}
    resp_dict = get_resp(args.subject, stories, stack = False)
    noise_model = np.zeros([n_channels, n_channels])
    heldout_pred, heldout_r = [], []
    for hstory in stories:
        tstim, hstim = np.vstack([stim_dict[tstory] for tstory in stories if tstory != hstory]), stim_dict[hstory]
        tresp, hresp = np.vstack([resp_dict[tstory] for tstory in stories if tstory != hstory]), resp_dict[hstory]
        bs_weights = ridge(tstim, tresp, alphas)
        heldout_pred.append(hstim.dot(bs_weights)) # hstory predicted by a model trained without it
        heldout_r.append((zs(heldout_pred[-1]) * zs(hresp)).mean(0)) # Pearson r per channel
        resids = hresp - heldout_pred[-1]
        bs_noise_model = resids.T.dot(resids)
        noise_model += bs_noise_model / np.diag(bs_noise_model).mean() / len(stories)
    heldout_r_pooled = (zs(np.vstack(heldout_pred)) * zs(np.vstack([resp_dict[story] for story in stories]))).mean(0)
    if args.null_shifts: null_r = shifted_null_r(heldout_pred, [resp_dict[story] for story in stories], args.null_shifts, int(10 / config.BIN_SIZE)) # comparison
    del stim_dict, resp_dict
    
    # save
    # save_location = os.path.join(config.MODEL_DIR, args.subject)
    save_location = os.path.join(args.model_dir, args.subject)
    os.makedirs(save_location, exist_ok = True)
    np.savez(os.path.join(save_location, "encoding_model_%s" % args.gpt), 
        weights = weights, noise_model = noise_model, alphas = alphas, stories = stories,
        bin_stats = np.array(bin_stats), word_stats = np.array(word_stats),
        bs_scores = bs_scores, heldout_r = np.array(heldout_r), heldout_r_pooled = heldout_r_pooled,
        heldout_pred = np.vstack(heldout_pred), heldout_bins = np.array([len(p) for p in heldout_pred]))
    if args.null_shifts: np.save(os.path.join(save_location, "encoding_model_%s_null_r" % args.gpt), null_r) # comparison