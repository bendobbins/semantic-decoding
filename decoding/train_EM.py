import os
import numpy as np
import json
import argparse

import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_resp import get_resp
from utils_ridge.ridge import ridge, bootstrap_ridge
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--sessions", nargs = "+", type = int, 
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--model_dir", type = lambda p: os.path.join(config.REPO_DIR, p), default = "models")
    parser.add_argument("--augment", type = str, default = None,
        help = "path to an aug manifest ({aug_story: base_story}); appends synonym-augmented "
               "copies of any listed base story that is in the session set. Fits the encoding "
               "weights on originals + augmentations, but estimates the noise model on originals only.")
    args = parser.parse_args()

    # training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])

    # data augmentation: originals fit the weights + augmentations; noise model uses originals only
    orig_stories = list(stories)
    aug_stories = []
    if args.augment is not None:
        with open(args.augment, "r") as f:
            aug_manifest = json.load(f)
        aug_stories = [aug for aug, base in aug_manifest.items() if base in orig_stories]
    train_stories = orig_stories + aug_stories

    # load gpt
    with open(os.path.join(config.DATA_LM_DIR, args.gpt, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    gpt = GPT(path = os.path.join(config.DATA_LM_DIR, args.gpt, "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    
    # estimate encoding model (weights fit on originals + augmentations)
    rstim, tr_stats, word_stats = get_stim(train_stories, features)

    # select alphas + voxels on originals only. An augmented story reuses its base
    # story's responses verbatim, so a bootstrap chunk held out from an augmented
    # story can have its identical twin sitting in the training half -- which
    # inflates the held-out correlations that pick the alphas and rank the voxels.
    # Without --augment this is the full training set, so the no-aug path is
    # unchanged (bootstrap_ridge's own final fit is exactly ridge() at valphas).
    sel_stim = get_stim(orig_stories, features, tr_stats = tr_stats) if aug_stories else rstim
    sel_resp = get_resp(args.subject, orig_stories, stack = True)
    nchunks = int(np.ceil(sel_resp.shape[0] / 5 / config.CHUNKLEN))
    _, alphas, bscorrs = bootstrap_ridge(sel_stim, sel_resp, use_corr = False, alphas = config.ALPHAS,
        nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)
    bscorrs = bscorrs.mean(2).max(0)
    vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])

    # fit the weights on originals + augmentations at those fixed alphas
    if aug_stories:
        del sel_stim, sel_resp                       # free before the larger load
        rresp = get_resp(args.subject, train_stories, stack = True)
    else:
        rresp = sel_resp
    weights = ridge(rstim, rresp, alphas)
    del rstim, rresp

    # estimate noise model (originals only: augmented twins would contaminate leave-one-story-out)
    stim_dict = {story : get_stim([story], features, tr_stats = tr_stats) for story in orig_stories}
    resp_dict = get_resp(args.subject, orig_stories, stack = False, vox = vox)
    noise_model = np.zeros([len(vox), len(vox)])
    for hstory in orig_stories:
        tstim, hstim = np.vstack([stim_dict[tstory] for tstory in orig_stories if tstory != hstory]), stim_dict[hstory]
        tresp, hresp = np.vstack([resp_dict[tstory] for tstory in orig_stories if tstory != hstory]), resp_dict[hstory]
        bs_weights = ridge(tstim, tresp, alphas[vox])
        resids = hresp - hstim.dot(bs_weights)
        bs_noise_model = resids.T.dot(resids)
        noise_model += bs_noise_model / np.diag(bs_noise_model).mean() / len(orig_stories)
    del stim_dict, resp_dict
    
    # save
    # save_location = os.path.join(config.MODEL_DIR, args.subject)
    save_location = os.path.join(args.model_dir, args.subject)
    os.makedirs(save_location, exist_ok = True)
    np.savez(os.path.join(save_location, "encoding_model_%s" % args.gpt), 
        weights = weights, noise_model = noise_model, alphas = alphas, voxels = vox, stories = train_stories,
        orig_stories = orig_stories, aug_stories = aug_stories,
        tr_stats = np.array(tr_stats), word_stats = np.array(word_stats))