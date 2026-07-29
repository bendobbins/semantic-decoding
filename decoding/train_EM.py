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


def load_aug_manifest(path, subject, sessions, aug_tag):
    """Read an augmentation manifest and resolve everything a run needs from it.

    The manifest written by make_augmented_stories.py carries the whole run descriptor,
    so this is the single source of truth for where the augmented stimuli, respdict
    overlay, responses and word rate model live. Anything the caller also passed on the
    command line is checked against it and a mismatch aborts, so a run can never be
    silently wired to another run's data.
    """
    with open(path, "r") as f:
        man = json.load(f)

    # tolerate a bare {aug_story: base_story} mapping from before the run descriptor
    if "stories" not in man:
        print("warning: %s has no run descriptor; falling back to the shared "
              "train_stimulus/respdict and train_response/%s" % (path, subject))
        return {"stories": man, "aug_stim_dirs": (), "respdict_extra": None,
                "resp_root": None, "word_rate_dir": "", "aug_tag": aug_tag}

    def check(field, passed, got):
        if passed is not None and got is not None and passed != got:
            raise SystemExit(
                "manifest mismatch on %s: you passed %r but %s was built with %r.\n"
                "  manifest: %s\n"
                "Use the values this run was generated with, or point --augment at the "
                "right manifest." % (field, passed, os.path.basename(path), got, path))

    check("--subject", subject, man.get("condition"))
    check("--sessions", list(sessions) if sessions is not None else None, man.get("sessions"))
    check("--aug_tag", aug_tag, man.get("aug_tag"))

    def abspath(rel):
        return os.path.join(config.REPO_DIR, rel) if rel else None

    respdict_path = abspath(man.get("respdict"))
    respdict_extra = None
    if respdict_path and os.path.exists(respdict_path):
        with open(respdict_path, "r") as f:
            respdict_extra = json.load(f)

    stim_dir = abspath(man.get("stim_dir"))
    return {
        "stories": man["stories"],
        "aug_stim_dirs": (stim_dir,) if stim_dir else (),
        "respdict_extra": respdict_extra,
        "resp_root": abspath(man.get("resp_root")),
        "word_rate_dir": man.get("word_rate_dir", ""),
        "aug_tag": aug_tag or man.get("aug_tag"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived")
    parser.add_argument("--sessions", nargs = "+", type = int,
        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20])
    parser.add_argument("--model_dir", type = str, default = "models")
    parser.add_argument("--aug_tag", type = str, default = None,
        help = "augmentation run tag, e.g. 3var-0.2swap-0seed; appended to the model save "
               "path as <model_dir>/<subject>/<aug_tag>. Defaults to the manifest's value "
               "when --augment is given, and to nothing otherwise.")
    parser.add_argument("--augment", type = str, default = None,
        help = "path to an aug manifest from make_augmented_stories.py; appends synonym-"
               "augmented copies of any listed base story that is in the session set. Fits "
               "the encoding weights on originals + augmentations, but selects alphas/voxels "
               "and estimates the noise model on originals only.")
    args = parser.parse_args()

    # training stories
    stories = []
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])

    # data augmentation: weights fit on originals + augmentations; everything that is
    # selected or estimated by holding data out uses originals only
    orig_stories = list(stories)
    aug_stories, aug_tag = [], args.aug_tag
    aug_stim_dirs, respdict_extra, resp_root, word_rate_dir = (), None, None, ""
    if args.augment is not None:
        aug = load_aug_manifest(args.augment, args.subject, args.sessions, args.aug_tag)
        aug_stories = [a for a, base in aug["stories"].items() if base in orig_stories]
        aug_stim_dirs = aug["aug_stim_dirs"]
        respdict_extra = aug["respdict_extra"]
        resp_root = aug["resp_root"]
        word_rate_dir = aug["word_rate_dir"]
        aug_tag = aug["aug_tag"]
    train_stories = orig_stories + aug_stories

    stim_kw = {"aug_stim_dirs": aug_stim_dirs, "respdict_extra": respdict_extra}

    # load gpt
    with open(os.path.join(config.DATA_LM_DIR, args.gpt, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    gpt = GPT(path = os.path.join(config.DATA_LM_DIR, args.gpt, "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)

    # estimate encoding model (weights fit on originals + augmentations)
    rstim, tr_stats, word_stats = get_stim(train_stories, features, **stim_kw)

    # select alphas + voxels on originals only. An augmented story reuses its base
    # story's responses verbatim, so a bootstrap chunk held out from an augmented
    # story can have its identical twin sitting in the training half -- which
    # inflates the held-out correlations that pick the alphas and rank the voxels.
    # Without --augment this is the full training set, so the no-aug path is
    # unchanged (bootstrap_ridge's own final fit is exactly ridge() at valphas).
    sel_stim = get_stim(orig_stories, features, tr_stats = tr_stats, **stim_kw) if aug_stories else rstim
    sel_resp = get_resp(args.subject, orig_stories, stack = True, resp_root = resp_root)
    nchunks = int(np.ceil(sel_resp.shape[0] / 5 / config.CHUNKLEN))
    _, alphas, bscorrs = bootstrap_ridge(sel_stim, sel_resp, use_corr = False, alphas = config.ALPHAS,
        nboots = config.NBOOTS, chunklen = config.CHUNKLEN, nchunks = nchunks)
    bscorrs = bscorrs.mean(2).max(0)
    vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])

    # fit the weights on originals + augmentations at those fixed alphas
    if aug_stories:
        del sel_stim, sel_resp                       # free before the larger load
        rresp = get_resp(args.subject, train_stories, stack = True, resp_root = resp_root)
    else:
        rresp = sel_resp
    weights = ridge(rstim, rresp, alphas)
    del rstim, rresp

    # estimate noise model (originals only: augmented twins would contaminate leave-one-story-out)
    stim_dict = {story : get_stim([story], features, tr_stats = tr_stats, **stim_kw) for story in orig_stories}
    resp_dict = get_resp(args.subject, orig_stories, stack = False, vox = vox, resp_root = resp_root)
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
    save_location = os.path.join(config.REPO_DIR, args.model_dir, args.subject)
    if aug_tag:
        save_location = os.path.join(save_location, aug_tag)
    os.makedirs(save_location, exist_ok = True)
    np.savez(os.path.join(save_location, "encoding_model_%s" % args.gpt),
        weights = weights, noise_model = noise_model, alphas = alphas, voxels = vox, stories = train_stories,
        orig_stories = orig_stories, aug_stories = aug_stories,
        # word rate is stimulus-agnostic, so augmented conditions reuse the subject's
        # existing model; run_decoder.py reads this when the model dir has no local copy
        word_rate_dir = word_rate_dir, aug_tag = aug_tag or "",
        aug_manifest = args.augment or "", sessions = args.sessions,
        tr_stats = np.array(tr_stats), word_stats = np.array(word_stats))
    print("saved -> %s" % os.path.relpath(save_location, config.REPO_DIR))
