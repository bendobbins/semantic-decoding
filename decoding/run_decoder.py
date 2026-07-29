import os
import numpy as np
import json
import argparse
import h5py
from pathlib import Path

import config
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel
from EncodingModel import EncodingModel
from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from utils_stim import predict_word_rate, predict_word_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--experiment", type = str, required = True)
    parser.add_argument("--task", type = str, required = True)
    parser.add_argument("--model_dir", type = str, default = "models")
    parser.add_argument("--aug_tag", type = str, default = None,
        help = "augmentation run tag, e.g. 3var-0.2swap-0seed; appended to the model load "
               "and result save paths as <model_dir>/<subject>/<aug_tag>")
    parser.add_argument("--word_rate_dir", type = str, default = None,
        help = "repo-relative dir holding word_rate_model_*.npz. Only needed to override "
               "the lookup: the model dir is used if it has one, otherwise the path the "
               "encoding model recorded at training time")
    args = parser.parse_args()
    rel_subject = os.path.join(args.subject, args.aug_tag) if args.aug_tag else args.subject
    
    # determine GPT checkpoint based on experiment
    if args.experiment in ["imagined_speech"]: gpt_checkpoint = "imagined"
    else: gpt_checkpoint = "perceived"

    # determine word rate model voxels based on experiment
    if args.experiment in ["imagined_speech", "perceived_movies"]: word_rate_voxels = "speech"
    else: word_rate_voxels = "auditory"

    # load responses (augmented conditions decode on the base subject's clean test data)
    subject = os.path.basename(args.subject).split("_")[0]
    hf = h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", subject, args.experiment, args.task + ".hf5"), "r")
    resp = np.nan_to_num(hf["data"][:])
    hf.close()
    
    # load gpt
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json"), "r") as f:
        gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json"), "r") as f:
        decoder_vocab = json.load(f)
    gpt = GPT(path = os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"), vocab = gpt_vocab, device = config.GPT_DEVICE)
    features = LMFeatures(model = gpt, layer = config.GPT_LAYER, context_words = config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass = config.LM_MASS, nuc_ratio = config.LM_RATIO)

    # load models
    # load_location = os.path.join(config.MODEL_DIR, args.subject)
    load_location = os.path.join(config.REPO_DIR, args.model_dir, rel_subject)
    encoding_model = np.load(os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint))

    # word rate is stimulus-agnostic, so an augmented condition has no model of its own and
    # reuses the one trained for the base subject. Prefer a local file (every non-augmented
    # run has one), then an explicit override, then the path recorded at training time.
    wr_name = "word_rate_model_%s.npz" % word_rate_voxels
    wr_candidates = [os.path.join(load_location, wr_name)]
    if args.word_rate_dir:
        wr_candidates.append(os.path.join(config.REPO_DIR, args.word_rate_dir, wr_name))
    if "word_rate_dir" in encoding_model.files:
        recorded = str(encoding_model["word_rate_dir"])
        if recorded:
            wr_candidates.append(os.path.join(config.REPO_DIR, recorded, wr_name))
    wr_path = next((p for p in wr_candidates if os.path.exists(p)), None)
    if wr_path is None:
        raise SystemExit("no %s found; searched:\n  %s" % (wr_name, "\n  ".join(wr_candidates)))
    word_rate_model = np.load(wr_path, allow_pickle = True)
    weights = encoding_model["weights"]
    noise_model = encoding_model["noise_model"]
    tr_stats = encoding_model["tr_stats"]
    word_stats = encoding_model["word_stats"]
    em = EncodingModel(resp, weights, encoding_model["voxels"], noise_model, device = config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    assert args.task not in encoding_model["stories"]
    
    # predict word times
    word_rate = predict_word_rate(resp, word_rate_model["weights"], word_rate_model["voxels"], word_rate_model["mean_rate"])
    if args.experiment == "perceived_speech": word_times, tr_times = predict_word_times(word_rate, resp, starttime = -10)
    else: word_times, tr_times = predict_word_times(word_rate, resp, starttime = 0)
    lanczos_mat = get_lanczos_mat(word_times, tr_times)

    # decode responses
    decoder = Decoder(word_times, config.WIDTH)
    sm = StimulusModel(lanczos_mat, tr_stats, word_stats[0], device = config.SM_DEVICE)
    for sample_index in range(len(word_times)):
        trs = affected_trs(decoder.first_difference(), sample_index, lanczos_mat)
        ncontext = decoder.time_window(sample_index, config.LM_TIME, floor = 5)
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1: continue
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))
            stim = sm.make_variants(sample_index, hyp.embs, extend_embs, trs)
            likelihoods = em.prs(stim, trs)
            local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose = False)
        
    if args.experiment in ["perceived_movie", "perceived_multispeaker"]: decoder.word_times += 10
    save_location = os.path.join(config.RESULT_DIR, args.model_dir, rel_subject, args.experiment)
    os.makedirs(save_location, exist_ok = True)
    decoder.save(os.path.join(save_location, args.task))