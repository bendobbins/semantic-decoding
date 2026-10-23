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
from StimulusModel import StimulusModel, get_onset_mat, affected_bins, LMFeatures
from utils_stim import predict_word_rate, predict_word_times
from utils_resp import bin_resp_decoding
from utils_ridge import stimulus_utils, DataSequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--experiment", type = str, required = True)
    parser.add_argument("--task", type = str, required = True)
    parser.add_argument("--model_dir", type = str, default = "models")
    args = parser.parse_args()
    
    # determine GPT checkpoint based on experiment
    if args.experiment in ["imagined_speech"]: gpt_checkpoint = "imagined"
    else: gpt_checkpoint = "perceived"

    # determine word rate model voxels based on experiment
    if args.experiment in ["imagined_speech", "perceived_movies"]: word_rate_voxels = "speech"
    else: word_rate_voxels = "auditory"

    # load response from test story
    subject = args.subject.split("_")[0]
    hf = h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", subject, args.experiment, args.task + ".mat"), "r")
    ref = hf["all_data"][0, 0]
    resp = np.nan_to_num(hf[ref][()].T)
    hf.close()
    resp = bin_resp_decoding(resp, args.subject, args.task)
    
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
    load_location = os.path.join(config.REPO_DIR, args.model_dir, args.subject)
    # word_rate_model = np.load(os.path.join(load_location, "word_rate_model_%s.npz" % word_rate_voxels), allow_pickle = True)
    encoding_model = np.load(os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint))
    weights = encoding_model["weights"]
    noise_model = encoding_model["noise_model"]
    bin_stats = encoding_model["bin_stats"]
    word_stats = encoding_model["word_stats"]
    em = EncodingModel(resp, weights, noise_model, device = config.EM_DEVICE)
    em.set_shrinkage(config.NM_ALPHA)
    assert args.task not in encoding_model["stories"]
    
    # predict word times
    # word_rate = predict_word_rate(resp, word_rate_model["weights"], word_rate_model["mean_rate"])
    if args.experiment == "perceived_speech": word_times, tr_times = predict_word_times(word_rate, resp, starttime = -10)
    # else: word_times, tr_times = predict_word_times(word_rate, resp, starttime = 0)
    # lanczos_mat = get_lanczos_mat(word_times, tr_times)

    # get actual word times (for checking)

    grid = stimulus_utils.load_test_textgrid(args.task, args.subject, config.DATA_TEST_DIR)
    grtranscript = grid.tiers[1].make_simple_transcript()
    goodtranscript = [x for x in grtranscript
                      if x[2].lower().strip("{}").strip() not in frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])]

    with open(os.path.join(config.DATA_TEST_DIR, "respdict.json"), "r") as f:
        respdict = json.load(f)[args.subject]
    bin_start_times = np.arange(int(respdict[args.task] / config.SAMPLE_RATE / config.BIN_SIZE)) * config.BIN_SIZE + grid.xmin

    d = DataSequence.DataSequence.from_grid(goodtranscript, bin_start_times)
    word_times = d.data_starts
    bin_times = d.bin_times

    onset_mat = get_onset_mat(word_times, bin_times)

    # decode responses
    decoder = Decoder(word_times, config.WIDTH)
    sm = StimulusModel(onset_mat, bin_stats, word_stats[0], device = config.SM_DEVICE)
    for sample_index in range(len(word_times)):
        bins = affected_bins(decoder.first_difference(), sample_index, onset_mat)
        ncontext = decoder.time_window(sample_index, config.LM_TIME, floor = 5)
        beam_nucs = lm.beam_propose(decoder.beam, ncontext)
        for c, (hyp, nextensions) in enumerate(decoder.get_hypotheses()):
            nuc, logprobs = beam_nucs[c]
            if len(nuc) < 1: continue
            extend_words = [hyp.words + [x] for x in nuc]
            extend_embs = list(features.extend(extend_words))
            stim = sm.make_variants(sample_index, hyp.embs, extend_embs, bins)
            likelihoods = em.prs(stim, bins)
            local_extensions = [Hypothesis(parent = hyp, extension = x) for x in zip(nuc, logprobs, extend_embs)]
            decoder.add_extensions(local_extensions, likelihoods, nextensions)
        decoder.extend(verbose = False)
        
    if args.experiment in ["perceived_movie", "perceived_multispeaker"]: decoder.word_times += 10
    save_location = os.path.join(config.RESULT_DIR, args.model_dir, args.subject, args.experiment)
    os.makedirs(save_location, exist_ok = True)
    decoder.save(os.path.join(save_location, args.task))