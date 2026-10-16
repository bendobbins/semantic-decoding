import os
import numpy as np
import h5py
import json

import config

def get_resp(subject, stories, stack = True):
    """loads response data
    """
    subject_dir = os.path.join(config.DATA_TRAIN_DIR, "train_response", subject)
    resp = {}
    for story in stories:
        resp_path = os.path.join(subject_dir, "%s.mat" % story)
        hf = h5py.File(resp_path, "r")
        ref = hf["all_data"][0, 0]
        resp[story] = np.nan_to_num(hf[ref][()].T)
        hf.close()
    # if stack: return bin_resp(np.vstack([resp[story] for story in stories]), subject, stories)
    return bin_resp(resp, subject, stack)


def bin_resp(resp, subject, stack):
    with open(os.path.join(config.DATA_TRAIN_DIR, "respdict.json"), "r") as f:
        respdict = json.load(f)[subject]

    binned_resp = {}
    for story in resp:
        # Bin size is in seconds (e.g. 0.05 is 50 ms/bin), multiply by sample rate
        # (2000 samples/s) for our ECoG to get samples per bin
        samples_per_bin = int(config.BIN_SIZE * config.SAMPLE_RATE)

        # Do if stepping incrementally (set STEP_EM in config)
        if config.STEP_EM:
            # Get step size in samples (same logic as above)
            step_size_samples = int(config.STEP_SIZE * config.SAMPLE_RATE)
            curr = 0
            bins = []
            while curr + samples_per_bin < respdict[story]:
                # Get a bin of samples. From current step place to step place + bin size
                # Take mean over time for each channel, left with 1 x n_channels array
                bins.append(resp[story][curr:curr+samples_per_bin].mean(axis=0))
                curr += step_size_samples
            # Should be n_steps x n_channels
            binned_resp[story] = np.array(bins)
        else:
            # No step, next bin is after previous, so amount of bins is
            # n_samples (samples) / sample rate (samples/s) / bin size (s/bin)
            amt_bins = int(respdict[story] / config.SAMPLE_RATE / config.BIN_SIZE)
            bins = []
            for b in range(amt_bins):
                # Get a bin of samples. From end of last bin to there + bin size
                # Take mean over time for each channel, left with 1 x n_channels array
                bins.append(resp[story][b*samples_per_bin:(b+1)*samples_per_bin].mean(axis=0))
            # Should be amt_bins x n_channels
            binned_resp[story] = np.array(bins)

    if stack: return np.vstack([binned_resp[story][config.TRIM:] for story in resp])
    else: return {story: binned_resp[story][config.TRIM:] for story in binned_resp}


def bin_resp_decoding(resp, subject, story):
    with open(os.path.join(config.DATA_TEST_DIR, "respdict.json"), "r") as f:
        respdict = json.load(f)[subject]
    
    samples_per_bin = int(config.BIN_SIZE * config.SAMPLE_RATE)

    if config.STEP_EM:
        step_size_samples = int(config.STEP_SIZE * config.SAMPLE_RATE) 
        curr = 0
        bins = []
        while curr + samples_per_bin < respdict[story]:
            bins.append(resp[curr:curr+samples_per_bin].mean(axis=0))
            curr += step_size_samples
    else:
        amt_bins = int(respdict[story] / config.SAMPLE_RATE / config.BIN_SIZE)
        bins = []
        for b in range(amt_bins):
            bins.append(resp[b*samples_per_bin:(b+1)*samples_per_bin].mean(axis=0))

    return np.array(bins)