import os
import numpy as np

# paths

REPO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_DIR, "data")
DATA_LM_DIR = os.path.join(DATA_DIR, "data_lm")
DATA_TRAIN_DIR = os.path.join(DATA_DIR, "data_train")
DATA_TEST_DIR = os.path.join(DATA_DIR, "data_test")
MODEL_DIR = os.path.join(REPO_DIR, "models")
RESULT_DIR = os.path.join(REPO_DIR, "results")
SCORE_DIR = os.path.join(REPO_DIR, "scores")

# GPT encoding model parameters

SAMPLE_RATE = 2000
# MUST BE IN SECONDS, NOT MILLISECONDS
BIN_SIZE = 0.05
STEP_SIZE = 0.01
STEP_EM = False

TRIM = 5
# Models will look at 50 * end of STIM_DELAYS amt of words ms, so if it is list(range(0, 5)) then they will look at 50 * 5 = 250 ms
STIM_DELAYS = list(range(0, 15))
RESP_DELAYS = [-num for num in STIM_DELAYS]
ALPHAS = np.logspace(1, 5, 17)
NBOOTS = 10
CHUNKLEN = 400
GPT_LAYER = 9
GPT_WORDS = 5

# decoder parameters

RANKED = True
WIDTH = 200
NM_ALPHA = 2/3
LM_TIME = 8
LM_MASS = 0.9
LM_RATIO = 0.1
EXTENSIONS = 5

# evaluation parameters

WINDOW = 20

# devices

import torch
def _best_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

GPT_DEVICE = _best_device()
EM_DEVICE = _best_device()
SM_DEVICE = _best_device()