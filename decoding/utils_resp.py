import os
import numpy as np
import h5py

import config

def get_resp(subject, stories, stack = True, vox = None, resp_root = None):
    """loads response data

    [resp_root] overrides the directory that holds per-subject response dirs; an
    augmentation run points it at its own train_response/aug/... tree.
    """
    if resp_root is None:
        resp_root = os.path.join(config.DATA_TRAIN_DIR, "train_response")
    subject_dir = os.path.join(resp_root, subject)
    resp = {}
    for story in stories:
        resp_path = os.path.join(subject_dir, "%s.hf5" % story)
        hf = h5py.File(resp_path, "r")
        resp[story] = np.nan_to_num(hf["data"][:])
        if vox is not None:
            resp[story] = resp[story][:, vox]
        hf.close()
    if stack: return np.vstack([resp[story] for story in stories]) 
    else: return resp