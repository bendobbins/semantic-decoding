import os
import shutil
import argparse
import numpy as np
import h5py

import config

"""
Create a copy of a test-response .hf5 file and add Gaussian noise to a random
subset of the voxels that are actually used for decoding (the best ~10,000
voxels stored in the encoding model's "voxels" array).

The decoder reads hf["data"] (shape: n_TRs x n_voxels) and selects columns by
encoding_model["voxels"], so we only need to perturb those columns of "data".
Noise for each chosen voxel is scaled to that voxel's own temporal std.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--experiment", type = str, required = True)
    parser.add_argument("--task", type = str, required = True)
    parser.add_argument("--frac", type = float, default = 0.1,
                        help = "fraction of used voxels to corrupt (default 0.1)")
    parser.add_argument("--noise_scale", type = float, default = 1.0,
                        help = "noise std as a multiple of each voxel's own std (default 1.0)")
    parser.add_argument("--seed", type = int, default = 0)
    args = parser.parse_args()

    # resp_dir = os.path.join(config.DATA_TRAIN_DIR, "train_response", args.subject, args.experiment)
    resp_dir = os.path.join(config.DATA_TRAIN_DIR, "train_response", args.subject)
    src_path = os.path.join(resp_dir, args.task + ".hf5")
    if not os.path.exists(src_path):
        raise SystemExit("source test file not found: %s" % src_path)

    # build a descriptive default name so variants (different frac/scale/seed) never collide

    # voxels actually used for decoding (best ~10,000 for this subject/experiment)
    em_path = os.path.join(config.MODEL_DIR, args.subject, "encoding_model_%s.npz"
                           % ("imagined" if args.experiment == "imagined_speech" else "perceived"))
    used_voxels = np.load(em_path)["voxels"]

    # add noise to the chosen columns of "data" in the copy
    with h5py.File(src_path, "r") as hf:
        block = hf["data"][:, used_voxels]                       # n_TRs x n_noisy
        print("block shape:", block.shape)
        print("block:", block[0:5, 0:10])
        vox_std = np.nan_to_num(block).std(axis = 0)        # per-voxel temporal std
        print("voxel std shape:", vox_std.shape)
        print("voxel std:", vox_std[0:10])
        # create a histogram of the stds for the voxels

        import matplotlib.pyplot as plt
        plt.hist(vox_std)
        plt.xlabel("Voxel Standard Deviation")
        plt.ylabel("Frequency")
        plt.title("Distribution of Voxel Temporal Standard Deviations")
        plt.show()
