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
    parser.add_argument("--out", type = str, default = None,
                        help = "output filename without extension (default encodes the noise "
                               "parameters, e.g. <task>_noise20)")
    args = parser.parse_args()

    resp_dir = os.path.join(config.DATA_TEST_DIR, "test_response", args.subject, args.experiment)
    src_path = os.path.join(resp_dir, args.task + ".hf5")
    if not os.path.exists(src_path):
        raise SystemExit("source test file not found: %s" % src_path)

    # build a descriptive default name so variants (different frac/scale/seed) never collide
    if args.out is not None:
        out_stem = args.out
    else:
        out_stem = "%s_noise%d" % (args.task, int(round(100 * args.frac)))
        if args.noise_scale != 1.0: out_stem += "_s%g" % args.noise_scale
        if args.seed != 0: out_stem += "_seed%d" % args.seed
    dst_path = os.path.join(resp_dir, out_stem + ".hf5")

    if os.path.exists(dst_path):
        raise SystemExit("refusing to overwrite existing file: %s" % dst_path)

    # voxels actually used for decoding (best ~10,000 for this subject/experiment)
    em_path = os.path.join(config.MODEL_DIR, args.subject, "encoding_model_%s.npz"
                           % ("imagined" if args.experiment == "imagined_speech" else "perceived"))
    used_voxels = np.load(em_path)["voxels"]

    # randomly pick the subset to corrupt
    rng = np.random.default_rng(args.seed)
    n_noisy = int(round(args.frac * len(used_voxels)))
    chosen = np.sort(rng.choice(used_voxels, size = n_noisy, replace = False))
    print("corrupting %d of %d used voxels (%.1f%%)" % (n_noisy, len(used_voxels), 100 * n_noisy / len(used_voxels)))

    # copy the original so it is preserved (content only; copy stays writable)
    print("copying %s -> %s" % (src_path, dst_path))
    shutil.copyfile(src_path, dst_path)

    # add noise to the chosen columns of "data" in the copy
    with h5py.File(dst_path, "r+") as hf:
        block = hf["data"][:, chosen]                       # n_TRs x n_noisy
        vox_std = np.nan_to_num(block).std(axis = 0)        # per-voxel temporal std
        noise = rng.normal(0.0, 1.0, size = block.shape) * (args.noise_scale * vox_std)
        hf["data"][:, chosen] = block + noise

    print("done. decode with --task %s" % out_stem)
