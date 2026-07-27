import os
import json
import shutil
import argparse
import numpy as np
import h5py

import config

"""
Make noisy copies of ALL of a subject's TRAINING response files so the encoding
model can be re-fit on corrupted data.

Noise model: stationary additive Gaussian per voxel. Each voxel gets a FRESH
random value at every TR (real sensor noise fluctuates each sample), but the
noise *level* (standard deviation) is constant across time for that voxel -- i.e.
each channel carries the same amount of noise throughout the whole task. A
constant value repeated across time would just be a baseline offset and would
barely touch the temporal signal the encoding model relies on, so that is NOT
what we do here.

Because training responses are z-scored per voxel (std == 1), noise_scale is
effectively the noise-to-signal amplitude ratio: 0.5 makes noise variance ~20%
of signal variance (clean/noisy corr ~0.89) -- affects the signal without
drowning it.

Unlike add_voxel_noise.py (which perturbs a single test file an already-fit
decoder reads), training noise only matters if you RE-RUN train_EM.py afterwards.
Copies go to a sibling subject folder so nothing original is touched:
    train_response/<subject>_<suffix>/<story>.hf5
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--gpt", type = str, default = "perceived",
                        help = "gpt checkpoint for the printed train_EM step")
    parser.add_argument("--sessions", nargs = "+", type = int,
                        default = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20],
                        help = "training sessions to copy + corrupt (match train_EM.py)")
    parser.add_argument("--noise_scale", type = float, default = 0.5,
                        help = "noise std as a multiple of each voxel's own std (default 0.5)")
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--suffix", type = str, default = None,
                        help = "new subject-name suffix (default noise_s<scale>)")
    args = parser.parse_args()

    suffix = args.suffix or ("noise_s%g" % args.noise_scale)
    new_subject = "%s_%s" % (args.subject, suffix)

    src_dir = os.path.join(config.DATA_TRAIN_DIR, "train_response", args.subject)
    dst_dir = os.path.join(config.DATA_TRAIN_DIR, "train_response", new_subject)
    if os.path.exists(dst_dir):
        raise SystemExit("refusing to overwrite existing dir: %s" % dst_dir)

    # training stories for the requested sessions
    with open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)
    stories = []
    for sess in args.sessions:
        stories.extend(sess_to_story[str(sess)])

    print("corrupting ALL voxels in %d training stories (noise_scale=%g)"
          % (len(stories), args.noise_scale))

    os.makedirs(dst_dir)
    for i, story in enumerate(stories):
        src_path = os.path.join(src_dir, story + ".hf5")
        if not os.path.exists(src_path):
            print("  skip (missing): %s" % story)
            continue
        dst_path = os.path.join(dst_dir, story + ".hf5")
        shutil.copyfile(src_path, dst_path)
        # independent, reproducible noise per story (a separate acquisition)
        srng = np.random.default_rng([args.seed, i])
        with h5py.File(dst_path, "r+") as hf:
            data = np.nan_to_num(hf["data"][:])                 # n_TRs x n_voxels (all voxels)
            vox_std = data.std(axis = 0)                        # per-voxel level; ~1 for z-scored train data
            # fresh value every TR, but a fixed per-voxel std => constant noise level over time
            noise = srng.normal(0.0, 1.0, size = data.shape) * (args.noise_scale * vox_std)
            hf["data"][:] = data + noise
        print("  noised %s" % story)

    print("\ndone -> train_response/%s" % new_subject)
    print("next steps:")
    print("  1. python train_EM.py --subject %s --gpt %s --sessions %s"
          % (new_subject, args.gpt, " ".join(map(str, args.sessions))))
    print("  2. mkdir -p models/%s && cp models/%s/word_rate_model_*.npz models/%s/"
          % (new_subject, args.subject, new_subject))
    print("  3. ln -s %s data_test/test_response/%s   # decode on CLEAN test data"
          % (args.subject, new_subject))
    print("  4. python run_decoder.py --subject %s --experiment perceived_speech --task <task>"
          % new_subject)
