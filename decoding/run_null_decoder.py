"""Generate null decoder runs: the beam search run_decoder.py performs, but with
the encoding model's brain likelihoods replaced by random values, so the word
sequence is driven by the language model alone.

evaluate_predictions.py already builds sequences like these internally for its
--null flag, but it keeps only the z-scores it derives from them -- the sequences
and their raw metric scores are discarded. This writes them out as ordinary
prediction files instead, so a null can be scored by evaluate_predictions.py
exactly like a real run and the two sets of raw scores compared directly.

Nulls are saved beside the run they baseline, with a _null<NN> suffix:

    results/<model_dir>/<subject>/<experiment>/wheretheressmoke_null00.npz

evaluate_predictions.py and utils_eval.load_transcript both split the task name
on "_" before looking up the eval segment and the reference transcript, so the
suffix needs no change on the evaluation side:

    python evaluate_predictions.py --subject S1 --experiment perceived_speech \
        --task wheretheressmoke_null00 --model_dir <model_dir> --null 2

That call's own --null only feeds its z-score denominator, which is meaningless
for a null run; 2 is the smallest value that avoids a divide-by-zero. The raw
scores it saves are the ones to compare.

Word times are copied from the matching real run by default. Holding them fixed
is what makes the comparison clean: the null then differs from the run only in
which words the search chose, not in how many it produced or where they fell
relative to the evaluation windows. Use --times predict when no real run exists
yet -- times then come from the word rate model, which predicts how many words
each TR contains and carries no information about which words they are.

Usage:
    python run_null_decoder.py --subject S1 --experiment perceived_speech \
        --task wheretheressmoke --model_dir 2hr-dataset-models --nulls 10

Each null is one beam search at width 2 * config.EXTENSIONS, far narrower than
the real decoder's config.WIDTH, so ten of them cost about what one
evaluate_predictions.py call at --null 10 already costs.
"""

import os
import time
import argparse
import numpy as np
import h5py

import config
from utils_eval import generate_null
from utils_stim import predict_word_rate, predict_word_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type = str, required = True)
    parser.add_argument("--experiment", type = str, required = True)
    parser.add_argument("--task", type = str, required = True)
    parser.add_argument("--model_dir", type = str, default = "models")
    parser.add_argument("--aug_tag", type = str, default = None,
        help = "augmentation run tag, e.g. 3var-0.2swap-0seed; must match what run_decoder.py "
               "was given so the null lands beside the run it baselines")
    parser.add_argument("--nulls", type = int, default = 10,
        help = "number of null runs to generate")
    parser.add_argument("--start_index", type = int, default = 0,
        help = "index of the first null, for fanning generation out over an array job")
    parser.add_argument("--seed", type = int, default = 42,
        help = "null i uses seed [seed] + i, so nulls split across array tasks reproduce "
               "the sequences a single batched call would have produced")
    parser.add_argument("--times", type = str, default = "run", choices = ["run", "predict"],
        help = "where word times come from: copied from the real run (default), or predicted "
               "from the response with the word rate model")
    parser.add_argument("--times_run", type = str, default = None,
        help = "path of the .npz to copy word times from; defaults to the real run for this "
               "subject and task. implies --times run")
    parser.add_argument("--word_rate_dir", type = str, default = None,
        help = "repo-relative dir holding word_rate_model_*.npz, for --times predict")
    parser.add_argument("--overwrite", action = "store_true",
        help = "regenerate nulls whose output file already exists")
    args = parser.parse_args()
    rel_subject = os.path.join(args.subject, args.aug_tag) if args.aug_tag else args.subject
    if args.times_run: args.times = "run"

    # determine GPT checkpoint based on experiment
    if args.experiment in ["imagined_speech"]: gpt_checkpoint = "imagined"
    else: gpt_checkpoint = "perceived"

    save_location = os.path.join(config.RESULT_DIR, args.model_dir, rel_subject, args.experiment)

    # word times. evaluate_predictions.py segments predictions into windows by absolute
    # time, so a null has to sit on the same time grid as the run it is compared against.
    if args.times == "run":
        times_path = args.times_run or os.path.join(save_location, args.task + ".npz")
        if not os.path.exists(times_path):
            raise SystemExit("no run to take word times from at %s\n"
                             "decode this task first, or pass --times predict to derive them "
                             "from the word rate model" % times_path)
        word_times = np.load(times_path)["times"]
        print("word times: %d words from %s" % (len(word_times), times_path), flush = True)
    else:
        # determine word rate model voxels based on experiment
        if args.experiment in ["imagined_speech", "perceived_movies"]: word_rate_voxels = "speech"
        else: word_rate_voxels = "auditory"

        subject = os.path.basename(args.subject).split("_")[0]
        hf = h5py.File(os.path.join(config.DATA_TEST_DIR, "test_response", subject,
                                    args.experiment, args.task + ".hf5"), "r")
        resp = np.nan_to_num(hf["data"][:])
        hf.close()

        # same lookup order run_decoder.py uses: the model dir, then an explicit override,
        # then the path the encoding model recorded at training time. that last candidate is
        # the only thing read out of the encoding model, and only to locate a file -- none of
        # its weights are loaded, which is the whole point of a null run.
        wr_name = "word_rate_model_%s.npz" % word_rate_voxels
        load_location = os.path.join(config.REPO_DIR, args.model_dir, rel_subject)
        wr_candidates = [os.path.join(load_location, wr_name)]
        if args.word_rate_dir:
            wr_candidates.append(os.path.join(config.REPO_DIR, args.word_rate_dir, wr_name))
        em_path = os.path.join(load_location, "encoding_model_%s.npz" % gpt_checkpoint)
        if os.path.exists(em_path):
            encoding_model = np.load(em_path)
            if "word_rate_dir" in encoding_model.files:
                recorded = str(encoding_model["word_rate_dir"])
                if recorded:
                    wr_candidates.append(os.path.join(config.REPO_DIR, recorded, wr_name))
        wr_path = next((p for p in wr_candidates if os.path.exists(p)), None)
        if wr_path is None:
            raise SystemExit("no %s found; searched:\n  %s" % (wr_name, "\n  ".join(wr_candidates)))
        word_rate_model = np.load(wr_path, allow_pickle = True)

        word_rate = predict_word_rate(resp, word_rate_model["weights"],
                                      word_rate_model["voxels"], word_rate_model["mean_rate"])
        if args.experiment == "perceived_speech":
            word_times, _ = predict_word_times(word_rate, resp, starttime = -10)
        else:
            word_times, _ = predict_word_times(word_rate, resp, starttime = 0)
        # run_decoder.py applies this shift after decoding, so a saved run for these
        # experiments already carries it; match it or the eval windows will not line up
        if args.experiment in ["perceived_movie", "perceived_multispeaker"]: word_times += 10
        print("word times: %d words predicted from %s" % (len(word_times), wr_path), flush = True)

    # generate nulls. one per generate_null call so each gets its own seed and is saved as
    # it finishes; the repeated GPT load that costs is small next to the search it wraps.
    os.makedirs(save_location, exist_ok = True)
    written = []
    for offset in range(args.nulls):
        index = args.start_index + offset
        out_path = os.path.join(save_location, "%s_null%02d" % (args.task, index))
        if os.path.exists(out_path + ".npz") and not args.overwrite:
            print("[%d/%d] %s.npz exists, skipping" % (offset + 1, args.nulls,
                                                       os.path.basename(out_path)), flush = True)
            continue
        np.random.seed(args.seed + index)
        start = time.time()
        null_words = generate_null(word_times, gpt_checkpoint, 1)[0]
        np.savez(out_path, words = np.array(null_words), times = np.array(word_times))
        written.append(os.path.basename(out_path))
        print("[%d/%d] wrote %s.npz  (%d words, %.1f min)" % (
            offset + 1, args.nulls, os.path.basename(out_path), len(null_words),
            (time.time() - start) / 60), flush = True)

    aug = " --aug_tag %s" % args.aug_tag if args.aug_tag else ""
    print("\n%d null runs written to %s" % (len(written), save_location))
    print("score them with, for each <run>:\n"
          "  python evaluate_predictions.py --subject %s --experiment %s --model_dir %s%s "
          "--task <run> --null 2" % (args.subject, args.experiment, args.model_dir, aug))
