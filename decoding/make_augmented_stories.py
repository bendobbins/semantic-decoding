"""Generate synonym-augmented copies of sessions-2&4 stories for the encoding-model
bake-off, one condition per substitution engine.

For each base story and each of N variants, a fixed set of content-word slots is
chosen once (seeded) and every requested engine fills those slots (matched dose, so
conditions differ only in *which* synonym each engine picks). For every engine this
writes:

    data_train/train_stimulus/<story>__<engine>_v<k>.TextGrid   (times identical to base)
    data_train/aug_manifest_<engine>.json                       ({aug_story: base_story})
    data_train/respdict.json                                     (+ aug_story -> base TR count)
    data_train/train_response/<subject>_aug_<engine>/           (symlinks: originals + aug->base)
    data_test/test_response/<subject>_aug_<engine> -> <subject>  (decode on clean test data)
    models/<subject>_aug_<engine>/word_rate_model_*.npz         (copied; WR is stimulus-agnostic)

It also builds a <subject>_base condition (response symlinks, no augmentation) so the
comparison is against a freshly-fit no-aug baseline rather than the paper's model.

Modeled on add_train_voxel_noise.py. Reuses only data files + a small train_EM hook;
nothing here modifies the original repo tree (all paths resolve inside the copy).
"""

import os
import csv
import glob
import json
import random
import shutil
import argparse

import config
import utils_augment as ua

TRAIN = config.DATA_TRAIN_DIR
STIM_DIR = os.path.join(TRAIN, "train_stimulus")
RESP_DIR = os.path.join(TRAIN, "train_response")
TEST_RESP_DIR = os.path.join(config.DATA_TEST_DIR, "test_response")


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {} if default is None else default


def _save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=0)


def _manifest_path(engine):
    return os.path.join(TRAIN, "aug_manifest_%s.json" % engine)


def clean_engine(engine, subject, respdict):
    """Remove all artifacts from a prior run of `engine` (for --force)."""
    for tg in glob.glob(os.path.join(STIM_DIR, "*__%s_v*.TextGrid" % engine)):
        os.remove(tg)
    manifest = _manifest_path(engine)
    if os.path.exists(manifest):
        for aug in _load_json(manifest):
            respdict.pop(aug, None)
        os.remove(manifest)
    for path in (os.path.join(RESP_DIR, "%s_aug_%s" % (subject, engine)),
                 os.path.join(config.MODEL_DIR, "%s_aug_%s" % (subject, engine))):
        if os.path.isdir(path):
            shutil.rmtree(path)
    tr = os.path.join(TEST_RESP_DIR, "%s_aug_%s" % (subject, engine))
    if os.path.islink(tr):
        os.unlink(tr)


def build_response_condition(subject, cond_name, aug_to_base):
    """Create train_response/<cond_name>/ with symlinks to all of the subject's
    training responses plus <aug>.hf5 -> <base>.hf5, then wire test_response + WR."""
    real = os.path.join(RESP_DIR, subject)
    pseudo = os.path.join(RESP_DIR, cond_name)
    os.makedirs(pseudo)
    for fn in os.listdir(real):                       # follows subject symlink -> originals
        if fn.endswith(".hf5"):
            os.symlink(os.path.join("..", subject, fn), os.path.join(pseudo, fn))
    for aug, base in aug_to_base.items():
        os.symlink(os.path.join("..", subject, base + ".hf5"),
                   os.path.join(pseudo, aug + ".hf5"))

    tr = os.path.join(TEST_RESP_DIR, cond_name)       # decode on clean test data
    if not os.path.exists(tr):
        os.symlink(subject, tr)

    mdir = os.path.join(config.MODEL_DIR, cond_name)  # WR model is stimulus-agnostic -> reuse
    os.makedirs(mdir, exist_ok=True)
    for wr in glob.glob(os.path.join(config.MODEL_DIR, subject, "word_rate_model_*.npz")):
        shutil.copyfile(wr, os.path.join(mdir, os.path.basename(wr)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--engines", nargs="+", default=["wordnet", "embedding", "llm", "random"])
    parser.add_argument("--sessions", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--stories", nargs="+", default=None,
                        help="override: augment these base stories instead of --sessions")
    parser.add_argument("--n_variants", type=int, default=3)
    parser.add_argument("--swap_rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpt", type=str, default="perceived")
    parser.add_argument("--llm_model", type=str, default="claude-opus-4-8")
    parser.add_argument("--no_matched", action="store_true",
                        help="disable matched-dose intersection (each engine fills all it can)")
    parser.add_argument("--no_baseline", action="store_true",
                        help="skip building the <subject>_base no-aug control condition")
    parser.add_argument("--dry_run", action="store_true",
                        help="report fill rates only; write nothing")
    parser.add_argument("--force", action="store_true",
                        help="remove prior artifacts for the requested engines first")
    args = parser.parse_args()

    matched = not args.no_matched
    ua.ensure_nltk()
    vocab, vocab_set = ua.load_vocab(args.gpt)

    # base stories
    if args.stories:
        base_stories = list(args.stories)
    else:
        sess_to_story = _load_json(os.path.join(TRAIN, "sess_to_story.json"))
        base_stories = []
        for s in args.sessions:
            base_stories.extend(sess_to_story[str(s)])
    print("augmenting %d base stories with engines=%s (matched=%s, N=%d, rate=%.2f)"
          % (len(base_stories), args.engines, matched, args.n_variants, args.swap_rate))

    # heavy: GPT only if the embedding engine is requested
    gpt = None
    if "embedding" in args.engines:
        from GPT import GPT
        print("loading GPT for embedding engine (device=%s) ..." % config.GPT_DEVICE)
        gpt = GPT(path=os.path.join(config.DATA_LM_DIR, args.gpt, "model"),
                  vocab=vocab, device=config.GPT_DEVICE)
    engines = ua.build_engines(args.engines, vocab, vocab_set, gpt=gpt, llm_model=args.llm_model)

    respdict = _load_json(os.path.join(TRAIN, "respdict.json"))

    # existing-artifact guard
    if not args.dry_run:
        for name in args.engines:
            pdir = os.path.join(RESP_DIR, "%s_aug_%s" % (args.subject, name))
            if os.path.isdir(pdir) or os.path.exists(_manifest_path(name)):
                if not args.force:
                    raise SystemExit("artifacts exist for engine '%s'; use --force to regenerate" % name)
                clean_engine(name, args.subject, respdict)

    manifest_data = {name: {} for name in args.engines}
    report_rows = []

    for story in base_stories:
        recs = ua.story_word_records(story)
        n_elig = len(ua.eligible_slots(recs, vocab_set))
        for k in range(1, args.n_variants + 1):
            slot_rng = random.Random("%d|%s|%d" % (args.seed, story, k))
            slots = ua.select_slots(recs, vocab_set, args.swap_rate, slot_rng)
            rng_by = {name: random.Random("%d|%s|%d|%s" % (args.seed, story, k, name))
                      for name in args.engines}
            per = ua.build_variant(recs, slots, engines, rng_by, matched=matched)
            kept = len(next(iter(per.values()))) if matched and per else None
            counts = {name: len(per[name]) for name in args.engines}
            print("  %-24s v%d: elig=%d selected=%d  %s%s"
                  % (story, k, n_elig, len(slots),
                     " ".join("%s=%d" % (n, counts[n]) for n in args.engines),
                     ("  matched=%d" % kept) if kept is not None else ""))
            for name in args.engines:
                subs = per[name]
                aug = "%s__%s_v%d" % (story, name, k)
                if not args.dry_run and subs:
                    ua.write_augmented_textgrid(story, os.path.join(STIM_DIR, aug + ".TextGrid"),
                                                recs, subs)
                    manifest_data[name][aug] = story
                    respdict[aug] = respdict[story]
                for gw, new in subs.items():
                    report_rows.append({"engine": name, "base_story": story, "variant": k,
                                        "gw": gw, "raw_idx": recs[gw]["raw_idx"],
                                        "original": recs[gw]["word"], "replacement": new})

    if args.dry_run:
        print("\n[dry run] nothing written. total proposed substitutions: %d" % len(report_rows))
        raise SystemExit(0)

    # persist registries
    _save_json(os.path.join(TRAIN, "respdict.json"), respdict)
    for name in args.engines:
        _save_json(_manifest_path(name), manifest_data[name])
        build_response_condition(args.subject, "%s_aug_%s" % (args.subject, name), manifest_data[name])
    if not args.no_baseline:
        base_cond = "%s_base" % args.subject
        if not os.path.isdir(os.path.join(RESP_DIR, base_cond)):
            build_response_condition(args.subject, base_cond, {})
    if "llm" in engines:
        engines["llm"]._flush()

    report_path = os.path.join(TRAIN, "augmentation_report.csv")
    with open(report_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "base_story", "variant", "gw",
                                          "raw_idx", "original", "replacement"])
        w.writeheader()
        w.writerows(report_rows)

    print("\ndone. %d substitutions across %d engine(s). report -> %s"
          % (len(report_rows), len(args.engines), report_path))
    print("next steps:")
    if not args.no_baseline:
        print("  # controlled no-aug baseline (same training code):")
        print("  python decoding/train_EM.py --subject %s_base --gpt %s" % (args.subject, args.gpt))
    for name in args.engines:
        print("  python decoding/train_EM.py --subject %s_aug_%s --gpt %s --augment %s"
              % (args.subject, name, args.gpt, _manifest_path(name)))
    print("  # then run_decoder.py / evaluate_predictions.py per condition, and scores_to_csv.py")
