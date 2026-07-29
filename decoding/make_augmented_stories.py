"""Generate synonym-augmented copies of a session set's stories, one condition per
substitution engine.

For each base story and each of N variants, a fixed set of content-word slots is chosen
once (seeded) and every requested engine fills those slots (matched dose, so conditions
differ only in *which* synonym each engine picks).

Everything a run produces lives under one directory, keyed by dataset, subject and
augmentation parameters, so any number of runs can be generated and trained in parallel
without touching each other or the base data:

    data/data_train/aug/<dataset_tag>/<subject>/<aug_tag>/
        run.json                    every parameter + resolved path for this run
        aug_manifest_<engine>.json  run header + {aug_story: base_story}
        respdict_aug.json           {aug_story: n_trs}  (base respdict is never written)
        augmentation_report.csv     every substitution made
        train_stimulus/             <story>__<engine>_v<k>.TextGrid

    data/data_train/train_response/aug/<dataset_tag>/<subject>/<aug_tag>/
        <subject>_aug_<engine>/     symlinks: originals + aug -> base

where aug_tag is "<n_variants>var-<swap_rate>swap-<seed>seed" and dataset_tag defaults to
--model_dir. Models, results and scores land under <dataset_tag>/<condition>/<aug_tag>/ by
passing "--model_dir <dataset_tag> --aug_tag <aug_tag>" to the downstream scripts.

Word rate models are NOT produced here: an augmented story reuses its base story's
responses and keeps its interval times, so words-per-TR is unchanged and the subject's
existing model at <dataset_tag>/<subject>/ applies as-is. Its path is recorded in the
manifest and run_decoder.py picks it up from there.
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
BASE_STIM_DIR = os.path.join(TRAIN, "train_stimulus")
BASE_RESP_DIR = os.path.join(TRAIN, "train_response")


def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {} if default is None else default


def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def make_aug_tag(n_variants, swap_rate, seed):
    return "%dvar-%gswap-%dseed" % (n_variants, swap_rate, seed)


def run_paths(dataset_tag, subject, aug_tag):
    """Every directory a run owns. Nothing outside these is ever written."""
    rel = os.path.join(dataset_tag, subject, aug_tag)
    run_dir = os.path.join(TRAIN, "aug", rel)
    return {
        "rel": rel,
        "run_dir": run_dir,
        "stim_dir": os.path.join(run_dir, "train_stimulus"),
        "resp_root": os.path.join(BASE_RESP_DIR, "aug", rel),
    }


def manifest_path(run_dir, engine):
    return os.path.join(run_dir, "aug_manifest_%s.json" % engine)


def clean_engine(engine, paths, subject):
    """Remove one engine's artifacts from THIS run only (for --force).

    Scoped to the run directory on purpose: the previous version globbed the shared
    stimulus tree, so regenerating one subject deleted the stories another subject's
    trained model referred to.
    """
    for tg in glob.glob(os.path.join(paths["stim_dir"], "*__%s_v*.TextGrid" % engine)):
        os.remove(tg)
    mpath = manifest_path(paths["run_dir"], engine)
    if os.path.exists(mpath):
        os.remove(mpath)
    cond = os.path.join(paths["resp_root"], "%s_aug_%s" % (subject, engine))
    if os.path.isdir(cond):
        shutil.rmtree(cond)


def build_response_condition(subject, cond_name, aug_to_base, resp_root):
    """Create <resp_root>/<cond_name>/ with links to the subject's real responses plus
    <aug>.hf5 -> <base>.hf5. Targets are relative so the tree stays portable and the
    links resolve at whatever depth the run directory sits."""
    real = os.path.join(BASE_RESP_DIR, subject)
    pseudo = os.path.join(resp_root, cond_name)
    os.makedirs(pseudo, exist_ok=True)

    def link(target, name):
        dst = os.path.join(pseudo, name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.unlink(dst)
        os.symlink(os.path.relpath(target, pseudo), dst)

    for fn in sorted(os.listdir(real)):          # follows a symlinked subject dir
        if fn.endswith(".hf5"):
            link(os.path.join(real, fn), fn)
    for aug, base in sorted(aug_to_base.items()):
        link(os.path.join(real, base + ".hf5"), aug + ".hf5")
    return pseudo


def merge_report(path, rows, engines):
    """Rewrite the run report, replacing only the engines regenerated this invocation."""
    fields = ["engine", "base_story", "variant", "gw", "raw_idx", "original", "replacement"]
    keep = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            keep = [r for r in csv.DictReader(f) if r.get("engine") not in engines]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
        w.writerows(rows)


def rebuild_respdict(run_dir, base_respdict):
    """Derive respdict_aug.json from every manifest in the run, so it always matches
    what is actually on disk regardless of which engines were regenerated."""
    out = {}
    for mpath in sorted(glob.glob(os.path.join(run_dir, "aug_manifest_*.json"))):
        for aug, base in _load_json(mpath).get("stories", {}).items():
            out[aug] = base_respdict[base]
    _save_json(os.path.join(run_dir, "respdict_aug.json"), out)
    return out


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
    parser.add_argument("--model_dir", type=str, default="models",
                        help="dataset-level model dir, e.g. 2hr-dataset-models; also the "
                             "default dataset_tag and where the word rate model is read from")
    parser.add_argument("--dataset_tag", type=str, default=None,
                        help="override the dataset directory name (defaults to --model_dir)")
    parser.add_argument("--aug_tag", type=str, default=None,
                        help="override the auto-generated <N>var-<R>swap-<S>seed tag")
    parser.add_argument("--llm_model", type=str, default="claude-opus-4-8")
    parser.add_argument("--llm_cache", type=str, default=None,
                        help="shared LLM synonym cache (default data_train/aug_llm_cache.json)")
    parser.add_argument("--no_matched", action="store_true",
                        help="disable matched-dose intersection (each engine fills all it can)")
    parser.add_argument("--baseline", action="store_true",
                        help="also build a <subject>_base no-aug condition. Off by default: it "
                             "is identical to training <subject> on the same sessions, which "
                             "you already have at <dataset_tag>/<subject>/")
    parser.add_argument("--dry_run", action="store_true",
                        help="report fill rates and resolved paths only; write nothing")
    parser.add_argument("--force", action="store_true",
                        help="remove this run's prior artifacts for the requested engines first")
    args = parser.parse_args()

    matched = not args.no_matched
    dataset_tag = (args.dataset_tag or args.model_dir).strip("/")
    aug_tag = args.aug_tag or make_aug_tag(args.n_variants, args.swap_rate, args.seed)
    paths = run_paths(dataset_tag, args.subject, aug_tag)
    model_root = os.path.join(config.REPO_DIR, args.model_dir)
    wr_dir = os.path.join(model_root, args.subject)

    print("run: %s" % paths["rel"])
    print("  data     -> %s" % os.path.relpath(paths["run_dir"], config.REPO_DIR))
    print("  response -> %s" % os.path.relpath(paths["resp_root"], config.REPO_DIR))
    print("  wordrate <- %s" % os.path.relpath(wr_dir, config.REPO_DIR))

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
    print("augmenting %d base stories with engines=%s (matched=%s, N=%d, rate=%.2f, seed=%d)"
          % (len(base_stories), args.engines, matched, args.n_variants, args.swap_rate, args.seed))

    if not glob.glob(os.path.join(wr_dir, "word_rate_model_*.npz")):
        print("  WARNING: no word_rate_model_*.npz in %s -- run train_WR.py --subject %s "
              "--model_dir %s --sessions %s before decoding"
              % (wr_dir, args.subject, args.model_dir,
                 " ".join(str(s) for s in args.sessions)))

    # existing-artifact guard, scoped to this run
    if not args.dry_run:
        clashes = [e for e in args.engines if os.path.exists(manifest_path(paths["run_dir"], e))]
        if clashes:
            if not args.force:
                raise SystemExit(
                    "artifacts already exist in this run for engine(s) %s.\n"
                    "  run dir: %s\n"
                    "  pass --force to regenerate them (only this run is touched), or use a "
                    "different --seed/--aug_tag." % (", ".join(clashes), paths["run_dir"]))
            for e in clashes:
                clean_engine(e, paths, args.subject)
        os.makedirs(paths["stim_dir"], exist_ok=True)
        os.makedirs(paths["resp_root"], exist_ok=True)

    # heavy: GPT only if the embedding engine is requested
    gpt = None
    if "embedding" in args.engines:
        from GPT import GPT
        print("loading GPT for embedding engine (device=%s) ..." % config.GPT_DEVICE)
        gpt = GPT(path=os.path.join(config.DATA_LM_DIR, args.gpt, "model"),
                  vocab=vocab, device=config.GPT_DEVICE)
    engines = ua.build_engines(args.engines, vocab, vocab_set, gpt=gpt, llm_model=args.llm_model)
    if "llm" in engines and args.llm_cache:
        engines["llm"].cache_path = args.llm_cache

    base_respdict = _load_json(os.path.join(TRAIN, "respdict.json"))
    manifest_data = {name: {} for name in args.engines}
    report_rows = []

    for story in base_stories:
        recs = ua.story_word_records(story)
        n_elig = len(ua.eligible_slots(recs, vocab_set))
        for k in range(1, args.n_variants + 1):
            # every parameter that changes the draw is in the key, so two runs can never
            # share a random stream by accident
            slot_key = "%d|%s|%d|%g" % (args.seed, story, k, args.swap_rate)
            slots = ua.select_slots(recs, vocab_set, args.swap_rate, random.Random(slot_key))
            rng_by = {name: random.Random("%s|%s" % (slot_key, name)) for name in args.engines}
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
                    ua.write_augmented_textgrid(
                        story, os.path.join(paths["stim_dir"], aug + ".TextGrid"), recs, subs)
                    manifest_data[name][aug] = story
                for gw, new in subs.items():
                    report_rows.append({"engine": name, "base_story": story, "variant": k,
                                        "gw": gw, "raw_idx": recs[gw]["raw_idx"],
                                        "original": recs[gw]["word"], "replacement": new})

    if args.dry_run:
        print("\n[dry run] nothing written. total proposed substitutions: %d" % len(report_rows))
        raise SystemExit(0)

    # run descriptor, shared by every manifest so train_EM has a single source of truth
    header = {
        "subject": args.subject,
        "dataset_tag": dataset_tag,
        "aug_tag": aug_tag,
        "model_dir": args.model_dir,
        "sessions": args.sessions if not args.stories else None,
        "base_stories": base_stories,
        "n_variants": args.n_variants,
        "swap_rate": args.swap_rate,
        "seed": args.seed,
        "matched": matched,
        "gpt": args.gpt,
        "engines": args.engines,
        # repo-relative so the tree can move between machines
        "stim_dir": os.path.relpath(paths["stim_dir"], config.REPO_DIR),
        "respdict": os.path.relpath(os.path.join(paths["run_dir"], "respdict_aug.json"),
                                    config.REPO_DIR),
        "resp_root": os.path.relpath(paths["resp_root"], config.REPO_DIR),
        "word_rate_dir": os.path.relpath(wr_dir, config.REPO_DIR),
    }

    conditions = {}
    for name in args.engines:
        cond = "%s_aug_%s" % (args.subject, name)
        _save_json(manifest_path(paths["run_dir"], name),
                   dict(header, engine=name, condition=cond, stories=manifest_data[name]))
        build_response_condition(args.subject, cond, manifest_data[name], paths["resp_root"])
        conditions[name] = cond
    if args.baseline:
        build_response_condition(args.subject, "%s_base" % args.subject, {}, paths["resp_root"])
        conditions["base"] = "%s_base" % args.subject

    rebuild_respdict(paths["run_dir"], base_respdict)
    merge_report(os.path.join(paths["run_dir"], "augmentation_report.csv"),
                 report_rows, set(args.engines))
    _save_json(os.path.join(paths["run_dir"], "run.json"), dict(header, conditions=conditions))
    if "llm" in engines:
        engines["llm"]._flush()

    print("\ndone. %d substitutions across %d engine(s) -> %s"
          % (len(report_rows), len(args.engines),
             os.path.relpath(paths["run_dir"], config.REPO_DIR)))
    print("next steps:")
    for name in args.engines:
        print("  python decoding/train_EM.py --subject %s --gpt %s --sessions %s \\\n"
              "      --model_dir %s --aug_tag %s \\\n"
              "      --augment %s"
              % (conditions[name], args.gpt, " ".join(str(s) for s in args.sessions),
                 args.model_dir, aug_tag,
                 os.path.relpath(manifest_path(paths["run_dir"], name), config.REPO_DIR)))
    print("  # then run_decoder.py / evaluate_predictions.py with the same "
          "--model_dir/--aug_tag, and scores_to_csv.py")
