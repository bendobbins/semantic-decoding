"""Delete artifacts written by make_augmented_stories.py.

Everything a run creates lives under two roots, so cleanup is just removing subtrees:

    data/data_train/aug/<dataset_tag>/<subject>/<aug_tag>/
    data/data_train/train_response/aug/<dataset_tag>/<subject>/<aug_tag>/

Nothing outside those is written by a run, so there is no shared registry to repair --
in particular data_train/respdict.json is never modified.

Select runs with any combination of --dataset_tag / --subject / --aug_tag (each defaults
to "all"), or pass --all to match everything. Dry run unless --yes is given.

    python decoding/clean_aug.py --dataset_tag 2hr-dataset-models --aug_tag 3var-0.2swap-0seed
    python decoding/clean_aug.py --all --yes
"""

import os
import glob
import json
import shutil
import argparse

import config

TRAIN = config.DATA_TRAIN_DIR
AUG_DATA_ROOT = os.path.join(TRAIN, "aug")
AUG_RESP_ROOT = os.path.join(TRAIN, "train_response", "aug")


def rel(path):
    return os.path.relpath(path, config.REPO_DIR)


def find_runs(dataset_tag, subject, aug_tag):
    """Run directories matching the selector, as (data_dir, resp_dir, rel_key) triples."""
    pattern = os.path.join(dataset_tag or "*", subject or "*", aug_tag or "*")
    out = []
    for data_dir in sorted(glob.glob(os.path.join(AUG_DATA_ROOT, pattern))):
        key = os.path.relpath(data_dir, AUG_DATA_ROOT)
        out.append((data_dir, os.path.join(AUG_RESP_ROOT, key), key))
    # a response tree can outlive its data dir if a previous cleanup was interrupted
    for resp_dir in sorted(glob.glob(os.path.join(AUG_RESP_ROOT, pattern))):
        key = os.path.relpath(resp_dir, AUG_RESP_ROOT)
        if not any(k == key for _, _, k in out):
            out.append((os.path.join(AUG_DATA_ROOT, key), resp_dir, key))
    return out


def derived_dirs(data_dir, key):
    """Model / result / score directories this run's conditions produced, from run.json."""
    run_json = os.path.join(data_dir, "run.json")
    if not os.path.exists(run_json):
        return []
    try:
        with open(run_json) as f:
            run = json.load(f)
    except (ValueError, OSError):
        return []
    model_dir, aug_tag = run.get("model_dir"), run.get("aug_tag")
    if not model_dir or not aug_tag:
        return []
    out = []
    for cond in run.get("conditions", {}).values():
        out.append(os.path.join(config.REPO_DIR, model_dir, cond, aug_tag))
        out.append(os.path.join(config.RESULT_DIR, model_dir, cond, aug_tag))
        out.append(os.path.join(config.REPO_DIR, "scores", model_dir, cond, aug_tag))
    return [p for p in out if os.path.isdir(p)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_tag", type=str, default=None, help="e.g. 2hr-dataset-models")
    parser.add_argument("--subject", type=str, default=None, help="e.g. S1")
    parser.add_argument("--aug_tag", type=str, default=None, help="e.g. 3var-0.2swap-0seed")
    parser.add_argument("--all", action="store_true", help="match every run")
    parser.add_argument("--models", action="store_true",
                        help="also delete the model/result/score dirs these runs produced "
                             "(listed either way; not deleted without this flag)")
    parser.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    args = parser.parse_args()

    if not (args.all or args.dataset_tag or args.subject or args.aug_tag):
        raise SystemExit("refusing to run without a selector; pass --all to match every run")

    runs = find_runs(args.dataset_tag, args.subject, args.aug_tag)
    if not runs:
        print("no augmentation runs matched.")
        raise SystemExit(0)

    to_delete, derived_all = [], []
    print("matched %d run(s):" % len(runs))
    for data_dir, resp_dir, key in runs:
        print("\n  %s" % key)
        for p in (data_dir, resp_dir):
            if os.path.isdir(p):
                print("    delete  %s" % rel(p))
                to_delete.append(p)
        d = derived_dirs(data_dir, key)
        derived_all.extend(d)
        for p in d:
            print("    %s  %s" % ("delete " if args.models else "keep   ", rel(p)))

    if args.models:
        to_delete.extend(derived_all)
    elif derived_all:
        print("\n%d model/result/score dir(s) left in place; pass --models to remove them too"
              % len(derived_all))

    if not args.yes:
        print("\n[dry run] nothing deleted. re-run with --yes to remove %d director%s."
              % (len(to_delete), "y" if len(to_delete) == 1 else "ies"))
        raise SystemExit(0)

    for p in to_delete:
        shutil.rmtree(p, ignore_errors=True)
    for root in (AUG_DATA_ROOT, AUG_RESP_ROOT):          # prune emptied parents
        if not os.path.isdir(root):
            continue
        # os.walk snapshots dirnames up front, so re-check the filesystem instead;
        # bottom-up order means a parent is already empty by the time it is visited
        for dirpath, _, _ in os.walk(root, topdown=False):
            try:
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)                    # includes the root once emptied
            except OSError:
                pass
    print("\ndeleted %d director%s." % (len(to_delete), "y" if len(to_delete) == 1 else "ies"))
