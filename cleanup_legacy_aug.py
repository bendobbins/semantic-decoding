"""THROWAWAY: remove artifacts left by the OLD make_augmented_stories.py.

The previous version scattered generated files through the shared data tree and mutated
data_train/respdict.json in place. Deleting the files is not enough on its own -- the
respdict entries survive and would keep pointing at TextGrids that no longer exist -- so
this also strips those keys and leaves a backup.

Once the tree is clean, delete this script. The current pipeline confines every run to
data_train/aug/... and data_train/train_response/aug/..., which decoding/clean_aug.py
removes; nothing here applies to it.

Dry run unless --yes is given:

    python cleanup_legacy_aug.py                 # show what would go
    python cleanup_legacy_aug.py --yes           # remove data artifacts
    python cleanup_legacy_aug.py --yes --models  # also remove aug model/result/score dirs
"""

import os
import re
import sys
import glob
import json
import shutil
import argparse

# this script lives at the repo root, so decoding/ is not on the path by default
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "decoding"))
import config

TRAIN = config.DATA_TRAIN_DIR
TEST = config.DATA_TEST_DIR

# old augmented story names were "<base_story>__<engine>_v<k>"; no real story name
# contains "__", so this cannot match a base story
AUG_STORY_RE = re.compile(r"^.+__.+_v\d+$")


def rel(path):
    return os.path.relpath(path, config.REPO_DIR)


def collect_data_targets():
    """Files and directories the old pipeline wrote into the shared data tree."""
    t = []
    t += sorted(glob.glob(os.path.join(TRAIN, "train_stimulus", "*__*_v*.TextGrid")))
    t += sorted(glob.glob(os.path.join(TRAIN, "aug_manifest_*.json")))
    for name in ("augmentation_report.csv",):
        p = os.path.join(TRAIN, name)
        if os.path.exists(p):
            t.append(p)
    for pat in ("*_aug_*", "*_base"):
        t += sorted(glob.glob(os.path.join(TRAIN, "train_response", pat)))
        t += sorted(glob.glob(os.path.join(TEST, "test_response", pat)))
    return t


def collect_model_targets():
    """Old per-condition model/result/score dirs, under any *models* tree."""
    roots = sorted(set(glob.glob(os.path.join(config.REPO_DIR, "*models*"))
                       + glob.glob(os.path.join(config.RESULT_DIR, "*"))
                       + glob.glob(os.path.join(config.REPO_DIR, "scores", "*"))))
    t = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for pat in ("*_aug_*", "*_base"):
            t += sorted(p for p in glob.glob(os.path.join(root, pat)) if os.path.isdir(p))
    return sorted(set(t))


def stale_respdict_keys(respdict):
    return sorted(k for k in respdict if AUG_STORY_RE.match(k))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true",
                        help="also remove <*models*>/<cond>_aug_*, results/ and scores/ dirs "
                             "for augmented conditions (off by default -- you delete models "
                             "yourself)")
    parser.add_argument("--llm-cache", dest="llm_cache", action="store_true",
                        help="also delete data_train/aug_llm_cache.json (kept by default: "
                             "rebuilding it re-spends API budget, and it is safe to reuse)")
    parser.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    args = parser.parse_args()

    targets = collect_data_targets()
    if args.llm_cache:
        p = os.path.join(TRAIN, "aug_llm_cache.json")
        if os.path.exists(p):
            targets.append(p)
    model_targets = collect_model_targets()

    respdict_path = os.path.join(TRAIN, "respdict.json")
    with open(respdict_path) as f:
        respdict = json.load(f)
    stale = stale_respdict_keys(respdict)

    print("data artifacts to remove: %d" % len(targets))
    for p in targets:
        print("   %s%s" % (rel(p), "/" if os.path.isdir(p) else ""))

    print("\nmodel/result/score dirs for augmented conditions: %d %s"
          % (len(model_targets), "(will remove)" if args.models else "(kept -- pass --models)"))
    for p in model_targets:
        print("   %s/" % rel(p))

    print("\nrespdict.json: %d base entries, %d stale augmented entries to strip"
          % (len(respdict) - len(stale), len(stale)))
    for k in stale[:10]:
        print("   %s" % k)
    if len(stale) > 10:
        print("   ... and %d more" % (len(stale) - 10))

    if args.models:
        targets += model_targets

    if not args.yes:
        print("\n[dry run] nothing changed. re-run with --yes to apply.")
        raise SystemExit(0)

    for p in targets:
        if os.path.islink(p) or os.path.isfile(p):
            os.unlink(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)

    if stale:
        shutil.copyfile(respdict_path, respdict_path + ".bak")
        for k in stale:
            respdict.pop(k)
        with open(respdict_path, "w") as f:
            json.dump(respdict, f, indent=0)
        print("wrote %s (backup at %s)" % (rel(respdict_path), rel(respdict_path) + ".bak"))

    print("removed %d path(s); respdict.json now has %d entries." % (len(targets), len(respdict)))
