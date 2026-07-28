"""Substitution engines for synonym-based data augmentation of the encoding-model
training set (sessions 2 & 4 bake-off).

Each engine proposes a single in-vocab replacement for one content word, keeping
the stimulus 1:1 with the original so the augmented "story" reuses the original
neural response bit-for-bit (see make_augmented_stories.py). Four engines share a
common interface so they can be compared head-to-head at matched dose:

    WordNetEngine    - same-POS WordNet lemmas, vocab-filtered (offline, deterministic-ish)
    EmbeddingEngine  - nearest in-vocab words in the model's own input-embedding space
    LLMEngine        - Claude proposes one in-context synonym (disk-cached)
    RandomEngine     - random in-vocab word of the same coarse POS (semantic control)

The critical constraint everywhere: the encoding model's GPT is word-level with a
fixed 17,379-word vocab (data_lm/perceived/vocab.json); anything outside it maps to
<unk>. Every proposal is filtered against that vocab.
"""

import os
import re
import json
import hashlib

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer

import config
from utils_ridge.textgrid import TextGrid
from utils_ridge.dsutils import DEFAULT_BAD_WORDS

_STEM = PorterStemmer()

# Penn Treebank prefixes for open-class (content) words, and proper-noun tags to skip.
_CONTENT_PREFIXES = ("NN", "VB", "JJ", "RB")
_PROPER_TAGS = ("NNP", "NNPS")


# ----------------------------------------------------------------------------- #
# vocab + nltk setup
# ----------------------------------------------------------------------------- #

def load_vocab(gpt_checkpoint="perceived"):
    """Return (vocab_list, vocab_set) for the encoding model's word-level GPT."""
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json")) as f:
        vocab = json.load(f)
    return vocab, set(vocab)


def ensure_nltk():
    """Download the corpora the engines need if they are not already present."""
    needed = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, pkg in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def _penn_to_wn(tag):
    if tag.startswith("N"):
        return wn.NOUN
    if tag.startswith("V"):
        return wn.VERB
    if tag.startswith("J"):
        return wn.ADJ
    if tag.startswith("R"):
        return wn.ADV
    return None


def _wn_pos_set(word):
    """Coarse WordNet POS tags a word can take ('n'/'v'/'a'/'r'); 's' folded to 'a')."""
    return {("a" if s.pos() == "s" else s.pos()) for s in wn.synsets(word)}


def _same_stem(a, b):
    return _STEM.stem(a) == _STEM.stem(b)


# ----------------------------------------------------------------------------- #
# TextGrid -> POS-tagged good-word records (with a map back to raw intervals)
# ----------------------------------------------------------------------------- #

def story_word_records(story):
    """Load a training story's word tier and tag it.

    Returns a list of records, one per GOOD word (bad words like sp/{NS} filtered
    exactly as dsutils.make_word_ds does), each carrying its index into the raw
    word-tier intervals so write_augmented_textgrid can target the right line:

        {gw, raw_idx, word, penn, wn_pos, is_content}
    """
    path = os.path.join(config.DATA_TRAIN_DIR, "train_stimulus", "%s.TextGrid" % story)
    grid = TextGrid(open(path).read())
    raw = grid.tiers[1].make_simple_transcript()  # [(xmin, xmax, text)], word tier

    good = []  # (raw_idx, lowercased word)
    for raw_idx, (_s, _e, txt) in enumerate(raw):
        norm = txt.lower().strip("{}").strip()
        if norm in DEFAULT_BAD_WORDS:
            continue
        good.append((raw_idx, norm))

    tags = nltk.pos_tag([w for _, w in good])
    records = []
    for (raw_idx, word), (_w, penn) in zip(good, tags):
        wn_pos = _penn_to_wn(penn)
        is_content = (
            penn[:2] in _CONTENT_PREFIXES
            and penn not in _PROPER_TAGS
            and wn_pos is not None
            and word.isalpha()
            and len(word) >= 3
        )
        records.append({
            "gw": len(records),
            "raw_idx": raw_idx,
            "word": word,
            "penn": penn,
            "wn_pos": wn_pos,
            "is_content": is_content,
        })
    return records


def context_window(records, gw, radius=7):
    """Space-joined words around slot `gw`, for the LLM engine's prompt/cache key."""
    lo, hi = max(0, gw - radius), min(len(records), gw + radius + 1)
    return " ".join(records[j]["word"] for j in range(lo, hi))


# ----------------------------------------------------------------------------- #
# engines
# ----------------------------------------------------------------------------- #

class SubstitutionEngine:
    name = "base"

    def propose(self, word, wn_pos, context, rng):
        """Return an in-vocab, single-token replacement != word, or None."""
        raise NotImplementedError


class WordNetEngine(SubstitutionEngine):
    name = "wordnet"

    def __init__(self, vocab_set):
        self.vocab = vocab_set

    def propose(self, word, wn_pos, context, rng):
        if wn_pos is None:
            return None
        cands = set()
        for syn in wn.synsets(word, pos=wn_pos):
            for lem in syn.lemmas():
                cand = lem.name().lower()
                if not cand.isalpha():           # drops multiword (has "_") and hyphenated
                    continue
                if cand == word or _same_stem(cand, word):
                    continue
                if cand in self.vocab:
                    cands.add(cand)
        return rng.choice(sorted(cands)) if cands else None


class EmbeddingEngine(SubstitutionEngine):
    name = "embedding"

    def __init__(self, vocab, vocab_set, gpt, topk=8):
        self.vocab = vocab
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.topk = topk
        self._pos_cache = {}
        emb = gpt.model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.E = emb / norms  # unit-normalized rows aligned to vocab order

    def _pos_ok(self, cand, wn_pos):
        if wn_pos is None:
            return True
        poss = self._pos_cache.get(cand)
        if poss is None:
            poss = _wn_pos_set(cand)
            self._pos_cache[cand] = poss
        return (not poss) or (wn_pos in poss)  # unknown-POS words are allowed

    def propose(self, word, wn_pos, context, rng):
        i = self.word2id.get(word)
        if i is None:
            return None
        sims = self.E @ self.E[i]
        order = np.argsort(-sims)
        cands = []
        for j in order[:200]:  # scan a bounded neighborhood
            if j == i:
                continue
            cand = self.vocab[j]
            if not cand.isalpha() or _same_stem(cand, word):
                continue
            if not self._pos_ok(cand, wn_pos):
                continue
            cands.append(cand)
            if len(cands) >= self.topk:
                break
        return rng.choice(cands) if cands else None


class RandomEngine(SubstitutionEngine):
    """Semantic control: any in-vocab word of the same coarse POS."""
    name = "random"

    def __init__(self, vocab, vocab_set):
        self.by_pos = {wn.NOUN: [], wn.VERB: [], wn.ADJ: [], wn.ADV: []}
        for w in vocab:
            if not w.isalpha() or len(w) < 3:
                continue
            for p in _wn_pos_set(w):
                if p in self.by_pos:
                    self.by_pos[p].append(w)

    def propose(self, word, wn_pos, context, rng):
        pool = self.by_pos.get(wn_pos)
        if not pool:
            return None
        for _ in range(10):
            cand = rng.choice(pool)
            if cand != word and not _same_stem(cand, word):
                return cand
        return None


class LLMEngine(SubstitutionEngine):
    name = "llm"

    def __init__(self, vocab_set, model="claude-opus-4-8", cache_path=None):
        import anthropic  # lazy: only needed for this engine
        self.client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / profile
        self.vocab = vocab_set
        self.model = model
        self.cache_path = cache_path or os.path.join(config.DATA_TRAIN_DIR, "aug_llm_cache.json")
        self.cache = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path) as f:
                self.cache = json.load(f)
        self._dirty = 0

    def _key(self, word, context):
        return hashlib.sha1(("%s||%s" % (word, context)).encode()).hexdigest()

    def _flush(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)
        self._dirty = 0

    def _valid(self, cand, word):
        return bool(cand) and cand in self.vocab and cand != word and not _same_stem(cand, word)

    def propose(self, word, wn_pos, context, rng):
        key = self._key(word, context)
        if key in self.cache:
            cand = self.cache[key]
            return cand if self._valid(cand, word) else None
        prompt = (
            'Sentence: "%s"\n'
            'Replace the word "%s" with a single common English synonym that fits this exact '
            "context and preserves the meaning. Reply with ONLY the one replacement word, "
            "lowercase, no punctuation." % (context, word)
        )
        cand = ""
        try:
            msg = self.client.messages.create(
                model=self.model, max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in msg.content if b.type == "text")
            cand = re.sub(r"[^a-z]", "", text.strip().lower().split()[0]) if text.strip() else ""
        except Exception:
            cand = ""
        self.cache[key] = cand
        self._dirty += 1
        if self._dirty >= 25:
            self._flush()
        return cand if self._valid(cand, word) else None


def build_engines(names, vocab, vocab_set, gpt=None, llm_model="claude-opus-4-8"):
    """Instantiate the requested engines by name."""
    engines = {}
    for name in names:
        if name == "wordnet":
            engines[name] = WordNetEngine(vocab_set)
        elif name == "embedding":
            if gpt is None:
                raise ValueError("embedding engine requires a loaded GPT model")
            engines[name] = EmbeddingEngine(vocab, vocab_set, gpt)
        elif name == "random":
            engines[name] = RandomEngine(vocab, vocab_set)
        elif name == "llm":
            engines[name] = LLMEngine(vocab_set, model=llm_model)
        else:
            raise ValueError("unknown engine: %s" % name)
    return engines


# ----------------------------------------------------------------------------- #
# slot selection + matched-dose intersection
# ----------------------------------------------------------------------------- #

def eligible_slots(records, vocab_set):
    """gw indices that are content words the model already knows (in vocab)."""
    return [r["gw"] for r in records if r["is_content"] and r["word"] in vocab_set]


def select_slots(records, vocab_set, swap_rate, rng):
    """Seeded subset of eligible slots, ~swap_rate of them."""
    elig = eligible_slots(records, vocab_set)
    if not elig:
        return []
    k = max(1, int(round(swap_rate * len(elig))))
    return sorted(rng.sample(elig, min(k, len(elig))))


def build_variant(records, slots, engines, rng_by_engine, matched=True):
    """For one variant, get every engine's replacement at each slot.

    Every engine is asked at every slot (so each engine's word choice is
    independent of the others). With matched=True the result is then restricted to
    the slots *every* engine could fill in-vocab, so all engines change the
    identical positions (dose held constant; only the replacement word differs).
    Returns {engine_name: {gw: new_word}}.
    """
    raw = {name: {} for name in engines}
    for gw in slots:
        r = records[gw]
        ctx = context_window(records, gw)
        for name, eng in engines.items():
            cand = eng.propose(r["word"], r["wn_pos"], ctx, rng_by_engine[name])
            if cand is not None:
                raw[name][gw] = cand
    if not matched:
        return raw
    common = None
    for name in engines:
        keys = set(raw[name].keys())
        common = keys if common is None else (common & keys)
    common = common or set()
    return {name: {gw: raw[name][gw] for gw in common} for name in engines}


# ----------------------------------------------------------------------------- #
# TextGrid writer (changes only word-tier labels; all times untouched)
# ----------------------------------------------------------------------------- #

_ITEM_RE = re.compile(r"\s*item \[\d+\]:")
_TEXT_RE = re.compile(r'(\s*text = ")(.*)("\s*)$')

# The training stimuli are not all in one Praat encoding, and utils_ridge.textgrid
# reads all three -- so story_word_records succeeds on every story and the writer
# has to cover the same ground. Across the 82 train_stimulus files:
#   long ooTextFile  (79) - keyed "item [n]:" blocks with `text = "..."`
#   short ooTextFile (1: life) - same header, no keys, bare values in fixed order
#   chronological    (2: legacy, exorcism) - both tiers interleaved in time order,
#                    each interval a "<tier-no> <xmin> <xmax>" line + quoted label
_QUOTED_RE = re.compile(r'(\s*")(.*)("\s*)$')
_CHRON_IVAL_RE = re.compile(r"\s*(\d+)\s+\d+\.?\d*\s+\d+\.?\d*\s*$")
_CHRON_TIER_RE = re.compile(r'\s*"\w*IntervalTier"\s+"([^"]*)"')
_SHORT_TIER_RE = re.compile(r'\s*"\w*IntervalTier"\s*$')


def _sub_ootextfile_short(lines, raw_to_new, base_story):
    """Short "ooTextFile" format: after the tier header come name/xmin/xmax/size,
    then `size` bare (xmin, xmax, text) triples -- so label k is 3 lines apart."""
    tiers = [i for i, ln in enumerate(lines) if _SHORT_TIER_RE.match(ln)]
    if len(tiers) < 2:
        raise ValueError("expected 2 tiers (phone, word) in %s" % base_story)
    start = tiers[1]  # word tier is the 2nd declared tier, matching tiers[1]
    try:
        n_intervals = int(lines[start + 4].strip())
    except (IndexError, ValueError):
        raise ValueError("could not read word-tier interval count in %s" % base_story)

    for k in range(n_intervals):
        if k not in raw_to_new:
            continue
        i = start + 5 + 3 * k + 2  # +2 past this interval's xmin/xmax
        if i >= len(lines):
            break
        m = _QUOTED_RE.match(lines[i])
        if m:
            lines[i] = "%s%s%s" % (m.group(1), raw_to_new[k].upper(), m.group(3))
    return lines


def _sub_ootextfile(lines, raw_to_new, base_story):
    """Standard "ooTextFile" long format: word tier is the 2nd item block."""
    item_lines = [i for i, ln in enumerate(lines) if _ITEM_RE.match(ln)]
    if len(item_lines) < 2:
        raise ValueError("expected 2 item blocks (phone, word) in %s" % base_story)

    interval_idx = -1
    for i in range(item_lines[1], len(lines)):
        m = _TEXT_RE.match(lines[i])
        if not m:
            continue
        interval_idx += 1
        if interval_idx in raw_to_new:
            lines[i] = "%s%s%s" % (m.group(1), raw_to_new[interval_idx].upper(), m.group(3))
    return lines


def _sub_chronological(lines, raw_to_new, base_story):
    """Praat chronological format: interleaved intervals keyed by tier number.

    Tier numbers are 1-based in declaration order, and story_word_records reads
    the word tier as tiers[1], so the 2nd declared tier is the one to rewrite.
    """
    decls = [i for i, ln in enumerate(lines) if _CHRON_TIER_RE.match(ln)]
    if len(decls) < 2:
        raise ValueError("expected 2 tiers (phone, word) in %s" % base_story)
    word_tier_no = "2"

    interval_idx = -1
    for i in range(decls[-1] + 1, len(lines) - 1):
        m = _CHRON_IVAL_RE.match(lines[i])
        if not m or m.group(1) != word_tier_no:
            continue
        interval_idx += 1
        if interval_idx not in raw_to_new:
            continue
        t = _QUOTED_RE.match(lines[i + 1])
        if t:
            lines[i + 1] = "%s%s%s" % (t.group(1), raw_to_new[interval_idx].upper(),
                                       t.group(3))
    return lines


def write_augmented_textgrid(base_story, out_path, records, subs_by_gw):
    """Write an augmented TextGrid: copy of base_story with the word-tier labels
    replaced at the substituted slots. Interval times and the phone tier are
    byte-identical, so downstream TR counts are unchanged.

    subs_by_gw maps gw -> new word; it is remapped to raw word-tier interval index.
    """
    raw_to_new = {records[gw]["raw_idx"]: new for gw, new in subs_by_gw.items()}
    # newline="" disables universal-newline translation, and split/join on "\n"
    # alone keeps any "\r" attached to the line, so a CRLF source (legacy is one)
    # stays CRLF instead of being silently rewritten as LF.
    with open(os.path.join(config.DATA_TRAIN_DIR, "train_stimulus",
                           "%s.TextGrid" % base_story), newline="") as f:
        src = f.read()
    lines = src.split("\n")

    if "chronological" in lines[0].lower():
        lines = _sub_chronological(lines, raw_to_new, base_story)
    elif any(_ITEM_RE.match(ln) for ln in lines):
        lines = _sub_ootextfile(lines, raw_to_new, base_story)
    else:
        lines = _sub_ootextfile_short(lines, raw_to_new, base_story)

    with open(out_path, "w", newline="") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    # light self-test: WordNet + Random on one sess-2 story (no GPT / no API needed)
    import random
    ensure_nltk()
    vocab, vocab_set = load_vocab("perceived")
    recs = story_word_records("avatar")
    elig = eligible_slots(recs, vocab_set)
    print("avatar: %d good words, %d eligible content slots" % (len(recs), len(elig)))
    wn_eng = WordNetEngine(vocab_set)
    rnd_eng = RandomEngine(vocab, vocab_set)
    rng = random.Random(0)
    shown = 0
    for gw in elig:
        r = recs[gw]
        w = wn_eng.propose(r["word"], r["wn_pos"], "", rng)
        if w:
            print("  %-14s (%s) -> wordnet=%-14s random=%s"
                  % (r["word"], r["penn"], w, rnd_eng.propose(r["word"], r["wn_pos"], "", rng)))
            shown += 1
        if shown >= 8:
            break
