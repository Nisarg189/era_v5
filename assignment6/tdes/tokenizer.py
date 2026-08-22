#!/usr/bin/env python3
"""Train the single tokenizer, freeze it, and hash it.

Session 6 requires that token ids mean exactly one thing for the life of a run.
The tokenizer is therefore trained once, written to disk, hashed, and never touched
again.  Every shard manifest carries that hash, and the loader refuses any shard
whose tokenizer hash does not match the tokenizer it is holding.

Two rules from earlier sessions are enforced here rather than assumed.

Character level, not byte level (Session 2).  A byte-level vocabulary spends three
tokens on one Devanagari character before it learns anything, which silently gives
the Indic lane about a third of the content per token.

Indic fertility (Session 2 and Session 3).  Fertility is tokens per word.  A
vocabulary trained on the natural lane mixture is dominated by English web text and
leaves Devanagari to be spelled out character by character.  The training sample is
therefore weighted toward Indic, matching the India-first vocabulary allocation the
programme committed to, and the resulting fertility is measured per lane on text the
tokenizer never saw.  ``FERTILITY_CEILING`` turns the result into a gate.
"""
import hashlib, json, os, random

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

UNK, EOS, PAD, BOS = "<unk>", "<eos>", "<pad>", "<bos>"
SPECIALS = [UNK, EOS, PAD, BOS]

# How much text each lane contributes to the vocabulary, relative to the others.
# Indic is deliberately over-represented against its 13 percent mixture share.
VOCAB_SAMPLE_WEIGHTS = {"web": 1.0, "code": 1.0, "indic": 2.6, "stem": 0.7,
                        "reasoning": 0.7, "agentic": 1.0}

# Measured fertility for the Indic lane must stay at or below this.  If a change to
# the vocabulary or the sample weights pushes Hindi past it, the run fails loudly.
# The value is set from measurement, not preference: at vocabulary 16,384 with the
# weights above, held-out Hindi measures about 1.48 tokens per word.
FERTILITY_CEILING = {"indic": 1.65}

# The India-first vocabulary allocation is only real if it pays.  Hindi must not cost
# more tokens per word than English web text does.  Unweighted, it does.
FERTILITY_MUST_NOT_EXCEED = {"indic": "web"}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _doc_text(doc):
    return "\n".join(s["text"] for s in doc["segments"])


def _weighted_sample(docs_by_lane, total_chars, seed=1):
    """A deterministic training sample, weighted per lane."""
    total_w = sum(VOCAB_SAMPLE_WEIGHTS.get(l, 1.0) for l in docs_by_lane)
    out = []
    for lane, docs in sorted(docs_by_lane.items()):
        budget = total_chars * VOCAB_SAMPLE_WEIGHTS.get(lane, 1.0) / total_w
        pool = list(docs)
        random.Random(seed).shuffle(pool)
        used = 0
        for d in pool:
            t = _doc_text(d)
            out.append(t)
            used += len(t)
            if used >= budget:
                break
    random.Random(seed + 1).shuffle(out)
    return out


def train(docs_by_lane, path, vocab_size=16384, sample_chars=3_000_000, log=print):
    """Train and save the tokenizer.  Returns its path."""
    sample = _weighted_sample(docs_by_lane, sample_chars)
    log(f"    vocabulary sample: {len(sample)} documents, "
        f"{sum(len(s) for s in sample) / 1e6:.1f} MB, Indic weight "
        f"{VOCAB_SAMPLE_WEIGHTS['indic']}x")

    tok = Tokenizer(models.BPE(unk_token=UNK))
    # Metaspace keeps the leading space attached to the word, the Session 2 shape.
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS,
                                  min_frequency=2, show_progress=False)
    tok.train_from_iterator(sample, trainer=trainer)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok.save(path)
    return path


def load(path):
    return Tokenizer.from_file(path)


def measure_fertility(tok, held_out_by_lane):
    """Fertility (tokens per word) and unknown rate per lane, on unseen text.

    Bytes per token is reported alongside because the lanes use different scripts
    and a single number would hide that Devanagari costs about three UTF-8 bytes a
    character.  Fertility is the figure the programme uses.
    """
    unk_id = tok.token_to_id(UNK)
    stats = {}
    for lane, docs in sorted(held_out_by_lane.items()):
        text = "\n\n".join(_doc_text(d) for d in docs)
        if not text.strip():
            continue
        ids = tok.encode(text).ids
        words = len(text.split())
        stats[lane] = {
            "chars": len(text),
            "bytes": len(text.encode()),
            "words": words,
            "tokens": len(ids),
            "fertility_tokens_per_word": round(len(ids) / max(1, words), 4),
            "bytes_per_token": round(len(text.encode()) / max(1, len(ids)), 4),
            "unk_rate": round(sum(1 for i in ids if i == unk_id) / max(1, len(ids)), 8),
        }
    return stats


def check_fertility(stats):
    """Return (ok, [failures]).  A breach is a hard failure, not a warning."""
    failures = []
    for lane, ceiling in FERTILITY_CEILING.items():
        got = stats.get(lane, {}).get("fertility_tokens_per_word")
        if got is None:
            failures.append(f"{lane}: no fertility measured")
        elif got > ceiling:
            failures.append(f"{lane}: fertility {got} exceeds ceiling {ceiling}")
    for lane, reference in FERTILITY_MUST_NOT_EXCEED.items():
        a = stats.get(lane, {}).get("fertility_tokens_per_word")
        b = stats.get(reference, {}).get("fertility_tokens_per_word")
        if a is not None and b is not None and a > b:
            failures.append(f"{lane}: fertility {a} is worse than {reference} at {b}")
    return (not failures), failures


def descriptor(path, vocab_size, stats):
    """The frozen record of this tokenizer, embedded in every shard manifest."""
    return {
        "tokenizer_path": os.path.basename(path),
        "tokenizer_hash": file_sha256(path),
        "vocab_size": vocab_size,
        "level": "character",
        "specials": SPECIALS,
        "vocab_sample_weights": VOCAB_SAMPLE_WEIGHTS,
        "fertility": stats,
        "fertility_ceiling": FERTILITY_CEILING,
        "fertility_must_not_exceed": FERTILITY_MUST_NOT_EXCEED,
    }
