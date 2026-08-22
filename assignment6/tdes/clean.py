#!/usr/bin/env python3
"""Text normalisation, carried forward from Session 4 without change.

This file is hashed and the digest is written into every shard manifest as the
``cleaning_pipeline_hash``.  A shard therefore records not just what text it holds
but which version of the cleaning code produced that text.  Editing this file
changes the hash, which invalidates every manifest that claims to have been built
by it, which is the intended behaviour.

The Indic rule from Session 4 is the reason this is vendored rather than
reimplemented: ZWNJ (U+200C) and ZWJ (U+200D) carry meaning in Brahmic scripts and
must survive cleaning.  Only the invisibles that carry no meaning are removed.
"""
import re, html, unicodedata

KEEP_INVISIBLE = {"‌", "‍"}          # ZWNJ, ZWJ: meaning in Brahmic scripts
NOISE_INVISIBLE = {
    "​",                                   # zero-width space
    "﻿",                                   # BOM / zero-width no-break space
    "‎", "‏",                         # LTR / RTL marks
    "‪", "‫", "‬", "‭", "‮",   # bidi embeddings and overrides
    "­",                                   # soft hyphen
    "�",                                   # replacement character
}
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WS_RE = re.compile("[ \\t\\xa0\\u2000-\\u200a\\u3000]+")   # spaces, NOT the joiners
NL_RE = re.compile(r"\n{3,}")


def normalize_text(text, counters=None):
    """Return normalised text.  If a counters dict is passed, tally what changed."""
    original = text
    if "&" in text and ";" in text:
        new = html.unescape(text)
        if counters is not None and new != text:
            counters["html"] = counters.get("html", 0) + 1
        text = new
    text = unicodedata.normalize("NFC", text)
    if counters is not None:
        counters["joiners"] = counters.get("joiners", 0) + sum(text.count(c) for c in KEEP_INVISIBLE)
    for c in NOISE_INVISIBLE:
        if c in text:
            if counters is not None:
                counters["noise"] = counters.get("noise", 0) + text.count(c)
            text = text.replace(c, "")
    ctrl = CONTROL_RE.findall(text)
    if counters is not None:
        counters["noise"] = counters.get("noise", 0) + len(ctrl)
    text = CONTROL_RE.sub("", text)
    text = WS_RE.sub(" ", text)
    text = NL_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n")).strip()
    return (text, text != original.strip()) if counters is not None else text
