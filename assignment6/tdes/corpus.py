#!/usr/bin/env python3
"""Fetch the real source documents for all six Session 5 capability lanes.

Every lane is drawn from a published, ungated dataset over the Hugging Face
datasets-server rows API.  Nothing here is synthetic and nothing is bundled: the
demonstration downloads the same documents on any machine that runs it.

Responses are cached under ``data/cache`` keyed by the exact request, so the first
run pays for the network and every run after it is offline and byte-identical.

The six lanes follow the Session 5 mixture, and each carries the structure its
data type actually has, because that structure is what decides the loss mask:

    lane       source                                    kind      loss applies to
    web        HuggingFaceFW/fineweb-edu                 pretrain  every token
    code       codeparrot/github-code-clean (Python)     pretrain  every token
    indic      ai4bharat/sangraha verified/hin           pretrain  every token
    stem       AI-MO/NuminaMath-CoT, short solutions     sft       the solution
    reasoning  AI-MO/NuminaMath-CoT, long solutions      sft       the solution
    agentic    NousResearch/hermes-function-calling-v1   agentic   assistant turns

The stem and reasoning lanes come from one dataset split by solution length.  That
is the "reasoning-length band" the session describes, and the band is applied to a
real field rather than assigned by hand.
"""
import hashlib, json, os, time, urllib.request, urllib.error

from .clean import normalize_text

ROWS_API = "https://datasets-server.huggingface.co/rows"
UA = "Mozilla/5.0 (compatible; era-v5-session6/1.0)"
PAGE = 100                      # the rows API maximum per request
MIN_CHARS = 200                 # drop stubs, same spirit as the Session 4 filter
MAX_DOC_CHARS = 12000           # cap one document so a blob cannot dominate a lane
REASONING_BAND_CHARS = 1100     # solutions longer than this go to the reasoning lane

SOURCES = {
    "web":       ("HuggingFaceFW/fineweb-edu", "default", "train"),
    "code":      ("codeparrot/github-code-clean", "Python-all", "train"),
    "indic":     ("ai4bharat/sangraha", "verified", "hin"),
    "stem":      ("AI-MO/NuminaMath-CoT", "default", "train"),
    "reasoning": ("AI-MO/NuminaMath-CoT", "default", "train"),
    "agentic":   ("NousResearch/hermes-function-calling-v1", "func_calling", "train"),
}

# Roles that carry loss.  Everything else is read for context but never graded.
LOSS_ROLES = {"text", "response", "assistant"}


def _cache_path(root, dataset, config, split, offset):
    key = f"{dataset}__{config}__{split}__{offset}".replace("/", "_")
    return os.path.join(root, "cache", key + ".json")


def fetch_rows(root, dataset, config, split, offset, log=print, tries=4):
    """One page of rows, from cache when present, otherwise from the API."""
    path = _cache_path(root, dataset, config, split, offset)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["rows"]
    url = (f"{ROWS_API}?dataset={dataset.replace('/', '%2F')}"
           f"&config={config}&split={split}&offset={offset}&length={PAGE}")
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"user-agent": UA, "accept": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                payload = json.loads(r.read().decode("utf-8"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(payload, f)
            return payload["rows"]
        except Exception as e:                                    # noqa: BLE001
            last = e
            log(f"    fetch retry {attempt + 1}/{tries} for {dataset} offset {offset}: {str(e)[:90]}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {dataset} offset {offset}: {last}")


# ------------------------------------------------------------------ row -> document
def _doc(lane, kind, source, row_id, segments):
    segments = [{"role": r, "text": t} for r, t in segments if t and t.strip()]
    if not segments:
        return None
    chars = sum(len(s["text"]) for s in segments)
    if chars < MIN_CHARS:
        return None
    return {"lane": lane, "kind": kind, "source": source, "source_row_id": str(row_id),
            "segments": segments, "chars": chars}


def _from_web(row, source):
    return _doc("web", "pretrain", source, row.get("id"),
                [("text", normalize_text(row.get("text") or "")[:MAX_DOC_CHARS])])


def _from_code(row, source):
    rid = f"{row.get('repo_name')}/{row.get('path')}"
    # code is not run through the Indic normaliser: whitespace is semantic in Python
    return _doc("code", "pretrain", source, rid, [("text", (row.get("code") or "")[:MAX_DOC_CHARS])])


def _from_indic(row, source):
    return _doc("indic", "pretrain", source, row.get("doc_id"),
                [("text", normalize_text(row.get("text") or "")[:MAX_DOC_CHARS])])


def _from_numina(row, source, want_lane):
    problem, solution = (row.get("problem") or "").strip(), (row.get("solution") or "").strip()
    if not problem or not solution:
        return None
    lane = "reasoning" if len(solution) >= REASONING_BAND_CHARS else "stem"
    if lane != want_lane:
        return None
    return _doc(lane, "sft", source, row.get("source", "") + ":" + hashlib.sha256(problem.encode()).hexdigest()[:12],
                [("prompt", "Problem: " + problem[:MAX_DOC_CHARS // 2]),
                 ("response", "Solution: " + solution[:MAX_DOC_CHARS])])


_HERMES_ROLE = {"system": "system", "human": "user", "tool": "tool", "gpt": "assistant"}


def _from_agentic(row, source):
    segs = []
    for turn in row.get("conversations") or []:
        role = _HERMES_ROLE.get(turn.get("from"))
        text = (turn.get("value") or "").strip()
        if role and text:
            segs.append((role, text[:MAX_DOC_CHARS // 2]))
    if not any(r == "assistant" for r, _ in segs):
        return None
    return _doc("agentic", "agentic", source, row.get("id"), segs)


def _builder(lane):
    if lane == "web":
        return _from_web
    if lane == "code":
        return _from_code
    if lane == "indic":
        return _from_indic
    if lane == "agentic":
        return _from_agentic
    return lambda row, source: _from_numina(row, source, lane)


# ------------------------------------------------------------------ lane assembly
def build_lane(root, lane, char_budget, start_offset=0, id_prefix="", log=print):
    """Documents for one lane, in dataset order, from ``start_offset`` onward.

    Returns (documents, next_offset).  Taking a later ``start_offset`` for the
    held-out sets guarantees they are different documents, not a resample, and
    ``id_prefix`` keeps their document ids in a separate namespace so a held-out id
    can never be confused with a training id by the never-train check.
    """
    dataset, config, split = SOURCES[lane]
    source = f"{dataset}:{config}/{split}"
    build = _builder(lane)
    docs, total, offset, empty_pages = [], 0, start_offset, 0
    while total < char_budget and empty_pages < 12:
        rows = fetch_rows(root, dataset, config, split, offset, log=log)
        if not rows:
            break
        got = 0
        for r in rows:
            d = build(r["row"], source)
            if d:
                d["doc_id"] = f"{id_prefix}{lane}-{len(docs):05d}"
                docs.append(d)
                total += d["chars"]
                got += 1
        empty_pages = empty_pages + 1 if got == 0 else 0
        offset += PAGE
        if total >= char_budget:
            break
    log(f"    {lane:<10} {len(docs):>5} docs  {total / 1e3:>8.0f} KB  from {source}")
    return docs, offset


def write_jsonl(path, docs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
