#!/usr/bin/env python3
"""Build the four-lane proxy corpus.

One lane per capability the proxy tests, each drawn from the dataset that the
Session 5 inventory names for that lane:

  web    FineWeb-Edu (sample/10BT)          inventory: general, 1.3T tokens
  code   codeparrot/github-code-clean       stand-in for The Stack v2, 900B
  indic  Sangraha verified/hin              inventory: Indic tier A, 64B
  math   NuminaMath-CoT                     inventory: reasoning, 0.5B

The inventory names The Stack v2 for the code lane, but every BigCode Stack
release is access-gated, so the code lane uses codeparrot/github-code-clean:
permissively-licensed GitHub source of the same provenance family, ungated.

Large parquet files are read a row group at a time over HTTP range requests, so
nothing is downloaded in full. The Hindi lane reuses the Session 4 cleaning
function verbatim, which keeps the ZWJ/ZWNJ joiners.

Output: data/<lane>.txt, documents separated by a blank line. Deterministic.

    python3 build_corpus.py [--mb-per-lane 140]
"""
import argparse, json, os, sys, time, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")


def find_assignment4():
    """Locate assignment4_work by walking up from here.

    It sits outside this repository, so the depth between the two is not fixed.
    Searching upward keeps this working wherever the repository is placed.
    """
    d = HERE
    for _ in range(6):
        d = os.path.dirname(d)
        cand = os.path.join(d, "assignment4_work")
        if os.path.isdir(cand):
            return cand
    raise SystemExit("assignment4_work not found above " + HERE)


A4 = find_assignment4()
# reuse the Session 4 cleaner so the Hindi lane is the same cleaned text
sys.path.insert(0, A4)
from hi_clean import normalize_text  # noqa: E402

SANGRAHA = os.path.join(A4, "data", "hin-0.parquet")
FINEWEB = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet"
NUMINA = "https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/resolve/main/data/train-00000-of-00005.parquet"
CODE = "https://huggingface.co/datasets/codeparrot/github-code-clean/resolve/main/data/train-00000-of-00880.parquet"

MIN_CHARS = 200          # drop stubs; same spirit as the Session 4 quality filter


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def write_lane(name, docs):
    path = os.path.join(DATA, f"{name}.txt")
    body = "\n\n".join(docs)
    with open(path, "w") as f:
        f.write(body)
    h = hashlib.sha256(body.encode()).hexdigest()[:16]
    log(f"{name:<6} {len(docs):>7,} docs  {len(body)/1e6:>7.1f} MB  sha256:{h}")
    return {"lane": name, "docs": len(docs), "chars": len(body), "sha256_16": h}


def remote_parquet_docs(url, column, budget_chars, transform=None):
    """Stream row groups over HTTP range requests until the char budget is met."""
    import fsspec, pyarrow.parquet as pq
    pf = pq.ParquetFile(fsspec.filesystem("http").open(url, "rb"))
    docs, total = [], 0
    for rg in range(pf.num_row_groups):
        for v in pf.read_row_group(rg, columns=[column]).column(column).to_pylist():
            t = transform(v) if transform else v
            if t and len(t) >= MIN_CHARS:
                docs.append(t)
                total += len(t)
        if total >= budget_chars:
            break
        if rg % 10 == 0:
            log(f"  ... row group {rg}, {total/1e6:.0f} MB")
    return docs


def build_web(budget):
    log("web: FineWeb-Edu sample/10BT")
    return remote_parquet_docs(FINEWEB, "text", budget)


def build_math(budget):
    log("math: NuminaMath-CoT")
    # 'solution' is the worked reasoning trace, 'problem' the prompt
    import fsspec, pyarrow.parquet as pq
    pf = pq.ParquetFile(fsspec.filesystem("http").open(NUMINA, "rb"))
    docs, total = [], 0
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg, columns=["problem", "solution"])
        for p, s in zip(tb.column("problem").to_pylist(), tb.column("solution").to_pylist()):
            if not p or not s:
                continue
            t = f"Problem: {p.strip()}\n\nSolution: {s.strip()}"
            if len(t) >= MIN_CHARS:
                docs.append(t)
                total += len(t)
        if total >= budget:
            break
    return docs


def build_code(budget):
    log("code: codeparrot/github-code-clean")
    # cap any one file so a handful of vendored blobs cannot dominate the lane
    return remote_parquet_docs(CODE, "code", budget,
                               transform=lambda t: t[:40000] if t else t)


def build_indic(budget):
    log("indic: Sangraha verified/hin (Session 4 cleaning reapplied)")
    import pyarrow.parquet as pq
    tbl = pq.read_table(SANGRAHA, columns=["text"])
    col = tbl.column("text")
    docs, total = [], 0
    for i in range(len(col)):
        t = normalize_text(col[i].as_py() or "")
        if len(t) >= MIN_CHARS:
            docs.append(t)
            total += len(t)
        if total >= budget:
            break
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb-per-lane", type=float, default=140)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    budget = int(a.mb_per_lane * 1e6)

    manifest = {"mb_per_lane": a.mb_per_lane, "min_chars": MIN_CHARS, "lanes": [],
                "sources": {"web": FINEWEB, "code": CODE, "indic": SANGRAHA, "math": NUMINA}}
    for name, fn in [("indic", build_indic), ("code", build_code),
                     ("math", build_math), ("web", build_web)]:
        if os.path.exists(os.path.join(DATA, f"{name}.txt")):
            log(f"{name}: exists, skipping")
            continue
        manifest["lanes"].append(write_lane(name, fn(budget)))

    with open(os.path.join(DATA, "corpus_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log("done")


if __name__ == "__main__":
    main()
