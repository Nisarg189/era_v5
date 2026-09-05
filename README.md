# ERA V5

Coursework for **ERA V5** (The School of AI), a build of a 40 billion parameter,
India-first language model: strong at coding and agentic work, and natively fluent in
Indian languages rather than treating them as an afterthought.

One folder per assignment. Each carries its own report and the code behind every number
in it.

## Assignments

| Session | Topic | Report |
|---|---|---|
| 5 | Data mixtures and curriculum | **[assignment5/README.md](assignment5/README.md)** |
| 6 | Building the training dataset | **[assignment6/README.md](assignment6/README.md)** |
| 7 | Embeddings and model internals | **[assignment7/README.md](assignment7/README.md)** |

## Session 5, in short

A mixture and curriculum specification for a 9 trillion token pretraining run, written
under one hard constraint: **no budget for crawling or optical character recognition**, so
every token must come from a dataset that has already been published and can be downloaded.

The plan: **web 40, code 29, Indic 13, STEM 11, reasoning 6, agentic 1**, delivered across
six stages, with a 15 percent protected floor and a 3 percent anneal reserve.

Three of its numbers were tested rather than argued for, by training **14 small models from
scratch**. Two survived and one was refuted:

- **The protected floor was confirmed.** An English-only data selector rejected
  **98.6 percent** of the Hindi it was offered, turning a balanced four-lane mixture into a
  nearly pure web run. The floor restored the lane completely.
- **The Indic share is a judgement call.** Returns per unit of cost fall from 27 to 9.5 to
  2.3 as the share rises, so 13 percent sits past the elbow and is defended on grounds
  other than the curve.
- **The anneal reserve did not replicate.** Three comparisons, including one with a
  quality-selected reserve, found no benefit from holding data back. It was cut from
  8 percent of the budget to 3, and is labelled the plan's weakest number.

Full reasoning, tables and raw results in [assignment5/README.md](assignment5/README.md).

## Session 6, in short

The system that executes the Session 5 plan and proves it did. One command builds
tokenised shards from six real datasets, compiles the curriculum into per-step lane
quotas, packs batches with correct loss and attention masks, trains a small
transformer on them, **crashes on purpose, resumes, replays an earlier interval and
forks a branch**, then writes an evidence bundle in which every entry is a comparison
between two artefacts the system produced independently.

```bash
cd assignment6 && python run_demo.py     # about 80 seconds
```

**Nineteen checks, all passing. 69 unit tests, run by the same command.** Three
results worth naming:

- **Crash recovery is exact.** The crash lands in the middle of step 140, 50
  microbatch records after the last checkpoint. The resumed loader reproduces all 50
  by hash, span and sample id. A resume that restores the model but not the
  dataloader restarts at step 0, which is the failure the ledger exists to catch.
- **The English-only selector rejects Hindi again, on real models.** Mean selector
  score is **+0.04 for web and -5.33 for Indic**. The protected floor rescued **336**
  Indic candidates. This is the Session 5 finding reproduced rather than restated.
- **The Indic tokenizer pays off.** Weighting the vocabulary sample toward Indian
  scripts gives Hindi a fertility of **1.444 tokens per word, better than English web
  text at 1.627**. Unweighted it was worse than English. The rule is a gate: the run
  fails if Hindi ever costs more per word than English.

Full architecture and design decisions in [assignment6/README.md](assignment6/README.md).

## Session 7, in short

Problem 5 of the Session 7 brief: make the Kronecker embedding reversible and use that to
delete the output head. The forward path is token to bytes to code to projection to
embedding; the code is exactly invertible, so tying the output head to the input codec
turns prediction into nearest-code, and the head then holds no parameter that grows with
the vocabulary.

Proven by training three small transformers from scratch on the six-lane mixture and by
auditing the real 9,975 token vocabulary:

- **The head can go.** Removing it costs 1.2 points of accuracy (0.339 against 0.351)
  while removing 100 percent of the vocabulary-dependent head parameters.
- **The codec is invertible.** Every in-window token round-trips exactly; the only loss
  is 17 collisions at pos_dim 32, fifteen of them Telugu, falling to 0 at 48.
- **The head is the codec adjoint**, equal to a code-space dot product to within 3e-14,
  so it is an operation rather than a stored matrix.
- **A million-token vocabulary is free.** The dense head at the V5 target of 262,144
  tokens is 2.12 billion parameters and 34 GB of optimizer state, and at 1,000,000 it is
  129.5 GB; the reversible head is zero in both. Growing the candidate vocabulary six
  times over at evaluation costs one point of accuracy and no parameters.

Full reasoning, tables and raw results in [assignment7/README.md](assignment7/README.md).

## Layout

```
assignment5/
  README.md            the plan, and the deliverable
  proxy/               the experiment: corpus build, tokenizer, trainer, reports
    inventory.json     the Session 5 dataset inventory, the supply figure behind the plan
    runs/              14 committed run records, one per trained model
  *.txt                committed script output, so every table can be checked

assignment6/
  README.md            the architecture and the design decisions
  run_demo.py          the one command that regenerates every artefact
  tdes/                the system, one module per responsibility
  tests/               69 invariant tests, offline, under a second
  submission_artifacts/  run.log, evidence.json, evidence.md, manifests, ledgers, reports

assignment7/
  README.md            the reversible-Kronecker result, and the deliverable
  run_demo.py          one command regenerates every number
  rkron/               codec + exact inverse, numpy autograd, transformer, data, training
  experiments/         the invertibility audit, the head comparison, the reversal demos
  tests/               gradient checks and the reversible-Kronecker invariants
  artifacts/           the JSON every table is read from
```

Corpora and model checkpoints are not committed. The scripts in each assignment's
`proxy/` regenerate them; see that assignment's report for the commands.
