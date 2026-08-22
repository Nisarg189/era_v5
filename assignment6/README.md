# A training data execution system for V5

**ERA V5, Session 6.** Session 5 produced a plan: what the model is trained on, in
what order, what is protected, what is held back. This is the system that carries
that plan out and proves it did.

The whole demonstration runs with one command and takes about eighty seconds:

```bash
pip install -r requirements.txt
python run_demo.py
```

It downloads real documents from six published datasets on the first run, caches
them, trains a real five million parameter transformer on them, crashes on purpose,
recovers, replays an earlier interval, forks a branch, and writes an evidence bundle
whose every entry is a comparison between two artefacts the system produced
independently. **Nineteen checks, all passing, 69 unit tests run by the same
command.**

Nothing here is simulated. Every number below was printed by
[`submission_artifacts/run.log`](submission_artifacts/run.log) on the run committed
with this repository.

---

## 01 What the system has to do, and why it is hard

The training loop consumes token windows. Humans produce documents. Between the two
sits a machine that must turn cleaned text into packed batches carrying the correct
loss masks, attention masks and position ids, in the proportions the curriculum
asks for, without ever letting held-out data through, while remaining able to answer
one question at any later date: **exactly which tokens produced this checkpoint.**

The difficulty is not any single step. It is that the answer must survive a crash.
A run restarted carelessly repeats or skips data, and both failures produce a loss
curve that looks perfectly healthy. The only defence is a record of what was
actually served, written as it happened, that a restart can be bound to.

## 02 The path a token takes

```
document  ->  screened  ->  tokenised  ->  sealed shard + manifest
          ->  mixture quota  ->  selector  ->  packed sequence
          ->  microbatch  ->  gradient  ->  consumption ledger
          ->  checkpoint  ->  crash  ->  resume  ->  replay  ->  audit
          ->  learning ledger  ->  the next corpus
```

Each arrow is a module in `tdes/`, and each is tested on its own.

| Module | Responsibility |
|---|---|
| `corpus.py` | Fetch real documents from six published datasets, cache them |
| `clean.py` | Normalise text, keeping the Indic joiners. Its own hash is the cleaning lineage |
| `tokenizer.py` | Train the tokenizer once, freeze it, hash it, gate on Indic fertility |
| `shards.py` | Tokenise into sealed shards, hash the content, deduplicate |
| `registry.py` | Admission gate, evaluation firewall, document screening |
| `mixture.py` | Compile the curriculum into per-step lane quotas with floors and warmup |
| `packing.py` | Six packing policies, loss masks, position ids, segment isolation |
| `opus.py` | Score candidates, accept, reject, defer, record the floor overrides |
| `scoring.py` | The excess loss proxy, built from a reference model trained on English only |
| `dataloader.py` | The deterministic stream, and the state a checkpoint must carry |
| `model.py` | A small decoder-only transformer that returns loss per token |
| `trainer.py` | Training, checkpoints, crash, resume, replay, fork, audit |
| `learning.py` | The learning ledger: what came back from each shard |
| `perf.py` | Throughput measured three ways, and reconstructed from the ledger |
| `evidence.py` | The bundle, assembled from artefacts rather than asserted |

---

## 03 The corpus: six real lanes, six real data types

The Session 5 mixture has six lanes. Each is drawn from a published, ungated dataset
over the Hugging Face rows API, which returns a page of rows for about a megabyte
rather than requiring a whole parquet file to be downloaded.

| Lane | Source | Data type | Loss applies to |
|---|---|---|---|
| web | `HuggingFaceFW/fineweb-edu` | pretraining | every token |
| code | `codeparrot/github-code-clean`, Python | pretraining | every token |
| indic | `ai4bharat/sangraha`, verified Hindi | pretraining | every token |
| stem | `AI-MO/NuminaMath-CoT`, short solutions | SFT pair | the solution |
| reasoning | `AI-MO/NuminaMath-CoT`, long solutions | SFT pair | the solution |
| agentic | `NousResearch/hermes-function-calling-v1` | tool trajectory | assistant turns |

Two of those choices carry an argument.

**The stem and reasoning lanes are split by solution length**, at 1,100 characters.
That is the reasoning-length band the session describes, applied to a real field
rather than assigned by hand.

**The agentic lane is a genuine tool-use dataset**, with `system`, `user`,
`assistant` and `tool` turns in their original order. It matters because the loss
mask for agentic data cannot be invented: the model must be graded on its own
planning and tool calls and left ungraded on the observations that come back.

Documents arrive with the roles their structure actually has, and every downstream
mask is derived from those roles. The lane never decides the mask. The data does.

## 04 The tokenizer, and the fertility rule

Two rules from earlier sessions are enforced in code rather than assumed.

**Character level, not byte level.** A byte-level vocabulary spends three tokens on
one Devanagari character before it learns anything.

**Indic fertility.** Fertility is tokens per word. A vocabulary trained on the
natural lane mixture is dominated by English web text and leaves Devanagari to be
spelled out. The training sample is therefore weighted toward Indic at 2.6 times,
matching the India-first vocabulary allocation the programme committed to.

Measured on held-out text the tokenizer never saw, at vocabulary 16,384:

| Lane | Fertility, tokens per word | Bytes per token | Unknown rate |
|---|---:|---:|---:|
| **indic** | **1.444** | 9.01 | 5.3e-05 |
| stem | 1.504 | 3.77 | 6.3e-04 |
| web | 1.627 | 3.82 | 1.4e-04 |
| reasoning | 1.698 | 3.42 | 3.4e-04 |
| agentic | 1.936 | 4.29 | 0.0 |
| code | 4.705 | 2.35 | 1.5e-03 |

**Hindi is the cheapest lane in the corpus, ahead of English web text.** That is the
whole point of weighting the sample, and it is measured rather than hoped for. An
unweighted sample was tried first and gave Indic 1.56 against web 1.52, the wrong way
round. Both the absolute ceiling and the relative rule are gates: if Hindi ever costs
more per word than English, the run fails at phase 02 and no shard is written.

Code reads at 4.7 tokens per word because a whitespace-delimited "word" in Python is
often a whole indented line. The figure is reported for completeness, not as a defect.

## 05 Shards, and what immutability means in practice

A shard is written once and sealed. Its content hash covers the token array **and**
the sample index together, so neither the tokens nor their segment boundaries can
move without the hash changing.

One storage layout serves all three data types, because the difference between them
is not how tokens are stored but which spans carry loss:

```
tokens.npy    one flat uint16 array for the whole shard
manifest      a sample index of (start, end, role) spans into that array
```

The manifest carries what the session asks for: source ids, document ids, token
count, language and script, capability lane, licence tier, dedup status,
contamination status, evaluation overlap, parent shards, and three hashes. The
**cleaning pipeline hash is the SHA-256 of `tdes/clean.py` itself**, so a shard does
not merely claim to have been cleaned, it names the version of the code that cleaned
it. Editing that file invalidates every manifest that claims to have been built by it.

Immutability is checked rather than declared. A **tamper probe** alters exactly one
token in the web shard and re-verifies: the change is detected. Unit tests extend
this to a reordered sample index and a truncated write, both of which also fail
verification.

On the committed run, **18 of 18 shards verified against their manifests.**

## 06 Contamination is screened before a shard is built, not after

This was the most instructive bug in the build, and the fix changed the pipeline
order.

The first design built the training shards, then offered each to the firewall. The
Indic shard was blocked. Its offending document was quarantined and the shard was
rebuilt, and the rebuilt shard was blocked again, five times in a row.

The cause was real, not a coding error. **Deduplication decides which near
duplicates survive**, first seen wins, so removing one document changes which of the
remaining documents are kept, which exposed a second contaminated document that
deduplication had previously been hiding.

Screening therefore happens on the full document set, once, before any shard is
written. On the committed run this removed **10 documents: 9 from the Indic lane by
exact content hash and 1 from the code lane by n-gram fingerprint.** The nine Indic
hits are genuine duplicates inside AI4Bharat Sangraha, found because the held-out
slice was taken from a different offset of the same dataset and the same text appears
in both.

The quarantine-and-rebuild path still exists and is still demonstrated, on a
candidate shard deliberately built to contain six held-out documents. It is blocked,
its clean samples are rewritten as `candidate-contaminated-001` with a **new content
hash naming the blocked shard as its parent**, and the child is admitted. A blocked
shard is never edited.

## 07 The evaluation firewall

One registry holds every shard at three permission levels.

| Split | Permission |
|---|---|
| train | enters batches, carries gradient |
| validation | may be read to measure, never gradient bearing |
| test | `never_train`, blocked at the gate |

Test data is registered precisely so that it can be recognised and refused. Nothing
can be blocked that was never fingerprinted.

Detection uses two independent methods. Exact content hashing catches copies.
Token n-gram fingerprinting, at Jaccard 0.35 over 8-grams with a minimum of four
shared n-grams, catches contamination that survived a small edit. A unit test proves
the second is necessary: a held-out document with three tokens appended passes the
exact check and is caught by the fingerprint.

Two probes run every time, and both must behave:

| Probe | Expected | Result |
|---|---|---|
| Candidate containing 6 held-out documents | blocked | blocked, 6 hits, all exact |
| Clean candidate | admitted | admitted |

A firewall that blocks everything is not a firewall, which is why the clean control
exists.

The final check is not at the gate but at the end. Every document id in every
consumed batch across the whole run is compared against the never-train registry:
**327 never-train documents indexed, 0 reached a loss-bearing batch.**

## 08 Compiling Session 5's curriculum into per-step quotas

The stage table is Session 5's own, verbatim, scaled from a nine trillion token run
down to the demonstration budget. Only the absolute token counts change. Every share,
floor, boundary and sequence-length step is the plan's number.

| Stage | Span | Seq | web | code | indic | stem | reason | agentic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 seed | 0 to 2% | 128 | 60 | 14 | 14 | 9 | 2 | 1 |
| S1 general | 2 to 34% | 128 | 56 | 22 | 13 | 7 | 1 | 1 |
| S2 capability | 34 to 66% | 256 | 36 | 33 | 12 | 15 | 3 | 1 |
| S3 reasoning | 66 to 84% | 256 | 26 | 31 | 13 | 12 | 17 | 1 |
| S4 long context | 84 to 97% | 512 | 31 | 35 | 14 | 9 | 10 | 1 |
| S5 anneal | 97 to 100% | 256 | 10 | 20 | 28 | 12 | 22 | 8 |

Turning a percentage into an instruction the loader can obey needs three things that
are easy to get wrong.

**Whole sequences.** A lane owed 3.84 sequences cannot be served 3.84 sequences. The
fraction is carried forward, and the average holds over many steps. The carried
remainder is loader state, which means it has to survive a crash. A unit test runs
400 steps and confirms each lane converges on its share to two decimal places.

**Warmup.** A stage boundary blends the two mixtures across a band rather than
switching in one step, because a hard distribution change moves the gradients.

**Floors below one sequence.** The agentic floor is one percent of a sixteen
sequence batch, which is 0.16 sequences. Rounding that to an integer gives zero and
the lane silently disappears. Floors are therefore applied by raising the lane's
effective share before allocation and carrying the fraction, and a unit test confirms
a one percent floor produces one percent over 400 steps.

Measured compliance on the committed run, planned share against the share the ledger
actually recorded:

| Lane | Planned | Actual | Error | Floor | Held |
|---|---:|---:|---:|---:|---|
| web | 0.469 | 0.470 | 0.001 | | |
| code | 0.268 | 0.254 | 0.015 | | |
| indic | 0.126 | 0.127 | 0.001 | 0.10 | yes |
| stem | 0.108 | 0.100 | 0.008 | | |
| reasoning | 0.020 | 0.040 | 0.020 | 0.04 | yes |
| agentic | 0.010 | 0.010 | 0.000 | 0.01 | yes |

**Worst lane error 0.020, every protected floor held.** The reasoning lane sits at
twice its planned share because the floor is above the plan for the early stages,
which is the floor doing its job rather than an error.

## 09 Packing, and the two numbers that are not the same

Six policies are implemented and compared on the real corpus. Each produces fixed
length sequences carrying four parallel arrays.

| Array | What it decides |
|---|---|
| `input_ids` | the tokens |
| `loss_mask` | 1 where graded, 0 where read for context only |
| `position_ids` | restart at 0 per packed sample |
| `segment_ids` | which sample a position belongs to, which blocks attention |

Utilisation is reported as two separate figures, because they answer different
questions. **Occupancy** is how much of the window is not padding. **Loss coverage**
is how much of the window is actually graded. On plain web text the two are the same.
On agentic trajectories they are not, and the gap is the point:

| Policy, agentic trajectories | Occupancy | Loss coverage | Sequences | Boundary crossings |
|---|---:|---:|---:|---:|
| pad_only | 1.000 | 0.749 | 120 | 0 |
| concat_chop | 0.998 | **0.461** | 369 | **102** |
| greedy | 1.000 | 0.749 | 120 | 0 |
| best_fit | 1.000 | 0.749 | 120 | 0 |
| structure_preserving | 1.000 | 0.749 | 120 | 0 |
| long_context | 1.000 | 0.749 | 120 | 0 |

Concatenate-and-chop reports the best occupancy and the worst outcome. It fills 99.8
percent of every window, but only 46 percent of those positions are graded, it lets
102 pairs of unrelated samples see each other through attention, and it cuts 368
trajectories mid stream. It is correct for plain web text and wrong for everything
with structure, which is why it is offered as a policy and never selected for
structured data.

Two design decisions came out of running the packer on real data.

**Long structured samples are truncated from the position that keeps the most graded
tokens, not from the front.** An agentic trajectory truncated from the front leaves a
window holding the system prompt, the user request and nothing the model is graded
on. It costs full compute and teaches nothing. `best_window` chooses the offset with
the highest graded-token coverage, so the assistant turns survive the cut.

**A sequence carrying no loss at all is never emitted.** Before this rule,
concatenate-and-chop produced 233 completely ungraded windows out of 369 on the
agentic lane. They are now dropped and the count is reported rather than hidden.

**Plain pretraining documents are chunked, not truncated.** A four thousand token web
document is not one training sequence, it is many. Chunking keeps every token and
keeps each chunk isolated, which concatenate-and-chop does not. Without it the web
lane under-delivered against its quota by half, because the quota is counted in
sequences and each long document was contributing only one.

Every packed sequence used in training is checked: **0 mask violations, 0 position
pairs where attention crosses a segment boundary.**

## 10 OPUS, and the bias it is built to expose

OPUS is the Session 5 batch selector, not the Anthropic model. It scores candidates
before they are served and keeps the useful fraction.

The score is **excess loss**, computed from two real models:

```
score = current model loss - reference model loss
```

The reference model is trained on the English web lane only, for 60 steps, inside the
same run. That choice is deliberate. Session 5 measured an English-proxy selector
rejecting 98.6 percent of the Hindi it was offered. This reproduces the effect from a
real model rather than describing it: Devanagari is enormously expensive for a
reference that has never seen the script, so the excess loss goes deeply negative and
the lane is rejected for a lack of merit it does not have.

Measured on the committed run, over 4,011 candidates:

| Quantity | Value |
|---|---:|
| Mean OPUS score, web lane | **+0.041** |
| Mean OPUS score, Indic lane | **-5.330** |
| Indic candidates retained by the protected floor | **336** |

The gap is five and a third nats. Nothing in the Indic data is worse. The proxy
simply cannot see it. Without the floor the lane would collapse, exactly as Session 5
predicted, and the floor is what stops it.

Every decision is recorded with its score, the checkpoint that scored it, the proxy
version, the status and the reason:

| Status | Count | Meaning |
|---|---:|---|
| accepted | 281 | served on merit |
| protected floor override | 558 | below threshold, retained by a floor |
| quota topup | 1,059 | lane short of its promised tokens after selection |
| deferred | 474 | held for a later phase |
| rejected | 1,639 | not served now, and kept rather than deleted |

Rejection reasons are separated rather than merged, because they mean different
things: 1,171 for low proxy utility, 468 for quota pressure.

The **quota topup** category is worth naming. When the selector rejects so much of a
lane that the lane cannot fill the tokens the mixture promised it, the shortfall is
made up rather than left. The mixture decides how many tokens a lane receives; the
selector decides which tokens within that allocation. That is the structural rule
Session 5's experiments forced into the plan, and topups are logged separately from
floor overrides so the two are never confused.

## 11 The two ledgers

**The consumption ledger** is append-only, one record per microbatch, flushed and
synced on every write because a ledger that loses its tail cannot be used to recover
from the crash that truncated it. Each record carries the run, branch, step,
checkpoint, rank, microbatch id, packed sample ids, shard ids, token span ids, batch
hash, loss mask hash, attention and position policy, lane, stage, tokenizer version,
dataloader version and the OPUS decision.

Each branch gets its own file, so an offset is unambiguous.

**The learning ledger** points the other way. Per shard: mean token loss, first and
last exposure, the change between them, the gradient norm of the steps the shard
appeared in, the loss at each repeated pass, and a classification.

Measured on the committed run:

| Shard | Lane | Graded tokens | Mean loss | Loss delta | Loss by pass | Repeat effect |
|---|---|---:|---:|---:|---|---:|
| train-web-000 | web | 333,555 | 7.50 | -2.89 | 8.02 / 7.53 / 7.29 / 7.14 | **-0.88** |
| train-code-000 | code | 180,519 | 6.21 | -3.42 | 7.13 / 5.95 / 5.69 | -1.44 |
| train-indic-000 | indic | 89,858 | 8.04 | -2.12 | 8.16 / 7.64 | -0.52 |
| train-stem-000 | stem | 56,973 | 7.65 | -1.78 | 7.65 | one pass only |
| train-reasoning-000 | reasoning | 28,500 | 7.66 | -2.05 | 7.66 | one pass only |
| train-agentic-000 | agentic | 5,608 | 8.33 | -1.54 | 8.33 | one pass only |

The web row is the interesting one. Four passes were completed, and the improvement
per pass decays: **-0.49, -0.24, -0.15.** That is the repetition budget being spent,
measured rather than assumed, and four passes is exactly the cap Session 5 set for
natural text.

**Token level detail is where the shard mean stops being useful.** Full token traces
are kept for the Indic and agentic shards and aggregates for the rest, which is the
storage tiering the session describes. The hardest tokens in the Indic lane, ranked
by their loss inside that lane rather than overall, are low-frequency proper nouns
and compound forms. The shard mean of 8.04 hides them completely. That pattern is the
instruction for what the next version of the corpus should contain, and it is the
signal that would be unrecoverable if it were not written down as it happened.

## 12 Crash, resume, replay and fork

The crash is injected in the **middle of a step**, after some microbatches have been
served and recorded but before the optimiser has applied anything. That is the
realistic case and the hard one: the ledger has run ahead of the model, and recovery
has to go back to the model's position rather than the ledger's end.

On the committed run:

| Fact | Value |
|---|---|
| Crash injected at step | 140 |
| Last checkpoint | `ckpt-main-00128`, ledger offset 512 |
| Microbatch records written after that checkpoint | 50, gradients lost |
| Expected next batch | step 128, `s00128-r0-a0`, hash `086893d78eba83ef` |
| Batch the resumed loader produced | step 128, `s00128-r0-a0`, hash `086893d78eba83ef` |
| Microbatches compared | **50, all identical** |

The comparison covers the batch hash, the token span ids and the packed sample ids,
not just the step number.

**The control matters as much as the result.** A second resume restores the model and
the optimiser correctly but leaves the dataloader fresh. It begins at step 0 and
produces a different batch. Two of the four saved fields were then removed in unit
tests, one at a time: dropping the lane cursors changes the stream, and dropping the
quota carry changes the stream. Every field in the saved state is load bearing.

**Replay** restores `ckpt-main-00096`, 32 steps further back than the resume did, and
rebuilds steps 96 to 99. All 16 microbatches match the original records by hash and
by span.

**Fork** restores the same checkpoint under a new branch id with the Indic share
raised ten points at web's expense. The Indic share moves from 0.125 to 0.219, the
streams diverge, and the divergence point is written to `branches.json` with the
parent branch, the step and the ledger offset. The difference between the two
branches is recorded rather than accidental, which is the only thing that makes the
comparison an experiment.

**Ledger hygiene** is checked across the whole run rather than at the crash alone.
Records carry a segment number, 0 before the crash and 1 after. Within each segment
no microbatch key repeats; across the effective stream, **steps 0 to 255 with no
gaps.**

## 13 Checkpoints

A checkpoint saves the model, the optimiser, the generator states, the dataloader
state and the ledger offset. A checkpoint without a data position can restart a run
but cannot resume one.

Two decisions keep a demonstration run from filling a laptop. The scoring snapshot is
not stored, because it is by definition the model at that checkpoint and a second
copy of the same weights doubles the file for nothing. **The optimiser is stored only
for the checkpoints a resume could land on.** Replay and fork rebuild the data stream
and never step the optimiser, so an older checkpoint needs the model, the dataloader
state, the generator states and the ledger offset, and nothing else. This alone cut
the run from 2.4 GB to 201 MB across four retained checkpoints.

Weights are not committed. `submission_artifacts/checkpoints/index.json` is committed
instead, and it names every checkpoint ever written with its step, ledger offset,
token count, size and a **SHA-256 digest of its weights**, so a checkpoint that is no
longer on disk is still identified. Running the command regenerates them, which is
step one of the evaluation anyway.

## 14 Throughput, measured three times

A loader can report a large tokens-per-second figure and deliver very little
learning. The three figures that matter are separated:

| Measure | Value |
|---|---:|
| Raw token positions per second | 9,422 |
| Useful loss-bearing tokens per second | 9,203 |
| Accepted tokens per second, after OPUS | 3,999 |
| Packing occupancy | 0.990 |
| Loss coverage of positions | 0.977 |
| Loader wait fraction | 0.269 |
| Selector scoring overhead | 9.5 s of 76.0 s |

The gap between the first two is padding and context, and it is small here because
packing occupancy is 99 percent. The gap between the second and the third is the
price of selection, and it is large: **OPUS costs 57 percent of the delivered token
rate.** That is the real trade-off the session names. Whether it is worth paying
depends on whether the retained tokens are more than twice as valuable, which is a
question the learning ledger exists to answer on a real run.

Loader wait at 27 percent is honest and would be unacceptable at scale. At this size
the model is small enough that batch construction is a real fraction of the step,
which is the opposite of the large-run regime.

**Every headline figure is recounted from the consumption ledger** and compared to
the counters that produced it. Raw positions: 715,776 from the report, 715,776 from
the ledger. A throughput claim that cannot be reconstructed receives no credit, so
the reconstruction is a check rather than a note.

---

## 15 The evidence bundle

`evidence.json` and `evidence.md` are generated. Nothing in `evidence.py` decides
that a requirement passed. Each check is handed two things the system produced
independently and reports whether they agree, together with the values compared, so
the bundle can be disproved by a reader rather than only trusted.

| # | Requirement | Result | What was compared |
|---|---|---|---|
| 1 | Tokenizer integrity | PASS | manifest hash against the file recomputed from disk |
| 2 | Indic fertility | PASS | measured tokens per word against the ceiling and against web |
| 3 | Shard immutability and manifests | PASS | 18 of 18 hashes, plus a tamper probe |
| 4 | Contamination screened before sharding | PASS | 10 documents removed, 6 lanes admitted |
| 5 | Blocked shard rebuilt rather than edited | PASS | parent and child content hashes |
| 6 | Evaluation firewall | PASS | contaminated candidate blocked, clean control admitted |
| 7 | Crash recovery | PASS | 50 microbatches, expected against resumed |
| 8 | No skipped or repeated batches | PASS | steps 0 to 255, no gaps, no repeats |
| 9 | Replay | PASS | 16 microbatches, original hashes against replay hashes |
| 10 | Fork with recorded divergence | PASS | branch record, parent share against fork share |
| 11 | Mixture compliance and protected floors | PASS | planned share against ledger share |
| 12 | OPUS audit trail | PASS | 4,011 decisions with score, status and reason |
| 13 | Packing correctness | PASS | 0 mask violations, 0 cross-segment attention pairs |
| 14 | Learning trace linked to source data | PASS | per-token loss attributed to shard and document |
| 15 | Validation and test data never trained | PASS | 327 never-train documents, 0 leaked |
| 16 | Throughput reconstructable from the ledger | PASS | report counts against ledger counts |
| 17 | End to end execution | PASS | 18 phases, one command |
| 18 | Invariant tests | PASS | 69 tests executed by this run |
| 19 | Checkpoints bound to ledger offsets | PASS | every checkpoint carries offset and digest |

## 16 Tests

```bash
python -m unittest discover -s tests
```

69 tests, under a second, no network. They build shards from generated token arrays
rather than from the corpus, so they test the system rather than the data.

| File | Covers |
|---|---|
| `test_packing.py` | mask derivation, position restarts, segment isolation, chunking, window choice |
| `test_shards.py` | tamper, reorder and truncation detection, deduplication, EOS, span consistency |
| `test_mixture.py` | quota sums, remainder convergence, sub-sequence floors, warmup blending |
| `test_ledger.py` | offsets, durability, gap and repeat detection, branch records, batch identity |
| `test_firewall.py` | exact and near-duplicate blocking, licence, tokenizer and lineage checks |
| `test_resume.py` | restore exactness, the naive control, field-by-field necessity, fork divergence |
| `test_evidence.py` | bundle assembly, markdown escaping, throughput reconstruction |

`run_demo.py` runs this suite as its seventeenth phase and records the result in the
evidence bundle, so the test result is produced by the submitted run rather than
quoted from an earlier one.

## 17 Honest limits

**Scale.** The model is 5,023,872 parameters and the run is 256 steps over 715,776
token positions. That is enough for real gradients, real per-token losses and real
loss deltas, and far too small for any of the loss numbers to say something about
language modelling. They are here to prove the data system reports what actually
happened, not to be good.

**Loader wait at 27 percent** would be a serious problem at scale and is a
consequence of the model being small enough that building a batch costs a real
fraction of a step.

**Three lanes complete only one pass**, so their repeated-pass columns are empty by
measurement rather than by omission. The mechanism is proved by unit tests that give
a lane four documents and confirm the pass number rises, survives a restore, and is
attached to every drawn item.

**OPUS scores on the first 192 tokens** of a candidate, chosen from the window that
carries the most graded tokens. A full-length score would be more accurate and would
cost more than the training step it is selecting for.

**The reference model is trained for 60 steps.** It is deliberately weak. A stronger
English-only reference would sharpen the proxy on English and would not change the
Indic result, which is driven by the script being absent from its training data
rather than by how well it was trained.

## 18 Layout

```
run_demo.py                    the one command
requirements.txt
tdes/                          the system, one module per responsibility
tests/                         69 invariant tests, offline, under a second
submission_artifacts/          regenerated by every run
  run.log                        the execution trace
  evidence.json                  machine readable, one entry per requirement
  evidence.md                    the same, for a reader
  performance.json               throughput, and its reconstruction from the ledger
  manifests/                     one per shard, plus the tokenizer and the schedule
  ledgers/                       consumption per branch, learning, token trace, branches
  checkpoints/index.json         every checkpoint, its offset and its weight digest
  reports/                       registry, screening, opus, packing, audit, loss curve, tests
data/                          downloaded and cached, not committed
```

## 19 Reproducing

```bash
pip install -r requirements.txt
python run_demo.py                  # about 80 seconds after the first run
python -m unittest discover -s tests
```

Useful flags: `--steps`, `--crash-at`, `--kb-per-lane`, `--vocab`, `--max-seq`,
`--ckpt-interval`, `--seed`, `--threads`.

The first run downloads roughly five megabytes of documents and caches them under
`data/cache`. Every run after that is offline. `run_demo.py` exits non-zero if any
check fails.

**Reproducibility was verified rather than assumed.** A clean checkout was extracted
to an empty directory with no cache, no shards and no access to the rest of the
repository, and `run_demo.py` was run from scratch. It completed in 110 seconds
including the downloads, 19 of 19 checks passed, and the artefacts it produced agree
with the committed ones exactly:

| Quantity | Committed run | Clean checkout |
|---|---|---|
| Tokenizer hash | `48e9fa5d02090d90...` | identical |
| Expected next batch after the crash | `086893d78eba83ef` | identical |
| Replay batch hashes | 16 hashes | identical |
| Indic fertility | 1.4445 | 1.4445 |
| Worst mixture lane error | 0.02047 | 0.02047 |
| Mean OPUS score, Indic | -5.32971 | -5.32971 |

The same tokenizer, the same shards, the same selector decisions and the same batches
come out of a fresh download on a different path.
