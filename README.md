# V5 Mixture and Curriculum Plan

ERA V5, Session 5. A mixture and curriculum specification for a 15 trillion token
pretraining run, composed backward from the benchmarks the model is meant to win,
sized against the real supply in the Session 5 dataset inventory, and tested by
proxy runs that were actually executed rather than only proposed.

Three numbers carry the plan. The mixture assigns **13 percent to Indic**, a
**17 percent protected floor** sits outside the data selector's control, and
**8 percent of the budget is held back** for the final anneal.

All three were tested on real training runs, not argued for. Two survived and one did
not. The protected floor is strongly confirmed: an English-only selector rejected
**98.6 percent** of the Hindi it was offered, and the floor restored the lane
completely. The Indic share is supported at the floor but sits past the elbow of its
own return curve at 13 percent. The anneal reserve **failed to replicate** across
three comparisons and is now the weakest number in the plan. Section 12 has the
numbers, section 13 has what happens next. Two parts of the plan, in sections 6 and 7,
were changed by results that contradicted them.

Every table below is printed by a script in this repository, not typed by hand.
`proxy/plan_arithmetic.py` produces sections 2 to 7 from the inventory. `proxy/report.py`
produces section 12 from the training runs.

---

## 1. What the mixture is buying

A capability that is never measured cannot be shown to exist, so each lane is
defined by the benchmarks it is meant to move.

| Lane | Benchmarks it feeds | Training shape the benchmark actually scores |
|---|---|---|
| Code | LiveCodeBench, Aider | loss on the generated patch or completion |
| Agentic | SWE-bench, tau-bench, BFCL, GAIA, BrowseComp | loss on planning, tool calls and the final answer, never on the tool observation |
| Reasoning | AIME, GPQA, HLE | loss on the worked trace and the final answer |
| STEM | AIME, GPQA | loss on derivations and notation |
| Indic | MILU, IndicGenBench | loss on native Indian-language text and generation |
| Long context | long-eval | loss on tokens whose evidence sits tens of thousands of tokens earlier |
| General web | MMLU | loss on broad world knowledge |

The masking rule is part of the specification, not an implementation detail. In an
agentic trajectory the user request and every tool observation are context and
receive no loss. Applying loss to a tool observation trains the model to invent
tool results instead of calling the tool.

## 2. The main pretraining mixture

15T tokens. Repetition is capped at four epochs for natural text, following
Muennighoff et al., *Scaling Data-Constrained Language Models* (NeurIPS 2023), which
found repeating up to roughly four epochs close to loss-free and returns decaying
sharply after. Model-generated text is capped at two, because its errors compound
under repetition.

| Lane | Share | Demand | Inventory | Added | Effective | Epochs | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| General web | 34% | 5.10T | 4.69T | 0 | 4.69T | 1.1 | covered |
| Code | 24% | 3.60T | 1.10T | 0 | 1.10T | 3.3 | repeat |
| Indic | 13% | 1.95T | 0.28T | 0.39T | 0.66T | 3.0 | repeat + acquire |
| STEM / math | 10% | 1.50T | 0.15T | 0.35T | 0.50T | 3.0 | repeat + mine |
| Long context | 8% | 1.20T | 0.10T | 0.60T | 0.70T | 1.7 | pack |
| Reasoning | 8% | 1.20T | 0.09T | 0.25T | 0.34T | 3.6 | repeat + generate |
| Agentic | 3% | 0.45T | 0.0006T | 0.13T | 0.13T | 3.4 | mine + build |
| **Total** | **100%** | **15.0T** | **6.40T** | **1.71T** | **8.12T** | **1.8** | |

The inventory holds 6.40T unique tokens against a 15T budget, so the run is
data-constrained before any lane is discussed. No lane exceeds the four-epoch cap.

## 3. Where the 1.71T of new tokens comes from

A share that exceeds supply is only legitimate if the shortfall is named and costed.

| Lane | Amount | Mechanism | Source |
|---|---:|---|---|
| Indic | 0.20T | acquire | licensed news archives, book and textbook digitisation, broadcast ASR, government and court records |
| Indic | 0.185T | generate | verified translation into tier C, templated and self-instruct into tier D |
| STEM | 0.25T | mine | classifier-selected STEM already inside DCLM and FineWeb-Edu, relabelled rather than re-sourced |
| STEM | 0.10T | acquire | arXiv full text, PubMed, open textbooks |
| Long context | 0.60T | pack | documents and repositories of 8K tokens or more already inside the web and code lanes |
| Reasoning | 0.25T | generate | teacher distillation filtered by verifiable rewards |
| Agentic | 0.10T | mine | issue to pull-request to diff chains, CI logs, notebooks, shell sessions |
| Agentic | 0.03T | generate | execution-verified trajectories in tool sandboxes |

Totals by mechanism: 0.60T packed, 0.46T generated, 0.35T mined, 0.30T acquired.
Only **0.46T of the 1.71T is model-generated**. Long context is a packing decision
rather than a sourcing one, because a long-context corpus is built by concatenating
documents the programme already holds.

## 4. The Indic slot, split by tier

13 percent of 15T is 1.95T. Split across the four provenance tiers from Session 3:

| Tier | Existing unique | New unique | Epochs | Tokens | Share of slot |
|---|---:|---:|---:|---:|---:|
| A verified native | 64.0B | 200B | 4x | 1056B | 54.2% |
| B unverified crawl | 44.9B | 0B | 4x | 180B | 9.2% |
| C translated | 5.0B | 60B | 2x | 130B | 6.7% |
| D synthetic | 162.0B | 125B | 2x | 574B | 29.4% |

Native text is 63 percent of the slot, translated 7 percent, model-generated 29 percent.

The single most consequential line in this plan is the **200B of newly digitised
native Indic text**. Without it, tier A can supply only 256B tokens, which is
13 percent of the slot, and the remainder has to be made up with translated and
synthetic text until roughly 60 percent of the Indic lane is machine-produced.
With it, tier A reaches 54 percent. For a model whose stated reason to exist is
native Indic fluency, acquisition is the lever and generation is the fallback.

Quality gate on the generated tiers: tier C is accepted only on round-trip
agreement above a fixed threshold, tier D only after native-speaker sampling at
1 in 10,000. Tier A is never diluted by either, and remains separately addressable
so the anneal can draw on it alone.

## 5. The agentic slot

3 percent of 15T is 450B tokens. The entire natural supply in the inventory is
0.63B tokens, which covers **0.6 percent of the slot even at four epochs**. This is
exactly the wishful accounting the session warns against, so the lane is built
rather than collected:

- 0.10T mined from the code lane. Issue to pull-request to diff chains, CI logs,
  notebooks and recorded shell sessions are naturally occurring multi-step traces
  with real observations and real failures. This is the largest honest source.
- 0.03T generated as execution-verified trajectories in sandboxes, kept only when
  the harness confirms the task succeeded. At 15K to 30K tokens per trajectory this
  is roughly 1.4M validated trajectories, against the 2.4K instances in SWE-Gym.
- 2.4B from natural trajectories at four epochs.

The share is held to 3 percent precisely because a larger one could not be
defended. Agentic capability is mostly purchased later, in the anneal at 12 percent
and in supervised fine-tuning and reinforcement learning, where token counts are
small and the data is at its most valuable.

## 6. Protected floors

The selector described in section 8 of the session defines usefulness through a
proxy, and an English-heavy proxy undervalues native Indic text and unfamiliar
agentic trajectories. V4 handled this with an always-on lane fixed at 8 percent for
Indic. V5 extends the protection and raises the Indic figure:

| Lane | Floor | Main-run share |
|---|---:|---:|
| Indic | 10% | 13% |
| Reasoning | 5% | 8% |
| Agentic | 2% | 3% |
| **Total protected** | **17%** | |

17 percent of every batch is outside the selector's control. The floor is a
guarantee against starvation, not a large allocation. Indic is set at 10 rather
than V4's 8 because experiment E1 shows the Indic lane still paying well above
break-even at 8 to 10 percent, and experiment E3 shows an unprotected lane losing
98.6 percent of its data to a selector that does not speak the language.

E3 also forced a correction to this section. **The selector runs inside each lane's
budget, not across lanes.** The mixture in section 2 decides how many tokens a lane
receives and the selector decides which tokens within that allocation. Floors are
kept as a second guarantee, but they are no longer the only thing standing between
the mixture and a selector that would otherwise overwrite it. The reason is in E3:
protecting only the scarce lanes pushed the entire cost onto code and STEM, which
fell to a 1.6 percent keep rate. A floor redistributes a biased selector, it does
not fix one.

## 7. The anneal reserve

**8 percent of the budget, 1.20T tokens, is held back** and spent in the final phase
at reduced learning rate. The reserve is identified now and protected from ordinary
sampling, because a reserve that the selector has already consumed does not exist
when the cooldown arrives.

This is the least defended number in the plan. Experiments E2 and E2b tested it three
times and found no benefit in any of them, including once with a reserve selected for
quality. It is retained rather than dropped because the proxy could not test the
mechanism that is believed to make annealing work, which is the coincidence of the
reserve with a sharp learning-rate decay, and because a reserve is cheap to keep and
impossible to recover once spent. Section 13 fixes the 3B test that decides it and the
share it falls to if that test is also flat.

| Lane | Anneal | Main run | Change |
|---|---:|---:|---:|
| Indic | 26% | 13% | +13pp |
| Reasoning | 18% | 8% | +10pp |
| Code | 16% | 24% | -8pp |
| Agentic | 12% | 3% | +9pp |
| STEM | 10% | 10% | 0pp |
| Long context | 10% | 8% | +2pp |
| General web | 8% | 34% | -26pp |

Reserve composition: the highest-scoring 40B tokens of tier A Indic, the whole of
SWE-Gym, SWE-smith and the OpenHands rollouts, OpenThoughts2 and
OpenMathReasoning, proof-pile-2 and peS2o. These are the Tier A datasets, and they
are spent once.

## 8. Curriculum

| Stage | Budget span | Tokens | Sequence | What changes |
|---|---|---:|---:|---|
| S0 seed | 0 to 2% | 0.30T | 4K | short, clean, high-frequency text only |
| S1 general | 2 to 34% | 4.80T | 4K | web-heavy, establishes language and world knowledge |
| S2 capability | 34 to 66% | 4.80T | 8K | code and STEM rise, web falls |
| S3 reasoning | 66 to 84% | 2.70T | 8K | reasoning traces rise, trace length grows |
| S4 long context | 84 to 92% | 1.20T | 32K then 128K | packed long documents and repositories |
| S5 anneal | 92 to 100% | 1.20T | 32K | the reserve, at decayed learning rate |

Long sequences arrive after the model can already read and reason, so it learns to
carry information across a long context without learning everything else at the
same time.

## 9. Difficulty bands

Each stage draws from a difficulty ladder. Bands are assigned by a classifier
calibrated on human-labelled samples, and the mixture within a stage moves up the
ladder as the stage progresses.

| Band | Definition | Concrete example |
|---|---|---|
| D1 elementary | single fact or single step, common vocabulary | "The capital of Bihar is Patna." |
| D2 standard | several steps, one domain, no ambiguity | A Python function that reads a CSV file and returns the column means |
| D3 advanced | multiple steps across domains, requires a choice between methods | Deriving the time complexity of a recursive algorithm and then rewriting it iteratively |
| D4 expert | genuine research or engineering difficulty, verification required | An AIME combinatorics problem, or a SWE-bench issue requiring a patch across three files that passes a hidden test suite |

## 10. Reasoning-length bands

Reasoning is reserved as a distribution of trace lengths, not one uniform slot,
because a model can only obey a reasoning-effort setting it has seen examples of.

| Band | Trace budget | Share of reasoning lane | Example |
|---|---:|---:|---|
| Low | under 200 tokens | 30% | "17 x 24. 17 x 24 = 17 x 25 - 17 = 425 - 17 = 408." |
| Medium | 200 to 1,000 | 35% | A two-variable word problem: set up the equations, solve, substitute back once to check |
| High | 1,000 to 5,000 | 25% | A geometry problem attempted by coordinates, found messy, restarted with a synthetic approach, answer verified against a special case |
| Ultra | 5,000 to 32,000 | 10% | A competition problem where three approaches are tried, two abandoned with the reason stated, intermediate results checked, and the final claim proved |

Each band spans mathematics, code and general problem solving, so that reasoning
depth does not become tied to one domain.

## 11. Mixture transitions

V4 saw the gradient norm rise roughly 150x when a sudden increase in the Hindi
share met frozen embeddings. Every boundary in section 8, and the entry to the
anneal, is therefore blended linearly across a **50B token warmup band**, about
0.33 percent of the run. No embedding is frozen across a mixture transition. The
gradient norm is monitored per step, and a rise above 4x the trailing median halts
the run.

## 12. The proxy experiments

Every number above is a hypothesis. Three were tested at a scale that fits on one
machine. The plan commits to repeating all three at 1B and 3B before the numbers
are trusted at full scale, but the results below are real, not proposed.

**Setup.** A 9.06M parameter decoder-only model (4.74M non-embedding), 6 layers,
256 hidden, 4 heads, 512 context. 30M tokens per run. Four lanes drawn from the
inventory's own datasets: FineWeb-Edu for web, GitHub source for code, Sangraha
verified Hindi for Indic reusing the Session 4 cleaning function, and NuminaMath-CoT
for reasoning. One character-level BPE vocabulary of 16,384 is trained once and
frozen across every run.

**Metric.** Held-out **bits per byte**, not loss. Devanagari costs about 8.3 bytes
per token against 3.7 for English in this vocabulary, so per-token loss is not
comparable across lanes while bits per byte is. Architecture, token budget, learning
rate schedule, batch size, seed and tokenizer are identical across arms. The
mixture is the only independent variable.

### E1. The Indic share sweep, and where the floor belongs

Indic share varied at a fixed budget, with the other three lanes held in the plan's
ratio. Held-out bits per byte, lower is better.

| Indic share | web | code | indic | math | non-Indic mean |
|---|---:|---:|---:|---:|---:|
| 0% | 2.1298 | 2.7039 | 1.6159 | 1.6730 | 2.1689 |
| 3% | 2.1483 | 2.7331 | 1.1438 | 1.6964 | 2.1926 |
| 8% | 2.1518 | 2.7528 | 1.0494 | 1.7132 | 2.2059 |
| 13% (plan) | 2.1722 | 2.7790 | 1.0093 | 1.7372 | 2.2295 |
| 30% | 2.2002 | 2.8369 | 0.9588 | 1.7883 | 2.2752 |

| Indic share | Indic gain vs 0% | Cost to other lanes | Gain per point of cost |
|---|---:|---:|---:|
| 3% | 29.2% | 1.09% | 26.8 |
| 8% | 35.1% | 1.71% | 9.5 |
| 13% (plan) | 37.5% | 2.79% | 2.2 |
| 30% | 40.7% | 4.90% | 1.5 |

The first three points of Indic are worth roughly 27 times their cost. The next five
are worth 9.5. Past 8 percent the exchange rate collapses to about 2, and past 13
percent it is 1.5.

**Effect on the plan.** The 10 percent floor is confirmed: the lane is still paying
well above break-even at 8 to 10 percent, so a floor below that gives away real
capability. The 13 percent main share is *not* strongly confirmed. It sits past the
elbow, and it buys 2.4 percent more Indic for 1.08 percent off every other lane. It
is retained because Indic is the model's stated differentiator and that exchange is
worth making for the one capability the programme exists to build, but it is retained
as a judgement call on top of a flat part of the curve, not as an optimum. The 1B run
in section 13 decides it, using MILU rather than bits per byte, because held-out loss
saturates earlier than generation quality does.

### E2 and E2b. The anneal reserve, which did not replicate

Both arms train on the same budget, the same learning-rate schedule and the same
number of tokens from every pool. Only the timing of the reserve differs. E2 uses an
arbitrary disjoint slice of Hindi, so quality is held constant and only ordering
varies. E2b repeats it with a reserve that is genuinely better data, the top 15
percent of Hindi by the Session 4 quality signal.

| Experiment | Reserve | Indic bpb, spread | Indic bpb, held back | Change |
|---|---|---:|---:|---:|
| E2 seed 0 | arbitrary slice | 1.0085 | 1.0112 | -0.27% |
| E2 seed 1 | arbitrary slice | 1.0038 | 1.0050 | -0.12% |
| E2b seed 0 | top 15% by quality | 1.0081 | 1.0171 | -0.89% |

**Holding the reserve back for the cooldown produced no benefit in any of the three
comparisons, and was marginally worse in all three.** By the decision rule fixed in
advance in section 13, a difference under 0.3 percent or in the wrong direction
refutes the hypothesis at this scale. It is refuted.

Four reasons this is a limited refutation, stated so a reviewer can weigh it:

1. Scale. 9M parameters and 30M tokens is far from convergence, and the annealing
   result the session cites is observed at 1B and above.
2. Size of the reserve. It is 2.2 percent of the run's tokens. A slice that small
   concentrated into the last tenth may not move a held-out loss measured over the
   whole distribution.
3. Metric. Bits per byte on held-out text of the same distribution is not what the
   anneal is believed to improve. The reported gains in the literature are downstream
   benchmark gains, which this proxy has no way to measure.
4. **The most likely reason.** The learning-rate schedule is a single cosine in both
   arms, which is the correct control for isolating data ordering but removes the
   mechanism. A real anneal is a joint decision about data *and* learning rate, where
   a sustained high rate is followed by a sharp decay over the reserve. Holding the
   data back while decaying the rate identically in both arms tests only half of it.

**Effect on the plan.** The 8 percent reserve is kept, and is now marked as the
weakest number in the plan. Section 13 replaces the proxy with a 3B test that uses a
warmup-stable-decay schedule rather than a cosine, so that the reserve and the decay
coincide, and that scores MILU rather than bits per byte. If that test is also flat,
the reserve drops from 8 percent to 3 and the difference returns to the main run.

### E3. The selector, and what a protected floor is actually worth

Candidate batches of 60 sequences are scored by excess loss against an English-only
reference model trained on 12M tokens of web, and the top 24 are kept. This is the
proxy failure the session describes, made concrete.

| Pool | Offered | Kept, no floor | Rate | Kept, with floor | Rate |
|---|---:|---:|---:|---:|---:|
| web | 28,956,160 | 27,545,600 | 95.1% | 24,467,456 | 84.5% |
| code | 20,603,904 | 1,261,568 | 6.1% | 334,848 | 1.6% |
| indic | 11,014,144 | 149,504 | **1.4%** | 4,971,008 | **45.1%** |
| math | 14,413,312 | 1,038,336 | 7.2% | 221,696 | 1.5% |

An English-only proxy **rejects 98.6 percent of the Hindi it is offered** and turns a
deliberate four-lane mixture into a nearly pure web run. Indic bits per byte degrades
from 1.0093 to 1.3279, a 31.6 percent loss, against the identical mixture trained
without a selector.

With a 14.8 percent always-on floor, the Indic lane recovers to **1.0093**, which is
the same number the unselected mixture reached. The floor does not partially protect
the lane. It makes the lane immune, because the floor fixes the token count at the
share the mixture designed.

**Effect on the plan, and a correction to it.** The floor works, but the cost lands
somewhere the plan did not anticipate. Protecting Indic pushes the damage onto the
unprotected lanes: code falls to a 1.6 percent keep rate and 14.6 percent worse bits
per byte, math to 1.5 percent and 19.6 percent worse. A floor on the scarce lanes does
not fix a biased selector, it redistributes the bias onto whatever is left unprotected.

The plan is therefore corrected: **the selector operates within each lane's budget, not
across lanes.** The mixture fixes how many tokens each lane receives, and OPUS chooses
which tokens inside that allocation. Protected floors remain as a second guarantee for
the scarce lanes, but the mixture itself is no longer something the selector can
overwrite. This change came out of the experiment and was not in the plan beforehand.

## 13. What would refute this plan

The proxy settled two of the three numbers and refuted the third. What remains is
specified at 1B and 3B, with the decision rule fixed in advance so that no result can
be reinterpreted after it arrives.

| Experiment | Scale | Metric | Confirms | Refutes |
|---|---|---|---|---|
| Indic share sweep, 6 / 10 / 13 / 18% | 1B, 20B tokens | MILU and IndicGenBench, not bits per byte | MILU still rising between 10% and 13%, non-Indic benchmarks down under 1% | MILU flat by 10%, or non-Indic cost above 1% |
| Anneal reserve, matched tokens, **warmup-stable-decay schedule** so the decay coincides with the reserve | 3B, 60B tokens | MILU and LiveCodeBench, plus Indic bits per byte | Held-back reserve beats even spending by 1% or more on benchmarks | Difference under 0.3%, or reversed, as at proxy scale |
| Selector inside lane budgets vs across lanes | 1B, 20B tokens | Kept-token rate per lane, MILU, LiveCodeBench | Per-lane selection holds every lane at its designed share and beats global selection with floors | Global selection with floors matches it on every lane |
| Agentic mining yield | offline | Unique tokens recovered from issue to pull-request chains, CI logs, notebooks | 0.10T or more at acceptable quality | Below 0.05T |

Pre-committed consequences:

- If MILU is flat by 10 percent, the Indic share drops from 13 to 10 and the freed
  3 points go to code. The 10 percent floor stands either way, because E1 and E3
  already support it.
- If the 3B anneal is flat under a warmup-stable-decay schedule as well, the reserve
  drops from 8 percent to 3 and the rest returns to the main run. Two flat results
  at two scales with two different reserve qualities would mean the mechanism is not
  worth 1.2T tokens of deferred data.
- If agentic mining yields under 0.05T unique tokens, the agentic share drops from
  3 percent to 2 rather than being topped up with more synthetic trajectories. This
  is the rule that stops the lane from becoming the wishful accounting the session
  warns about.
- If per-lane selection does not beat global selection with floors, the correction in
  E3 is reverted and the floors are extended to code and STEM instead.

## 14. Reproducing

```bash
python3 proxy/build_corpus.py      # four lanes from the inventory's datasets
python3 proxy/train_tokenizer.py   # one frozen 16,384 character-level BPE
python3 proxy/tokenize_corpus.py     # uint16 arrays plus a held-out tail per lane
python3 proxy/build_quality_split.py # the quality-ranked Hindi pools used by E2b
python3 proxy/run_experiments.py --tokens 30e6
python3 proxy/report.py              # section 12
python3 proxy/plan_arithmetic.py   # sections 2 to 7
```

14 runs, roughly 4 hours on one Apple M4 Pro. `proxy/inventory.json` is the Session 5
dataset inventory, extracted from the session's inventory widget, and is the supply
figure behind every table in sections 2 to 7. `proxy_results.txt` and
`plan_arithmetic.txt` are the raw output of the two report scripts, committed so the
tables above can be checked against them. Corpus files and run outputs are not
committed; the scripts regenerate them.
