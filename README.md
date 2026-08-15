# V5 Mixture and Curriculum Plan

ERA V5, Session 5. A mixture and curriculum specification for a 15 trillion token
pretraining run, composed backward from the benchmarks the model is meant to win,
sized against the real supply in the Session 5 dataset inventory, and tested by
proxy runs that were actually executed rather than only proposed.

Three numbers carry the plan. The mixture assigns **13 percent to Indic**, a
**17 percent protected floor** sits outside the data selector's control, and
**8 percent of the budget is held back** for the final anneal. Each of the three
was tested, and the results are in section 12.

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
than V4's 8 because experiment E1 shows the Indic lane still improving between
8 and 13 percent, and experiment E3 shows what an unprotected lane does under a
selector that does not speak the language.

## 7. The anneal reserve

**8 percent of the budget, 1.20T tokens, is held back** and spent in the final phase
at reduced learning rate. The reserve is identified now and protected from ordinary
sampling, because a reserve that the selector has already consumed does not exist
when the cooldown arrives.

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

RESULTS_PLACEHOLDER

## 13. What would refute this plan

The proxy runs above are small. The same three experiments are specified at 1B and
3B, with the decision rule fixed in advance so the result cannot be reinterpreted
after the fact.

| Experiment | Metric | Confirms the plan | Refutes it |
|---|---|---|---|
| Indic share sweep at 1B, shares 6 / 10 / 13 / 18% | MILU, IndicGenBench, Indic bits per byte | Indic gains continue to 13% and non-Indic loss rises under 0.5% | Indic gains flatten by 8%, or non-Indic cost exceeds 1% |
| Anneal ordering at 3B, matched tokens | Indic and reasoning bits per byte, MILU | Held-back reserve beats even spending by 1% or more | Difference under 0.3%, or reversed |
| Selector floor at 1B | Indic kept-token rate, MILU | Unprotected Indic retention falls below 3%, floor restores capability | Selector retains Indic without a floor |

If the 1B sweep flattens by 8 percent, the Indic share drops to 10 and the freed
3 percent goes to code. If the anneal shows no effect at 3B, the reserve falls from
8 percent to 3 and the rest returns to the main run. If the agentic mining yield
comes in below 0.05T unique tokens, the agentic share drops to 2 percent rather
than being filled with more synthetic data.

## 14. Reproducing

```bash
python3 proxy/build_corpus.py      # four lanes from the inventory's datasets
python3 proxy/train_tokenizer.py   # one frozen 16,384 character-level BPE
python3 proxy/tokenize_corpus.py   # uint16 arrays plus a held-out tail per lane
python3 proxy/run_experiments.py --tokens 30e6
python3 proxy/report.py            # section 12
python3 proxy/plan_arithmetic.py   # sections 2 to 7
```

`proxy/inventory.json` is the Session 5 dataset inventory, extracted from the
session's inventory widget, and is the supply figure behind every table above.
Corpus files and run outputs are not committed; the scripts regenerate them.
