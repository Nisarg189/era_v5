# V5 Mixture and Curriculum Plan

**ERA V5, Session 5.** This document decides what the model is trained on: how the
15 trillion token pretraining budget is divided between capabilities, in what order
those capabilities are taught, what is protected from being dropped along the way,
and what is held back for the end.

Three of its numbers were not argued for but tested, by training 14 small models from
scratch and measuring what changed. Two survived. One was refuted, and the plan was
altered because of it. Sections 15 and 16 give those results, including the one that
went against the plan.

Every table here is printed by a script in this repository rather than typed by hand.
`proxy/plan_arithmetic.py` produces sections 5 to 13 from the dataset inventory.
`proxy/report.py` produces section 15 from the training runs. Both outputs are
committed as `plan_arithmetic.txt` and `proxy_results.txt` so the tables can be
checked against their source.

---

## 01 What a mixture is, and why it is the whole assignment

A model learns only from the text it is shown. The corpus does not decide what the
model becomes; the **proportions** do. The same clean data and the same compute
produce a very different model depending on how much of each kind of text goes in.

Because the total is fixed, every share given to one capability is taken from another.
Designing the mixture is therefore not bookkeeping. It is the decision about which
capabilities matter enough to pay for with something else.

Two words recur below. A **lane** is one category of training data, such as code or
Indic text. The **mixture** is the set of percentages assigned to those lanes.

## 02 The token budget, and where 15 trillion comes from

The figure is carried forward from the Session 3 plan for this model, which set a
target of roughly 15 trillion pretraining tokens for a 40 billion parameter model.
The reasoning there was that 15T is about 375 tokens per parameter, far past
Chinchilla's compute-optimal ratio of roughly 20 tokens per parameter, which for 40B
would be only 800 billion tokens. Training well beyond the compute-optimal point is a
deliberate trade: it costs more once, during training, and produces a smaller model
that is cheaper on every inference request afterwards. Gemma-class models sit in the
same 13 to 15 trillion range.

Session 3 shows the V5 target as a range of 10 to 30 trillion tokens. 15T is the
conservative end of that range, used here for consistency with the earlier plan rather
than re-derived.

## 03 What data actually exists

The Session 5 dataset inventory lists 32 named corpora with token counts. Summed, it
holds **6.40 trillion unique tokens against a 15 trillion budget**.

That number needs a qualification, because on its own it misleads. The inventory is a
shortlist of corpora that are already filtered, deduplicated, licensed and ready to
train on. It is not a measure of how much text exists. Session 3 cites Epoch AI's
estimate of roughly **300 trillion tokens** of public human text, so the inventory is
about 2 percent of it.

The distinction matters because it separates two very different problems:

| Lane | Is the shortage real? |
|---|---|
| General web | No. Abundant in the world. The inventory row is a quality-filtered slice, and more can be filtered from Common Crawl at a lower quality bar. |
| STEM / maths | No. Largely a labelling and extraction problem rather than a collection one. |
| Long context | No. Not a sourcing problem at all. Long-context data is built by packing documents that already sit in other lanes. |
| Indic | **Yes.** Genuinely scarce in the world, not merely uncollected. |
| Reasoning | **Yes**, though it is the category most legitimately produced by generation. |
| Agentic | **Yes, severely.** Barely exists anywhere, at any price. |

So the constraint is not that text has run out. It is that the specific kinds of text
this model needs most are the hardest to get. Everything below follows from that.

Where demand exceeds what is on hand, only three honest options exist: repeat the data,
acquire more, or manufacture it. Repetition is bounded. Following Muennighoff et al.,
*Scaling Data-Constrained Language Models* (NeurIPS 2023), repeating natural text up to
roughly **four passes** costs almost nothing in quality, and returns decay sharply
after. That four-epoch ceiling is applied to every lane below. Model-generated text is
held to two passes, because its errors compound when repeated.

## 04 Working backward from the benchmarks

A capability nobody measures cannot be shown to exist. Each lane therefore exists to
move specific benchmarks, and the training data has to take the shape the benchmark
actually scores.

| Lane | Benchmarks | What the training example must look like |
|---|---|---|
| Code | LiveCodeBench, Aider | loss on the generated patch or completion |
| Agentic | SWE-bench, tau-bench, BFCL, GAIA, BrowseComp | loss on the model's planning, tool calls and final answer, and never on the tool's response |
| Reasoning | AIME, GPQA, HLE | loss on the worked trace and the final answer |
| STEM | AIME, GPQA | loss on derivations and notation |
| Indic | MILU, IndicGenBench | loss on native Indian-language text |
| Long context | long-eval | loss on tokens whose supporting evidence sits tens of thousands of tokens earlier |
| General web | MMLU | loss on broad world knowledge |

The masking rule in the agentic row is part of the specification, not an implementation
detail. In a tool-using trajectory, the user's request and every tool observation are
context the model reads but is never trained to produce. Training on a tool observation
teaches the model to invent tool results instead of calling the tool.

## 05 The curriculum: six stages, each with its own mixture

The mixture is not one set of percentages held constant for 15 trillion tokens. It
moves. Training begins on broad general text to establish language and world knowledge,
shifts toward code and reasoning once that foundation exists, introduces very long
sequences only after the model can already read and reason, and ends with a short
concentrated phase on reserved high-quality data.

**How to read this table.** *Span* is the share of the whole 15 trillion token run
that the stage occupies, so S1 runs from 2 percent to 34 percent of training and
consumes 4.80T tokens. The seven lane columns are **percentages of that stage's own
tokens**, and each row therefore adds to 100. They are not percentages of the whole
run. Example: during S2, out of every 100 tokens the model sees, 30 are general web,
31 are code and 12 are Indic.

| Stage | Span of run | Tokens | web | code | indic | stem | reason | longctx | agentic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 seed | 0 to 2% | 0.30T | 52 | 14 | 15 | 10 | 3 | 2 | 4 |
| S1 general | 2 to 34% | 4.80T | 46 | 20 | 14 | 11 | 4 | 2 | 3 |
| S2 capability | 34 to 66% | 4.80T | 30 | 31 | 12 | 12 | 8 | 4 | 3 |
| S3 reasoning | 66 to 84% | 2.70T | 22 | 26 | 12 | 9 | 17 | 11 | 3 |
| S4 long context | 84 to 97% | 1.95T | 18 | 24 | 12 | 6 | 11 | 26 | 3 |
| S5 anneal | 97 to 100% | 0.45T | 8 | 16 | 26 | 10 | 18 | 10 | 12 |

Sequence length rises with the stages: 4K through S0 and S1, 8K through S2 and S3, 32K
then 128K in S4, and 32K in the anneal.

Reading down a column shows what the plan is doing. General web falls from 52 to 8 as
its job, teaching basic language, is finished early. Code rises to a peak in S2 and
stays high. Reasoning is near zero at the start, because reasoning traces teach nothing
to a model that cannot yet form sentences. Indic holds between 12 and 15 throughout
rather than spiking, then rises sharply in the anneal. Agentic stays flat and small
until the anneal, where it quadruples.

This shape follows V4, which moved general web from roughly 70 percent down toward 18
percent across its growth stages while code went from 13 to 35 percent, and kept one
protected channel fixed at 8 percent throughout.

## 06 The headline mixture, and whether the data exists to fill it

The often-quoted single mixture is not a separate decision here. It is what the six
stages average out to:

| Lane | Share | Demand | Inventory | Added | Effective | Epochs | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| General web | 32% | 4.80T | 4.69T | 0 | 4.69T | 1.0 | covered |
| Code | 25% | 3.75T | 1.10T | 0 | 1.10T | 3.4 | repeat |
| Indic | 13% | 1.95T | 0.28T | 0.39T | 0.66T | 3.0 | repeat and acquire |
| STEM | 10% | 1.50T | 0.15T | 0.35T | 0.50T | 3.0 | repeat and mine |
| Reasoning | 9% | 1.35T | 0.09T | 0.28T | 0.37T | 3.7 | repeat and generate |
| Long context | 8% | 1.20T | 0.10T | 0.60T | 0.70T | 1.7 | pack |
| Agentic | 3% | 0.45T | 0.0006T | 0.13T | 0.13T | 3.4 | mine and build |

"Added" is new unique data the plan commits to producing or acquiring, itemised in the
next section. "Epochs" is how many times the effective supply is read. **No lane
exceeds the four-epoch ceiling**, which is the test this table exists to pass.

## 07 Closing the gap: 1.75T of tokens that do not exist yet

The plan requires 1.75 trillion unique tokens beyond the inventory. Naming where each
one comes from is the difference between a plan and a wish.

| Lane | Amount | Mechanism | Source |
|---|---:|---|---|
| Long context | 0.60T | pack | documents and repositories of 8K tokens or more already inside the web and code lanes |
| Reasoning | 0.28T | generate | teacher distillation, filtered by checking the final answer against a verifier |
| STEM | 0.25T | mine | classifier-selected STEM already inside DCLM and FineWeb-Edu, relabelled rather than re-sourced |
| Indic | 0.20T | harvest | re-mine the full Common Crawl archive for Indian-language pages, plus open public-sector text: court judgments, Parliament and assembly proceedings, gazettes, NCERT and state textbooks, NPTEL and SWAYAM transcripts, Indian-language Wikipedia and Wikisource, National Digital Library public-domain scans |
| Indic | 0.185T | generate | translation of curated English with IndicTrans2 into tier C, templated and self-instruct text into tier D |
| STEM | 0.10T | harvest | arXiv bulk access, PubMed Central open-access subset, OpenStax and other open textbooks |
| Agentic | 0.10T | mine | issue to pull-request to diff chains, CI logs, notebooks and shell sessions |
| Agentic | 0.03T | generate | execution-verified trajectories in tool sandboxes |

By mechanism: 0.60T packed, 0.49T generated, 0.35T mined, 0.30T harvested. **Only 0.49T
of the 1.75T is model-generated.** That ratio is deliberate. Synthetic text is the
cheapest option and the one most likely to degrade the model, so it is used where it is
most defensible, in reasoning traces whose final answer can be checked automatically,
and avoided where it is least, in replacing native Indic prose.

**Every mechanism here is executable with compute alone.** Nothing in this plan
requires a purchase, a licence negotiation or an outside vendor. Harvesting means
crawling and extracting from sources that are already public domain, openly licensed or
government-published. Translation uses IndicTrans2, which AI4Bharat released openly,
rather than a translation agency. This constraint was applied deliberately: a plan that
depends on a procurement budget is not a plan this programme can execute, and a share
that can only be filled by spending money the programme does not have is the same
wishful accounting as a share with no data behind it.

## 08 The Indic slot, tier by tier

13 percent of 15T is 1.95 trillion tokens, split across the four provenance tiers
established in Session 3.

| Tier | Existing unique | New unique | Epochs | Tokens | Share of slot |
|---|---:|---:|---:|---:|---:|
| A verified native | 64.0B | 200B | 4x | 1056B | 54.2% |
| B unverified crawl | 44.9B | 0B | 4x | 180B | 9.2% |
| C translated | 5.0B | 60B | 2x | 130B | 6.7% |
| D synthetic | 162.0B | 125B | 2x | 574B | 29.4% |

Native human text is 63 percent of the slot, translated 7 percent, machine-generated
29 percent.

**The most consequential line in this plan is the 200 billion tokens of additional
native text in tier A.** Using only the inventory, tier A supplies 256B tokens even at
four passes, which is 13 percent of the slot. Everything else would have to be filled
with translated or generated text until roughly 60 percent of the Indic lane was
machine-produced. For a model whose stated reason to exist is native fluency in Indian
languages, training it mostly on machine-produced Indian-language text defeats the
purpose. Adding 200B tokens of real text moves tier A from 13 percent of the slot to
54 percent.

**Where those 200 billion tokens come from, without buying anything.** The inventory's
Indic rows are the output of one pipeline (AI4Bharat's) over a subset of the crawl. They
are not the ceiling on Indian-language text that is freely available:

| Source | Why it is available and large |
|---|---|
| Common Crawl, re-mined for Indian languages | The largest single source. Indian-language pages are present across the full archive but under-extracted, because language identification and quality filters tuned for English discard them. Session 4 measured this directly: an English-tuned quality filter wrongly rejected 97 percent of good Hindi. Re-running extraction with script-aware filters over the whole archive rather than a subset is compute, not procurement. |
| Court judgments | Every High Court and Supreme Court judgment is public, and a large share is in Indian languages. Decades of it, already digitised. |
| Parliament and state assembly proceedings | Verbatim transcripts, public by law, in multiple Indian languages. |
| Government gazettes, ministry reports, data.gov.in | Public-sector text, published openly, and written in formal registers the crawl under-represents. |
| NCERT and state board textbooks | Free to download, professionally edited, and pitched exactly at the difficulty bands in section 10. |
| NPTEL and SWAYAM transcripts | Publicly funded lecture material with transcripts, covering technical vocabulary in Indian languages. |
| Indian-language Wikipedia and Wikisource | Wikisource in particular holds large volumes of already-digitised public-domain Indian literature. |
| National Digital Library of India | Public-domain scans, needing optical character recognition rather than licensing. |

The work is crawling, extraction, script-aware filtering and optical character
recognition. All of it is compute and engineering the programme already does. The one
genuine cost beyond compute is proofreading a sample of the recognised text to measure
its error rate, which is a small human task rather than a procurement exercise.

Licensing commercial news archives would add more, and is worth doing if a budget ever
exists, but **this plan does not depend on it.** For an India-first model, extraction is
the lever and generation is the fallback.

Quality gates on the generated tiers: tier C is accepted only above a fixed round-trip
translation agreement threshold, which is an automatic check, and tier D only after
native-speaker sampling at one document in ten thousand, which the cohort can do itself. Tier A is never diluted with either and stays separately
addressable, so the anneal can draw on it alone.

## 09 The agentic slot, the hardest number in the plan

Agentic data means complete trajectories: a model planning a task, calling a tool,
reading the result, recovering when a call fails, and continuing until it finishes.

3 percent of 15T is 450 billion tokens. **The entire natural supply in the inventory is
0.63 billion tokens**, which covers 0.6 percent of the slot even at four passes. A
larger agentic share would be exactly the wishful accounting this session warns
against, so the share is held low deliberately and the lane is filled three ways:

- **0.10T mined from the code lane.** An issue, its discussion, the resulting pull
  request, the diff and the test run form a naturally occurring multi-step trace with
  genuine observations and genuine failures. CI logs, notebooks and recorded shell
  sessions are the same. This is the largest honest source and requires no generation.
- **0.03T generated** as trajectories executed in sandboxes and kept only when the test
  harness confirms the task actually succeeded. At 15K to 30K tokens per trajectory
  that is roughly 1.4 million validated trajectories, against the 2,400 instances in
  SWE-Gym. This is a large engineering programme, and is stated as one.
- **2.4B from the natural trajectories** at four passes.

Agentic capability is mostly bought after pretraining, which is why the anneal raises
this lane to 12 percent and why supervised fine-tuning and reinforcement learning
matter more here than the pretraining share does. Those later stages use very few
tokens, but the most valuable ones.

## 10 Difficulty bands

Within each stage the data also climbs a difficulty ladder, so the model is not shown
graduate material before it can handle grade-school material.

| Band | Level | Example |
|---|---|---|
| B0 | Nursery | "The sun is hot. The sky is blue." |
| B1 | Grade school | A short paragraph explaining why it rains, or a Python function that adds two numbers |
| B2 | High school | A history passage on Chola administration, or a function that reads a CSV file and returns column means |
| B3 | Undergraduate | Deriving the time complexity of a recursive algorithm, then rewriting it iteratively |
| B4 | Graduate | A proof by induction with base case and inductive step stated separately, or a concurrency bug traced through a lock ordering |
| B5 | Research | An AIME combinatorics problem, or a SWE-bench issue needing a patch across three files that passes a hidden test suite |

Early stages sit mostly in B0 to B2. S2 and S3 move the mass to B3 and B4. B5 material
is concentrated in S3 and the anneal, where the model can extract something from it.

## 11 Reasoning-length bands

A model can only obey a reasoning-effort setting at inference time if it was trained on
examples across the full range of depths. Reserving a reasoning share therefore means
reserving a **distribution of trace lengths**, not one uniform kind of example.

| Band | Trace budget | Share of reasoning lane | Example |
|---|---:|---:|---|
| Low | under 200 tokens | 30% | "17 x 24 = 17 x 25 - 17 = 425 - 17 = 408." |
| Medium | 200 to 1,000 | 35% | A two-variable word problem: set up both equations, solve, substitute back once to check |
| High | 1,000 to 5,000 | 25% | A geometry problem attempted with coordinates, found messy, restarted with a synthetic argument, then verified against a special case |
| Ultra | 5,000 to 32,000 | 10% | A competition problem where three approaches are tried, two abandoned with the reason recorded, intermediate results checked, and the final claim proved |

Each band spans mathematics, code and general problem solving, so reasoning depth does
not become welded to one domain. The curriculum introduces short traces first; longer
ones arrive in S3.

## 12 The selector, and the protected floor

Modern training does not fix the recipe and walk away. A selector scores candidate
batches while the run is in progress and keeps the most useful fraction, so the same
compute buys more learning. In V4 this retained about 40 percent of candidate data for
roughly a sixfold increase in effective token value.

The danger is that the selector defines "useful" through a proxy, and a proxy trained
mostly on English does not recognise value in native Indic text or in unfamiliar
agentic trajectories. It will quietly discard exactly the capabilities the model exists
to build. Section 15 shows this happening and measures it.

The defence is a **protected floor**: a share of every batch reserved for scarce lanes
regardless of what the selector scores them. V4 used an always-on lane fixed at
8 percent for Indic. This plan extends the protection and raises the Indic figure:

| Lane | Floor | Headline share |
|---|---:|---:|
| Indic | 10% | 13% |
| Reasoning | 5% | 9% |
| Agentic | 2% | 3% |

17 percent of every batch sits outside the selector's control. A floor is a guarantee
against starvation, not a large allocation.

The floor sits on top of a structural rule that the experiments forced into the plan:
**the selector operates inside each lane's budget, never across lanes.** The mixture
decides how many tokens a lane receives; the selector decides which tokens within that
allocation. Section 16 explains why.

## 13 The anneal reserve

The final stage trains briefly at a decayed learning rate on a reserve of the best
data, held back from the main run. The capability gain is reported to be
disproportionate to its size. That gain is only available if the reserve survives,
which is why it is a mixture decision made now: if the selector consumes the best Indic
and agentic data early, nothing special is left when the cooldown arrives.

**3 percent of the budget, 0.45 trillion tokens, is held back.** Session 5 places this
phase at about 2 percent of tokens within a 1 to 5 percent band, and 3 percent sits
inside it. An earlier draft of this plan used 8 percent, which was outside the course's
range and which the proxy experiments then failed to support. Both reasons brought it
down.

Reserve composition: the highest-scoring 40B tokens of tier A Indic, the whole of
SWE-Gym, SWE-smith and the OpenHands rollouts, OpenThoughts2 and OpenMathReasoning,
proof-pile-2 and peS2o. These are spent once.

This remains the least evidenced number in the plan, for reasons section 15 sets out.

## 14 Keeping the run stable when the mixture moves

Every stage boundary changes the data distribution, which changes the gradients. V4 saw
the gradient norm rise roughly 150 times when a sudden increase in the Hindi share met
embeddings that had been frozen, and an event that size can destroy a run.

Every boundary in section 5, and the entry to the anneal, is therefore blended linearly
across a **50 billion token warmup band**, about 0.33 percent of the run, rather than
switched in one step. No embedding is frozen across a mixture transition. Gradient norm
is monitored per step and a rise above four times the trailing median halts the run.

## 15 The proxy experiments

Every number above is a hypothesis until something cheap has tested it. Three were
tested for real.

**What was built.** A 9.06 million parameter decoder-only transformer (4.74M
non-embedding), 6 layers, 256 hidden, 4 heads, 512 context. Each run trains from
scratch on 30 million tokens. Four lanes were assembled from the inventory's own
datasets: FineWeb-Edu for web, permissively licensed GitHub source for code, Sangraha
verified Hindi for Indic (reusing the Session 4 cleaning function unchanged, so the
joiners are preserved), and NuminaMath-CoT for reasoning. One character-level BPE
vocabulary of 16,384 was trained once and frozen across every run. 14 runs, about four
hours on one Apple M4 Pro.

The plan's seven lanes collapse to these four, with STEM and reasoning combined. The
Indic share in the projection is 14.8 percent against the plan's 13, and that share is
the variable E1 actually sweeps.

**A note on the proxy tokenizer, since the byte figures invite a misreading.** This
vocabulary encodes Hindi at **8.29 bytes per token** against 3.80 for English. That is
not a fertility problem, it is a property of UTF-8: Devanagari costs about 2.5 bytes per
character where English costs 1, so a Hindi token covering 3.27 characters covers 8.29
bytes. Measured the way this course defines it, as **tokens per word**, the proxy
tokenizer gives Hindi **1.570** against English **1.618**, so Hindi is marginally
*better*, not worse.

| Lane | Tokens per word | Characters per token | Bytes per token |
|---|---:|---:|---:|
| web | 1.618 | 3.78 | 3.80 |
| indic | **1.570** | 3.27 | 8.29 |
| math | 1.636 | 3.52 | 3.52 |
| code | 4.185 | 2.51 | 2.53 |

For context, the Assignment 4 tokenizer, a 32,000-vocabulary character-level BPE trained
on Hindi alone, reached 1.178 tokens per word on the same corpus, against 4.984 for an
English-first tokenizer. The proxy's 1.570 is worse than 1.178 because this vocabulary
is only 16,384 entries and is shared across four very different lanes including code.
That is the correct trade for an experiment: the tokenizer here is a controlled
variable, held identical across all 14 runs, not the production tokenizer. The
production vocabulary remains the 262,144 entries specified in Session 3.

**How results are measured.** Held-out **bits per byte**, not loss. Because the lanes use
different scripts, a per-token loss would compare encodings rather than models. Bits per
byte divides out the tokenizer entirely, which is why the byte figures above cannot
distort any comparison below. Absolute values are still not comparable *between* lanes, because
languages differ in how much information a byte carries, so every claim below compares
the same lane across different runs. Architecture, token budget, learning-rate
schedule, batch size, seed and tokenizer are identical across arms. The mixture is the
only thing that varies.

### E1. How much Indic is worth including

The same model was trained five times from scratch, changing only the Indic share and
letting the other three lanes absorb the difference in their planned ratio.

**How to read this table.** Each row is one separately trained model. The four lane
columns are that model's **held-out bits per byte on each lane, where lower is
better**: how many bits the model needs, on average, to predict one byte of text it
has never seen. The last column averages the three non-Indic lanes, which is what the
Indic share is being paid for out of. Reading row three: a model trained with 8 percent
Indic scores 1.0494 on Hindi and 2.2059 averaged across the other three lanes.

| Indic share | web | code | indic | math | non-Indic mean |
|---|---:|---:|---:|---:|---:|
| 0% | 2.1298 | 2.7039 | 1.6159 | 1.6730 | 2.1689 |
| 3% | 2.1483 | 2.7331 | 1.1438 | 1.6964 | 2.1926 |
| 8% | 2.1518 | 2.7528 | 1.0494 | 1.7132 | 2.2059 |
| 13% (plan) | 2.1722 | 2.7790 | 1.0093 | 1.7372 | 2.2295 |
| 30% | 2.2002 | 2.8369 | 0.9588 | 1.7883 | 2.2752 |

The same result restated as a trade. "Indic improvement" is how much better the Hindi
score is than the 0 percent model. "Cost to other lanes" is how much worse the other
three became. The last column is simply the first divided by the second, which is the
exchange rate being offered at that share.

| Indic share | Indic improvement | Cost to other lanes | Improvement per unit of cost |
|---|---:|---:|---:|
| 3% | 29.2% | 1.09% | 26.8 |
| 8% | 35.1% | 1.71% | 9.5 |
| 13% (plan) | 37.5% | 2.79% | 2.2 |
| 30% | 40.7% | 4.90% | 1.5 |

The first three points of Indic return roughly 27 times what they cost. The next five
return 9.5. After 8 percent the exchange rate collapses to about 2.

**Finding.** The 10 percent floor is confirmed: the lane is still well above
break-even at 8 to 10 percent, so a lower floor gives away real capability. The
13 percent share is **not** strongly confirmed. It sits past the elbow and buys
2.4 percent more Indic for about 1 percent off every other lane. It is kept, because
Indic is the one capability this model exists for and that trade is worth making for
it, but it is kept as a judgement call on a flat part of the curve rather than as an
optimum. Section 17 specifies the test that settles it.

### E2 and E2b. The anneal reserve, which did not replicate

Both arms train on the same budget, the same learning-rate schedule, and give every
data pool the same number of tokens. Only **when** the reserve is used differs: spread
evenly through the run, or held back entirely for the final tenth. E2 uses an arbitrary
slice of Hindi, holding quality constant so only timing varies. E2b repeats the test
with a reserve that is genuinely better data, the top 15 percent of Hindi by the
Session 4 quality signal.

| Run | Reserve | Spread evenly | Held for anneal | Difference |
|---|---|---:|---:|---:|
| E2 seed 0 | arbitrary slice | 1.0085 | 1.0112 | -0.27% |
| E2 seed 1 | arbitrary slice | 1.0038 | 1.0050 | -0.12% |
| E2b seed 0 | top 15% by quality | 1.0081 | 1.0171 | -0.89% |

**Holding the reserve back produced no benefit in any of the three comparisons, and was
marginally worse in all three.** The threshold for refutation was written down before
the runs, and this clears it.

Four limits on that conclusion, stated so a reviewer can weigh it properly:

1. **Scale.** 9M parameters and 30M tokens is very far from convergence. The annealing
   result being tested is observed at 1B parameters and above.
2. **Size of the reserve.** It is 2.2 percent of the run's tokens. A slice that small,
   concentrated into the last tenth, may not move a loss averaged over everything.
3. **Metric.** Held-out bits per byte on the same distribution is not what annealing is
   believed to improve. Reported gains elsewhere are downstream benchmark gains, which
   this proxy cannot measure.
4. **The most likely explanation.** The learning rate follows a single cosine curve in
   both arms. That is the correct control for isolating data ordering, but it removes
   the mechanism. A real anneal changes two things together: the data becomes higher
   quality *and* the learning rate decays sharply so the model consolidates on it.
   Holding the data back while decaying identically in both arms tests only half of it.

**Finding.** The reserve is kept but cut from 8 percent to 3, and is now the weakest
number in the plan. Section 17 replaces this proxy with a 3B test using a
warmup-stable-decay schedule, so the reserve and the decay coincide, scored on
benchmarks rather than on loss.

### E3. What a protected floor is actually worth

Candidate batches of 60 sequences are scored by a reference model trained only on
English web, and the best 24 are kept. This is the proxy-bias failure made concrete.

| Lane | Offered | Kept, no floor | Rate | Kept, with floor | Rate |
|---|---:|---:|---:|---:|---:|
| web | 28,956,160 | 27,545,600 | 95.1% | 24,467,456 | 84.5% |
| code | 20,603,904 | 1,261,568 | 6.1% | 334,848 | 1.6% |
| indic | 11,014,144 | 149,504 | **1.4%** | 4,971,008 | **45.1%** |
| math | 14,413,312 | 1,038,336 | 7.2% | 221,696 | 1.5% |

An English-only proxy **rejects 98.6 percent of the Hindi it is offered** and turns a
deliberately balanced four-lane mixture into a nearly pure web run. Indic bits per byte
degrades from 1.0093 to 1.3279, a **31.6 percent loss**, against the identical mixture
trained with no selector.

Switch on a 14.8 percent always-on floor and Indic recovers to **1.0093**, the figure
the unselected mixture reached. The floor does not partly protect the lane; it makes it
immune, because the floor pins the token count to the share the mixture designed.

**Finding, and a correction.** The floor works exactly as claimed. But the cost lands
somewhere the plan had not anticipated: protecting Indic pushed the damage onto the
lanes that were *not* protected. Code fell to a 1.6 percent keep rate and 14.6 percent
worse bits per byte, maths to 1.5 percent and 19.6 percent worse. A floor does not
repair a biased selector, it redistributes the bias onto whatever is left unguarded.

## 16 What the experiments changed

Three parts of this plan are different because results contradicted them.

1. **The selector now runs inside each lane's budget rather than across lanes**
   (section 12). E3 showed that global selection with floors protects the guarded lanes
   by destroying the unguarded ones. Making the mixture something the selector cannot
   overwrite fixes the cause rather than the symptom. This was not in the plan
   beforehand.
2. **The anneal reserve fell from 8 percent to 3** (section 13), for two independent
   reasons: it was outside the 1 to 5 percent band the session states, and three
   separate comparisons found no benefit from it.
3. **The 13 percent Indic share is now presented as a judgement call, not a result**
   (section 08), because E1 put the elbow of the return curve nearer 8 percent.

The 10 percent floor is the one headline number that came out of testing stronger than
it went in.

## 17 How the remaining numbers get checked

This section exists because of a failure mode that is easy to fall into. Once a result
arrives, it is very tempting to reinterpret it so that it agrees with the plan already
written. The defence is to state, **before running anything**, exactly what outcome
would count as confirmation and what outcome would force a change, and to name the
consequence of each in advance.

That is what the table does. Each row is a test not yet run, at 1 billion or 3 billion
parameters rather than the 9 million used so far. "Confirms" and "Refutes" are the
thresholds fixed now. The list beneath states what actually changes in this document if
a test refutes its row, so the answer is already decided rather than argued about later.

This is the same discipline that produced the anneal result in section 15: the
refutation threshold had been written down first, which is why a disappointing result
became a finding rather than something to explain away.

| Test | Scale | Metric | Confirms | Refutes |
|---|---|---|---|---|
| Indic share sweep at 6 / 10 / 13 / 18% | 1B, 20B tokens | MILU and IndicGenBench, not bits per byte | MILU still rising between 10% and 13%, other benchmarks down under 1% | MILU flat by 10%, or non-Indic cost above 1% |
| Anneal reserve, matched tokens, **warmup-stable-decay schedule** so the decay coincides with the reserve | 3B, 60B tokens | MILU and LiveCodeBench | Reserve beats even spending by 1% or more | Under 0.3%, or reversed, as at proxy scale |
| Selection inside lane budgets vs across lanes with floors | 1B, 20B tokens | Kept-token rate per lane, MILU, LiveCodeBench | Per-lane selection holds every lane at its designed share and wins | Global selection with floors matches it on every lane |
| Agentic mining yield | offline | Unique tokens recovered from issue-to-pull-request chains, CI logs, notebooks | 0.10T or more at acceptable quality | Below 0.05T |

Pre-committed consequences:

- If MILU is flat by 10 percent, the Indic share drops from 13 to 10 and the freed three
  points go to code. The 10 percent floor stands regardless, because E1 and E3 both
  support it.
- If the 3B anneal is also flat under a decay schedule, the reserve drops from 3 percent
  to 1 and the rest returns to the main run. Two flat results at two scales with two
  reserve qualities would mean the mechanism is not worth deferring data for.
- If agentic mining yields under 0.05T unique tokens, the agentic share drops from
  3 percent to 2 rather than being topped up with more synthetic trajectories. This is
  the rule that stops the lane becoming the wishful accounting the session warns about.
- If per-lane selection does not beat global selection with floors, the correction in
  section 16 is reverted and floors are extended to code and STEM instead.

## 18 Conclusion

The mixture is the model. Working backward from the benchmarks and forward from what
data can actually be obtained, this plan commits to the following.

**The shape of the run.** 15 trillion tokens across six stages, with the mixture moving
from 52 percent general web at the start to 8 percent at the end, code peaking in the
middle third, reasoning entering only after language is established, long sequences
after that, and a short concentrated final phase. The single headline mixture, web 32,
code 25, Indic 13, STEM 10, reasoning 9, long context 8, agentic 3, is the average of
those stages rather than an independent decision.

**The binding constraint is supply, not preference.** The inventory holds 6.40T unique
tokens for a 15T budget. Every lane is checked against a four-epoch repetition ceiling,
and the 1.75T shortfall is itemised by mechanism, with only 0.49T of it model-generated
and none of it requiring a purchase.

**The Indic lane is won by extraction, not generation.** 200 billion additional tokens
of native text, obtained by re-mining the crawl with script-aware filters and harvesting
open public-sector, educational and public-domain sources, moves verified native text
from 13 percent of the Indic slot to 54 percent. No volume of synthetic Hindi
substitutes for that, and Session 4 already measured why the crawl under-delivers Indian
languages: English-tuned filters discard 97 percent of good Hindi.

**The agentic lane is the honest weak point**, at 3 percent, held deliberately low
because the entire natural supply covers 0.6 percent of even that. It is mined from
naturally occurring multi-step traces in the code lane rather than inflated with
synthetic trajectories, and the capability is mostly bought after pretraining.

**What testing changed.** Fourteen models were trained to check three numbers. The
protected floor came out stronger than it went in: an English-only selector discarded
98.6 percent of the Hindi it was offered, and the floor restored the lane completely.
The Indic share of 13 percent came out weaker, sitting past the elbow of its own return
curve, and is now labelled a judgement call. The anneal reserve was refuted three times
and fell from 8 percent to 3. One structural correction came out of the experiments
rather than the reading: the selector must operate inside each lane's budget, because
protecting only the scarce lanes simply moves the damage to the unprotected ones.

**What this plan is not.** It is not a set of numbers that survived because nobody
tested them. Three were tested, one failed, and the document says so. The remaining
open questions in section 17 carry their refutation thresholds and their consequences
written down in advance.

## 19 Reproducing

```bash
python3 proxy/build_corpus.py        # four lanes from the inventory's own datasets
python3 proxy/train_tokenizer.py     # one frozen 16,384 character-level BPE
python3 proxy/tokenize_corpus.py     # uint16 arrays plus a held-out tail per lane
python3 proxy/build_quality_split.py # the quality-ranked Hindi pools used by E2b
python3 proxy/run_experiments.py --tokens 30e6
python3 proxy/report.py              # section 15
python3 proxy/plan_arithmetic.py     # sections 5 to 13
```

`proxy/inventory.json` is the Session 5 dataset inventory, extracted from the session's
inventory widget, and is the supply figure behind every table in sections 3 to 13. The
14 run records in `proxy/runs/` are committed, each holding its full configuration,
per-lane evaluation, tokens seen per pool and loss history, so every number in
section 15 can be checked without retraining anything. Corpus files and model
checkpoints are not committed; the scripts regenerate them.
