# V5 Mixture and Curriculum Plan

**ERA V5, Session 5.** This document decides what the model is trained on: how the
pretraining token budget is divided between capabilities, in what order they are
taught, what is protected from being dropped along the way, and what is held back for
the end.

It is written under one hard operating constraint. **The programme has no budget for
crawling or for optical character recognition**, so every token must come from a
dataset that has already been published and can simply be downloaded. No purchases, no
licence negotiations, no outside vendors, no new collection of any kind. That
constraint turns out to change the most important number in the plan, and section 02
works through why.

Three of the plan's numbers were not argued for but tested, by training 14 small models
from scratch. Two survived, one was refuted, and the plan changed because of it.
Sections 15 and 16 report that, including the result that went against the plan.

Every table here is printed by a script in this repository rather than typed by hand.
`proxy/supply_constraint.py` and `proxy/plan_arithmetic.py` produce sections 02 to 13.
`proxy/report.py` produces section 15. Their outputs are committed as
`plan_arithmetic.txt`, `supply_constraint.txt` and `proxy_results.txt`.

**The plan in one line:** 9 trillion tokens, mixed **web 40, code 29, Indic 13, STEM 11,
reasoning 6, agentic 1**, delivered through six stages, with a 15 percent protected
floor and a 3 percent anneal reserve. Section 18 gives the full specification as
numbers.

---

## 01 What a mixture is, and why it is the whole assignment

A model learns only from the text it is shown. The corpus does not decide what the
model becomes; the **proportions** do. The same clean data and the same compute produce
a very different model depending on how much of each kind of text goes in.

Because the total is fixed, every share given to one capability is taken from another.
Designing the mixture is the decision about which capabilities matter enough to pay for
with something else.

Two words recur below. A **lane** is one category of training data, such as code or
Indic text. The **mixture** is the set of percentages assigned to those lanes.

## 02 The token budget, and why it came down from 15 trillion to 9

Session 3 set a target of roughly **15 trillion** pretraining tokens for this 40 billion
parameter model, reasoning that 15T is about 375 tokens per parameter, far past
Chinchilla's compute-optimal ratio of about 20, and that training past the optimum is a
deliberate trade: it costs more once, and produces a smaller model that is cheaper on
every inference request afterwards. Session 3 showed the V5 target as a 10 to 30
trillion range.

That figure does not survive the no-collection constraint, and the reason is worth
following carefully because it is the central finding of this plan.

Adding up every published dataset that can be downloaded, including everything in the
Session 5 inventory and the further corpora listed in section 03, the **ceiling** on
each lane is fixed. It is the unique tokens available multiplied by how many times they
can be repeated (four passes for natural text, two for machine-generated). Those
ceilings are absolute: no arrangement of the mixture can exceed them.

Expressed as the largest share of a run each lane could fill, at four candidate budgets:

| Budget | web | code | Indic | STEM | reasoning | agentic |
|---|---:|---:|---:|---:|---:|---:|
| 15T | 128% | 29% | **8.4%** | 7.2% | 3.6% | 0.4% |
| 12T | 160% | 37% | **10.5%** | 9.0% | 4.5% | 0.5% |
| 9T | 213% | 49% | **14.0%** | 12.0% | 6.0% | 0.7% |
| 8T | 240% | 55% | **15.7%** | 13.5% | 6.8% | 0.8% |

Read the Indic column. **At 15 trillion tokens the Indic lane cannot exceed 8.4 percent
however the mixture is arranged**, because that is all the Indian-language text there is
to download. The same holds for STEM at 7.2 percent and reasoning at 3.6 percent.

Whatever the scarce lanes cannot fill has to be filled with general web, the one lane
with room to spare at any budget. So a 15T budget does not merely permit the web-heavy
mixture this session warns against, **it forces it, by arithmetic rather than by
choice**. Chasing a large token count with a fixed data supply is the same mistake as
assigning a large share to a lane with no data behind it, arriving by a different route.

Shrinking the budget raises every scarce lane's reachable share, because the ceiling is
fixed in tokens while the denominator falls. **9 trillion tokens is the point at which
the Indic lane can hold 13 percent, STEM 11 and reasoning 6, entirely from downloadable
data.** That is 225 tokens per parameter, in the same territory as Llama-3-70B's roughly
214, and still ten times past Chinchilla-optimal. It sits just below the 10 to 30T range
Session 3 named, and this document argues Session 3's lower bound was set before the
supply was counted.

**The trade being made is explicit: fewer total tokens, in exchange for a mixture that
still contains the capabilities the model exists for.** A 15T run under this constraint
would be a larger, more general, more English model. A 9T run is a smaller, sharper,
genuinely multilingual one.

## 03 What data actually exists

The Session 5 dataset inventory lists 32 named corpora holding 6.40 trillion unique
tokens. That inventory is a shortlist of what the course assembled, not a census. A
further 308 billion unique tokens are available from datasets that are already published
and simply were not on the list:

| Lane | Added | Datasets |
|---|---:|---|
| Indic | 120B | FineWeb-2 Indian-language subsets (60B), CulturaX Indian-language subsets (40B), HPLT v2 (12B), Varta Indian news corpus (5B), Indian-language Wikipedia dumps (3B) |
| STEM | 123B | FineMath (34B), RedPajama arXiv (28B), PubMed Central open access (26B), StackExchange (20B), OpenWebMath (15B) |
| Reasoning | 50B | OpenR1-Math-220k and other open R1 distillations (30B), Nemotron and AM reasoning post-training sets (20B) |
| Agentic | 15B | the-stack-github-issues |

**These figures are estimates of unique tokens surviving deduplication against the
inventory, not measured counts.** They are the least certain numbers in the document,
because these corpora overlap heavily with each other and with Sangraha, all being
derived from the same crawls. Section 17 states what happens to the plan if a
measurement comes in below them.

Note what is *not* here. There is no line for digitising books, licensing news
archives, or re-crawling the web for Indian-language pages. Those would be the highest
yielding sources by a wide margin, and Session 4 measured exactly why the existing
crawl under-delivers Indian languages: an English-tuned quality filter wrongly discarded
**97 percent** of good Hindi. Re-extracting the crawl with script-aware filters would
likely add hundreds of billions of tokens. It is excluded because it requires crawling
compute this programme does not have, and the plan is built to be executed rather than
admired.

For completeness, Session 3 cites Epoch AI's estimate of roughly 300 trillion tokens of
public human text in existence. The gap between that and the 6.7 trillion reachable here
is almost entirely a collection-budget gap, not a scarcity of writing.

## 04 Working backward from the benchmarks

A capability nobody measures cannot be shown to exist. Each lane exists to move specific
benchmarks, and the training data must take the shape the benchmark actually scores.

| Lane | Benchmarks | What the training example must look like |
|---|---|---|
| Code | LiveCodeBench, Aider | loss on the generated patch or completion |
| Agentic | SWE-bench, tau-bench, BFCL, GAIA, BrowseComp | loss on the model's planning, tool calls and final answer, never on the tool's response |
| Reasoning | AIME, GPQA, HLE | loss on the worked trace and the final answer |
| STEM | AIME, GPQA | loss on derivations and notation |
| Indic | MILU, IndicGenBench | loss on native Indian-language text |
| General web | MMLU | loss on broad world knowledge |

The masking rule in the agentic row is part of the specification. In a tool-using
trajectory the user's request and every tool observation are context the model reads but
is never trained to produce. Training on a tool observation teaches the model to invent
tool results instead of calling the tool.

Long context has no row because it is not a lane. It is a **sequence-length policy**
applied to the web and code lanes during stage S4. A long-context corpus is built by
selecting long documents already inside those lanes and packing them, so giving it its
own token budget would double-count the same tokens. An earlier draft of this plan made
that mistake and credited itself with 0.60T of tokens that did not exist.

## 05 The curriculum: six stages, each with its own mixture

The mixture is not one set of percentages held constant for the whole run. It moves.
Training begins on broad general text to establish language and world knowledge, shifts
toward code and STEM once that foundation exists, brings in reasoning traces only after
the model can form sentences, stretches the context window late, and ends with a short
concentrated phase on reserved high-quality data.

**How to read this table.** *Span of run* is the portion of the whole 9T run the stage
occupies, so S1 runs from the 2 percent mark to the 34 percent mark and consumes 2.88T
tokens. The six lane columns are **percentages of that stage's own tokens**, so each row
totals 100. They are not percentages of the whole run. During S2, out of every 100 tokens
the model sees, 36 are general web, 33 are code and 12 are Indic.

| Stage | Span of run | Tokens | web | code | indic | stem | reason | agentic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 seed | 0 to 2% | 0.18T | 60 | 14 | 14 | 9 | 2 | 1 |
| S1 general | 2 to 34% | 2.88T | 56 | 22 | 13 | 7 | 1 | 1 |
| S2 capability | 34 to 66% | 2.88T | 36 | 33 | 12 | 15 | 3 | 1 |
| S3 reasoning | 66 to 84% | 1.62T | 26 | 31 | 13 | 12 | 17 | 1 |
| S4 long context | 84 to 97% | 1.17T | 31 | 35 | 14 | 9 | 10 | 1 |
| S5 anneal | 97 to 100% | 0.27T | 10 | 20 | 28 | 12 | 22 | 8 |

Sequence length: 4K through S0 and S1, 8K through S2 and S3, then 32K rising to 128K in
S4, and 32K in the anneal. S4 is where the long-document packing happens, which is why
its web and code shares rise again: those are the two lanes with documents long enough
to pack.

Reading down a column shows the design. General web falls from 60 to 10 as its job,
teaching basic language, finishes early. Code peaks in the middle third. Reasoning is
held near zero at the start because reasoning traces teach nothing to a model that
cannot yet form sentences, then rises sharply in S3. Indic holds steady between 12 and 14
throughout rather than spiking, so the capability is built continuously, then doubles in
the anneal. Agentic stays at 1 until the anneal, where it goes to 8.

This shape follows V4, which moved general web from roughly 70 percent down toward 18
across its growth stages while code went from 13 to 35, and kept one protected channel
fixed at 8 percent throughout.

## 06 The headline mixture, and whether the data exists to fill it

The single quoted mixture is not a separate decision. It is what the six stages average
to, which is why it is computed rather than asserted.

| Lane | Share | Demand | Ceiling | Headroom | Epochs | Verdict |
|---|---:|---:|---:|---:|---:|---|
| General web | 40% | 3600B | 19164B | 81% | 0.8 | fits |
| Code | 29% | 2610B | 4412B | 41% | 2.4 | fits |
| Indic | 13% | 1170B | 1260B | 7% | 3.0 | fits |
| STEM | 11% | 990B | 1076B | 8% | 3.7 | fits |
| Reasoning | 6% | 540B | 540B | 0% | 4.0 | fits, exactly at the limit |
| Agentic | 1% | 90B | 63B | -44% | 5.8 | **over** |

**Ceiling** is unique tokens times the repetition cap. **Epochs** is how many times the
unique data is read. Following Muennighoff et al., *Scaling Data-Constrained Language
Models* (NeurIPS 2023), repeating natural text up to about four passes costs almost
nothing and returns decay sharply after; machine-generated text is held to two because
its errors compound.

General web runs at 0.8 epochs, meaning the plan does not even need all the web text
available. That is the visible sign that 9T is the right budget: the abundant lane has
slack while the scarce lanes are near their limits.

Every lane fits except agentic, which section 09 addresses directly.

## 07 The Indic slot, tier by tier

13 percent of 9T is 1,170 billion tokens, split across the four provenance tiers from
Session 3.

| Tier | Unique | Epochs | Tokens | Share of slot |
|---|---:|---:|---:|---:|
| A verified native | 72.0B | 4x | 288B | 24.6% |
| B unverified crawl | 156.9B | 4x | 628B | 53.6% |
| C translated | 5.0B | 4x | 20B | 1.7% |
| D synthetic | 117.2B | 2x | 234B | 20.0% |

Tier A is Sangraha's verified human-origin split plus Varta news and Indian-language
Wikipedia, both human-written and edited. Tier B is crawl-derived multilingual text:
Sangraha unverified, IndicCorpV2, FineWeb-2, CulturaX, HPLT. Tier D is drawn entirely
from Sangraha's existing synthetic split.

**Native human text is 78 percent of the Indic lane, translated 2 percent, and
machine-generated 20 percent. No new generation is required at all**, and only 117B of
the 162B of synthetic text already available is needed.

This is a better outcome than the earlier 15T draft of this plan achieved, and it is
worth being clear about why, because it is counter-intuitive. At 15T the Indic lane
needed 1.95T tokens against the same fixed supply of native text, so the gap had to be
filled with generated text until roughly 40 percent of the lane was machine-produced. At
9T the lane needs 1.17T, which the native and existing synthetic supply covers
comfortably. **Reducing the budget made the Indic data more authentic, not less.** For a
model whose stated reason to exist is native fluency in Indian languages, that is the
argument for 9T on its own.

Quality gates: tier C is accepted only above a fixed round-trip translation agreement
threshold, which is an automatic check. Tier D is accepted only after native-speaker
sampling at one document in ten thousand, which the cohort can do itself. Tier A is
never diluted with either and stays separately addressable, so the anneal can draw on it
alone.

## 08 The agentic slot, which is genuinely short

**Yes. The agentic lane is short, and it is the one place this plan cannot close the
gap.** Stating that plainly is the point of this section.

Agentic data means complete trajectories: a model planning a task, calling a tool,
reading the result, recovering when a call fails, continuing until it finishes. It is
the newest capability slot and the one V4 had almost none of.

The arithmetic. The lane is 1 percent of 9T, which is 90 billion tokens. Everything
downloadable amounts to **15.6 billion unique tokens**: 0.6B of real trajectories from
SWE-Gym, SWE-smith, OpenHands, ToolBench and the rest, plus roughly 15B from
the-stack-github-issues, where an issue, its discussion and the resulting fix form a
naturally occurring multi-step trace with real observations and real failures.

At four passes that supplies 63 billion tokens against a demand of 90. **The lane runs at
5.8 epochs, 44 percent over its ceiling.**

Three options existed and the third was chosen:

1. **Cut the lane to 0.7 percent** so it fits at four epochs. Rejected: agentic ability
   is a headline goal, and 0.7 percent is below any plausible threshold for teaching it.
2. **Generate the difference**, roughly 7 billion unique trajectory tokens from sandboxed
   execution. Rejected under the operating constraint: generating validated trajectories
   at that volume needs inference compute the programme cannot fund, and unvalidated
   trajectories teach the model to hallucinate tool results.
3. **Accept the overage and say so.** Chosen. The lane is 1 percent of the run, so the
   cost of over-repeating it is bounded and small, and the alternative is a model with no
   agentic capability at all.

The honest position: **this plan does not fully fund agentic capability during
pretraining, and it does not pretend to.** What pretraining buys here is familiarity with
the *shape* of a trajectory. The capability itself is bought after pretraining, in
supervised fine-tuning and reinforcement learning, where token counts are tiny and
execution feedback does the teaching. That is also why the anneal raises this lane from
1 percent to 8. If agentic performance has to improve within pretraining, the only
honest lever is a collection budget, and section 17 records that as a decision point
rather than hiding it inside a percentage.

## 09 Difficulty bands

Within each stage the data climbs a difficulty ladder, so the model is not shown graduate
material before it can handle grade-school material.

| Band | Level | Example |
|---|---|---|
| B0 | Nursery | "The sun is hot. The sky is blue." |
| B1 | Grade school | A paragraph explaining why it rains, or a Python function that adds two numbers |
| B2 | High school | A passage on Chola administration, or a function that reads a CSV file and returns column means |
| B3 | Undergraduate | Deriving the time complexity of a recursive algorithm, then rewriting it iteratively |
| B4 | Graduate | A proof by induction with base case and inductive step stated separately, or a concurrency bug traced through a lock ordering |
| B5 | Research | An AIME combinatorics problem, or a SWE-bench issue needing a patch across three files that passes a hidden test suite |

S0 and S1 sit mostly in B0 to B2. S2 and S3 move the mass to B3 and B4. B5 material is
concentrated in S3 and the anneal, where the model can extract something from it.

## 10 Reasoning-length bands

A model can only obey a reasoning-effort setting at inference time if it has seen
examples across the full range of depths. Reserving a reasoning share therefore means
reserving a **distribution of trace lengths**.

| Band | Trace budget | Share of reasoning lane | Example |
|---|---:|---:|---|
| Low | under 200 tokens | 30% | "17 x 24 = 17 x 25 - 17 = 425 - 17 = 408." |
| Medium | 200 to 1,000 | 35% | A two-variable word problem: set up both equations, solve, substitute back once to check |
| High | 1,000 to 5,000 | 25% | A geometry problem attempted with coordinates, found messy, restarted with a synthetic argument, verified against a special case |
| Ultra | 5,000 to 32,000 | 10% | A competition problem where three approaches are tried, two abandoned with the reason recorded, intermediate results checked, and the final claim proved |

Each band spans mathematics, code and general problem solving, so reasoning depth does
not become welded to one domain. Short traces come first; longer ones arrive in S3.

## 11 The selector, and the protected floor

Modern training does not fix the recipe and walk away. A selector scores candidate
batches during the run and keeps the most useful fraction, so the same compute buys more
learning. In V4 this retained about 40 percent of candidates for roughly a sixfold
increase in effective token value.

The danger is that the selector defines "useful" through a proxy, and a proxy trained
mostly on English does not recognise value in native Indic text or unfamiliar agentic
trajectories. Section 15 measures this happening.

The defence is a **protected floor**: a share of every batch reserved for scarce lanes
regardless of score.

| Lane | Floor | Headline share |
|---|---:|---:|
| Indic | 10% | 13% |
| Reasoning | 4% | 6% |
| Agentic | 1% | 1% |

15 percent of every batch sits outside the selector's control. The agentic floor equals
its share, because a lane that small cannot absorb any erosion at all.

The floor sits on top of a structural rule the experiments forced into this plan:
**the selector operates inside each lane's budget, never across lanes.** The mixture
decides how many tokens a lane receives; the selector decides which tokens within that
allocation. Section 16 explains why.

## 12 The anneal reserve

The final stage trains briefly at a decayed learning rate on a reserve of the best data
held back from the main run. The gain is reported to be disproportionate to its size,
and it is only available if the reserve survives, which is why it is decided now: if the
selector consumes the best Indic and agentic data early, nothing special is left when the
cooldown arrives.

**3 percent of the budget, 270 billion tokens.** Session 5 places this phase at about 2
percent within a 1 to 5 percent band, and 3 sits inside it. An earlier draft used 8
percent, which was outside the stated range and which the experiments then failed to
support.

Reserve composition: the highest-scoring tier A Indic, the whole of SWE-Gym, SWE-smith
and the OpenHands rollouts, OpenThoughts2 and OpenMathReasoning, proof-pile-2 and peS2o.
Spent once.

This remains the least evidenced number in the plan, for reasons section 15 gives.

## 13 Keeping the run stable when the mixture moves

Every stage boundary changes the data distribution, which changes the gradients. V4 saw
the gradient norm rise roughly 150 times when a sudden increase in the Hindi share met
frozen embeddings, and an event that size can destroy a run.

Every boundary in section 05, and the entry to the anneal, is blended linearly across a
**30 billion token warmup band**, about 0.33 percent of the run, rather than switched in
one step. No embedding is frozen across a mixture transition. Gradient norm is monitored
per step and a rise above four times the trailing median halts the run.

## 14 What was tested, and how

Every number above is a hypothesis until something cheap has tested it. Three were tested
for real.

**What was built.** A 9.06 million parameter decoder-only transformer (4.74M
non-embedding), 6 layers, 256 hidden, 4 heads, 512 context. Each run trains from scratch
on 30 million tokens. Four lanes assembled from the inventory's own datasets: FineWeb-Edu
for web, permissively licensed GitHub source for code, Sangraha verified Hindi for Indic
(reusing the Session 4 cleaning function unchanged, joiners preserved), NuminaMath-CoT
for reasoning. One character-level BPE vocabulary of 16,384 trained once and frozen
across every run. 14 runs, about four hours on one Apple M4 Pro.

The plan's six lanes collapse to these four, with STEM and reasoning combined. The Indic
share in the projection is 14.8 percent against the plan's 13, and that share is the
variable E1 sweeps.

### What "held-out bits per byte" means

This is the measurement everything below rests on, so it is worth building up carefully.

A language model does one thing: given some text, it produces a probability for what the
next token will be. If it is confident and right, it assigned a high probability. If it
is surprised, it assigned a low one.

**Step 1, surprise per token.** The training loss, cross-entropy, is the average
surprise: the negative logarithm of the probability the model gave to the token that
actually appeared. It comes out in units called nats. A model that has learned the
language well is less surprised, so its loss is lower.

**Step 2, held out.** The surprise is measured on text the model has never been trained
on, the last 2 percent of each lane, set aside before training. Measuring on training
text would reward memorisation instead of learning.

**Step 3, convert to bits.** Dividing nats by the natural logarithm of 2 converts to
bits, which is just a more familiar unit: it is literally the number of yes-or-no
questions needed, on average, to pin down the next token.

**Step 4, divide by bytes.** Here is the important step. Different lanes use different
scripts, and the tokenizer chops them differently, so "per token" is not comparable
across lanes. Dividing by the number of bytes those tokens covered gives **bits per
byte**: how many bits the model needs to predict one byte of text. This cancels the
tokenizer out of the measurement entirely.

A worked example, the plan mixture's Indic score:

```
loss            = 5.7989 nats per token      (measured on held-out Hindi)
bits per token  = 5.7989 / ln(2) = 8.3661
bytes per token = 8.289                       (a property of the tokenizer)
bits per byte   = 8.3661 / 8.289 = 1.0093
```

**Lower is better.** It means the model finds the text less surprising, which is what it
means for a model to have learned a language.

**Why it moves when the Indic share moves.** A model trained on more Hindi has seen more
Hindi words, more Devanagari spelling patterns, more Hindi grammar. It is therefore less
surprised by unseen Hindi, so its Indic bits per byte falls. Meanwhile, because the total
budget is fixed, every token of Hindi added is a token of English, code or maths removed,
so those lanes get slightly more surprising and their bits per byte rises. **The
experiment is measuring that trade directly.**

One caution. Bits per byte is comparable for the **same lane across different runs**,
which is all that is claimed below. It is not meaningful to compare Hindi's 1.01 against
English's 2.17 and conclude the model is better at Hindi. Those numbers differ mainly
because Devanagari uses about 2.5 UTF-8 bytes per character where English uses 1, so a
byte of Hindi carries less information to begin with.

### A note on fertility, since the byte figures invite a misreading

The proxy tokenizer encodes Hindi at 8.29 bytes per token against 3.80 for English. That
is **not** a fertility problem. It is the same UTF-8 property just described: Devanagari
costs about 2.5 bytes per character, so a Hindi token covering 3.27 characters covers
8.29 bytes. A high bytes-per-token means each token carries *more* text, which is good.

Measured as this course defines fertility, **tokens per word**:

| Lane | Tokens per word | Characters per token | Bytes per token |
|---|---:|---:|---:|
| web | 1.618 | 3.78 | 3.80 |
| indic | **1.570** | 3.27 | 8.29 |
| math | 1.636 | 3.52 | 3.52 |
| code | 4.185 | 2.51 | 2.53 |

Hindi is marginally **better** than English. For reference, the Assignment 4 tokenizer, a
32,000-entry character-level BPE trained on Hindi alone, reached 1.178 tokens per word on
the same corpus against 4.984 for an English-first tokenizer. The proxy's 1.570 is worse
than 1.178 because this vocabulary is only 16,384 entries shared across four lanes
including code. That is the correct trade for an experiment: the tokenizer here is a
controlled variable held identical across all 14 runs, not the production tokenizer,
which remains the 262,144 entries specified in Session 3.

Architecture, token budget, learning-rate schedule, batch size, seed and tokenizer are
identical across arms. The mixture is the only thing that varies.

## 15 The experiments and their results

### E1. How much Indic is worth including

The same model trained five times from scratch, changing only the Indic share and letting
the other three lanes absorb the difference in their planned ratio.

Each row is one separately trained model. The lane columns are that model's held-out bits
per byte on each lane, lower being better. The last column averages the three non-Indic
lanes, which is what the Indic share is paid for out of.

| Indic share | web | code | indic | math | non-Indic mean |
|---|---:|---:|---:|---:|---:|
| 0% | 2.1298 | 2.7039 | 1.6159 | 1.6730 | 2.1689 |
| 3% | 2.1483 | 2.7331 | 1.1438 | 1.6964 | 2.1926 |
| 8% | 2.1518 | 2.7528 | 1.0494 | 1.7132 | 2.2059 |
| 13% (plan) | 2.1722 | 2.7790 | 1.0093 | 1.7372 | 2.2295 |
| 30% | 2.2002 | 2.8369 | 0.9588 | 1.7883 | 2.2752 |

Restated as percentage changes against the 0 percent model. "Indic gain" is how much
lower that model's Indic bits per byte is than the 0 percent model's. "Cost" is how much
higher its non-Indic mean is.

Worked example for the 13 percent row: Indic gain is (1.6159 − 1.0093) / 1.6159 = 37.5
percent. Cost is (2.2295 − 2.1689) / 2.1689 = 2.79 percent.

| Indic share | Indic gain | Cost to other lanes |
|---|---:|---:|
| 3% | 29.2% | 1.09% |
| 8% | 35.1% | 1.71% |
| 13% (plan) | 37.5% | 2.79% |
| 30% | 40.7% | 4.90% |

The cumulative view hides the shape, so the same result step by step. Each row is what
that particular increment added and cost, obtained by subtracting the previous row above.
The last column divides one by the other, giving the exchange rate on offer at that point.

| Step | Indic gain in this step | Cost in this step | Gain per unit of cost |
|---|---:|---:|---:|
| 0% to 3% | 29.2 | 1.09 | **26.8** |
| 3% to 8% | 5.8 | 0.62 | **9.5** |
| 8% to 13% | 2.5 | 1.08 | **2.3** |
| 13% to 30% | 3.1 | 2.11 | **1.5** |

The first three points of Indic return roughly 27 times what they cost. The next five
return 9.5. After 8 percent the exchange rate collapses to about 2.

**Finding.** The 10 percent floor is confirmed: the lane is still well above break-even at
8 to 10 percent, so a lower floor gives away real capability. The 13 percent share is
**not** strongly confirmed by this evidence: it sits past the elbow and buys 2.5 units of
Indic for 1.08 units off everything else. It is kept for two reasons stated separately so
they can be argued with separately. First, Indic is the one capability this model exists
for, and a 2.3-to-1 exchange rate is still positive for the capability the whole
programme is organised around. Second, and independently, section 02 shows 13 percent is
what the supply supports at a 9T budget, so nothing is being taken from a lane that could
have used it. Section 17 specifies the test that settles it.

### E2 and E2b. The anneal reserve, which did not replicate

Both arms train on the same budget, the same learning-rate schedule, and give every data
pool the same number of tokens. Only **when** the reserve is used differs: spread evenly
through the run, or held back entirely for the final tenth. E2 uses an arbitrary slice of
Hindi, holding quality constant so only timing varies. E2b repeats it with a reserve that
is genuinely better data, the top 15 percent of Hindi by the Session 4 quality signal.

| Run | Reserve | Spread evenly | Held for anneal | Difference |
|---|---|---:|---:|---:|
| E2 seed 0 | arbitrary slice | 1.0085 | 1.0112 | -0.27% |
| E2 seed 1 | arbitrary slice | 1.0038 | 1.0050 | -0.12% |
| E2b seed 0 | top 15% by quality | 1.0081 | 1.0171 | -0.89% |

**Holding the reserve back produced no benefit in any of the three comparisons, and was
marginally worse in all three.** The refutation threshold was written down before the runs
and this clears it.

Four limits on that conclusion:

1. **Scale.** 9M parameters and 30M tokens is far from convergence. The annealing result
   being tested is observed at 1B parameters and above.
2. **Size of the reserve.** It is 2.2 percent of the run's tokens. A slice that small,
   concentrated into the last tenth, may not move a loss averaged over everything.
3. **Metric.** Held-out bits per byte on the same distribution is not what annealing is
   believed to improve. Reported gains elsewhere are downstream benchmark gains, which
   this proxy cannot measure.
4. **The most likely explanation.** The learning rate follows a single cosine curve in
   both arms. That is the correct control for isolating data ordering, but it removes the
   mechanism. A real anneal changes two things together: the data becomes higher quality
   *and* the learning rate decays sharply so the model consolidates on it. Holding the
   data back while decaying identically in both arms tests only half of it.

**Finding.** The reserve is kept but cut from 8 percent to 3, and is now the weakest
number in the plan.

### E3. What a protected floor is actually worth

Candidate batches of 60 sequences are scored by a reference model trained only on English
web, and the best 24 are kept.

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

Switch on a 14.8 percent always-on floor and Indic recovers to **1.0093**, the figure the
unselected mixture reached. The floor does not partly protect the lane, it makes it
immune, because the floor pins the token count to the share the mixture designed.

**Finding, and a correction.** The floor works exactly as claimed. But the cost lands
somewhere the plan had not anticipated: protecting Indic pushed the damage onto the lanes
that were *not* protected. Code fell to a 1.6 percent keep rate and 14.6 percent worse
bits per byte, maths to 1.5 percent and 19.6 percent worse.

## 16 What the experiments changed, and why

This section exists because a plan that reports experiments but never revises itself is
not really using them. Four things in this document are different from the draft written
before the runs.

**1. The selector now operates inside each lane's budget, not across lanes.**

The original design was the one Session 5 describes: let the selector choose freely from
all candidate batches, and add floors underneath so the scarce lanes cannot be starved.
E3 shows why that is not enough. The floor did protect Indic, completely. But the
selector still had to fill the remaining 85 percent of each batch, and it filled it with
what its English proxy liked, which was web. Code collapsed from 6.1 percent of offered
tokens to 1.6, and maths from 7.2 to 1.5. Their bits per byte got 14.6 and 19.6 percent
worse.

In other words, the floor did not repair the biased selector. It moved the bias onto
whichever lanes had no floor. Adding floors to those lanes too would just move it again,
onto whatever remains, until every lane has a floor and the selector has nothing left to
decide.

The fix is structural. The mixture already specifies how many tokens each lane gets. Let
that be binding, and let the selector choose only *within* each lane's allocation: which
Hindi documents, which repositories, which reasoning traces. The selector keeps its
value, which is picking better examples, and loses the ability it should never have had,
which is silently rewriting the mixture. **This idea was not in the plan beforehand. It
came out of noticing what the floor did to the unprotected lanes.**

**2. The anneal reserve fell from 8 percent of the budget to 3.**

Two independent reasons. Session 5 states the anneal phase is about 2 percent of tokens
within a 1 to 5 percent band, so 8 was outside the range the course gives. And three
separate comparisons, including one with a quality-selected reserve, found no benefit
from holding data back at all. Neither reason alone would be decisive; together they are.

**3. The 13 percent Indic share is now presented as a judgement call rather than a
result.**

E1 put the elbow of the return curve nearer 8 percent than 13. The share did not change,
but the justification did, and it is now stated as a deliberate choice to pay past the
elbow for the model's defining capability, rather than implied to be optimal. A reviewer
can now disagree with the choice without having to catch a hidden assumption first.

**4. The 10 percent floor came out of testing stronger than it went in.**

It is the only headline number with two independent pieces of evidence behind it: E1
shows the lane still paying well above break-even at 8 to 10 percent, and E3 shows what
happens without protection.

## 17 How the remaining numbers get checked

Once a result arrives it is tempting to reinterpret it so that it agrees with the plan
already written. The defence is to state, **before running anything**, what outcome counts
as confirmation, what counts as refutation, and what specifically changes in this document
if the refutation happens.

That discipline produced the anneal result above: the threshold was fixed first, which is
why a disappointing outcome became a finding rather than something to explain away.

**None of the tests below have been run.** They are specified, not reported. The benchmarks
named are public: MILU is AI4Bharat's multi-task Indic understanding benchmark, available
on HuggingFace as `ai4bharat/MILU`, and IndicGenBench, LiveCodeBench and SWE-bench all ship
public evaluation harnesses. The proxy in sections 14 and 15 could not run any of them,
because a 9 million parameter model scores at chance on all of them; that is precisely why
these tests need 1B and 3B scale.

| Test | Scale | Metric | Confirms | Refutes |
|---|---|---|---|---|
| Indic share sweep at 6 / 10 / 13 / 18% | 1B, 20B tokens | MILU, IndicGenBench | MILU still rising between 10% and 13%, other benchmarks down under 1% | MILU flat by 10%, or non-Indic cost above 1% |
| Anneal reserve, matched tokens, **warmup-stable-decay schedule** so the decay coincides with the reserve | 3B, 60B tokens | MILU, LiveCodeBench | Reserve beats even spending by 1% or more | Under 0.3%, or reversed, as at proxy scale |
| Selection inside lane budgets against selection across lanes with floors | 1B, 20B tokens | Kept-token rate per lane, MILU, LiveCodeBench | Per-lane selection holds every lane at its designed share and wins | Global selection with floors matches it on every lane |
| **Deduplicated token count of the added corpora in section 03** | offline, no training | Unique tokens after dedup against the inventory | Indic yields 120B or more | Below 80B |

Pre-committed consequences:

- If MILU is flat by 10 percent, the Indic share drops from 13 to 10 and the freed three
  points go to code. The 10 percent floor stands regardless.
- If the 3B anneal is flat under a decay schedule too, the reserve drops from 3 percent to
  1 and the rest returns to the main run.
- If per-lane selection does not beat global selection with floors, the correction in
  section 16 is reverted and floors are extended to code and STEM instead.
- **If the Indic dedup yields under 80B**, the whole plan moves down a budget step, from 9T
  to about 7.5T, rather than the Indic share being cut. Section 02's logic runs in both
  directions: the budget is a function of the supply, so a smaller supply means a smaller
  budget, not a thinner mixture.
- If agentic performance at 1B is unacceptable, the only remaining lever is a collection
  budget for trajectory generation. That is recorded as a decision for the programme rather
  than absorbed into a percentage.

## 18 The plan, as numbers

| Item | Value |
|---|---|
| Pretraining budget | **9T tokens**, reduced from 15T because supply caps the scarce lanes (section 02) |
| Model | 40B dense, 225 tokens per parameter |
| Headline mixture | web 40, code 29, Indic 13, STEM 11, reasoning 6, agentic 1 |
| Stage S0, seed | 0 to 2%, 0.18T, web 60 / code 14 / Indic 14 / STEM 9 / reason 2 / agentic 1, seq 4K, bands B0 to B1 |
| Stage S1, general | 2 to 34%, 2.88T, web 56 / code 22 / Indic 13 / STEM 7 / reason 1 / agentic 1, seq 4K, bands B0 to B2 |
| Stage S2, capability | 34 to 66%, 2.88T, web 36 / code 33 / Indic 12 / STEM 15 / reason 3 / agentic 1, seq 8K, bands B2 to B3 |
| Stage S3, reasoning | 66 to 84%, 1.62T, web 26 / code 31 / Indic 13 / STEM 12 / reason 17 / agentic 1, seq 8K, bands B3 to B5 |
| Stage S4, long context | 84 to 97%, 1.17T, web 31 / code 35 / Indic 14 / STEM 9 / reason 10 / agentic 1, seq 32K to 128K |
| Stage S5, anneal | 97 to 100%, 0.27T, web 10 / code 20 / Indic 28 / STEM 12 / reason 22 / agentic 8, seq 32K, decayed learning rate |
| Indic tier split | A verified native 24.6%, B unverified crawl 53.6%, C translated 1.7%, D synthetic 20.0% |
| Indic authenticity | 78% native human text, 20% machine-generated, no new generation required |
| Protected floors | Indic 10%, reasoning 4%, agentic 1%, total 15% of every batch |
| Selector rule | operates inside each lane's budget, never across lanes |
| Anneal reserve | 3% of budget, 270B tokens, spent once at decayed learning rate |
| Repetition caps | 4 epochs natural, 2 epochs machine-generated |
| Lane at its limit | reasoning, exactly 4.0 epochs |
| Lane over its limit | agentic, 5.8 epochs, accepted deliberately (section 08) |
| Mixture transitions | blended across a 30B token band, no frozen embeddings, halt above 4x trailing median gradient norm |
| Reasoning-length mix | low 30%, medium 35%, high 25%, ultra 10% |
| Difficulty ladder | six bands, B0 nursery to B5 research |
| Evidence | 14 proxy runs: floor confirmed, Indic share partly supported, anneal refuted |
| Operating constraint | published datasets only, no crawling, no OCR, no purchases, no new generation |

## 19 Reproducing

```bash
python3 proxy/build_corpus.py        # four lanes from the inventory's own datasets
python3 proxy/train_tokenizer.py     # one frozen 16,384 character-level BPE
python3 proxy/tokenize_corpus.py     # uint16 arrays plus a held-out tail per lane
python3 proxy/build_quality_split.py # the quality-ranked Hindi pools used by E2b
python3 proxy/run_experiments.py --tokens 30e6
python3 proxy/report.py              # sections 14 and 15
python3 proxy/supply_constraint.py   # section 02 and 03
python3 proxy/plan_arithmetic.py     # sections 05 to 12
```

`proxy/inventory.json` is the Session 5 dataset inventory, extracted from the session's
inventory widget. The 14 run records in `proxy/runs/` are committed, each holding its full
configuration, per-lane evaluation, tokens seen per pool and loss history, so every number
in section 15 can be checked without retraining anything. Corpus files and model
checkpoints are not committed; the scripts regenerate them.
