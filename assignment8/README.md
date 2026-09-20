# Attention, in the order it was built

**Session 8.** A web app that shows every attention mechanism from the session in the
order it was launched, not the order it was taught. Each one is presented as an answer to
a problem that existed at that moment, with what it buys, what it gives up, and when it
should be chosen.

**Live app:** `<paste the Netlify link here after the first deploy>/assignment8/`
**Repository:** https://github.com/Nisarg189/era_v5 (this folder)

The page is a route on the existing course site rather than a separate deployment. The
site folder is `webapp/`, which is not tracked by git, so this folder is the source of
truth and `webapp/assignment8/` is a copy of it.

Twenty-nine mechanisms span eleven years, from September 2014 to December 2025. The
eighteen on the required list are all present. Nine more were added. Every date was read
from the primary record, and the full source table is at the end of this file.

## What the app contains

| Part | What it does |
|---|---|
| Start here | Standard scaled dot-product attention first, before any variant, as the brief requires. The six steps are named: score, scale, mask, softmax, read, cache. The live demo runs all six on one sentence. |
| The timeline | Twenty-nine dots at their true dates, in five rows by the cost each one attacks. Gaps in a row are the point, not a defect. |
| Twenty-nine cards | The problem it answered, the mechanism, what it changes, Pros, Cons, and when to pick it. Every quoted figure belongs to the paper, and the paper is linked. |
| What it changes | Every card marks which of the six baseline steps it modifies, so no mechanism is presented in isolation. Two cards mark none, and that is the interesting part: FlashAttention changes how attention is computed, not what it computes. |
| Answers a limit of | Twenty-seven cards name the earlier entry whose weakness they respond to, and link to it. The chain is walkable in both directions. |
| Nine live demos | Each computes its numbers in the browser from the values on screen. Nothing shown is a stored result. |
| The source table | One row per mechanism, with the record the date came from. |

The nine demos are the attention layer itself, the four position schemes compared, the
RoPE rotation, the three ways to stretch RoPE, the cache bill, the sliding window with
attention sinks, the regrouping that softmax forbids, the delta rule against an add-only
write, and where the cost of sparse attention actually sits.

### How the brief is answered

| The brief asks for | Where it is |
|---|---|
| Start with plain scaled dot-product attention with a softmax, and make it obvious first | The **Start here** section, above the timeline. The six-step flow from the brief, then the live layer. |
| Do not start with GQA, RoPE or DeltaNet and assume the reader knows what they modify | Every card carries a **What it changes** strip against those same six steps. |
| Everything else as a response to some limitation of what came before | **Answers a limit of** on twenty-seven cards, linked to the entry it responds to. |
| Chronological by launch date, not taught order, not grouped by family | The timeline and the card order are both by date. The five rows are a filter, not the ordering. |
| Pros and cons honestly written, and when would I actually choose it | **Pros**, **Cons** and **Pick it when** on all twenty-nine cards. |
| The eighteen named mechanisms, at minimum | All present. See the source table. |
| Anything missed counts in your favour | Nine added, listed below with dates and sources. |
| A README saying which sources the dates came from | The source table at the end of this file. |

## Question 2: what the timeline shows that a list cannot

Six things are visible only once the mechanisms are in date order.

**1. The bills arrive about two years after the method.** Attention is published in June
2017. Nothing attacks its cost until January 2019. That gap is how long it took to build
models large enough for the cost to hurt. Every card after the gap is a reaction to a
problem that did not exist when attention was designed.

**2. Decode memory was solved once, then ignored for three and a half years.**
Multi-query attention appears alone in November 2019. The row is then empty until May
2023. The idea was not wrong and was not missed. The key-value cache does not hurt until
a model is served to many people at once. Grouped-query attention arrives in 2023 with a
recipe for converting an existing checkpoint at 5 percent of pretraining compute, and
that recipe, rather than the idea, is what made it spread.

**3. Position is the row that never closes.** It is busy in 2017, busy again in 2021, and
then four entries land between 27 June and 29 September 2023, inside fourteen weeks. That
cluster is not a research programme. It is the open-weights community trying to make an
8K checkpoint reach 32K on one machine, and two of the four entries are not papers at
the time they appear. The row is still open in December 2025.

**4. Recurrence never left. It was waiting for a kernel.** The field drops recurrent
state in 2017 and starts rebuilding it in 2020. The delta rule is published in February
2021 and does not reach a real model until June 2024, when a way to train it in parallel
is found. Three years of silence in that row is not three years of doubt. It is three
years of an idea being correct and unaffordable.

**5. The most useful change of the decade was not a new mechanism.** FlashAttention, May
2022, changes nothing about the model. It computes the same attention and moves less
memory. It made long context affordable with no quality argument to win, and it retired a
generation of approximations that had been beating attention only on paper. A timeline
that hides this is dishonest about how the field actually works.

**6. Where the next one comes from.** Read the rows left to right and the priority moves:
exactness, then decode memory, then length, then memory again, then sparsity that is
trained rather than bolted on. The two newest entries move the same way. Sparse attention
stopped being an inference patch and became part of pretraining. DroPE stopped repairing
the positional embedding and removed it. Both promote a workaround into the training
recipe. The next one will come from the same place: whatever is applied after training
today and works.

## Question 2: mechanisms not covered in class

Nine were added. Each is held to the same standard as the required list: a primary date,
the problem it answered, the mechanism, the cost, and a place on the timeline.

| Date | Mechanism | Why it belongs on the timeline | Source |
|---|---|---|---|
| 1 Sep 2014 | Additive attention | Attention exists three years before the Transformer. Starting at 2017 hides that the mechanism was invented to fix a translation bottleneck, not a cost problem. | [arXiv 1409.0473](https://arxiv.org/abs/1409.0473) |
| 9 Jan 2019 | Transformer-XL | The first cross-segment memory, and the first working relative position scheme. Both ideas return later. | [arXiv 1901.02860](https://arxiv.org/abs/1901.02860) |
| 27 May 2022 | FlashAttention | The exact method wins a round. Without it the timeline reads as a straight line of approximations getting better, which is false. | [arXiv 2205.14135](https://arxiv.org/abs/2205.14135) |
| 27 Jun 2023 | Position interpolation | The missing link between RoPE and the two extension methods on the required list. NTK-aware scaling is a direct answer to its weakness. | [arXiv 2306.15595](https://arxiv.org/abs/2306.15595) |
| 10 Oct 2023 | Mistral 7B sliding window | Where windowed attention stopped being research and met a serving bill, with a rolling cache that stops growing. | [arXiv 2310.06825](https://arxiv.org/abs/2310.06825) |
| 1 Dec 2023 | Mamba | Content-based selection restored to a fixed-state model, which is what linear attention could not do. | [arXiv 2312.00752](https://arxiv.org/abs/2312.00752) |
| 31 May 2024 | Mamba-2, state space duality | A class of state-space models and a class of masked attention are shown to be the same computation. The two branches of the timeline turn out to be one family. | [arXiv 2405.21060](https://arxiv.org/abs/2405.21060) |
| 10 Jun 2024 | Parallelising the delta rule | Explains the three-year gap between the delta rule and Gated DeltaNet. Without this paper neither exists in a real model. | [arXiv 2406.06484](https://arxiv.org/abs/2406.06484) |
| 7 Oct 2024 | Differential attention | The only entry after 2019 that attacks a quality problem rather than a bill. It shows the field is no longer only paying down cost. | [arXiv 2410.05258](https://arxiv.org/abs/2410.05258) |

## How the dates were checked

The date used is the **first public appearance**: for a paper, the v1 entry in the arXiv
submission history, not the conference year and not a later revision. Each arXiv page was
opened and the submission history read. No date was taken from memory or from a summary.

Two traps were found during the check, and both are the kind the brief warns about.

**DroPE is two different methods.** A search for the name returns
[arXiv 2503.15029](https://arxiv.org/abs/2503.15029) first, which is DRoPE, Directional
Rotary Position Embedding, submitted 19 March 2025. That paper is about agent trajectories
in autonomous driving and has nothing to do with context length. The DroPE in this session
is from Sakana AI, it drops positional embeddings after pretraining, and it is
[arXiv 2512.12167](https://arxiv.org/abs/2512.12167), submitted 13 December 2025. Taking
the first search result would have put the mechanism on the wrong date by nine months and
described the wrong method.

**NTK-aware scaling has no paper.** It was a post to r/LocalLLaMA by a user named bloc97.
No dated primary record for it could be reached: Reddit was not fetchable, and the YaRN
paper cites it without a day. The earliest dated public reference found is a Hugging Face
issue opened on 30 June 2023 that links to the post. It is therefore listed as **late June
2023** and marked approximate in the app rather than given a false precision. This is the
one date in the table that is not exact, and it is marked as such on its card.

### One note for the session notes

Section 9 presents DroPE through the V4 cookbook and states that the available record does
not establish the algorithm or which rotary dimensions it changes. That was accurate for
an internal record. A public method of that name has since been published, and it is worth
adding because it works the other way round from what the section heading suggests. It
does not recalibrate RoPE. It keeps RoPE for pretraining, removes positional embeddings
from every layer afterwards, and runs a short recalibration at the original context length,
so position then reaches the model only through the causal mask. The paper reports
adaptation at as little as 0.5 percent of pretraining cost on models up to 7B. This is
offered as an addition to the record, not a correction to the V4 result, which is a
separate claim.

## Every date and its source

| Date | Mechanism | Primary source |
|---|---|---|
| 1 Sep 2014 | Additive attention | Bahdanau, Cho, Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, [arXiv 1409.0473](https://arxiv.org/abs/1409.0473) v1 |
| 8 May 2017 | Learned absolute positions | Gehring, Auli, Grangier, Yarats, Dauphin, *Convolutional Sequence to Sequence Learning* section 3.1, [arXiv 1705.03122](https://arxiv.org/abs/1705.03122) v1 |
| 12 Jun 2017 | Scaled dot-product attention | Vaswani et al., *Attention Is All You Need*, [arXiv 1706.03762](https://arxiv.org/abs/1706.03762) v1 |
| 12 Jun 2017 | Sinusoidal positions | Same paper, section 3.5 |
| 9 Jan 2019 | Transformer-XL segment recurrence | Dai et al., [arXiv 1901.02860](https://arxiv.org/abs/1901.02860) v1 |
| 23 Apr 2019 | Sparse Transformer | Child, Gray, Radford, Sutskever, [arXiv 1904.10509](https://arxiv.org/abs/1904.10509) v1 |
| 6 Nov 2019 | Multi-query attention | Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, [arXiv 1911.02150](https://arxiv.org/abs/1911.02150) v1 |
| 25 Dec 2019 | Top-k attention | Zhao et al., *Explicit Sparse Transformer*, [arXiv 1912.11637](https://arxiv.org/abs/1912.11637) v1 |
| 10 Apr 2020 | Sliding window attention | Beltagy, Peters, Cohan, *Longformer*, [arXiv 2004.05150](https://arxiv.org/abs/2004.05150) v1 |
| 29 Jun 2020 | Linear attention | Katharopoulos, Vyas, Pappas, Fleuret, *Transformers are RNNs*, [arXiv 2006.16236](https://arxiv.org/abs/2006.16236) v1 |
| 22 Feb 2021 | The delta rule | Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, [arXiv 2102.11174](https://arxiv.org/abs/2102.11174) v1 |
| 20 Apr 2021 | RoPE | Su et al., *RoFormer*, [arXiv 2104.09864](https://arxiv.org/abs/2104.09864) v1 |
| 27 Aug 2021 | ALiBi | Press, Smith, Lewis, *Train Short, Test Long*, [arXiv 2108.12409](https://arxiv.org/abs/2108.12409) v1 |
| 27 May 2022 | FlashAttention | Dao, Fu, Ermon, Rudra, Ré, [arXiv 2205.14135](https://arxiv.org/abs/2205.14135) v1 |
| 22 May 2023 | Grouped-query attention | Ainslie et al., [arXiv 2305.13245](https://arxiv.org/abs/2305.13245) v1 |
| 27 Jun 2023 | Position interpolation | Chen, Wong, Chen, Tian, [arXiv 2306.15595](https://arxiv.org/abs/2306.15595) v1 |
| late Jun 2023 | NTK-aware scaled RoPE | bloc97, [r/LocalLLaMA post](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/). No paper. Corroborated by [text-generation-inference issue 512](https://github.com/huggingface/text-generation-inference/issues/512), opened 30 June 2023. **Approximate.** |
| 31 Aug 2023 | YaRN | Peng, Quesnelle, Fan, Shippole, [arXiv 2309.00071](https://arxiv.org/abs/2309.00071) v1 |
| 29 Sep 2023 | Attention sinks | Xiao, Tian, Chen, Han, Lewis, *Efficient Streaming Language Models with Attention Sinks*, [arXiv 2309.17453](https://arxiv.org/abs/2309.17453) v1 |
| 10 Oct 2023 | Sliding window in production | Jiang et al., *Mistral 7B*, [arXiv 2310.06825](https://arxiv.org/abs/2310.06825) v1 |
| 1 Dec 2023 | Mamba | Gu, Dao, [arXiv 2312.00752](https://arxiv.org/abs/2312.00752) v1 |
| 7 May 2024 | Multi-head latent attention | DeepSeek-AI, *DeepSeek-V2*, [arXiv 2405.04434](https://arxiv.org/abs/2405.04434) v1 |
| 31 May 2024 | Mamba-2, state space duality | Dao, Gu, *Transformers are SSMs*, [arXiv 2405.21060](https://arxiv.org/abs/2405.21060) v1 |
| 10 Jun 2024 | DeltaNet made trainable | Yang, Wang, Zhang, Shen, Kim, [arXiv 2406.06484](https://arxiv.org/abs/2406.06484) v1 |
| 7 Oct 2024 | Differential attention | Ye et al., *Differential Transformer*, [arXiv 2410.05258](https://arxiv.org/abs/2410.05258) v1 |
| 9 Dec 2024 | Gated DeltaNet | Yang, Kautz, Hatamizadeh, [arXiv 2412.06464](https://arxiv.org/abs/2412.06464) v1 |
| 16 Feb 2025 | Native sparse attention | Yuan, Gao, Dai et al., [arXiv 2502.11089](https://arxiv.org/abs/2502.11089) v1 |
| 29 Sep 2025 | DeepSeek sparse attention | DeepSeek-AI, [V3.2-Exp release note dated 2025/09/29](https://api-docs.deepseek.com/news/news250929/). Report: [arXiv 2512.02556](https://arxiv.org/abs/2512.02556) v1, 2 December 2025 |
| 13 Dec 2025 | DroPE | Gelberg, Eguchi, Akiba, Cetin (Sakana AI), [arXiv 2512.12167](https://arxiv.org/abs/2512.12167) v1. Announced [12 January 2026](https://sakana.ai/drope/) |

Figures quoted on the cards were read from the paper that made the claim: the 93.3 percent
cache reduction and 42.5 percent training saving of DeepSeek-V2, the 5 percent uptraining
compute of GQA, ALiBi at length 1024 matching a sinusoidal model trained at 2048 while
training 11 percent faster, YaRN at 10 times fewer tokens and 2.5 times fewer steps,
StreamingLLM at 4 million tokens and 22.2 times faster than recomputation, Position
interpolation reaching 32768 within 1000 fine-tuning steps, Mamba at 5 times the
generation throughput of a Transformer, Mamba-2 at 2 to 8 times faster than Mamba,
DeltaNet at 1.3B parameters on 100B tokens, Differential Transformer at about 65 percent
of the model size or training tokens, and DroPE adapting at 0.5 percent of pretraining
cost.

## Running it

The page is static, with no build step and no external dependency. Every script and
stylesheet is in this folder.

```bash
cd assignment8
python3 -m http.server 8000     # then open http://localhost:8000
```

## Deploying it

The page ships as part of the existing course site. Edit the files here, then copy them
across and drag the site folder onto app.netlify.com.

```bash
./deploy.sh                     # copies this folder into webapp/assignment8/
```

Then drag `webapp/` onto app.netlify.com. The page answers at
`<site>.netlify.app/assignment8/`, and the site front page links to it.

Run `deploy.sh` after any edit. It is the only thing that keeps the two copies identical,
because `webapp/` is not tracked by git and cannot be kept in step by the repository.

## Files

```
assignment8/
  index.html        the page
  css/app.css       one stylesheet, light and dark
  js/data.js        the twenty-nine mechanisms, their dates and their sources
  js/app.js         timeline, cards and source table
  js/demos.js       the nine live demos
  deploy.sh         copies the five files above into webapp/assignment8/
  README.md         this file
```
