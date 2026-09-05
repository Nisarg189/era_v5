# Reversible Kronecker embeddings: removing the output head

**Session 7, Problem 5.** This report shows that the Kronecker embedding can be run
backwards, and that doing so lets the model's output head be removed entirely. The
result is a language model whose cost, at both the input and the output, no longer grows
with the size of the vocabulary. Every number below comes from a run on real course data.

## Problem

A language model turns each token into a vector at the input, and at the output turns a
vector back into a token by scoring it against every token in the vocabulary. Both steps
are usually large stored matrices of shape `V x D` (vocabulary size by model width). At
the V5 reference shape each is about 1.06 billion parameters.

The Kronecker embedding taught in Session 7 removes the input matrix. Instead of storing
one row per token, it builds each token's vector from the token's own bytes, through a
single shared projection whose size does not depend on `V`. This solves the input side.
It does not touch the output side. The output head remains a full `V x D` matrix, so at a
large vocabulary the model still pays the full price once, and if the head is untied it
pays it twice.

Problem 5 asks whether the Kronecker embedding can be made reversible, so that a token
can be recovered from a vector rather than looked up in a stored matrix. If so, the
output head is no longer needed, and a vocabulary of a million tokens becomes affordable.

## Idea

The forward path is `token -> bytes -> code -> projection -> vector`. The **code** is the
step that matters. Each token's bytes are written onto a fixed grid of 256 byte values by
32 positions, one mark per byte, and that grid is the code. The code is sparse and
deterministic, and it is exactly invertible: reading the marks off the grid recovers the
bytes, and the bytes are the token. Only the final projection, which maps the 8192-long
code down to the model width, loses information.

The proposal is to tie the output head to that same input projection. The score the model
assigns to a candidate token then equals the similarity between the model's output vector
and the candidate's code, where the code is generated from the candidate's bytes on
demand. No `V x D` matrix is stored. Predicting the next token becomes choosing the
nearest code, and the codec turns that code back into a string. This is the forward
Kronecker path run in reverse.

Intuitively: instead of keeping a dictionary with one page per word and flipping to the
right page, the model learns to *spell*. It emits something close to a word's byte
pattern, and the pattern is read directly. Spelling does not need a bigger book when the
language has more words.

## Method

The claim is established by five measurements. Three are exact and require no training
(the codec is invertible, the head equals a code comparison, the head has no
vocabulary-sized parameters). Two require a trained model (removing the head barely
changes quality, and an unbounded vocabulary works at test time).

Because no deep-learning framework was reachable on the training machine, the transformer
was written from scratch in numpy with a hand-written autograd. Every gradient was
verified against finite differences to within 1e-10, so the training arithmetic is
inspectable rather than taken on faith.

## Data

Training used the same six-lane mixture defined in Sessions 5 and 6, assembled from the
course data already on disk: fineweb-edu for English web text, AI4Bharat Sangraha for
Hindi, NuminaMath for mathematics, GitHub Python for code, and Hermes for agentic
function calling, plus the Session 2 English, Hindi, Gujarati and Telugu samples. The
corpus was tokenised to a 1,200-token vocabulary, giving 967,490 training tokens and
107,502 validation tokens. Vocabulary tokens were restricted to 32 bytes or fewer so the
codec is lossless over the vocabulary; the separate long-token case belongs to Problem 3.

## Model

A small pre-LayerNorm transformer: width 96, two layers, single-head causal attention,
block length 48. It has two interchangeable input paths (Kronecker codec, or a dense
table) and two interchangeable output heads (a dense `V x D` matrix, or the reversible
tied-Kronecker head). Training used Adam at 1e-3 with gradient clipping at norm 1.0 for
600 steps. To compare fairly, the models differ in one component at a time and share the
same initialisation seed, the same batches in the same order, and the same optimizer.

## Results

### The codec is invertible, and loses information only by collision

Encoding and decoding the real 9,975-token Session 2 vocabulary, every token that fits
the byte window is recovered exactly. The only loss is a collision: two tokens that share
their first 32 bytes receive the same code.

| Byte window | Unique codes | Collisions | Exact round-trip |
|---|---|---|---|
| 32 | 9,958 / 9,975 | 17 | every in-window token |
| 48 | 9,975 / 9,975 | 0 | every in-window token |
| 64 | 9,975 / 9,975 | 0 | every in-window token |

The collisions are not evenly distributed. At the shipped window of 32 they fall almost
entirely on Indic scripts, which UTF-8 encodes at three bytes per character.

| Script | Tokens | Collisions at window 32 |
|---|---|---|
| Latin / ASCII | 5,984 | 1 |
| Telugu | 2,087 | 15 |
| Devanagari | 970 | 0 |
| Gujarati | 778 | 1 |

Reversibility is therefore exact in the regime the head needs. The residual loss is a
byte-window question that costs a script rather than a rounding error that costs everyone.

### The output head is a code comparison, not a stored matrix

For a trained model, the head's scores factor exactly as `logits = q @ C^T`, where
`q = H @ W_in^T` is the model's output projected into code space and `C` are codes
generated from candidate bytes. Measured on a trained model, the largest difference
between the model's logits and this code comparison is `3.4e-14`, i.e. equal to machine
precision. The head is an operation over generated codes, not a `V x D` parameter.

### Removing the head costs little quality

Three models were trained identically, differing only in how tokens enter and leave.
Perplexity is lower-is-better; accuracy is next-token top-1.

| Model | Total params | Head params (grow with V) | Val perplexity | Val accuracy |
|---|---|---|---|---|
| Kronecker input + reversible head | 1,014,145 | **0** | 17.43 | 0.339 |
| Kronecker input + dense head | 1,129,344 | 115,200 | 14.67 | 0.351 |
| dense input + dense head (classic) | 458,112 | 115,200 | 12.23 | 0.396 |

The controlled comparison is the first two rows, which share an identical input path and
differ only in the head. Removing the head costs 1.2 points of accuracy and 0.17 nats,
and removes 100 percent of the vocabulary-sized head parameters. The reversible head keeps
about 97 percent of the dense head's accuracy for none of its cost. Behaviour is stable
across the mixture rather than collapsing on one script:

| Model | web (en) | indic | math | code | agentic |
|---|---|---|---|---|---|
| reversible head | 2.96 | 3.21 | 3.29 | 2.15 | 2.78 |
| dense head | 2.81 | 2.96 | 3.06 | 2.08 | 2.61 |

Two caveats keep this honest. At a 1,200-token vocabulary the Kronecker *input* saving has
not yet begun (it starts once `V` exceeds 8,192), which is why the classic model is
smallest here; the input saving is asymptotic and is shown by the cost table below. And
the absolute perplexities are high because the model and data are deliberately small. The
findings that carry are the relative comparison and the exact measurements, not the levels.

### The impact: a large vocabulary becomes free

Computed at the V5 width of 8,096 with full AdamW training-state accounting (16 bytes per
parameter):

| Vocabulary | Dense head params | Dense head training memory | Reversible head |
|---|---|---|---|
| 131,072 (V5 today) | 1.06 B | 17.0 GB | **0** |
| 262,144 (V5 target) | 2.12 B | 34.0 GB | **0** |
| 1,000,000 | 8.10 B | 129.5 GB | **0** |

The reversible head reuses the 66.3 million-parameter projection the input path already
carries and adds nothing that scales with `V`. At the V5 target it removes 34 GB of
optimizer state from a single matrix; at a million tokens it removes 129.5 GB, more than a
single 80 GB accelerator holds.

Two behaviours confirm this is operational, not only arithmetic. First, the candidate
vocabulary can be expanded at test time with tokens the model never trained on, and
prediction holds while the head gains nothing:

| Candidate vocabulary | Unseen tokens added | Val accuracy | Head params added |
|---|---|---|---|
| 1,200 | 0 | 0.375 | 0 |
| 4,200 | 3,000 | 0.367 | 0 |
| 7,200 | 6,000 | 0.364 | 0 |

Growing the candidate set six times over costs about one point of accuracy and no
parameters and no retraining. A dense head cannot do this, because it has no row for a
token it never allocated. Second, the head scores unseen tokens by language rather than by
noise: after a Hindi context, forty unseen real Hindi words receive probability mass 0.189
against 0.0002 for forty random Devanagari byte strings, a factor of roughly 900. The
reversible head performs open-vocabulary prediction, which a fixed head cannot.

## Intuitive summary

Kronecker turns a word into a vector by reading its letters onto a grid, so the input
never needs a per-word table. The letter grid is reversible, so the same trick works in
reverse: the model produces a vector, the vector is read back to a code, and the code is
read back to letters. That reverse reading replaces the output head. Because letters are
generated on demand, neither end of the model stores anything that grows with the
vocabulary, and a million-word vocabulary costs the same as a small one. The price is a
small drop in quality, because the input and output now share one set of weights and have
less room to specialise.

## Limitations

The reversible head is a modest quality trade, 1.2 accuracy points behind a dense head in
the controlled comparison, because tying the output to the input codec removes degrees of
freedom. It requires gradient clipping to train, since tying couples the input embedding
and the output logits through one matrix. Reversibility is exact only within the byte
window, so Problem 5 depends on Problem 3: at window 32 the head cannot separate tokens
that already collide, and 15 of the 9,975 real tokens are affected, all Indic. The model
is deliberately small, so the results that carry are the exact and relative ones: the
collision audit, the parameter counts, the code-comparison identity, and the head-to-head
comparison.

## Reproduce

```
assignment7/
  README.md            this report
  run_demo.py          one command regenerates every number above
  rkron/
    codec.py           the Kronecker codec and its exact inverse
    autograd.py        a small reverse-mode autograd over numpy, gradient checked
    model.py           the transformer, both input paths and both output heads
    data.py            the six-lane corpus and vocabulary builder
    train.py           training and evaluation helpers
  experiments/
    exp_invertibility.py   the invertibility and collision audit
    exp_heads.py           the three-model comparison and the cost table
    exp_reversal.py        the code-comparison identity, decoding, open vocabulary
  tests/
    test_autograd.py       finite-difference gradient checks
    test_reversible.py     round-trip, collision, code-identity, nearest-code invariants
  artifacts/             the JSON and logs every table is read from
```

```bash
cd assignment7
python3 run_demo.py     # gradient checks, invariant tests, then the three experiments
```

The run is CPU-only, uses numpy alone, reaches no network, and takes roughly fifteen to
twenty minutes, dominated by training the three models. The corpus is rebuilt from the
course data already on disk.
