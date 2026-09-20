/* Attention, in the order it was built.
   Every date is the first public appearance of the mechanism, taken from the
   primary record. Sources are listed in README.md and linked on every card. */

const LANES = {
  found: { name: 'Exactness',     note: 'Attention itself, and work that keeps it exact.' },
  pos:   { name: 'Position',      note: 'How a model learns where a token is, and how far it can go.' },
  mem:   { name: 'Decode memory', note: 'The key-value cache. What generation must store per token.' },
  comp:  { name: 'Compute',       note: 'The all-pairs score bill. Read fewer keys, or read them better.' },
  state: { name: 'Recurrent state', note: 'Replace the growing cache with one fixed-size state.' }
};

const MECHANISMS = [
{
  id: 'additive',
  date: '2014-09-01', dateText: '1 Sep 2014',
  name: 'Additive attention',
  sub: 'Neural Machine Translation by Jointly Learning to Align and Translate',
  who: 'Bahdanau, Cho, Bengio',
  url: 'https://arxiv.org/abs/1409.0473', src: 'arXiv 1409.0473 v1',
  lane: 'found', req: false,
  problem: "A translation model read the whole source sentence, squeezed it into a single fixed-length vector, and generated the translation from that vector alone. A twelve-word sentence and a sixty-word sentence were given exactly the same amount of space. Detail was therefore lost as sentences got longer, and quality fell with length.",
  touches: ["score", "read"],
  mechanism: 'The decoder scores its current state against every encoder state with a small feed-forward network, turns the scores into weights with a softmax, and reads back a weighted sum. The alignment is learned rather than fixed.',
  buys: [
    'The fixed-length bottleneck is gone. Quality stops falling as the sentence gets longer.',
    'The weights are readable, so the alignment can be inspected.'
  ],
  costs: [
    'One score for every source position at every output step.',
    'The scoring network is an extra learned module, not a plain dot product.',
    'It still sits on a recurrent model, so the sequence is still processed one step at a time.'
  ],
  pick: 'Historical. The idea survives everywhere. The form does not.',
  demo: null
},
{
  id: 'learned-abs',
  date: '2017-05-08', dateText: '8 May 2017',
  name: 'Learned absolute positions',
  sub: 'Convolutional Sequence to Sequence Learning, section 3.1',
  who: 'Gehring, Auli, Grangier, Yarats, Dauphin',
  url: 'https://arxiv.org/abs/1705.03122', src: 'arXiv 1705.03122 v1',
  lane: 'pos', req: true,
  problem: "An attention score is a dot product between two token vectors, so it compares meaning and nothing else. If the word bank appears at position 2 and again at position 20, both produce the same vector, and the score cannot tell which one is nearby. Once recurrence is removed, nothing in the model carries order, so position has to be supplied from outside.",
  touches: ["position"],
  mechanism: 'Store one trainable vector for each absolute position index and add it to the token embedding. The paper writes the input as e = (w1 + p1, ... , wm + pm), where p is the position embedding.',
  buys: [
    'It is the simplest thing that works. No function to design.',
    'The model learns whatever positional structure the data rewards.'
  ],
  costs: [
    'The table has a last row. Position L+1 has no vector at all, so the context length is a wall rather than a slope.',
    'Positions near the end of the table are seen rarely in training, so they are trained badly.',
    'The signal is absolute. Distance between two tokens has to be worked out indirectly.'
  ],
  pick: 'A short fixed context that will not move, such as an encoder with a set input size.',
  demo: 'position'
},
{
  id: 'sdpa',
  date: '2017-06-12', dateText: '12 Jun 2017',
  name: 'Scaled dot-product attention',
  sub: 'Attention Is All You Need',
  who: 'Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin',
  url: 'https://arxiv.org/abs/1706.03762', src: 'arXiv 1706.03762 v1',
  lane: 'found', req: true,
  problem: "Recurrent models read a sentence one token at a time, so training could not be spread across a processor, and a link between two distant words had to survive every step in between. Both limits have the same cause: two tokens had no way to interact directly. They could only pass information along the chain.",
  touches: ["baseline"],
  answers: 'additive',
  mechanism: 'Project each token three ways into a query, a key and a value. Score every query against every key with a dot product, divide by the square root of the head width, add minus infinity at every future position, take a softmax across the row, and read back the weighted sum of values. Several heads do this at once on separate slices of the width.',
  buys: [
    'Any token reaches any other token in one step.',
    'The whole sequence trains in parallel, which is what made scale possible.',
    'The weights are exact and specific to each query.'
  ],
  costs: [
    'The score matrix is T by T. Ten thousand tokens is one hundred million scores.',
    'Generation must keep one key and one value per earlier token, per layer, per head. That store is private to each conversation.',
    'The mechanism itself carries no order information. Position has to be added from outside.'
  ],
  pick: 'The baseline, always. At short context nothing beats it for quality per unit of engineering. Everything below is a response to one of its two bills.',
  demo: 'sdpa'
},
{
  id: 'sinusoidal',
  date: '2017-06-12', dateText: '12 Jun 2017',
  name: 'Sinusoidal positions',
  sub: 'Attention Is All You Need, section 3.5',
  who: 'Vaswani et al.',
  url: 'https://arxiv.org/abs/1706.03762', src: 'arXiv 1706.03762 v1',
  lane: 'pos', req: true,
  problem: "A learned position table holds one vector per position, so it has a fixed number of rows. A model trained with 512 rows has no vector at all for position 513 and cannot run there. The authors wanted a position signal that is defined at every position, including positions never seen during training.",
  touches: ["position"],
  answers: 'learned-abs',
  mechanism: 'Take the sine and the cosine of the position at a set of geometrically spaced frequencies, and add the result to the embedding. Nothing is trained.',
  buys: [
    'Defined at every position, so there is no table wall.',
    'No parameters at all.',
    'A shift by a fixed offset is a linear function of the encoding, so relative distance is in principle recoverable.'
  ],
  costs: [
    'It is added to the embedding, so position and content share the same dimensions and compete for them.',
    'The paper found it about equal to the learned table, so it bought no quality.',
    'Being defined past the training length did not make models work there. This is the first appearance of the gap that sections below spend eight years closing.'
  ],
  pick: 'Historical. Useful mainly because it names the problem that RoPE later solves properly.',
  demo: 'position'
},
{
  id: 'txl',
  date: '2019-01-09', dateText: '9 Jan 2019',
  name: 'Transformer-XL: segment recurrence',
  sub: 'Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context',
  who: 'Dai, Yang, Yang, Carbonell, Le, Salakhutdinov',
  url: 'https://arxiv.org/abs/1901.02860', src: 'arXiv 1901.02860 v1',
  lane: 'state', req: false,
  problem: "A model with a fixed window splits a long document into separate chunks and starts each chunk from nothing. A fact stated in chunk one is invisible in chunk two. Evaluation is wasteful as well, because predicting each next token means recomputing a whole window that mostly repeats the last one.",
  touches: ["position", "cache"],
  answers: 'sdpa',
  mechanism: 'Cache the hidden states of the previous segment and let the current segment attend to them as extra keys and values, with no gradient flowing back across the boundary. Absolute positions break under this scheme, so the paper replaces them with a relative position term inside the score.',
  buys: [
    'Context reaches past one window.',
    'Evaluation gets much faster, because a segment is computed once instead of being recomputed for every shifted window.',
    'Relative position arrives as a working idea, two years before RoPE.'
  ],
  costs: [
    'The cached states were produced by older weights, so they are slightly stale.',
    'Memory grows with the number of cached segments.',
    'The relative term adds arithmetic to every score.'
  ],
  pick: 'Streaming over long documents where a hard boundary is not acceptable. Its stop-gradient cache is the ancestor of every cross-chunk memory since.',
  demo: null
},
{
  id: 'sparse-tx',
  date: '2019-04-23', dateText: '23 Apr 2019',
  name: 'Sparse Transformer',
  sub: 'Generating Long Sequences with Sparse Transformers',
  who: 'Child, Gray, Radford, Sutskever',
  url: 'https://arxiv.org/abs/1904.10509', src: 'arXiv 1904.10509 v1',
  lane: 'comp', req: true,
  problem: "Attention compares every token with every other token. At 1,000 tokens that is 1 million scores. At 12,000 tokens it is 144 million. Images and raw audio need sequences of that length, so the all-pairs comparison did not make those tasks slow, it made them impossible.",
  touches: ["select"],
  answers: 'sdpa',
  mechanism: 'Factorise each attention head into a fixed pattern. One head reads a local block, another reads every n-th position. Two such steps connect any pair of positions. The cost falls to about T times the square root of T.',
  buys: [
    'A large saving with exact softmax kept inside the chosen pattern.',
    'The pattern is fixed, so the kernel is predictable and fast.',
    'It made tens of thousands of steps practical for the first time.'
  ],
  costs: [
    'A person chooses the pattern, not the data.',
    'A pair that matters but is not in the pattern cannot be connected in one layer.',
    'It works because raster-ordered images have structure that matches the stride. Text is less obliging.'
  ],
  pick: 'Data with a known regular structure. For text it is the idea, not the implementation, that carried forward.',
  demo: null
},
{
  id: 'mqa',
  date: '2019-11-06', dateText: '6 Nov 2019',
  name: 'Multi-query attention',
  sub: 'Fast Transformer Decoding: One Write-Head is All You Need',
  who: 'Noam Shazeer',
  url: 'https://arxiv.org/abs/1911.02150', src: 'arXiv 1911.02150 v1',
  lane: 'mem', req: true,
  problem: "When a model writes a reply one token at a time, every new token has to read the key and value stored for every token before it. The arithmetic is small but the amount of memory moved is large, so generation is limited by memory bandwidth rather than by compute. Adding more heads makes that stored history proportionally larger.",
  touches: ["project", "cache"],
  answers: 'sdpa',
  mechanism: 'Keep every query head, but give the whole layer one key head and one value head. All query heads search the same keys and values.',
  buys: [
    'The cache shrinks by the number of heads, and so does the bandwidth needed to read it.',
    'Decoding gets much faster at long context.'
  ],
  costs: [
    'Every head is forced to share one view of the past, which removes capacity.',
    'Quality falls measurably, and the paper reports training instability.',
    'It cannot be turned back up. The choice is all or nothing.'
  ],
  pick: 'Where decode speed decides everything and quality can give a little. In most places GQA replaced it four years later.',
  demo: 'cache'
},
{
  id: 'topk',
  date: '2019-12-25', dateText: '25 Dec 2019',
  name: 'Top-k attention',
  sub: 'Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection',
  who: 'Zhao, Lin, Zhang, Ren, Su, Sun',
  url: 'https://arxiv.org/abs/1912.11637', src: 'arXiv 1912.11637 v1',
  lane: 'comp', req: true,
  problem: "Softmax gives every key a weight above zero, so tokens with no relevance still contribute a little to the output. With thousands of tokens in context, those small contributions add up and blur the few tokens that actually carry the answer.",
  touches: ["select"],
  answers: 'sdpa',
  mechanism: 'Score every key, keep the k largest, set the rest to minus infinity, then softmax over the survivors. The paper states that the mask is applied to select the top-k contributive elements, and finds k = 8 best on its tasks.',
  buys: [
    'Attention concentrates on what matters, so this is a quality method as much as a cost method.',
    'The value read shrinks to k vectors.'
  ],
  costs: [
    'The selection needs all T scores first. The quadratic bill is not paid down at all. This is the catch that every later sparse method exists to fix.',
    'A useful key just outside the top k is lost with no recourse.',
    'k is fixed, but the number of relevant tokens is not.'
  ],
  pick: 'As a quality device, or downstream of a cheap proposal step that supplies the candidates. On its own it saves the wrong half of the bill.',
  demo: 'sparse'
},
{
  id: 'swa',
  date: '2020-04-10', dateText: '10 Apr 2020',
  name: 'Sliding window attention',
  sub: 'Longformer: The Long-Document Transformer',
  who: 'Beltagy, Peters, Cohan',
  url: 'https://arxiv.org/abs/2004.05150', src: 'arXiv 2004.05150 v1',
  lane: 'comp', req: true,
  problem: "Documents of several thousand tokens do not fit in an all-pairs score matrix. At the same time, most of what a token needs is close to it: the neighbouring words, the current sentence, the current paragraph. Paying for all-pairs comparison buys long-range access that the majority of tokens never use.",
  touches: ["select"],
  answers: 'sparse-tx',
  mechanism: 'Each token attends to w neighbours on each side. A few chosen tokens attend to everything and are attended to by everything. The cost becomes T times w, which is linear in length.',
  buys: [
    'Linear in sequence length, with no learned selection to get wrong.',
    'Stacking layers widens the reach by w per layer, so distant information can still arrive.',
    'The global tokens keep one exact path open for task-level information.'
  ],
  costs: [
    'A distant token reaches the query only after travelling up through layers, and that path is lossy.',
    'Exact one-hop long-range attention is gone.',
    'Which tokens become global is a decision made by hand.'
  ],
  pick: 'Long documents whose dependencies are mostly local. It is now the local branch inside almost every hybrid design.',
  demo: 'window'
},
{
  id: 'linear',
  date: '2020-06-29', dateText: '29 Jun 2020',
  name: 'Linear attention',
  sub: 'Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention',
  who: 'Katharopoulos, Vyas, Pappas, Fleuret',
  url: 'https://arxiv.org/abs/2006.16236', src: 'arXiv 2006.16236 v1',
  lane: 'state', req: true,
  problem: "Softmax divides each score by the sum of every score in that row, so the weight given to key 1 depends on keys 2 and 3. No weight can be worked out until every key is present. That single fact is what forces the model to keep every past key and value, and why that stored history grows with every token generated.",
  touches: ["score", "cache"],
  answers: 'sdpa',
  mechanism: 'Replace the exponential of the dot product with a product of feature maps, so the score factorises. Association then allows the key-value products to be summed before the query arrives. One state matrix replaces the whole history, and generation becomes a recurrent network.',
  buys: [
    'Sequence work becomes linear in length.',
    'The decode state has a fixed size. It is the same after a thousand tokens and after a million.',
    'Cost per generated token stops depending on how long the conversation is.'
  ],
  costs: [
    'No query-specific distribution over exact old tokens, so recall of one specific earlier detail gets weak.',
    'The state is a sum, so memories interfere with each other.',
    'The write rule only adds. It cannot revise an association it already holds. That is the next card.'
  ],
  pick: 'Very long context where decode memory binds harder than quality at 2K. In practice it is used as part of a hybrid, not alone.',
  demo: 'linear'
},
{
  id: 'delta',
  date: '2021-02-22', dateText: '22 Feb 2021',
  name: 'The delta rule',
  sub: 'Linear Transformers Are Secretly Fast Weight Programmers',
  who: 'Schlag, Irie, Schmidhuber',
  url: 'https://arxiv.org/abs/2102.11174', src: 'arXiv 2102.11174 v1',
  lane: 'state', req: true,
  problem: "A fixed-size state stores an association by adding it in. Suppose key A was written with the answer 40, and it now needs to return 55. Adding 55 makes a read return 95, because the 40 is still in the sum. The state has no way to take back what it wrote, so every old answer stays mixed into the new one.",
  touches: ["cache"],
  answers: 'linear',
  mechanism: 'Before writing, read what the state currently returns for this key. Take the difference between the wanted value and that reading. Write only the difference. The paper replaces the purely additive outer product with a delta rule instruction so the memory can correct its own mapping from keys to values.',
  buys: [
    'The state can now overwrite, not only accumulate, so its capacity holds current information instead of stale sums.',
    'Associative recall improves at the same state size.'
  ],
  costs: [
    'Read, then write, is sequential in the obvious form. This kept the rule out of large models for three years.',
    'A fixed state still has a capacity limit. Correcting is not the same as growing.',
    'The write strength is one more thing to learn.'
  ],
  pick: 'Whenever a fixed-state layer is used at all. Add-only linear attention is strictly weaker for the same cost.',
  demo: 'delta'
},
{
  id: 'rope',
  date: '2021-04-20', dateText: '20 Apr 2021',
  name: 'RoPE',
  sub: 'RoFormer: Enhanced Transformer with Rotary Position Embedding',
  who: 'Su, Lu, Pan, Murtadha, Wen, Liu',
  url: 'https://arxiv.org/abs/2104.09864', src: 'arXiv 2104.09864 v1',
  lane: 'pos', req: true,
  problem: "Sinusoidal and learned position vectors are added to the token embedding, so position and meaning are held in the same numbers and compete for the same space. The attention score is a dot product of those mixed vectors, so distance only reaches the score indirectly, as a side effect of a comparison that was about meaning.",
  touches: ["position"],
  answers: 'sinusoidal',
  mechanism: 'Split each head into pairs of dimensions and treat each pair as an arrow on a plane. Rotate the arrow by an angle proportional to the token position, at a different rate for each pair. When a query at position i meets a key at position j, the shared rotation cancels in the dot product and only the turn by i minus j is left.',
  buys: [
    'Relative distance appears directly inside the score, with no extra parameters and no extra score term.',
    'It is a function, so it is defined at any position and has no table wall.',
    'It works inside the standard fast attention kernels, which is why it won.'
  ],
  costs: [
    'Defined beyond the training length is not the same as trained there. The fast-rotating pairs complete many turns inside the window, and past it the model meets angle patterns it never saw.',
    'The base frequency is a quiet hyper-parameter that decides the usable range.',
    'It still rewards nothing about which distances matter. It only makes distance visible.'
  ],
  pick: 'The default for decoder models since 2022. Everything later in the position lane is repair work on this one failure mode.',
  demo: 'rope'
},
{
  id: 'alibi',
  date: '2021-08-27', dateText: '27 Aug 2021',
  name: 'ALiBi',
  sub: 'Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation',
  who: 'Press, Smith, Lewis',
  url: 'https://arxiv.org/abs/2108.12409', src: 'arXiv 2108.12409 v1',
  lane: 'pos', req: true,
  problem: "Every position method so far had been tested only at the length it was trained on. Nobody had asked the plain question: train a model on 1,024 tokens, then show it 2,048, and does it still work? The authors ran that test and found that both learned and sinusoidal positions degrade badly.",
  touches: ["position"],
  answers: 'rope',
  mechanism: 'Use no position vectors at all. Subtract a penalty from the score that grows with the distance between query and key, with a different fixed slope for each head.',
  buys: [
    'It extrapolates. The paper trains a 1.3 billion parameter model on length 1024 and matches, at length 2048, the perplexity of a sinusoidal model trained at 2048, while training 11 percent faster and using 11 percent less memory.',
    'No parameters, and almost no arithmetic.'
  ],
  costs: [
    'The penalty only ever decays, so a preference for recent tokens is built into the architecture rather than learned.',
    'An important token far back is punished for being far back.',
    'It cannot express a pattern that is not monotone in distance, which a rotation can.'
  ],
  pick: 'Length generalisation matters more than exact long-range recall, or the position budget is near zero. RoPE plus an extension method took most of its ground.',
  demo: 'position'
},
{
  id: 'flash',
  date: '2022-05-27', dateText: '27 May 2022',
  name: 'FlashAttention',
  sub: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness',
  who: 'Dao, Fu, Ermon, Rudra, Ré',
  url: 'https://arxiv.org/abs/2205.14135', src: 'arXiv 2205.14135 v1',
  lane: 'found', req: false,
  problem: "The field had assumed the cost of attention was the arithmetic, and spent five years approximating it to avoid that arithmetic. Measurement showed the assumption was wrong. The score matrix is written out to GPU main memory and read back several times, and it is that traffic, not the multiplying, that the clock is spent on.",
  touches: [],
  answers: 'sparse-tx',
  mechanism: 'Tile the work so each tile stays in on-chip memory, and use an online softmax so the full score matrix is never built. The backward pass recomputes tiles instead of storing them. The output is exactly the same attention, to the bit.',
  buys: [
    'Large real speedups and a memory footprint linear in length, with no approximation at all.',
    'It made long context affordable without changing the model, so no quality argument was needed.'
  ],
  costs: [
    'The arithmetic is still quadratic. It moved the wall, it did not remove it.',
    'It is a kernel, so it has to be rewritten for each hardware generation and each attention variant.',
    'It raised the bar every approximate method now has to clear, and several did not.'
  ],
  pick: 'Always, where a kernel exists. This card is the honest counterweight to the rest of the timeline: exact attention, engineered properly, beat most of its approximations.',
  demo: null
},
{
  id: 'gqa',
  date: '2023-05-22', dateText: '22 May 2023',
  name: 'Grouped-query attention',
  sub: 'GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints',
  who: 'Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai',
  url: 'https://arxiv.org/abs/2305.13245', src: 'arXiv 2305.13245 v1',
  lane: 'mem', req: true,
  problem: "Multi-head attention stores one key and one value per head, which is too much to serve. Multi-query stores one for the whole layer, which is small but measurably worse and unstable in training. There was nothing in between, and moving an existing model to either setting looked as though it needed a fresh pretraining run.",
  touches: ["project", "cache"],
  answers: 'mqa',
  mechanism: 'Divide the query heads into G groups and give each group one key head and one value head. G equal to one is MQA and G equal to the head count is ordinary attention. The paper also converts an existing checkpoint by mean-pooling its key and value heads, then uptrains with 5 percent of the original pre-training compute.',
  buys: [
    'Most of the cache saving of MQA with quality close to full multi-head.',
    'The conversion recipe means an existing model can be moved over cheaply.',
    'One dial instead of a binary choice.'
  ],
  costs: [
    'It changes what is stored per token. It does not change that something is stored for every token, so the cache still grows linearly with context.',
    'The group count is another thing to tune.',
    'Some quality is still given up against full multi-head.'
  ],
  pick: 'The default for any decoder that has to be served. Read this card as the baseline the long-context methods have to beat, not as the answer to long context.',
  demo: 'cache'
},
{
  id: 'pi',
  date: '2023-06-27', dateText: '27 Jun 2023',
  name: 'Position interpolation',
  sub: 'Extending Context Window of Large Language Models via Positional Interpolation',
  who: 'Chen, Wong, Chen, Tian',
  url: 'https://arxiv.org/abs/2306.15595', src: 'arXiv 2306.15595 v1',
  lane: 'pos', req: false,
  problem: "RoPE is a formula rather than a table, so it still returns an angle at a position far beyond the training length. That angle is the problem. It is a pattern the model has never seen, and the result is not a gentle loss of quality. Output collapses.",
  touches: ["position"],
  answers: 'rope',
  mechanism: 'Do not extrapolate. Compress. Divide every position index by the extension factor so the new range maps inside the range the model already knows, then fine-tune briefly. The paper reports context windows up to 32768 within 1000 fine-tuning steps.',
  buys: [
    'Every angle stays inside the trained range, so the collapse does not happen.',
    'A short fine-tune is enough, which made long context reachable for people with one machine.'
  ],
  costs: [
    'The same angular range now has to hold more positions, so neighbouring tokens get harder to tell apart.',
    'High-frequency detail, which is what encodes local order, is crushed the hardest.',
    'It needs fine-tuning. It is not a switch.'
  ],
  pick: 'The extension factor is known in advance and a short fine-tune is affordable. Read it as the parent of the two cards below.',
  demo: 'extend'
},
{
  id: 'ntk',
  date: '2023-06-28', dateText: 'late June 2023',
  dateApprox: true,
  name: 'NTK-aware scaled RoPE',
  sub: 'NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning',
  who: 'bloc97, posted to r/LocalLLaMA',
  url: 'https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/',
  src: 'community post, no paper',
  lane: 'pos', req: true,
  problem: "Position interpolation divides every position index by the same number, which compresses every rotation frequency by the same amount. The fastest-rotating dimensions are the ones that tell a token apart from its immediate neighbour, and they are damaged most by that squeeze. The model reaches further and reads local word order worse.",
  touches: ["position"],
  answers: 'pi',
  mechanism: 'Do not scale the positions. Change the RoPE base instead, so the squeeze is spread unevenly across the dimensions. The fastest pairs are left almost untouched and the slowest pairs absorb nearly all of the interpolation.',
  buys: [
    'It extends context with no fine-tuning at all, which position interpolation cannot do.',
    'Local detail survives, because the pairs that encode it are not compressed.'
  ],
  costs: [
    'The slowest dimensions still get pushed a little past their trained range.',
    'Perplexity still degrades. It degrades less.',
    'There is no correction for the way a longer context changes attention entropy. YaRN adds that.'
  ],
  pick: 'An existing checkpoint has to reach further today and there is no training budget. YaRN supersedes it two months later.',
  dateNote: 'This is the one entry with no paper and no dated primary record. It was a Reddit post. The date is given as late June 2023 because the earliest dated public reference found is a Hugging Face issue opened on 30 June 2023 that links to the post. It is stated here as approximate rather than guessed at precisely.',
  demo: 'extend'
},
{
  id: 'yarn',
  date: '2023-08-31', dateText: '31 Aug 2023',
  name: 'YaRN',
  sub: 'YaRN: Efficient Context Window Extension of Large Language Models',
  who: 'Peng, Quesnelle, Fan, Shippole',
  url: 'https://arxiv.org/abs/2309.00071', src: 'arXiv 2309.00071 v1',
  lane: 'pos', req: true,
  problem: "NTK-aware scaling protects the fast dimensions with a single change to the base, applied across the board. Two problems are left. The slowest dimensions are still pushed past the range they trained in, and nothing accounts for the fact that a longer context spreads the softmax thinner and makes attention less decisive.",
  touches: ["position", "score"],
  answers: 'ntk',
  mechanism: 'Split the RoPE dimensions by how many full turns each one completes inside the training window. Leave the fast ones alone, interpolate the slow ones fully, and ramp between the two. Then scale the attention logits by a temperature that compensates for the longer context.',
  buys: [
    'The strongest member of the RoPE extension family. The paper reports the same extension with 10 times fewer tokens and 2.5 times fewer training steps than previous methods.',
    'It works with fine-tuning and without it.'
  ],
  costs: [
    'It is still extension, so it inherits a ceiling. A factor that worked is not evidence for a larger one.',
    'Two turn thresholds and a temperature, all to be set.',
    'The model is still being asked to work in a regime it never trained in.'
  ],
  pick: 'Extending an existing RoPE checkpoint is the plan. It is the method to reach for in that family.',
  demo: 'extend'
},
{
  id: 'sinks',
  date: '2023-09-29', dateText: '29 Sep 2023',
  name: 'Attention sinks',
  sub: 'Efficient Streaming Language Models with Attention Sinks',
  who: 'Xiao, Tian, Chen, Han, Lewis',
  url: 'https://arxiv.org/abs/2309.17453', src: 'arXiv 2309.17453 v1',
  lane: 'mem', req: true,
  problem: "A conversation that never ends cannot let its cache grow forever, so the obvious fix is to drop the oldest entries and keep a window of recent ones. Do that and the model does not degrade gently. Output quality collapses the moment the very first tokens are dropped, even though those tokens are far away and look unimportant.",
  touches: ["select", "cache"],
  answers: 'swa',
  mechanism: 'The authors found that models pour large attention weight onto the first few tokens whatever those tokens say, because softmax has to put its weight somewhere even when there is nothing it wants. Those tokens act as a sink. Keep four of them in the cache permanently, plus a sliding window of recent tokens, and stability returns.',
  buys: [
    'Streaming at constant memory, with no fine-tuning. The paper reports stable modelling up to 4 million tokens.',
    'Up to 22.2 times faster than recomputing the window each time.',
    'It explained a failure that had been treated as a bug.'
  ],
  costs: [
    'It does not extend context. It extends stream length. Anything that left the window is still gone.',
    'It buys fluency, not memory, and the two are easy to confuse when reading the result.',
    'The cleanest version needs a dedicated sink token trained in from the start.'
  ],
  pick: 'An endless chat or feed where recent context is what matters and old context can genuinely be dropped.',
  demo: 'window'
},
{
  id: 'mistral-swa',
  date: '2023-10-10', dateText: '10 Oct 2023',
  name: 'Sliding window in production',
  sub: 'Mistral 7B',
  who: 'Jiang, Sablayrolles, Mensch, and the Mistral AI team',
  url: 'https://arxiv.org/abs/2310.06825', src: 'arXiv 2310.06825 v1',
  lane: 'comp', req: false,
  problem: "Windowed attention had been published three years earlier and was used in encoders, but no widely deployed decoder model had shipped with it. What it really costs in production, in quality and in serving, was therefore unknown outside research papers.",
  touches: ["select", "cache"],
  answers: 'swa',
  mechanism: 'A 4096-token sliding window over a 32K sequence, combined with grouped-query attention and a rolling buffer cache that overwrites its oldest slot in place.',
  buys: [
    'The cache stops growing. It is capped at the window size no matter how long the sequence gets.',
    'Stacked layers give an effective reach of roughly window times depth.',
    'It put a fixed per-request memory budget within reach of ordinary serving.'
  ],
  costs: [
    'Information past the window survives only by being carried up through layers, which is lossy.',
    'Exact recall of a distant token is not available at any layer.',
    'The advertised sequence length and the attended length are different numbers, which is easy to misread.'
  ],
  pick: 'Serving at scale with a fixed per-request cache budget. This card is here because it is where the research idea met a bill.',
  demo: 'window'
},
{
  id: 'mamba',
  date: '2023-12-01', dateText: '1 Dec 2023',
  name: 'Mamba: selective state spaces',
  sub: 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
  who: 'Gu, Dao',
  url: 'https://arxiv.org/abs/2312.00752', src: 'arXiv 2312.00752 v1',
  lane: 'state', req: false,
  problem: "Linear attention and earlier state-space models update their state in the same way for every token. The update does not depend on what the token says, so the model cannot choose to hold on to an important word or throw away a filler word. A state that cannot choose fills up with whatever happens to arrive.",
  touches: ["score", "cache"],
  answers: 'linear',
  mechanism: 'Make the state-space parameters functions of the current token, so the model decides per token what to keep and what to forget. That choice destroys the convolutional form, so the paper supplies a hardware-aware parallel scan instead.',
  buys: [
    'Linear time and a constant decode state, with content-based selection restored.',
    'The paper reports 5 times higher generation throughput than a Transformer, and a 3B model matching Transformers twice its size.'
  ],
  costs: [
    'Exact recall of a specific earlier token stays the weak point against softmax.',
    'The state size is a hard ceiling on how much can be held at once.',
    'It needs its own kernels, which slowed adoption.'
  ],
  pick: 'Throughput-bound long-sequence work. In practice it appears as the cheap layer inside a hybrid rather than on its own.',
  demo: null
},
{
  id: 'mla',
  date: '2024-05-07', dateText: '7 May 2024',
  name: 'Multi-head latent attention',
  sub: 'DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model',
  who: 'DeepSeek-AI',
  url: 'https://arxiv.org/abs/2405.04434', src: 'arXiv 2405.04434 v1',
  lane: 'mem', req: true,
  problem: "Grouped-query attention shrinks the cache by making several query heads share one key and one value head. The saving comes from the sharing, and so does the cost: heads that share cannot hold different views of the past. The open question was whether the cache can be made smaller without taking anything away from the heads.",
  touches: ["project", "cache"],
  answers: 'gqa',
  mechanism: 'Compress the keys and values of each token into one low-rank latent vector, and cache only that. The matrices that expand it again are folded into the query and output projections, so the full keys and values never have to be rebuilt in memory. RoPE does not survive that folding, so a small separate group of dimensions carries the rotary part.',
  buys: [
    'A much smaller cache than GQA at equal or better quality. The paper reports the KV cache reduced by 93.3 percent against its own previous model, and 42.5 percent of training cost saved.',
    'Every head keeps its own view of the past. Nothing is shared away.'
  ],
  costs: [
    'Considerably harder to implement and to serve than GQA.',
    'The decoupled rotary dimensions are a patch. They exist because the algebra does not work otherwise.',
    'Arithmetic per token goes up even as memory goes down.',
    'An existing GQA checkpoint cannot be converted in any straightforward way.'
  ],
  pick: 'High-concurrency serving where cache is the binding cost and the engineering effort is affordable.',
  demo: 'cache'
},
{
  id: 'mamba2',
  date: '2024-05-31', dateText: '31 May 2024',
  name: 'Mamba-2 and state space duality',
  sub: 'Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality',
  who: 'Dao, Gu',
  url: 'https://arxiv.org/abs/2405.21060', src: 'arXiv 2405.21060 v1',
  lane: 'state', req: false,
  problem: "State-space models and attention had grown up as separate fields, with separate theory, separate code and separate kernels, so an improvement to one did not help the other. Mamba also had to be computed as a sequential scan, which cannot use the matrix-multiply units that make modern accelerators fast.",
  touches: ["score", "cache"],
  answers: 'mamba',
  mechanism: 'Show that a class of state-space models and a class of masked attention are the same computation. That equivalence yields a block-decomposed algorithm which runs as matrix multiplication.',
  buys: [
    'The core layer is 2 to 8 times faster than Mamba while staying competitive with Transformers.',
    'The two families can now borrow from each other, and several ideas move straight across.'
  ],
  costs: [
    'The duality holds for a restricted class, which forces the state transition down to a scalar decay per head.',
    'That is less expressive than the general recurrence it replaces.'
  ],
  pick: 'As the recurrent layer of a hybrid. Its real value on this timeline is conceptual: the recurrence branch and the attention branch turn out to be one family, seen from two sides.',
  demo: null
},
{
  id: 'deltanet-parallel',
  date: '2024-06-10', dateText: '10 Jun 2024',
  name: 'DeltaNet, made trainable',
  sub: 'Parallelizing Linear Transformers with the Delta Rule over Sequence Length',
  who: 'Yang, Wang, Zhang, Shen, Kim',
  url: 'https://arxiv.org/abs/2406.06484', src: 'arXiv 2406.06484 v1',
  lane: 'state', req: false,
  problem: "The delta rule reads the state before it writes to it, and that read depends on every write before it. The steps are therefore strictly ordered and cannot be spread across a processor the way attention can. A rule published in 2021 was still missing from large models three years later for this reason alone.",
  touches: ["cache"],
  answers: 'delta',
  mechanism: 'Rewrite the sequence of rank-one corrections as a product of Householder-like matrices using the WY representation, so a whole chunk of steps can be applied as matrix multiplications.',
  buys: [
    'The delta rule becomes trainable at scale. The paper trains a 1.3B model on 100B tokens and beats Mamba and GLA on perplexity and zero-shot tasks.',
    'Hybrids of DeltaNet layers with attention layers beat strong transformer baselines.'
  ],
  costs: [
    'Parallelism is chunked, so the chunk size becomes a new knob.',
    'It removes nothing from the fixed-state capacity limit. It only makes the rule affordable.'
  ],
  pick: 'Not a choice, a prerequisite. Every modern delta-rule layer exists because of this paper.',
  demo: null
},
{
  id: 'diff',
  date: '2024-10-07', dateText: '7 Oct 2024',
  name: 'Differential attention',
  sub: 'Differential Transformer',
  who: 'Ye, Dong, Xia, Sun, Zhu, Huang, Wei',
  url: 'https://arxiv.org/abs/2410.05258', src: 'arXiv 2410.05258 v1',
  lane: 'found', req: false,
  problem: "Softmax weights have to add up to one, so attention must put all of its weight somewhere even when nothing in the context is relevant. The leftover weight spreads across unrelated tokens as noise. In a long document that noise is what buries the one fact the model was asked to find, and it feeds answers that read well and are not supported.",
  touches: ["score"],
  answers: 'sdpa',
  mechanism: 'Compute two separate softmax attention maps and subtract one from the other, scaled by a learned value. What is common to both cancels, in the way a differential amplifier rejects common-mode noise. The result is sparse without anything being selected.',
  buys: [
    'Better retrieval inside long context, less sensitivity to the order of the prompt, and reported reductions in hallucination.',
    'The paper reports that it needs about 65 percent of the model size or training tokens of a standard Transformer for comparable language modelling.'
  ],
  costs: [
    'Two attention maps, so more compute for the same number of heads.',
    'A new schedule to tune for the subtraction weight.',
    'It reduces neither the cache nor the asymptotic cost by any amount.'
  ],
  pick: 'The failing metric is retrieval accuracy inside a long context rather than speed. This card is worth noticing because it attacks a quality problem, not a bill. By 2024 not every change is about cost.',
  demo: null
},
{
  id: 'gated-delta',
  date: '2024-12-09', dateText: '9 Dec 2024',
  name: 'Gated DeltaNet',
  sub: 'Gated Delta Networks: Improving Mamba2 with Delta Rule',
  who: 'Yang, Kautz, Hatamizadeh',
  url: 'https://arxiv.org/abs/2412.06464', src: 'arXiv 2412.06464 v1',
  lane: 'state', req: true,
  problem: "The delta rule corrects one stored association precisely, but it cannot clear the state, so unrelated old material stays in place. A decay gate can clear the state, but it fades everything at once and cannot make a targeted correction. Each mechanism was missing exactly what the other one had.",
  touches: ["cache"],
  answers: 'deltanet-parallel',
  mechanism: 'Do both in one update. A forget gate scales the whole state down, and the delta correction writes the specific change on top of it.',
  buys: [
    'Precise correction and the ability to forget, in one rule.',
    'Reported to beat Mamba2 and DeltaNet on language modelling, in-context recall and length extrapolation.'
  ],
  costs: [
    'Still a fixed state, so exact recall from far back remains bounded.',
    'Two more quantities per step to learn.',
    'It needs its own chunked kernel to be fast.'
  ],
  pick: 'As the recurrent layer of a hybrid. Qwen3-Next shipped it in September 2025 in a three-to-one layout against gated attention, which is the production evidence for this row.',
  demo: 'delta'
},
{
  id: 'nsa',
  date: '2025-02-16', dateText: '16 Feb 2025',
  name: 'Native sparse attention',
  sub: 'Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention',
  who: 'Yuan, Gao, Dai, and the DeepSeek team',
  url: 'https://arxiv.org/abs/2502.11089', src: 'arXiv 2502.11089 v1',
  lane: 'comp', req: true,
  problem: "Sparse attention was being applied at inference to models that had been trained densely, so the model had never learned to work with it and quality suffered. The kernels were also too slow to use during training, which ruled out fixing that. On top of both, top-k selection still had to score every key before it could choose any.",
  touches: ["select", "cache"],
  answers: 'topk',
  mechanism: 'Three branches run for each query and a learned gate combines them: compressed coarse blocks for global context, a small number of fine-grained blocks selected by their block scores, and a sliding window for local detail. The block structure is aligned to what the hardware reads in one go, and the model is trained this way from the start.',
  buys: [
    'Sparsity that is trained rather than bolted on, so the model learns to use it.',
    'Substantial speedups over full attention on 64k-length sequences in decoding, in the forward pass and in the backward pass.',
    'It matches or exceeds full attention on general, long-context and reasoning benchmarks.'
  ],
  costs: [
    'Three branches and a gate is real architectural complexity.',
    'Selection happens per block, so it is coarse by construction.',
    'The gains depend on custom kernels, which ties the design to a hardware generation.',
    'The compressed branch still loses token-level detail.'
  ],
  pick: 'A long-context model is being trained from scratch and the kernels are available.',
  demo: 'sparse'
},
{
  id: 'dsa',
  date: '2025-09-29', dateText: '29 Sep 2025',
  name: 'DeepSeek sparse attention',
  sub: 'DeepSeek-V3.2-Exp release, then the DeepSeek-V3.2 report',
  who: 'DeepSeek-AI',
  url: 'https://api-docs.deepseek.com/news/news250929/', src: 'vendor release note, 29 Sep 2025; report arXiv 2512.02556',
  url2: 'https://arxiv.org/abs/2512.02556',
  lane: 'comp', req: true,
  problem: "Native sparse attention showed that trained sparsity works, but it was a research architecture. A production model also has to keep the latent cache compression it already depends on, run on serving infrastructure that exists, and save enough that the saving reaches the price list rather than a benchmark table.",
  touches: ["select", "cache"],
  answers: 'nsa',
  mechanism: 'A lightweight indexer scores which earlier tokens matter for the current query, and fine-grained top-k selection passes only those to the expensive attention, on top of multi-head latent attention. The cost falls from growing with the square of the length to growing with the length times k.',
  buys: [
    'It shipped. API prices were cut by more than half at release, which is the plainest public statement of the saving.',
    'It keeps the latent cache compression and adds sparsity on top, rather than choosing between them.'
  ],
  costs: [
    'The indexer is another learned component, and it can be wrong.',
    'Approximate selection can miss a block that mattered.',
    'It was released as an experimental model, and its quality claims are the vendor’s own.'
  ],
  pick: 'Serving long context at scale. Read this one because it is the first widely deployed fine-grained sparse attention, which makes its trade-offs measurable rather than argued.',
  demo: 'sparse'
},
{
  id: 'drope',
  date: '2025-12-13', dateText: '13 Dec 2025',
  name: 'DroPE',
  sub: 'Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings',
  who: 'Gelberg, Eguchi, Akiba, Cetin (Sakana AI)',
  url: 'https://arxiv.org/abs/2512.12167', src: 'arXiv 2512.12167 v1',
  url2: 'https://sakana.ai/drope/',
  lane: 'pos', req: true,
  problem: "Every position method since 2021 repairs RoPE so that it survives past the length it trained on. All of that work assumed the positional embedding has to be there, and that the only question is how to stretch it. The opposite idea had not been tested: that the embedding itself is what stops a model generalising to longer inputs.",
  touches: ["position"],
  answers: 'yarn',
  mechanism: 'Keep RoPE during pretraining, where it helps the model converge, then remove the positional embeddings from every layer and run a short recalibration at the original context length. After that, order reaches the model only through the causal mask.',
  buys: [
    'Usable context extends with no long-context fine-tuning.',
    'The adaptation can cost as little as 0.5 percent of the pretraining budget.',
    'It is reported to beat both context extension methods and architectures designed for length, on models up to 7B trained on trillions of tokens.'
  ],
  costs: [
    'The recalibration phase is required, so this is a training recipe and not an inference switch.',
    'Removing position entirely gives up the explicit distance signal, and what replaces it is implicit.',
    'It is very new. It has not been reproduced widely yet, and that is the correct amount of caution to hold.'
  ],
  pick: 'A pretraining recipe is being written now, or a checkpoint has to reach further on a small budget.',
  dateNote: 'A different and unrelated method for autonomous driving is also called DRoPE (arXiv 2503.15029, 19 March 2025). It rotates trajectory headings and has nothing to do with context length. The two are easy to confuse, and searching the name returns the wrong one first.',
  demo: null
}
];
