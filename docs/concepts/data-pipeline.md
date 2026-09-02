# Concept: the data pipeline — shards, windows, and the resumable shuffler

> **Canonical** for the shard/window contract, the no-+1-shift rationale,
> and deterministic-resume data order. `DIFFUSION.md` §6 stays authoritative
> for rulings; this page derives the pipeline from the bytes up.

**Depends on:** [foundations](foundations.md) §17 ·
[diffusion-core](diffusion-core.md) §3 ·
**Read next:** [training](../training.md)

---

## Table of Contents

1. [The contract: shards → windows → batches](#1-the-contract)
2. [The producer: tokenization, cleaning, packing](#2-the-producer)
3. [Shard layout and window math](#3-shard-layout-and-window-math)
4. [Why there is no +1 shift](#4-why-there-is-no-1-shift)
5. [The resumable shuffler](#5-the-resumable-shuffler)
6. [Worked example: indexing a window at toy scale](#6-worked-example-indexing-a-window-at-toy-scale)
7. [What breaks if you change this](#7-what-breaks-if-you-change-this)
8. [Glossary](#8-glossary)
9. [Interview Q&A](#9-interview-qa)

---

## 1. The contract

Three objects move bytes to the trainer
(`data/dataset.py:ShardWindows`, `data/dataset.py:ShuffledRangeSampler`,
`data/dataset.py:build_dataloader`):

```
shared_data pipeline ──► shard_*.bin (uint32 memmap, 50M tokens each)
        │                                  │
        └ data/prepare_data.py main()      └ ShardWindows (flat seq_len windows)
                (tokenizer: gpt2, V=50,257,          │
                 EOS=PAD=50,256)                     ShuffledRangeSampler (seed 42)
                                                     │
                                            build_dataloader → DataLoader (drop_last=True)
                                                     │
                                              batches of B windows  ──► trainer
                                              q_sample corrupts x0 → xt; chunked_x0_ce ◄──┘
                                              scores the posterior against clean x0
```

`data/dataset.py:ShardWindows` memmaps every `shard_*.bin` under
`train_data_path` (`configs/pretrain_a100_380m.yaml` data block), slices each
into `len(mm) // seq_len` windows of exactly `seq_len = 4,096`, and indexes
across shards via a cumulative-start table (`bisect`). The dataset emits the
**clean** window (x0); corruption into `xt` happens in the trainer
(`models/diffusion.py:q_sample`) and the chunked cross-entropy consumes x0 as
the target (`training/losses.py:chunked_x0_ce`) — the dataset is deliberately
corruption-agnostic, so the same windows feed training, the held-out NLL eval,
and hand inspection.

### 1.1 Shape table, toy and production

| quantity | toy (tests) | production (8.0B config) |
|---|---|---|
| shards | 2 × 10 tokens | 160 × 50,000,000 tokens |
| shard file size | 40 B | 200 MB (50M × 4 B uint32) |
| `seq_len` | 4 | 4,096 (= 16 canvases × 256) |
| windows per shard | `10 // 4 = 2` | `50,000,000 // 4,096 = 12,207` |
| tail dropped per shard | 2 tokens | 128 tokens (`50M − 12,207·4,096`) |
| total windows (`n_windows`) | 4 | 160 × 12,207 = **1,953,120** |
| usable tokens | 16 of 20 | 7,999,387,648 of 8.0B (99.9997%) |
| micro-batch shape | `(2, 4)` | `(16, 4096)` int64 |
| batches per full pass | 2 | 122,070 micro / 61,035 effective (32×4096) |
| windows the 61,000-step run consumes | — | 61,000 × 2 × 16 = 1,952,000 (fits one pass) |

Two boundaries exist in this corpus and only one is forbidden. Windows cross
**document** boundaries freely — documents are concatenated inside a shard
before windowing, so a 4,096-token window typically straddles many documents,
separated by EOS tokens. Windows never cross **shard** boundaries: windowing
is per-shard, so a window is always a contiguous slice of one memmap.
Conflating the two is the most common misreading of
`cross_document_boundary_ok: false` — that contract governs how documents are
*packed into shards* (§2.4), not how windows are cut from shards.

## 2. The producer

`data/prepare_data.py:main` is the producer: it materializes a project-local
`data_config.yaml` pinning the GPT-2 tokenizer contract
(`vocab 50,257, EOS = PAD = 50,256`) and delegates download → clean →
tokenize → pack to the workspace `shared_data` pipeline
(`data/prepare_data.py:_require_shared_data` raises a vendor-me error when
the package is absent). The repo never reimplements tokenization or packing;
it pins a contract and delegates.

### 2.1 Why the shim exists at all

`shared_data` is a workspace-wide pipeline serving every LLM Lite in the
portfolio. The per-project variance is the **tokenizer contract**: the
embedding matrix is sized `V = 50,257`, the head is weight-tied to it, and the
corruption draws noise uniformly over those same 50,257 ids — if the packed
shards used any other tokenizer, every id would be wrong by construction, with
no adaptation step downstream. So the shim's whole job is to make the *shared*
pipeline emit *this project's* ids:
`data/prepare_data.py:_apply_diffusiongemma_defaults` →
`data/prepare_data.py:_ensure_diffusiongemma_data_config` reads shared_data's
universal config, overrides the tokenizer name/vocab/EOS/PAD with the
DiffusionGemma constants, writes `data/data_config.yaml` (checked in as the
contract record, stamped with `_generator`/`_tokenizer_family` provenance),
and then `run_pipeline` runs with it.

`data/data_config.yaml` is the persisted result: `pipeline.tokenizer`
(gpt2, 50,257, EOS = PAD = 50,256), `sharding` (50M-token uint32 shards,
8.0B target), `dedup.enabled: true`, the quality filters, and
`pack.cross_document_boundary_ok: false`. Pinned by
`tests/test_data.py::test_data_config_contract`.

### 2.2 The tokenizer contract and why EOS = PAD

GPT-2 BPE, `V = 50,257`, `EOS = PAD = 50,256`. The EOS id is load-bearing:
the pack stage joins documents inside a shard with EOS separators, so the
model sees explicit document terminators in every window — a 4,096-token
window crossing k documents contains k EOS tokens, and the model learns
document-length structure from them.

PAD shares the id because **the data path never pads**: windows are exact
flat slices (`len(mm) // seq_len` complete windows, tail dropped,
`drop_last=True` at the batch level), so no position ever holds filler. A
distinct PAD token would reserve a vocab slot that uniform-state corruption
wastes noise draws on — a never-trained-as-content id consumes softmax
probability mass and appears as noise the model must learn to ignore. Pinning
`pad_token_id = eos_token_id` satisfies the config contract for downstream
code that asks for a pad id, at zero vocab cost. This is a property of the
*flat-window* design: a padded-batch pipeline would need a real, separate PAD.

### 2.3 Cleaning and dedup

`data/data_config.yaml` quality block: `drop_empty: true`,
`min_unique_chars_ratio: 0.05`, `max_digit_ratio: 0.5`,
`max_punct_ratio: 0.5`, `max_whitespace_ratio: 0.5`, plus `dedup: enabled`.
Why bother at 8.0B tokens: a from-scratch model trained on one corpus pass
cannot average over repeated garbage across epochs — every duplicate page is
budget spent memorizing duplication instead of language statistics. The
filters are deliberately cheap and recall-biased (ratio thresholds, not
classifiers): a false positive that drops a quirky-but-real document costs
less than a heavy quality model over billions of tokens, and cheap shared
filters keep the mixture uniform across every Lite in the portfolio.

### 2.4 The packing strategy: concatenate, then window

The pack contract (`pack.cross_document_boundary_ok: false`) means a document
is **never split across two shards**: the packer fills a shard's 50M-token
budget with whole documents (joined by EOS) and closes the shard when the next
document would not fit. The alternative designs, and why each loses:

| strategy | what it costs here |
|---|---|
| **Doc-aligned variable-length batches** (BERT-style: batch documents, pad to longest) | needs a real PAD id (§2.2's vocab-mass problem); pad positions are dead FLOPs under a loss that reconstructs *every* position; window shapes vary per batch |
| **One global token stream, windowed across the whole corpus** | windows straddle shard boundaries → indexing needs cross-memmap reads; resume math must track byte offsets, not window ids; per-shard parallel prep impossible; a 1-token append shifts every later window's alignment |
| **Per-document windows with truncation** | short documents waste window capacity or force truncation at 4,096; long documents lose their tails; the block-AR model never sees document-spanning context |
| **Concatenate within shard + fixed flat windows** (chosen) | 128 tokens of tail per shard (0.003%); every position is real text; O(1) memmap indexing; windows are byte-identical slices, so caching/inspection is trivial |

The chosen strategy's only loss is the per-shard tail — 128 tokens × 160
shards = 20,480 tokens out of 8.0B (0.00026%), bought for a single `bisect`
per lookup (`data/dataset.py:ShardWindows.__getitem__`).

## 3. Shard layout and window math

### 3.1 Why 50M-token uint32 shards

`shard_size_tokens: 50000000`, `dtype: uint32`. Arithmetic first: 8.0B / 50M
= 160 shards exactly; each file is 50M × 4 B = 200 MB. The number sits at the
intersection of three constraints:

- **mmap-friendliness.** 200 MB fits page cache alongside the model's weight
  reads; `np.memmap` opens a shard lazily and the OS pages in only the touched
  16 KB pages, so a random-window workload touches a tiny fraction of the file.
- **Pipeline restartability.** The shared pipeline stages (download → clean →
  tokenize → pack) checkpoint at shard granularity: a killed pack run resumes
  at a shard boundary — fine-grained progress without 8,000 tiny files or one
  unresumable blob.
- **Granularity vs. open cost.** `data/dataset.py:ShardWindows.__init__`
  opens *every* shard (160 handles — trivial). At 4M-token shards you would
  hold 2,000 handles and lose up to 4,095 tokens per shard as tail — 2.9 GB
  corpus-wide instead of 20,480 tokens.

Why uint32 when GPT-2's ids fit in uint16 (50,256 < 65,535): the dtype is the
`shared_data` house format shared across the portfolio, where sibling Lites
use larger vocabs that uint16 cannot hold. Spending 2× disk (200 MB vs
100 MB per shard) buys one dtype contract across every model that reads these
shards — and the dataset code never needs a dtype branch. It is a deliberate
over-allocation, not an oversight.

### 3.2 The cumulative-start table

`data/dataset.py:ShardWindows.__init__` builds the global index:

```
self.shards        = [np.memmap(p) for p in sorted(glob("shard_*.bin"))]
self.shard_windows = [len(mm) // seq_len for mm in self.shards]
self._starts       = [0, n₀, n₀+n₁, ...]      # cumulative window counts
self.n_windows     = sum(self.shard_windows)
```

`sorted(glob)` fixes the shard order lexically — shard_000.bin before
shard_001.bin — so the global window id space is stable across machines and
restarts as long as the shard set is fixed. `_starts` has one entry per shard
plus a leading 0, and window `w` belongs to shard
`bisect.bisect_right(_starts, w) − 1` with local index `w − _starts[shard]`:
O(log 160) comparisons per lookup, and `bisect_right` returning
`len(_starts)` for `w ≥ n_windows` is prevented from ever arising by the
modulo in `__getitem__`. `data/dataset.py:ShardWindows.__len__` returns
`n_windows`; the sampler and `DataLoader` both read length from it.

### 3.3 What `__getitem__` returns

`data/dataset.py:ShardWindows.__getitem__`:

```
w      = int(idx) % self.n_windows          # out-of-range indices cycle
shard  = bisect_right(_starts, w) - 1
local  = w - _starts[shard]
start  = local * seq_len
chunk  = np.array(self.shards[shard][start:start+seq_len], copy=True)
return torch.from_numpy(chunk).long()
```

Three deliberate details:

1. **`np.array(..., copy=True)`** — the memmap slice is a view into
   OS-managed pages; without the copy, the returned tensor aliases the mmap
   and a later page eviction or file change would corrupt in-flight batches.
   The copy is 16 KB per window — noise next to a 32×4096 transformer step.
2. **`.long()`** — `torch.nn.Embedding` requires int64 indices; the uint32
   numpy buffer is converted once at fetch. `(16, 4096)` int64 = 512 KB per
   micro-batch on the host, pinned if CUDA (`pin_memory=True`).
3. **`% self.n_windows`** — a hand-rolled out-of-range index cycles instead of
   raising; the sampler never emits such an index (it walks a permutation of
   `range(n_windows)`), so this is a user-facing guard, not load-bearing
   logic. `tests/test_data.py::test_shard_windows_shape_dtype_and_range` pins
   shape/dtype/range; `tests/test_data.py::test_partial_tail_window_dropped`
   pins the tail-drop; `tests/test_data.py::test_missing_shards_raises` pins
   the empty-directory error.

## 4. Why there is no +1 shift

An AR dataset returns `(x[i:i+seq], x[i+1:i+seq+1])` — inputs and targets
offset by one, because AR predicts token `i+1` from tokens `≤ i`:

```
L_AR = Σᵢ −log p(x_{i+1} | x_{≤i})       # target window ≠ input window
```

Diffusion training reconstructs **x0 at every position simultaneously**: the
corruption (`models/diffusion.py:q_sample`) and the target are the *same
window* — each position is independently kept-or-corrupted given per-canvas
`t`, and the loss is a sum of per-position cross-entropies against the clean
tokens:

```
L_diff = E_{t, x0, xt}[ Σᵢ −log p(x0_i | xt, t) ]    # target = input's clean version
window w:  tokens[start : start+seq_len]             # input = target = x0
```

So `data/dataset.py:ShardWindows.__getitem__` returns a flat
`seq_len`-token slice — no shift, no `+1` offset, no second tensor. The +1
would be wrong three ways:

1. **It wastes a token per window** — with the pack contract already paying a
   128-token tail per shard, shifting costs another 1/4,096 of every window's
   positions (≈ 1.95M tokens corpus-wide) for nothing.
2. **It implies a coupling the objective does not have.** Position i's *input*
   never predicts position i+1's target — every position is independently
   corrupted and independently reconstructed
   ([diffusion-core §3](diffusion-core.md)). A shifted target tensor reads as
   if `x0[i+1]` were supervised from `xt[i]`, which is not any term of the loss.
3. **It desynchronizes input and target corruption.** The trainer corrupts the
   window it receives and uses it as *both* `xt` source and CE target
   (`training/losses.py:chunked_x0_ce`). With a shift, the corrupted version
   of window w would be paired with the clean version of window w+1 — a
   different token stream entirely.

What the trainer does with the returned window: seed a per-micro-step
generator, sample per-canvas `t`, corrupt x0 → xt, forward, score against the
*uncorrupted* x0 — the dataset cannot know `t` or the draw, so a data-side
target has no analogue here. Pinned:
`tests/test_data.py::test_producer_consumer_shard_path_wiring`.

## 5. The resumable shuffler

`data/dataset.py:ShuffledRangeSampler` — the determinism story in one class:

```
__init__:  indices = np.random.default_rng(seed).permutation(n_windows)
           offset  = int(offset) % n_windows
__iter__:  yields indices[offset:]          # resume mid-permutation
__len__:   len(indices) - offset            # windows left in this pass
```

- **The permutation is a pure function of `(seed, n_windows)`** — seed 42
  yields the *same* order on every machine, every restart, in the pinned
  environment. No RNG state is consumed beyond the seed; two
  `ShuffledRangeSampler` instances with equal `(seed, n)` walk identical
  orders forever. Pinned: `tests/test_data.py::test_loader_seed42_repeatability`.
- **Resume = offset.** The trainer passes `offset_batches · batch_size`
  (`data/dataset.py:build_dataloader`): after a checkpoint at batch `n`, the
  loader restarts at window `n · batch_size` in the permutation *without
  regenerating any draws*. The offset is **batch-granular**, the right unit: a
  checkpoint's progress is measured in optimizer steps, each worth
  `grad_accum · micro_bs` windows, and integer arithmetic on batch counts is
  exact where a "fraction of an epoch" float would drift. Pinned:
  `tests/test_data.py::test_loader_resumable_offset`.
- **Wrap is a safety net, not a schedule.** `offset % n_windows`
  (`data/dataset.py:ShuffledRangeSampler.__init__`) keeps a pathological
  offset from indexing out of range; production never touches it — the
  61,000-step run consumes 1,952,000 of 1,953,120 windows, 1,120 short of a
  wrap.
- **No cross-process state**: the sampler holds `indices` and an integer;
  two ranks with the same seed walk identical orders (data-parallel
  sharding would be a later concern; this repo trains single-process).

### 5.1 Why not the obvious alternatives

| alternative | why it loses |
|---|---|
| `DataLoader(shuffle=True)` (RandomSampler) | draws a *fresh* permutation from the global torch RNG each epoch: order depends on how much RNG state was consumed before, so it is not a function of `(seed, n)`; resume needs the RNG state serialized; epoch boundaries replay differently |
| persistent stateful shuffler (buffer + position in checkpoint) | checkpoint must carry sampler state; two processes reconstructing from the same seed diverge if the state is missing; the fp64 resume test then compares against a moving target |
| re-shuffle per epoch with a different seed | after a resume that crosses an epoch boundary you cannot recompute the epoch-2 order from `(seed, step)` alone — you need the epoch count, which the checkpoint may not have |
| sorted/sequential windows (no shuffle) | correlation between consecutive windows (same document region) inflates gradient correlation; 4,096-token windows from the same shard arrive adjacent |
| shuffle once at pack time (pre-shuffled shards) | order is frozen into the bytes: re-shuffling means re-packing 8B tokens, and per-shard windowing no longer aligns with the shuffle |

The chosen design keeps *all* order information in `(seed, n_windows,
offset)` — three integers. Combined with the training side — per-micro-step
corruption generators seeded from the step count
(`training/pretrain.py:Pretrainer._step_rng`) — the *entire* data+corruption
pipeline is a pure function of `(seed, step)`, which is what makes
`tests/test_training.py::test_checkpoint_resume_determinism` a bit-equality
test rather than a statistical one.

## 6. Worked example: indexing a window at toy scale

Toy: 2 shards of 10 tokens, `seq_len = 4` → windows per shard:
`10 // 4 = 2`, 2 tokens of each shard dropped.

```
shard0:  [t0 t1 t2 t3 | t4 t5 t6 t7 | t8 t9]   shard1: [u0 ... u9]
windows:  shard0[0:4]=w0, shard0[4:8]=w1,  shard1[0:4]=w2, shard1[4:8]=w3
_starts  = [0, 2]          (cumulative window counts)
n_windows = 4
```

`__getitem__(3)`:

```
w = 3 % 4 = 3
shard = bisect_right([0, 2], 3) - 1 = 2 - 1 = 1     # window 3 lives in shard 1
local = 3 - _starts[1] = 3 - 2 = 1                  # second window of shard 1
start = 1 · 4 = 4  → tokens[4:8] of shard 1 = [u4 u5 u6 u7]
```

Production trace, same arithmetic, `__getitem__(1_000_000)` over the 160-shard
corpus (equal shard sizes → `_starts[k] = k · 12,207`):

```
w      = 1,000,000 % 1,953,120 = 1,000,000
shard  = bisect_right(_starts, 1,000,000) - 1 = 81   # 81·12,207 = 988,767 ≤ w
local  = 1,000,000 - 988,767 = 11,233
start  = 11,233 · 4,096 = 46,009,088
       → shard_081.bin[46,009,088 : 46,013,184]  (16 KB of a 200 MB file)
```

Take-aways:

- global window id → shard via `bisect` on cumulative starts, local id → byte
  offset via one multiply: O(log 160) comparisons, zero copies at open time
  (memmap), one `np.array(..., copy=True)` per fetch to detach from the mmap
  (`data/dataset.py:ShardWindows.__getitem__`). The whole index is two Python
  lists rebuilt in `__init__` — no sidecar metadata file to fall out of sync;
  the shard *names and sizes are* the metadata.
- At batch level, `data/dataset.py:build_dataloader` stacks 16 windows into
  `(16, 4096)` int64; `drop_last=True` drops nothing at production scale
  (1,953,120 = 61,035 × 32 exactly) but protects toy/test shapes where the
  division is ragged (`tests/test_data.py::test_loader_batch_shapes`).

## 7. What breaks if you change this

| change | immediate effect | the test / guard that catches it |
|---|---|---|
| add the AR +1 shift | windows misaligned by one vs the x0 target; loss measures a hybrid objective | `tests/test_data.py::test_producer_consumer_shard_path_wiring` |
| regenerate the permutation on resume | data order after resume ≠ order before; resume no longer bit-deterministic | `tests/test_data.py::test_loader_resumable_offset` |
| permute per epoch (`random.shuffle` per pass) | resume across an epoch boundary replays a different order | `tests/test_data.py::test_loader_seed42_repeatability` |
| window across shard boundaries | corrupts the no-cross-boundary pack contract; indexing needs cross-memmap reads | `data/dataset.py:ShardWindows` per-shard windowing |
| shrink shards below ~4,096-token multiples per shard | tail loss grows from 128 tok/shard toward a full window per shard | `tests/test_data.py::test_partial_tail_window_dropped` |
| switch shards to uint16 | GPT-2 ids fit, but the dtype diverges from the shared_data house format and every sibling consumer | `data/data_config.yaml` `sharding.dtype` |
| drop the quality filters / dedup | duplicated and degenerate text consumes the single-pass 8.0B budget | `tests/test_data.py::test_data_config_contract` |
| pick a different EOS/PAD split (separate PAD id) | PAD occupies vocab mass the uniform corruption wastes noise draws on | `data/prepare_data.py:main` constants |
| `seq_len` not a multiple of `canvas_len` | the block-causal mask asserts divisibility | `models/mask.py:build_block_causal_mask` |
| `drop_last=False` in `build_dataloader` | ragged final batch breaks the (B, S) shape contract | `data/dataset.py:build_dataloader` |
| pass `pin_memory=False` on CUDA | `non_blocking=True` H2D copies become synchronous | `data/dataset.py:build_dataloader` docstring |
| `offset` not taken modulo n_windows | negative/out-of-range start after wrap | `data/dataset.py:ShuffledRangeSampler` (offset % n_windows) |
| remove the `np.array(copy=True)` in `__getitem__` | batch tensors alias mmap pages; page eviction corrupts in-flight data | `data/dataset.py:ShardWindows.__getitem__` |

## 8. Glossary

| symbol | meaning | code |
|---|---|---|
| shard | 50M-token uint32 memmap (`shard_*.bin`, 200 MB) | `data/prepare_data.py:main` output |
| window | flat `seq_len`-token x0 slice, no shift | `data/dataset.py:ShardWindows` |
| `_starts` | cumulative window offsets across shards | `data/dataset.py:ShardWindows.__init__` |
| `n_windows` | total windows = Σ `len(mm) // seq_len` | `data/dataset.py:ShardWindows.__len__` |
| `offset` | resume point: windows already consumed | `data/dataset.py:ShuffledRangeSampler` |
| `seed=42` | the fixed permutation seed | `data/dataset.py:build_dataloader` |
| `LLM_DATA_ROOT` | env var the pack subprocess reads | `data/prepare_data.py:main` |
| pack contract | `cross_document_boundary_ok: false` — documents never split across shards | `data/data_config.yaml` |
| EOS=PAD | one id (50,256) ends documents; no padding exists in the flat-window path | `data/data_config.yaml` `pipeline.tokenizer` |
| tail | sub-`seq_len` remainder of a shard, dropped at windowing (128 tok/shard) | `data/dataset.py:ShardWindows.__init__` |

## 9. Interview Q&A

**Q: Why no +1 shift in the dataset?**
A: AR predicts token i+1 from prefix ≤ i; uniform-state diffusion
reconstructs every position of x0 independently — input and target are the
same window (`data/dataset.py:ShardWindows`). A +1 shift would waste a token
and imply a coupling the objective doesn't have
([diffusion-core §2](diffusion-core.md)); corruption belongs to the training
step, not the data, so a data-side target has no analogue here.

**Q: How does resume-with-identical-data-order work?**
A: The permutation is `default_rng(seed).permutation(n_windows)` — fixed by
(seed, n_windows). Resume passes `offset_batches · batch_size`, and the
sampler yields `indices[offset:]` — no draws regenerated
(`data/dataset.py:ShuffledRangeSampler`). Combined with per-step corruption
generators, resume is bit-exact
(`tests/test_training.py::test_checkpoint_resume_determinism`).

**Q: What does a shard contain and how big is the corpus?**
A: 50M-token uint32 memmaps (200 MB each) packed by the shared pipeline;
the 8.0B-token config → 160 shards → 1,953,120 windows of 4,096, 128 tokens
per shard dropped as tail (0.000%)
(`data/dataset.py:ShardWindows.__init__`).

**Q: Why do windows cross document boundaries but not shard boundaries?**
A: The pack contract is per-shard (`cross_document_boundary_ok: false`
governs documents *inside* a shard's text stream, not window boundaries);
windowing is per-shard by construction so a window never spans two memmaps
(`data/dataset.py:ShardWindows.__getitem__`).

**Q: What breaks first if the shuffler is made stateful?**
A: Resume determinism: a shuffle regenerated from a fresh RNG would replay a
different order after restart, and the fp64 resume-determinism test — which
compares full training trajectories — fails
(`tests/test_data.py::test_loader_resumable_offset`,
`tests/test_training.py::test_checkpoint_resume_determinism`).

**Q: Why `pin_memory=True` on CUDA?**
A: The trainer's `non_blocking=True` H2D copy is only actually asynchronous
with pinned host memory; without the flag it silently degrades to a
synchronous copy (`data/dataset.py:build_dataloader` docstring).

**Q: Why GPT-2 BPE for a diffusion model?**
A: The corruption state space *is* the vocabulary — uniform-state diffusion
draws noise uniformly over all V ids (`models/diffusion.py:q_sample`), so V
sets the noise entropy and the embedding/head width. GPT-2 BPE (50,257) is
the shared_data house tokenizer shared by the sibling Lites, which keeps the
from-scratch comparison honest: identical ids, mixture, and budget across
every control. A byte-level tokenizer would inflate sequence length ~4× and
blow the Chinchilla budget at the same sequence count.

**Q: Why 50M-token shards instead of fewer big files or many small ones?**
A: 160 × 200 MB balances mmap/page-cache behavior, per-shard pipeline
restartability, and handle count (`data/dataset.py:ShardWindows.__init__`
opens every shard). Smaller shards multiply the 4,096-token tail loss and the
memmap handle count; one 8B blob kills resumable packing.

**Q: How much data does the fixed-window packing throw away, and why is that
acceptable?**
A: The sub-4,096-token tail of each shard: 128 × 160 = 20,480 tokens of 8.0B,
0.00026%. The alternative — windows straddling shards — would complicate every
lookup for a loss orders of magnitude smaller than the cross-memmap indexing
it would add (`data/dataset.py:ShardWindows.__getitem__`).

**Q: Where exactly do the training targets come from?**
A: The dataset returns the clean window once; the trainer corrupts it into xt
with a per-micro-step generator and scores the model's x0 posterior against
the *uncorrupted* window with the chunked cross-entropy
(`training/losses.py:chunked_x0_ce`). Targets are x0 — the plan text's
`chunked_x0_ce(hidden, E, xt)` signature is a known typo recorded in the SDD
ledger.