# KV Latent Cache

A memory optimisation for **Multi-head Latent Attention (MLA)** models that replaces the standard KV cache with a **latent cache** — storing the compressed latent vector `kv_a_norm` and the positional key `k_pe_roped` instead of the full key and value tensors.

Tested on **DeepSeek-V2-Lite** (deepseek-ai/DeepSeek-V2-Lite, 15.7 B parameters).

---

## Table of Contents

1. [Background — Why This Works](#background--why-this-works)
2. [Project Structure](#project-structure)
3. [Quickstart](#quickstart)
4. [Verification Tests](#verification-tests)
5. [CLI Reference](#cli-reference)
6. [Module Reference](#module-reference)
7. [Memory Estimates](#memory-estimates)
8. [Known Limitations](#known-limitations)

---

## Background — Why This Works

### MLA Attention Recap

DeepSeek-V2-Lite uses Multi-head Latent Attention (MLA). The key and value projections share a **two-stage compression**:

```
hidden_states
  → kv_a_proj_with_mqa  →  split  →  kv_a_out [512 dims]  +  k_pe_raw [64 dims]
                                            ↓
                                    kv_a_layernorm
                                            ↓
                                       kv_a_norm                   ← THE SHARED LATENT
                                            ↓
                                       kv_b_proj
                                            ↓
                               k_nope [16 heads × 128]  +  v [16 heads × 128]
```

Both `k_nope` and `v` are derived from **the same 512-dimensional latent** `kv_a_norm`. This means:

- Storing `kv_a_norm` (512 dims) is sufficient to reconstruct both `k_nope` and `v` exactly at attention time
- Only `k_pe` (64 dims) needs to be stored separately — it carries position information via RoPE

### What Gets Cached

| Approach | Stored per token | Dims |
|---|---|---|
| Standard `DynamicCache` | `k_full` [16 × 192] + `v` [16 × 128] | **5120** |
| `KVLatentCache` (this work) | `kv_a_norm` [512] + `k_pe_roped` [64] | **576** |

```
saving = (5120 − 576) / 5120 ≈ 88.8%
```

### Exact Reconstruction

At attention time, both `k_nope` and `v` are reconstructed by running the accumulated `kv_a_norm` through `kv_b_proj`:

```python
kv = kv_b_proj(kv_a_norm_acc)        # [B, S, 16 × 256]
kv = kv.view(B, S, 16, 256)
k_nope, v = split(kv, [128, 128])    # exact — same weights, same input
```

This is **lossless**: `kv_b_proj` is a deterministic linear layer, so the reconstruction is bit-for-bit identical to what the original forward would have computed.

### Why Not the N-Matrix Approach?

An earlier version tried to reconstruct `k_nope` from `v` via a static matrix `N = pinv(W_V.T) @ W_K.T`. This is **fundamentally approximate**: `v = kv_a_norm @ W_V.T` maps 512D → 128D, discarding 384 dimensions that `k_nope` also depends on. The latent cache avoids this loss entirely.

---

## Project Structure

```
.
├── README.md
├── kv_patch.py                   # Patches all decoder layers to use KVLatentCache
├── requirements.txt
│
├── ops/
│   ├── kv_latent_cache.py        # DynamicCache subclass storing (kv_a_norm, k_pe_roped)
│   ├── kv_nopeless_cache.py      # Kept for reference — superseded by kv_latent_cache.py
│   └── kv_relation_module.py     # Kept for reference — not used in current approach
│
├── src/
│   └── compute_n_matrix.py       # Kept for reference — not used in current approach
│
├── scripts/
│   ├── setup_runtime.sh          # Create venv, install all dependencies
│   └── download_model.sh         # Download DeepSeek-V2-Lite from HuggingFace
│
└── tests/
    └── verify_kv_relation.py     # Full 5-test verification suite
```

---

## Quickstart

### Step 1 — Set up the runtime environment

```bash
bash scripts/setup_runtime.sh
```

Creates a virtual environment at `.venv` and installs all dependencies from `requirements.txt`. Only needs to be run once.

### Step 2 — Activate the environment

```bash
source .venv/bin/activate
```

### Step 3 — Download the model

```bash
bash scripts/download_model.sh
```

Downloads **DeepSeek-V2-Lite** (~31 GB, BF16) to `models/DeepSeek-V2-Lite/`.

### Step 4 — Run the verification suite

```bash
python tests/verify_kv_relation.py \
    --model_path ./models/DeepSeek-V2-Lite \
    --skip-throughput
```

All 4 correctness tests should pass with exact zero logit difference. Typical output:

```
════════════════════════════════════════════════════════════════
  KV Latent Cache — Verification Suite
  Target: DeepSeek-V2-Lite
════════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────
  Test 1 — Latent round-trip  (pre-patch)
──────────────────────────────────────────────────────────────
  [ PASS ]  kv_b_proj(kv_a_norm) == kv_b_out  (max_diff=0.00e+00)
  [ PASS ]  kv_b_out last dim == num_heads×(qk_nope+v_dim) = 4096

──────────────────────────────────────────────────────────────
  Test 2 — Output equivalence  (patched vs unpatched)
──────────────────────────────────────────────────────────────
  [ PASS ]  max |Δlogit| = 0.0000   (exact match)
  [ PASS ]  top-1 token unchanged  ← primary correctness gate

──────────────────────────────────────────────────────────────
  Test 3 — Cache shape  (latent + k_pe must be stored)
──────────────────────────────────────────────────────────────
  [ PASS ]  Actual key   dim : 512  (kv_a_norm ✓)
  [ PASS ]  Actual value dim : 64   (k_pe_roped ✓)

──────────────────────────────────────────────────────────────
  Test 4 — Memory savings across context lengths
──────────────────────────────────────────────────────────────
  Context       Standard cache    Latent cache    Saving
  ─────────────────────────────────────────────────────────
  2048 tok           565.95 MB        63.67 MB     88.8%
  4096 tok          1132.19 MB       127.37 MB     88.8%
  ─────────────────────────────────────────────────────────
  [ PASS ]  Saving is constant across lengths
  [ PASS ]  Observed saving ≈ theoretical 88.8%

════════════════════════════════════════════════════════════════
  ALL TESTS PASSED
════════════════════════════════════════════════════════════════
```

---

## Verification Tests

### Test 1 — Latent Round-Trip (pre-patch)

**Goal:** Confirm that `kv_b_proj(kv_a_norm)` reproduces `k_nope` and `v` exactly — validating that the latent extraction in `kv_patch.py` captures the right tensor.

**How it works:** Hooks both the input and output of `kv_b_proj` during one forward pass, then verifies `kv_b_proj(hooked_input) == hooked_output` to machine precision.

**Pass gate:** `max_diff == 0.0` (exact).

---

### Test 2 — Output Equivalence (post-patch)

**Goal:** Confirm the patched model produces identical outputs to the unpatched model at every decode step.

**How it works:** Collects reference logits with `DynamicCache` (unpatched), then runs the same prompt through the patched model with `KVLatentCache`. Compares logits at every step.

**Pass gate:** Top-1 token must match at every position. Because the reconstruction is exact and we replicate the original forward's manual matmul + FP32 softmax, `max |Δlogit|` is **0.0000** — not just close, but bit-identical.

**Note:** The patched forward uses manual `matmul + F.softmax(fp32)` to match the original DeepSeek forward exactly. Using `F.scaled_dot_product_attention` produces ~0.4 logit differences due to its fused kernel using different internal precision.

---

### Test 3 — Cache Shape

**Goal:** Prove the cache stores `kv_a_norm` (512 dims) and `k_pe_roped` (64 dims) — not the full key and value.

**Pass gate:** `cache.latent_dim(0) == 512` and `cache.kpe_dim(0) == 64`.

---

### Test 4 — Memory Savings Across Context Lengths

**Goal:** Measure the actual memory reduction and confirm it is consistent.

**Key insight:** The saving is a pure function of model dimensions:
```
saving = (num_heads × (qk_nope + qk_rope) + num_heads × v_dim − kv_lora_rank − qk_rope)
       / (num_heads × (qk_nope + qk_rope) + num_heads × v_dim)
       = (5120 − 576) / 5120 ≈ 88.8%
```

**Pass gate:** Drift between min and max saving < 1 pp, observed mean within 5 pp of theoretical.

---

### Test 5 — Throughput (informational only)

**Goal:** Report decode tokens/sec for `DynamicCache` vs `KVLatentCache`.

**Note:** The patched model runs `kv_b_proj` over the full accumulated sequence at each decode step. This is the main compute overhead of the latent cache approach. The memory saving pays off at larger batch sizes where fitting more sequences per step outweighs the per-step overhead.

Requires ~62 GB VRAM (two model copies). Use `--skip-throughput` if unavailable.

---

## CLI Reference — verify_kv_relation.py

```
python tests/verify_kv_relation.py [OPTIONS]

Required:
  --model_path PATH          Path to HuggingFace model directory

Optional:
  --prompt TEXT              Prompt used for all tests
  --layer_idx INT            Layer index to hook for Test 1 (default: 0)
  --context_lengths INT...   Context lengths for Test 4 table
                             (default: 2048 4096 8192 16384 32768)
  --gen_tokens INT           New tokens to generate in Tests 2 and 5 (default: 64)
  --throughput_tokens INT    Tokens per model in Test 5 benchmark (default: 256)
  --skip-throughput          Skip Test 5; avoids reloading the model
  --output_abs_tol FLOAT     |Δlogit| tolerance for Test 2 (default: 0.6)
```

---

## Module Reference

### `ops/kv_latent_cache.py`

Drop-in replacement for HuggingFace `DynamicCache` storing `(kv_a_norm, k_pe_roped)` per layer.

```python
from ops.kv_latent_cache import KVLatentCache

cache = KVLatentCache()
output = model.generate(**inputs, past_key_values=cache, use_cache=True)

print(cache.report())
# KVLatentCache | 27 layers populated
#   kv_a_norm : [1, 1, 512, 512]  dtype=bfloat16
#   k_pe      : [1, 1, 512,  64]  dtype=bfloat16
#   latent mem:   14.15 MB
#   k_pe mem  :    1.75 MB
#   total mem :   15.89 MB

cache.latent_dim(layer_idx=0)   # returns kv_lora_rank (512)
cache.kpe_dim(layer_idx=0)      # returns qk_rope_head_dim (64)
cache.cache_size_bytes()        # {'latent_bytes': ..., 'kpe_bytes': ..., 'total_bytes': ...}
```

Overrides `update()` and `get_seq_length()` to manage storage explicitly, bypassing transformers 4.47+ internal cache refactor.

---

### `kv_patch.py`

Patches all decoder layers in-place to use `KVLatentCache`.

```python
from kv_patch import patch_kv_model
from ops.kv_latent_cache import KVLatentCache

model = patch_kv_model(model)

output = model.generate(
    **inputs,
    past_key_values=KVLatentCache(),
    use_cache=True,
)
```

For each layer, `patch_kv_model` replaces `attn.forward` with a closure that:

1. Extracts `kv_a_norm` (stops before `kv_b_proj`) and `k_pe_raw`
2. Applies RoPE to `q_pe` and `k_pe` using the model's own `apply_rotary_pos_emb`
3. Stores `(kv_a_norm, k_pe_roped)` in the cache
4. Reconstructs `k_nope` and `v` exactly via `kv_b_proj(kv_a_norm_acc)` at attention time
5. Computes attention using manual `matmul + F.softmax(fp32)` to match the original numerics

---

## Memory Estimates

BF16, batch size 1. DeepSeek-V2-Lite: 27 layers, `kv_lora_rank=512`, `qk_rope=64`.

| Context | Standard cache | Latent cache | Saving |
|---|---|---|---|
| 2 048 tokens | 565.95 MB | 63.67 MB | **88.8%** |
| 4 096 tokens | 1132.19 MB | 127.37 MB | **88.8%** |
| 8 192 tokens | ~2264 MB | ~255 MB | **88.8%** |
| 32 768 tokens | ~9056 MB | ~1019 MB | **88.8%** |

The saving is constant at every context length — it is a pure function of model dimensions, not sequence length.

---

## Known Limitations

**`kv_b_proj` over full context at each decode step.** At decode step N, `kv_b_proj` is applied to all N accumulated tokens to reconstruct `k_nope` and `v`. This is O(N) compute per step vs O(1) for the standard cache. The memory saving enables larger batch sizes, but per-step latency may increase at long contexts.

**No batch-size scaling test.** Test 4 measures cache size at batch=1. The 88.8% saving scales linearly with batch size, but this is not automatically verified by the test suite.

**Throughput vs memory trade-off.** The latent cache trades compute (rerunning `kv_b_proj`) for memory (88.8% KV cache reduction). At large batch sizes or memory-constrained deployments, the memory saving dominates. At small batch sizes with short contexts, the compute overhead may outweigh the benefit.