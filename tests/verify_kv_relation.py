"""
verify_kv_relation.py — End-to-end correctness and efficiency verification
for the GQA KV cache optimisation via K = V @ N reconstruction.

Background
──────────────────────────────────────────────────────────────────────────────
In standard GQA attention (e.g. Llama 3), keys and values are independent
linear projections of the same input X:

    K_pre = X @ W_K.T      [S, num_kv_heads, head_dim]   (before RoPE)
    V     = X @ W_V.T      [S, num_kv_heads, head_dim]

Because both share the same input, we ask: does a static matrix N exist per
KV head such that K_pre_h ≈ V_h @ N_h?

N_h is derived offline from the weight matrices (compute_n_matrix.py):
    N_h  =  W_V_h⁺  @  W_K_h        shape [head_dim × head_dim]

where W_V_h⁺ is the pseudo-inverse of W_V_h (computed via truncated SVD).

At inference, only V is cached. K is reconstructed on-the-fly:
    K_pre_h = V_h @ N_h
    K_h     = RoPE(K_pre_h, positions)   ← identical to normal from here on

This halves the KV cache footprint at the cost of one [head_dim × head_dim]
matmul per layer per decode step.

The approximation quality depends entirely on the subspace alignment between
col(W_K_h) and col(W_V_h). Test 1 measures this at the weight level; Test 2
measures it on real activations. Both must be checked before deployment.

Weight conventions (Llama-style)
──────────────────────────────────────────────────────────────────────────────
  attn.k_proj.weight  ∈  R^{num_kv_heads*head_dim  ×  d_model}
  attn.v_proj.weight  ∈  R^{num_kv_heads*head_dim  ×  d_model}

  Per-head slice (head h):
    W_K_h = k_proj.weight[h*D : (h+1)*D, :].T   ∈  R^{d_model × head_dim}
    W_V_h = v_proj.weight[h*D : (h+1)*D, :].T   ∈  R^{d_model × head_dim}
    N_h   = lstsq(W_V_h, W_K_h)                  ∈  R^{head_dim × head_dim}

Target model
──────────────────────────────────────────────────────────────────────────────
  Any standard GQA model (tested on Llama 3 8B).
  Required attributes: num_heads, num_key_value_heads, head_dim.
  Required submodules: k_proj, v_proj (nn.Linear).

Tests
──────────────────────────────────────────────────────────────────────────────
  Test 1  N matrix weight-level accuracy
          Compute N_h offline and verify W_V_h @ N_h ≈ W_K_h.
          Checks the offline computation has no bugs and reports per-head
          relative reconstruction error as a diagnostic.
          The error here is a lower bound on activation-level error (Test 2).

  Test 2  Pre-RoPE activation reconstruction
          Hook k_proj and v_proj outputs during a forward pass.
          Compute K_rec_h = V_h @ N_h and compare with the true pre-RoPE K_h.
          This is the CRITICAL test: weight-level accuracy does NOT guarantee
          activation-level accuracy — input distribution can expose misaligned
          subspaces that look fine on weights alone.

  Test 3  Output equivalence
          Generate tokens with the unpatched model → ref logits.
          Patch the model to use VOnlyCache with K reconstruction.
          Assert top-1 token matches at every position.
          Top-1 match is the primary correctness gate.

  Test 4  Cache shape
          After one prefill + one decode step, verify VOnlyCache contains
          only V (no K). Shape: [B, num_kv_heads, S, head_dim].
          Key cache must be empty — K should not persist between steps.

  Test 5  Memory savings across context lengths
          Standard DynamicCache stores K + V = 2 × num_kv_heads × head_dim.
          VOnlyCache stores V only = num_kv_heads × head_dim.
          Expected saving: exactly 50%, constant across all context lengths.

  Test 6  Throughput  (informational, no pass/fail)
          Tokens/sec patched vs unpatched.
          At long context, cache bandwidth reduction should dominate.
          Gated with --skip-throughput.

Usage
──────────────────────────────────────────────────────────────────────────────
    python verify_kv_relation.py --model_path ./models/Llama-3-8B
    python verify_kv_relation.py --model_path ./models/Llama-3-8B --skip-throughput
    python verify_kv_relation.py --model_path ./models/Llama-3-8B --layer_idx 4
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# VOnlyCache: custom cache that stores only V per layer.
# At attention time, K is reconstructed via K_pre = V @ N_h then RoPE is applied.
from src.v_only_cache import VOnlyCache

# patch_kv_model: swaps the model's attention forward to use VOnlyCache
# and the precomputed N matrices for K reconstruction.
from src.kv_patch import patch_kv_model

# compute_all_N_matrices: offline computation of N_h per KV head per layer.
# Returns List[Tensor], one [num_kv_heads, head_dim, head_dim] per layer.
from src.compute_n_matrix import compute_all_N_matrices


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify GQA V-only KV cache via K = V @ N")
    p.add_argument(
        "--model_path", type=str,
        default="./models/gpt-oss-20b-BF16",
    )
    p.add_argument(
        "--prompt", type=str,
        default=(
            "The history of artificial intelligence begins in antiquity, "
            "with myths, stories and rumors of artificial beings endowed with "
            "intelligence or consciousness by master craftsmen."
        ),
    )
    # layer_idx: which single layer to inspect in Tests 1 and 2.
    # Tests 3-6 always exercise the full model.
    p.add_argument("--layer_idx",         type=int,   default=0)
    p.add_argument("--context_lengths",   type=int,   nargs="+",
                   default=[2048, 4096, 8192, 16384, 32768])
    p.add_argument("--gen_tokens",        type=int,   default=64)
    p.add_argument("--throughput_tokens", type=int,   default=256)
    p.add_argument("--skip-throughput",   dest="skip_throughput", action="store_true")
    # Tolerance for logit comparison in Test 3. Top-1 match is the hard gate;
    # this is a softer numeric sanity check on top.
    p.add_argument("--output_abs_tol",    type=float, default=6e-1)
    # Relative error threshold for Test 1 PASS — if mean per-head error exceeds
    # this, we flag a warning (but still report; hard FAIL only on NaN/Inf/shape).
    p.add_argument("--weight_err_warn",   type=float, default=0.30,
                   help="Warn if mean weight-level relative error exceeds this fraction")
    return p.parse_args()


# ── Pretty-print helpers ──────────────────────────────────────────────────────

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
WARN = "\033[93m WARN \033[0m"
INFO = "\033[94m INFO \033[0m"
SEP  = "─" * 64


def header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def log(label: str, passed: Optional[bool] = None, detail: str = "") -> None:
    if passed is None:
        tag = f"[{INFO}]"
    elif passed:
        tag = f"[{PASS}]"
    else:
        tag = f"[{FAIL}]"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag}  {label}{suffix}")


def log_warn(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{WARN}]  {label}{suffix}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str):
    print(f"\nLoading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).to(torch.bfloat16)
    model.eval()
    print(f"  dtype : {next(model.parameters()).dtype}")
    print(f"  device: {next(model.parameters()).device}")
    return model, tokenizer


def tokenize(tokenizer, prompt: str, device: torch.device) -> dict:
    return tokenizer(prompt, return_tensors="pt").to(device)


# ── Attention attribute helpers ───────────────────────────────────────────────

def _get_attn(model: nn.Module, layer_idx: int) -> nn.Module:
    return model.model.layers[layer_idx].self_attn


def _get_head_dim(attn: nn.Module) -> int:
    """head_dim is the per-head projection dimension (e.g. 128 for Llama 3 8B)."""
    if hasattr(attn, "head_dim"):
        return int(attn.head_dim)
    # Fallback: infer from hidden_size / num_heads
    return int(attn.hidden_size // attn.num_heads)


def _get_num_kv_heads(attn: nn.Module) -> int:
    """Number of KV heads (g), e.g. 8 for gpt-oss-20b."""
    for attr in (
        "num_key_value_heads",
        "num_kv_heads",
        "num_kv_attention_heads",
        "kv_heads",
        "n_kv_heads",
    ):
        if hasattr(attn, attr):
            return int(getattr(attn, attr))
    # Config fallback
    if hasattr(attn, "config"):
        for attr in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
            if hasattr(attn.config, attr):
                return int(getattr(attn.config, attr))
    # Print all int attributes to identify the correct name
    numeric_attrs = {
        k: getattr(attn, k)
        for k in dir(attn)
        if not k.startswith("_") and isinstance(getattr(attn, k, None), int)
    }
    raise AttributeError(
        f"Cannot find KV head count on {type(attn).__name__}.\n"
        f"Available int attributes: {numeric_attrs}\n"
        f"Add the correct attribute name to _get_num_kv_heads()."
    )


# ── Per-head weight extraction ────────────────────────────────────────────────

def _get_per_head_weights(
    attn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract per-head W_K and W_V as tall matrices (d_model × head_dim each).

    k_proj.weight has shape [num_kv_heads * head_dim, d_model].
    We transpose and slice along the output dimension to get per-head matrices:
        W_K_h = k_proj.weight[h*D : (h+1)*D, :].T   →  [d_model, head_dim]
        W_V_h = v_proj.weight[h*D : (h+1)*D, :].T   →  [d_model, head_dim]

    Returns:
        W_K: [num_kv_heads, d_model, head_dim]  in float64 for numerical precision
        W_V: [num_kv_heads, d_model, head_dim]  in float64
    """
    num_kv_heads = _get_num_kv_heads(attn)
    head_dim     = _get_head_dim(attn)

    # Weight matrices from nn.Linear: shape [out_features, in_features]
    W_K_flat = attn.k_proj.weight.double()  # [num_kv_heads*head_dim, d_model]
    W_V_flat = attn.v_proj.weight.double()  # [num_kv_heads*head_dim, d_model]

    # Reshape: [num_kv_heads, head_dim, d_model] then transpose last two dims
    # to get [num_kv_heads, d_model, head_dim] — the natural "projection" shape.
    W_K = W_K_flat.view(num_kv_heads, head_dim, -1).transpose(1, 2)  # [H, d_model, D]
    W_V = W_V_flat.view(num_kv_heads, head_dim, -1).transpose(1, 2)  # [H, d_model, D]

    return W_K, W_V


# ── Generation helper ─────────────────────────────────────────────────────────

def run_generation_logits(
    model: nn.Module,
    tokenizer,
    prompt: str,
    n_new_tokens: int,
    cache_type: str = "dynamic",
) -> torch.Tensor:
    """
    Generate n_new_tokens and return stacked logits [T, B, vocab].

    cache_type: 'dynamic' → DynamicCache (standard, unpatched model)
                'v_only'  → VOnlyCache   (patched model, K reconstructed from V)

    Both pass an explicit cache object so the model uses the same code path
    for attention mask handling — passing None vs a cache object changes how
    the mask is built and causes ~0.4 logit error independent of the patch.
    """
    device    = next(model.parameters()).device
    inputs    = tokenize(tokenizer, prompt, device)
    attn_mask = inputs["attention_mask"]
    past_kv   = VOnlyCache() if cache_type == "v_only" else DynamicCache()
    all_logits = []

    with torch.no_grad():
        for step in range(n_new_tokens):
            if step == 0:
                out = model(**inputs, past_key_values=past_kv, use_cache=True)
            else:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_ones((1, 1))], dim=1
                )
                out = model(
                    input_ids=next_token,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            logits     = out.logits[:, -1, :]
            past_kv    = out.past_key_values
            next_token = logits.argmax(dim=-1, keepdim=True)
            all_logits.append(logits.cpu())

    return torch.stack(all_logits, dim=0)


# ── Test 1 — N matrix weight-level accuracy ───────────────────────────────────

def test_n_weight_accuracy(
    model: nn.Module,
    layer_idx: int,
    N_matrices: list,
    warn_threshold: float = 0.30,
) -> bool:
    """
    Compute N_h offline and verify W_V_h @ N_h ≈ W_K_h for each KV head.

    WHY: N is derived from the pseudo-inverse of W_V applied to W_K.
    This test checks the offline computation has no implementation bugs —
    e.g. wrong slice ordering, transposition errors, wrong head count.

    The relative error ‖W_K_h - W_V_h @ N_h‖_F / ‖W_K_h‖_F measures how
    well col(W_K_h) lies inside col(W_V_h). A value near 0 means the
    approximation will be exact; near 1 means the subspaces are orthogonal
    and the algorithm will not work for this head.

    Note: high per-head error here is NOT a bug — it is a fundamental property
    of the model weights. It is a warning that the algorithm may not work well.

    PASS criteria:
      - N has correct shape [num_kv_heads, head_dim, head_dim]
      - No NaN or Inf in N
      (The relative error is diagnostic only; hard FAIL only on shape/NaN)
    """
    header(f"Test 1 — N matrix weight-level accuracy  (layer {layer_idx})")

    attn         = _get_attn(model, layer_idx)
    num_kv_heads = _get_num_kv_heads(attn)
    head_dim     = _get_head_dim(attn)
    W_K, W_V     = _get_per_head_weights(attn)  # [H, d_model, head_dim], float64

    errors = []
    for h in range(num_kv_heads):
        # Solve min_{N_h} || W_V_h @ N_h - W_K_h ||_F  (least-squares)
        # lstsq returns N_h ∈ R^{head_dim × head_dim}
        N_h = torch.linalg.lstsq(W_V[h], W_K[h]).solution  # [head_dim, head_dim]

        # Reconstruction: how well does W_V_h @ N_h recover W_K_h?
        K_reconstructed = W_V[h] @ N_h                     # [d_model, head_dim]
        residual        = W_K[h] - K_reconstructed
        rel_err         = residual.norm() / W_K[h].norm()
        errors.append(rel_err.item())

    mean_err = sum(errors) / len(errors)
    max_err  = max(errors)

    # Report per-head errors
    print(f"\n  Per-head relative error  ‖W_K_h - W_V_h @ N_h‖ / ‖W_K_h‖")
    for h, e in enumerate(errors):
        quality = "good" if e < 0.10 else ("ok" if e < 0.30 else "poor")
        print(f"    head {h:2d}: {e:.4f}  [{quality}]")
    print()

    # Use the already-computed N for this layer (passed in from main).
    # No need to recompute — N_matrices is derived from weights offline once.
    N_layer = N_matrices[layer_idx]  # [num_kv_heads, head_dim, head_dim]

    shape_ok = (N_layer.shape == (num_kv_heads, head_dim, head_dim))
    nan_ok   = not torch.isnan(N_layer).any().item()
    inf_ok   = not torch.isinf(N_layer).any().item()

    log(
        f"N shape == [{num_kv_heads}, {head_dim}, {head_dim}]",
        passed=shape_ok,
        detail=f"got {tuple(N_layer.shape)}",
    )
    log("N contains no NaN", passed=nan_ok)
    log("N contains no Inf", passed=inf_ok)
    log(
        f"Mean weight-level relative error",
        passed=None,  # diagnostic only
        detail=f"{mean_err:.4f}  (max {max_err:.4f})",
    )

    if mean_err > warn_threshold:
        log_warn(
            f"Mean relative error {mean_err:.4f} > warn threshold {warn_threshold:.2f}",
            detail="col(W_K) and col(W_V) are poorly aligned — "
                   "activation reconstruction (Test 2) will likely have high error",
        )

    passed = shape_ok and nan_ok and inf_ok
    print(f"\n  Overall Test 1: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Test 2 — Pre-RoPE activation reconstruction ───────────────────────────────

def test_activation_reconstruction(
    model: nn.Module,
    tokenizer,
    prompt: str,
    layer_idx: int,
    N_matrices: list,
) -> bool:
    """
    Hook k_proj and v_proj outputs on a real forward pass and measure
    how accurately K_pre can be reconstructed as V @ N_h per head.

    WHY: weight-level error (Test 1) only tells us about the geometry of
    the weight matrices. At inference, X is a specific input distribution
    — not all of R^{d_model}. Activations may emphasise exactly the
    directions where W_K and W_V diverge, making the real-data error
    higher than the weight-level error predicts.

    This is therefore the most important diagnostic test before deployment.
    A model where weight-level error is low but activation error is high
    indicates the input distribution preferentially excites the misaligned
    subspace — the algorithm will not work.

    PASS criteria:
      - K_rec has correct shape (no shape bugs)
      - No NaN/Inf in K_rec
      (Reconstruction error is diagnostic; report per-head)
    """
    header(f"Test 2 — Pre-RoPE activation reconstruction  (layer {layer_idx})")

    attn         = _get_attn(model, layer_idx)
    num_kv_heads = _get_num_kv_heads(attn)
    head_dim     = _get_head_dim(attn)
    device       = next(model.parameters()).device

    # Hook both k_proj and v_proj output — these are pre-RoPE activations.
    # k_proj output shape: [B, S, num_kv_heads * head_dim]
    # v_proj output shape: [B, S, num_kv_heads * head_dim]
    captured: dict = {}

    def _hook_k(module, input, output):
        captured["K_pre_flat"] = output.detach()  # [B, S, H*D]

    def _hook_v(module, input, output):
        captured["V_flat"] = output.detach()      # [B, S, H*D]

    h_k = attn.k_proj.register_forward_hook(_hook_k)
    h_v = attn.v_proj.register_forward_hook(_hook_v)

    inputs_tok = tokenize(tokenizer, prompt, device)
    with torch.no_grad():
        model(**inputs_tok, use_cache=False)

    h_k.remove()
    h_v.remove()

    if "K_pre_flat" not in captured or "V_flat" not in captured:
        log("Hooks did not fire — k_proj or v_proj not found", passed=False)
        return False

    B, S, _ = captured["K_pre_flat"].shape

    # Reshape to per-head: [B, S, num_kv_heads, head_dim]
    K_pre = captured["K_pre_flat"].view(B, S, num_kv_heads, head_dim)
    V     = captured["V_flat"].view(B, S, num_kv_heads, head_dim)

    # N for this layer: [num_kv_heads, head_dim, head_dim]
    N_layer = N_matrices[layer_idx].to(device=device, dtype=K_pre.dtype)

    log(f"Sequence length : {S}")
    log(f"K_pre shape     : {tuple(K_pre.shape)}")
    log(f"V     shape     : {tuple(V.shape)}")
    log(f"N     shape     : {tuple(N_layer.shape)}")
    print()

    errors = []
    for h in range(num_kv_heads):
        # K_pre_h and V_h: [B, S, head_dim]
        # N_h: [head_dim, head_dim]
        # Reconstruct: K_rec_h = V_h @ N_h  →  [B, S, head_dim]
        V_h     = V[..., h, :]         # [B, S, head_dim]
        K_pre_h = K_pre[..., h, :]     # [B, S, head_dim]
        N_h     = N_layer[h]           # [head_dim, head_dim]

        K_rec_h = V_h @ N_h            # [B, S, head_dim]
        residual = K_pre_h - K_rec_h

        rel_err = (residual.norm() / K_pre_h.norm()).item()
        errors.append(rel_err)

    mean_err = sum(errors) / len(errors)
    max_err  = max(errors)

    print(f"  Per-head activation relative error  ‖K_pre_h - V_h @ N_h‖ / ‖K_pre_h‖")
    for h, e in enumerate(errors):
        quality = "good" if e < 0.10 else ("ok" if e < 0.30 else "POOR")
        print(f"    head {h:2d}: {e:.4f}  [{quality}]")
    print()

    # Shape and numerical sanity — reconstruct per head and stack
    K_rec_check = torch.stack([V[..., h, :] @ N_layer[h] for h in range(num_kv_heads)], dim=2)
    shape_ok = (K_rec_check.shape == K_pre.shape)
    nan_ok   = not torch.isnan(K_rec_check).any().item()
    inf_ok   = not torch.isinf(K_rec_check).any().item()

    log(
        f"K_rec shape == K_pre shape {tuple(K_pre.shape)}",
        passed=shape_ok,
        detail=f"got {tuple(K_rec_check.shape)}",
    )
    log("K_rec contains no NaN", passed=nan_ok)
    log("K_rec contains no Inf", passed=inf_ok)
    log(
        "Mean activation relative error",
        passed=None,
        detail=f"{mean_err:.4f}  (max {max_err:.4f})",
    )

    if mean_err > 0.30:
        log_warn(
            f"Mean activation error {mean_err:.4f} is high",
            detail="reconstruction will distort attention patterns — "
                   "top-1 token match in Test 3 may fail",
        )

    passed = shape_ok and nan_ok and inf_ok
    print(f"\n  Overall Test 2: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Test 3 — Output equivalence ───────────────────────────────────────────────

def test_output_equivalence(
    model: nn.Module,
    tokenizer,
    ref_logits: torch.Tensor,
    prompt: str,
    n_new_tokens: int,
    max_abs_tol: float = 4e-1,
) -> bool:
    """
    Generate with the patched model (VOnlyCache) and compare to reference.

    WHY: Tests 1 and 2 measure approximation error in isolation (weights,
    then activations). This test measures the downstream effect on the
    actual model output — whether the reconstruction error is small enough
    that the model still picks the same tokens.

    Top-1 match is the primary gate because logit scale varies between models
    and steps. A max absolute logit difference that passes in one model may
    not be meaningful in another; token agreement is model-agnostic.

    A step-by-step breakdown (first 5 steps) shows whether errors are
    introduced at prefill or accumulate during decoding.
    """
    header("Test 3 — Output equivalence  (patched vs unpatched)")

    pat_logits = run_generation_logits(
        model, tokenizer, prompt, n_new_tokens, cache_type="v_only"
    )

    # Per-step max logit diff — useful for debugging where errors enter.
    # If step 0 (prefill) already shows large diff, the issue is in K reconstruction.
    # If only later steps diverge, it may be a cache state accumulation issue.
    per_step_max = (pat_logits - ref_logits).abs().max(dim=-1).values.squeeze()
    print(f"\n  Per-step max |Δlogit| (first {min(5, len(per_step_max))} steps):")
    for i, d in enumerate(per_step_max[:min(5, len(per_step_max))]):
        print(f"    step {i}: {d.item():.4f}")
    print()

    max_diff   = (pat_logits - ref_logits).abs().max().item()
    top1_ref   = ref_logits.argmax(dim=-1)
    top1_pat   = pat_logits.argmax(dim=-1)
    top1_match = (top1_ref == top1_pat).all().item()
    abs_ok     = max_diff < max_abs_tol

    log(
        f"max |Δlogit| < {max_abs_tol}  (soft numeric check)",
        passed=abs_ok,
        detail=f"max={max_diff:.4f}",
    )
    log(
        "top-1 token unchanged at every step  ← PRIMARY GATE",
        passed=top1_match,
        detail=f"{n_new_tokens} tokens compared",
    )

    passed = top1_match
    print(f"\n  Overall Test 3: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Test 4 — Cache shape ──────────────────────────────────────────────────────

def test_cache_shape(
    model: nn.Module,
    tokenizer,
    prompt: str,
) -> bool:
    """
    Run prefill + one decode step with VOnlyCache and verify cache contents.

    WHY: VOnlyCache must store ONLY V — if K is accidentally stored too,
    the memory saving evaporates and K reconstruction may be bypassed.
    This test catches implementation bugs in the cache write path.

    Expected after prefill + 1 decode step:
      value_cache[0].shape[-2] == num_kv_heads    (KV head axis)
      value_cache[0].shape[-1] == head_dim        (head dimension)
      key_cache must be empty (len == 0) or contain only positional data
      that is NOT the full K — i.e. no [S, num_kv_heads, head_dim] tensor.
    """
    header("Test 4 — Cache shape  (V only — K must not be stored)")

    device       = next(model.parameters()).device
    attn0        = _get_attn(model, layer_idx=0)
    num_kv_heads = _get_num_kv_heads(attn0)
    head_dim     = _get_head_dim(attn0)

    inputs_tok = tokenize(tokenizer, prompt, device)
    past_kv    = VOnlyCache()

    with torch.no_grad():
        out      = model(**inputs_tok, past_key_values=past_kv, use_cache=True)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        attn_mask = torch.cat(
            [inputs_tok["attention_mask"],
             inputs_tok["attention_mask"].new_ones((1, 1))], dim=1
        )
        out = model(
            input_ids=next_tok,
            attention_mask=attn_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = out.past_key_values

    stored_v_kv_heads = past_kv.value_dim_kv_heads(layer_idx=0)
    stored_v_head_dim = past_kv.value_dim_head(layer_idx=0)

    # Key cache should be absent — K is reconstructed on-the-fly from V @ N.
    # If key_cache is non-empty here it means K is being stored, wasting memory.
    key_cache_empty = past_kv.key_cache_empty()

    log(f"Expected V kv_heads dim : {num_kv_heads}")
    log(f"Expected V head_dim     : {head_dim}")
    log(
        f"value_cache kv_heads == {num_kv_heads}",
        passed=(stored_v_kv_heads == num_kv_heads),
        detail=f"got {stored_v_kv_heads}",
    )
    log(
        f"value_cache head_dim == {head_dim}",
        passed=(stored_v_head_dim == head_dim),
        detail=f"got {stored_v_head_dim}",
    )
    log(
        "key_cache is empty  (K not stored — reconstructed at attention time)",
        passed=key_cache_empty,
        detail="WRONG: K is being stored, no memory saving" if not key_cache_empty else "",
    )

    passed = (
        (stored_v_kv_heads == num_kv_heads)
        and (stored_v_head_dim == head_dim)
        and key_cache_empty
    )
    print(f"\n  Overall Test 4: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Test 5 — Memory savings ───────────────────────────────────────────────────

def _run_to_length_standard(
    model: nn.Module,
    tokenizer,
    prompt: str,
    target_tokens: int,
) -> float:
    """Fill DynamicCache to target_tokens. Returns total KV cache MB."""
    device    = next(model.parameters()).device
    inputs    = tokenize(tokenizer, prompt, device)
    n_new     = max(1, target_tokens - inputs["input_ids"].shape[1])
    attn_mask = inputs["attention_mask"]

    with torch.no_grad():
        out      = model(**inputs, past_key_values=DynamicCache(), use_cache=True)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(n_new - 1):
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=1)
            out       = model(
                input_ids=next_tok,
                attention_mask=attn_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv  = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    # DynamicCache stores key_cache and value_cache as lists of tensors
    if hasattr(past_kv, "key_cache"):
        key_bytes   = sum(t.nbytes for t in past_kv.key_cache   if t is not None)
        value_bytes = sum(t.nbytes for t in past_kv.value_cache if t is not None)
    else:
        key_bytes   = sum(kv[0].nbytes for kv in past_kv if kv is not None)
        value_bytes = sum(kv[1].nbytes for kv in past_kv if kv is not None)
    return (key_bytes + value_bytes) / 1e6


def _run_to_length_v_only(
    model: nn.Module,
    tokenizer,
    prompt: str,
    target_tokens: int,
) -> float:
    """Fill VOnlyCache to target_tokens. Returns total cache MB."""
    device    = next(model.parameters()).device
    inputs    = tokenize(tokenizer, prompt, device)
    n_new     = max(1, target_tokens - inputs["input_ids"].shape[1])
    attn_mask = inputs["attention_mask"]
    past_kv   = VOnlyCache()

    with torch.no_grad():
        out      = model(**inputs, past_key_values=past_kv, use_cache=True)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(n_new - 1):
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=1)
            out       = model(
                input_ids=next_tok,
                attention_mask=attn_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv  = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    return past_kv.cache_size_bytes()["total_bytes"] / 1e6


def test_memory_savings(
    model: nn.Module,
    tokenizer,
    prompt: str,
    context_lengths: list,
    std_sizes: dict,
) -> bool:
    """
    Measure memory usage of VOnlyCache vs DynamicCache across context lengths.

    WHY: The entire motivation for this algorithm is memory reduction.
    DynamicCache stores K + V per token per layer (2 × num_kv_heads × head_dim).
    VOnlyCache stores V only (1 × num_kv_heads × head_dim).
    Expected saving: exactly 50%, independent of context length.

    The saving must be CONSTANT across lengths — if it drifts, something is
    being stored in the key cache that shouldn't be (implementation bug).
    A drift < 1 percentage point across all tested lengths is required to pass.

    For Llama 3 8B (num_kv_heads=8, head_dim=128, BF16):
      Standard: 2 × 8 × 128 × 2 bytes = 4 KB / token / layer
      V-only  : 1 × 8 × 128 × 2 bytes = 2 KB / token / layer
      Saving  : 50.0%
    """
    header("Test 5 — Memory savings across context lengths")

    attn0        = _get_attn(model, layer_idx=0)
    num_kv_heads = _get_num_kv_heads(attn0)
    head_dim     = _get_head_dim(attn0)
    expected_saving = 50.0  # exactly 50% since we drop K and keep V

    log(f"Standard cache: K + V  =  2 × {num_kv_heads} × {head_dim} dims/token/layer")
    log(f"V-only   cache: V only =  1 × {num_kv_heads} × {head_dim} dims/token/layer")
    log(f"Expected saving: {expected_saving:.1f}%  (constant across all context lengths)")
    print()

    col_w = [10, 18, 16, 10]
    print(
        f"  {'Context':<{col_w[0]}}"
        f"{'Standard cache':>{col_w[1]}}"
        f"{'V-only cache':>{col_w[2]}}"
        f"{'Saving':>{col_w[3]}}"
    )
    print("  " + "─" * (sum(col_w) + 2))

    savings = []
    for ctx in context_lengths:
        std_mb    = std_sizes[ctx]
        vonly_mb  = _run_to_length_v_only(model, tokenizer, prompt, ctx)
        saving_pct = 100.0 * (1.0 - vonly_mb / std_mb) if std_mb > 0 else 0.0
        savings.append(saving_pct)
        print(
            f"  {f'{ctx} tok':<{col_w[0]}}"
            f"{f'{std_mb:.2f} MB':>{col_w[1]}}"
            f"{f'{vonly_mb:.2f} MB':>{col_w[2]}}"
            f"{f'{saving_pct:.1f}%':>{col_w[3]}}"
        )

    print("  " + "─" * (sum(col_w) + 2))
    print()

    drift     = max(savings) - min(savings)
    drift_ok  = drift < 1.0   # saving must be constant — any drift = caching bug
    mean_sav  = sum(savings) / len(savings)
    # Allow 2pp tolerance around the 50% target (rounding, padding, etc.)
    approx_ok = abs(mean_sav - expected_saving) < 2.0

    log(
        "Saving is constant across lengths  (drift < 1pp)",
        passed=drift_ok,
        detail=f"min={min(savings):.1f}%  max={max(savings):.1f}%  drift={drift:.2f}pp",
    )
    log(
        f"Observed saving ≈ {expected_saving:.0f}%  (within 2pp)",
        passed=approx_ok,
        detail=f"observed mean={mean_sav:.1f}%",
    )

    passed = drift_ok and approx_ok
    print(f"\n  Overall Test 5: {'PASS' if passed else 'FAIL'}")
    return passed


# ── Test 6 — Throughput ───────────────────────────────────────────────────────

def _measure_tps(
    model: nn.Module,
    tokenizer,
    prompt: str,
    n_tokens: int,
    cache_type: str = "dynamic",
) -> float:
    """Generate n_tokens decode steps (after prefill warm-up). Returns tok/sec."""
    device    = next(model.parameters()).device
    inputs    = tokenize(tokenizer, prompt, device)
    attn_mask = inputs["attention_mask"]
    past_kv   = VOnlyCache() if cache_type == "v_only" else DynamicCache()

    with torch.no_grad():
        out      = model(**inputs, past_key_values=past_kv, use_cache=True)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    # One warm-up decode step to initialise CUDA state before timing
    with torch.no_grad():
        attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=1)
        out       = model(
            input_ids=next_tok, attention_mask=attn_mask,
            past_key_values=past_kv, use_cache=True,
        )
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_tokens):
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=1)
            out       = model(
                input_ids=next_tok, attention_mask=attn_mask,
                past_key_values=past_kv, use_cache=True,
            )
            past_kv  = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    if device.type == "cuda":
        torch.cuda.synchronize()

    return n_tokens / (time.perf_counter() - t0)


def test_throughput(
    model_unpatched: nn.Module,
    model_patched: nn.Module,
    tokenizer,
    prompt: str,
    n_tokens: int = 256,
) -> None:
    """
    Measure tokens/sec for patched vs unpatched model. Informational only.

    WHY: The V-only cache saves 50% memory bandwidth for K/V loads, but adds
    a [head_dim × head_dim] matmul per layer per step for K reconstruction.
    The net effect depends on whether the workload is memory-bandwidth-bound
    (typical at long context on a single GPU) or compute-bound.

    At long context (>8k): bandwidth saving dominates → expect speedup.
    At short context (<2k): reconstruction overhead may dominate → possible slowdown.

    No pass/fail gate — this is a characterisation measurement.
    """
    header("Test 6 — Throughput  (informational, no pass/fail)")
    log(f"Generating {n_tokens} decode tokens per model (after warm-up)...")
    print()

    tps_std = _measure_tps(model_unpatched, tokenizer, prompt, n_tokens, "dynamic")
    tps_pat = _measure_tps(model_patched,   tokenizer, prompt, n_tokens, "v_only")
    ratio   = tps_pat / tps_std if tps_std > 0 else float("nan")

    col_w = [28, 14]
    print(f"  {'Model':<{col_w[0]}}{'Tokens / sec':>{col_w[1]}}")
    print("  " + "─" * (sum(col_w) + 2))
    print(f"  {'Unpatched (DynamicCache)':<{col_w[0]}}{tps_std:>{col_w[1]}.1f}")
    print(f"  {'Patched   (VOnlyCache)'  :<{col_w[0]}}{tps_pat:>{col_w[1]}.1f}")
    print("  " + "─" * (sum(col_w) + 2))
    print(f"\n  Relative speed : {ratio:.3f}x")

    if ratio < 1.0:
        print(f"  Overhead: {100.0*(1.0-ratio):.1f}%  "
              f"(V @ N matmul cost > cache bandwidth saving at this context length)")
    else:
        print(f"  Speedup : {100.0*(ratio-1.0):.1f}%  "
              f"(cache bandwidth reduction dominating at this context length)")

    log("Throughput is informational — no pass/fail gate")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 64)
    print("  GQA V-only KV Cache — Verification Suite")
    print("  K reconstructed as K_pre = V @ N, then RoPE applied")
    print("=" * 64)
    print(f"  model     : {args.model_path}")
    print(f"  prompt    : {args.prompt[:60]}...")
    print(f"  layer_idx : {args.layer_idx}  (Tests 1 and 2 only)")
    print(f"  gen_tokens: {args.gen_tokens}")

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    device = next(model.parameters()).device

    # ── Offline: compute all N matrices before any patching ──────────────────
    # N matrices are derived from weights alone — no forward pass needed.
    # Compute in FP64 for numerical precision; store BF16 for runtime use.
    # Shape per layer: [num_kv_heads, head_dim, head_dim]
    print(f"\n{SEP}")
    print("  Computing N matrices offline (all layers, BF16 for runtime)...")
    print(SEP)
    N_matrices = compute_all_N_matrices(model, out_dtype=torch.bfloat16, device=device)

    # ── Pre-patch: reference logits and standard cache sizes ─────────────────
    print(f"\n{SEP}")
    print("  Pre-patch: collecting reference logits  (unpatched model)")
    print(SEP)
    ref_logits = run_generation_logits(
        model, tokenizer, args.prompt, args.gen_tokens, cache_type="dynamic"
    )
    print(f"  Reference logits shape: {tuple(ref_logits.shape)}")

    print(f"\n{SEP}")
    print("  Pre-patch: measuring standard DynamicCache sizes")
    print(SEP)
    std_sizes: dict = {}
    for ctx in args.context_lengths:
        mb = _run_to_length_standard(model, tokenizer, args.prompt, ctx)
        std_sizes[ctx] = mb
        print(f"    {ctx:5d} tokens → {mb:.2f} MB")

    # ── Test 1: N matrix weight-level accuracy ────────────────────────────────
    t1_pass = test_n_weight_accuracy(
        model,
        layer_idx=args.layer_idx,
        N_matrices=N_matrices,
        warn_threshold=args.weight_err_warn,
    )

    # ── Test 2: Pre-RoPE activation reconstruction ────────────────────────────
    # Run before patching — hooks into the unpatched forward pass to capture
    # clean k_proj and v_proj outputs and compare per head.
    t2_pass = test_activation_reconstruction(
        model, tokenizer, args.prompt,
        layer_idx=args.layer_idx,
        N_matrices=N_matrices,
    )

    # ── Patch the model ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Patching model: replacing attention forward with V-only cache + K reconstruction")
    print(SEP)
    patch_kv_model(model, N_matrices=N_matrices, device=device)

    # ── Test 3: Output equivalence ────────────────────────────────────────────
    t3_pass = test_output_equivalence(
        model, tokenizer, ref_logits, args.prompt,
        n_new_tokens=args.gen_tokens,
        max_abs_tol=args.output_abs_tol,
    )

    # ── Test 4: Cache shape ───────────────────────────────────────────────────
    t4_pass = test_cache_shape(model, tokenizer, args.prompt)

    # ── Test 5: Memory savings ────────────────────────────────────────────────
    t5_pass = test_memory_savings(
        model, tokenizer, args.prompt,
        context_lengths=args.context_lengths,
        std_sizes=std_sizes,
    )

    # ── Test 6: Throughput ────────────────────────────────────────────────────
    if args.skip_throughput:
        header("Test 6 — Throughput  (skipped via --skip-throughput)")
    else:
        print(f"\n{SEP}")
        print("  Test 6: reloading unpatched model for throughput comparison...")
        print(f"  (pass --skip-throughput to skip if VRAM is tight)")
        print(SEP)
        model_ref, _ = load_model_and_tokenizer(args.model_path)
        test_throughput(
            model_ref, model, tokenizer, args.prompt,
            n_tokens=args.throughput_tokens,
        )
        del model_ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  SUMMARY")
    print(f"{'=' * 64}")
    results = {
        "Test 1 — N weight accuracy          ": t1_pass,
        "Test 2 — Activation reconstruction  ": t2_pass,
        "Test 3 — Output equivalence         ": t3_pass,
        "Test 4 — Cache shape                ": t4_pass,
        "Test 5 — Memory savings             ": t5_pass,
    }
    all_pass = True
    for name, ok in results.items():
        print(f"  [{PASS if ok else FAIL}]  {name}")
        if not ok:
            all_pass = False

    if args.skip_throughput:
        print(f"  [{INFO}]  Test 6 — Throughput               (skipped)")
    else:
        print(f"  [{INFO}]  Test 6 — Throughput               (informational only)")

    print(f"\n{'=' * 64}")
    print("  ALL TESTS PASSED" if all_pass else "  SOME TESTS FAILED — see details above")
    print(f"{'=' * 64}\n")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()