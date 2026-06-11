"""
benchmark_kv_cache.py — End-to-end correctness and efficiency verification
for the KV cache optimisation.

Tests
──────────────────────────────────────────────────────────────────────────────
  Test 1  N matrix accuracy
          Hook kv_b_proj (or fused_kv_b) output during one forward pass.
          Split into k_nope_ref and v_ref.
          Check max |k_nope_ref - v_ref @ N| and top-1 argmax match.
          This validates compute_n_matrix.py independently of the cache patch.

  Test 2  Output equivalence
          Generate tokens with the unpatched model → ref logits.
          Patch the model; generate with KVNopelessCache.
          Assert top-1 token matches at every position.
          This confirms k_nope reconstruction doesn't change predictions.

  Test 3  Cache shape
          After one prefill + one decode step, inspect key tensors in
          KVNopelessCache.  Assert shape[-1] == qk_rope_head_dim, not
          qk_nope_head_dim + qk_rope_head_dim.

  Test 4  Memory savings across context lengths
          Measure DynamicCache (pre-patch) and KVNopelessCache (post-patch)
          at context lengths [128, 512, 1024, 2048, 4096].
          Print a table and assert the saving % is constant (within 1%).

  Test 5  Throughput  (informational, no pass/fail)
          Tokens/sec for the patched vs unpatched model.
          Gated with --skip-throughput.

Usage
──────────────────────────────────────────────────────────────────────────────
    python benchmark_kv_cache.py --model_path ./models/DeepSeek-V2-Lite
    python benchmark_kv_cache.py --model_path ./models/DeepSeek-V2-Lite --skip-throughput
    python benchmark_kv_cache.py --model_path ./models/DeepSeek-V2-Lite --layer_idx 2
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# ── Make project root importable regardless of where the script is invoked from
# This file lives in tests/; root is one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from src.compute_n_matrix import compute_N_matrix
from ops.kv_nopeless_cache import KVNopelessCache
from kv_patch import patch_kv_model


# ── Test 1 — N matrix accuracy (DeepSeek-V2 Patched) ──────────────────────────

def test_n_accuracy(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    layer_idx: int = 0,
    max_abs_tol: float = 3e-1,
) -> bool:
    """
    Hook into kv_b_proj or fused_kv_b at layer_idx for DeepSeek-V2 (MLA).
    """
    header("Test 1 — N matrix accuracy")

    attn = model.model.layers[layer_idx].self_attn
    
    # DeepSeek-V2 uncompresses keys/values into the total number of attention heads
    num_heads = getattr(attn, "num_heads", getattr(attn, "num_attention_heads", None))
    qk_nope_head_dim = attn.qk_nope_head_dim
    v_head_dim = attn.v_head_dim
    device = next(model.parameters()).device

    # ── Compute N from weights ────────────────────────────────────────────────
    N = compute_N_matrix(attn, out_dtype=torch.bfloat16, verbose=False).to(device)
    # Target shape expectation: [num_heads, v_head_dim, qk_nope_head_dim]
    log(f"N shape   : {tuple(N.shape)}", detail=f"layer {layer_idx}")
    log(f"N dtype   : {N.dtype}")

    # ── Register forward hook ─────────────────────────────────────────────────
    captured: dict = {}

    def _hook(module, input, output):
        # DeepSeek-V2 output can be a tuple depending on the custom implementation; handle safely
        if isinstance(output, tuple):
            output = output[0]
        captured["kv_b_out"] = output.detach()

    if hasattr(attn, "fused_kv_b"):
        hook_mod = attn.fused_kv_b
        log("Weight source: fused_kv_b.W_new  (fused patch model)")
    elif hasattr(attn, "kv_b_proj"):
        hook_mod = attn.kv_b_proj
        log("Weight source: kv_b_proj.weight  (standard HF model)")
    else:
        log("Cannot find kv_b_proj or fused_kv_b — skipping Test 1", passed=False)
        return False

    handle = hook_mod.register_forward_hook(_hook)

    inputs_tok = tokenize(tokenizer, prompt, device)
    with torch.no_grad():
        model(**inputs_tok, use_cache=False)

    handle.remove()

    if "kv_b_out" not in captured:
        log("Hook did not fire — module not reached during forward", passed=False)
        return False

    kv_out = captured["kv_b_out"]   # [B, S, num_heads * (qk_nope_head_dim + v_head_dim)]
    B, S, _ = kv_out.shape

    # ── Split safely into k_nope_ref and v_ref ────────────────────────────────
    # Reshape matching standard Hugging Face MLA slicing order
    kv_reshaped = kv_out.view(B, S, num_heads, qk_nope_head_dim + v_head_dim)
    k_nope_ref  = kv_reshaped[..., :qk_nope_head_dim]   # [B, S, H, qk_nope]
    v_ref       = kv_reshaped[..., qk_nope_head_dim:]   # [B, S, H, v_dim]

    # Transpose to standard [B, H, S, dim] structure
    k_nope_ref = k_nope_ref.transpose(1, 2)   # [B, H, S, qk_nope]
    v_ref      = v_ref.transpose(1, 2)        # [B, H, S, v_dim]

    # ── Reconstruct k_nope via v @ N ──────────────────────────────────────────
    # If N is [H, v_dim, qk_nope], we use batched matrix multiplication (bmm) 
    # over the head dimension 'H'.
    # v_ref: [B, H, S, v_dim] -> permute to [H, B*S, v_dim]
    # N: [H, v_dim, qk_nope]
    v_ref_batched = v_ref.permute(1, 0, 2, 3).reshape(num_heads, B * S, v_head_dim)
    k_nope_hat_batched = torch.bmm(v_ref_batched, N)  # [H, B*S, qk_nope]
    
    # Restore layout back to [B, H, S, qk_nope]
    k_nope_hat = k_nope_hat_batched.view(num_heads, B, S, qk_nope_head_dim).permute(1, 0, 2, 3)

    # ── Match Validation ──────────────────────────────────────────────────────
    diff       = (k_nope_hat - k_nope_ref).abs()
    max_diff   = diff.max().item()
    mean_diff  = diff.mean().item()
    top1_match = (
        k_nope_hat.argmax(dim=-1) == k_nope_ref.argmax(dim=-1)
    ).all().item()

    abs_ok = max_diff < max_abs_tol

    log(
        f"max  |k_nope - v @ N| < {max_abs_tol}",
        passed=abs_ok,
        detail=f"max={max_diff:.4f}  mean={mean_diff:.4f}",
    )
    log(
        "top-1 argmax match  ← primary correctness gate",
        passed=top1_match,
    )

    passed = top1_match
    print(f"\n  Overall Test 1: {'PASS' if passed else 'FAIL'}")
    return passed



# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 64)
    print("  KV Nopeless Cache — Sequential Verification Suite")
    print("=" * 64)
    print(f"  Directory : {args.models_dir}")
    print(f"  Prompt    : {args.prompt[:60]}...")
    print(f"  Layer Idx : {args.layer_idx}  (Test 1 hook)")

    # Resolve paths from your main/models/ directory structure
    fused_model_path = os.path.join(args.models_dir, "fused")
    base_model_path = os.path.join(args.models_dir, "non-fused")

    # ──────────────────────────────────────────────────────────────────────────
    # ── STAGE 1: Fused Model Loading & Test 1 ─────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  [STAGE 1] Loading Fused Model into VRAM...")
    print(SEP)
    
    model_fused, tokenizer = load_model_and_tokenizer(fused_model_path)
    
    print(f"\n{SEP}")
    print("  Executing Test 1 — N Matrix Accuracy")
    print(SEP)
    
    t1_pass = test_n_accuracy(
        model_fused, tokenizer, args.prompt,
        layer_idx=args.layer_idx,
        max_abs_tol=args.n_abs_tol,
    )
    
    # ── Strict VRAM Clean up of Fused Model ───────────────────────────────────
    print(f"\n{SEP}")
    print("  Purging Fused Model from VRAM to make room...")
    print(SEP)
    
    del model_fused
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Force the host to wait for GPU cleanup to finish

    # ──────────────────────────────────────────────────────────────────────────
    # ── Summary (Test 1 Only) ─────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  STAGE 1 SUMMARY")
    print(f"{'=' * 64}")
    
    tag = PASS if t1_pass else FAIL
    print(f"  [{tag}]  Test 1 — N matrix accuracy")
    print(f"\n{'=' * 64}")
    
    # Optional: Exit early if Test 1 fails to avoid loading the second large model
    if not t1_pass:
        print("  Test 1 failed. Aborting remaining stages.")
        sys.exit(1)

    print("  Stage 1 successful. Ready to proceed to baseline collection.")
    
    # We will write the subsequent stages (Base model loading, Tests 2-5) next.

if __name__ == "__main__":
    main()