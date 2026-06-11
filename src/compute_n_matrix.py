"""
compute_n_matrix.py — Offline computation of the KV relation matrix N.

Background
──────────────────────────────────────────────────────────────────────────────
In standard GQA attention, K and V are independent linear projections of the
same input X through separate weight matrices:

    K_pre = X @ W_K.T      [S, num_kv_heads, head_dim]   (before RoPE)
    V     = X @ W_V.T      [S, num_kv_heads, head_dim]

Because both share the same input X, there exists a static matrix N per KV
head such that:

    K_pre_h  ≈  V_h @ N_h

N_h is the least-squares solution to:

    min_{N_h}  ‖ W_V_h @ N_h  −  W_K_h ‖_F

which is solved via the pseudo-inverse of W_V_h:

    N_h  =  W_V_h⁺  @  W_K_h           shape [head_dim × head_dim]

At inference, only V is cached. K is reconstructed on-the-fly:

    K_pre_h = V_h @ N_h
    K_h     = RoPE(K_pre_h, positions)  ← identical to the normal path from here

The approximation is exact iff col(W_K_h) ⊆ col(W_V_h). The relative error
‖W_K_h − W_V_h @ N_h‖_F / ‖W_K_h‖_F quantifies subspace misalignment
and should be measured per head per layer before deployment.

Difference from MLA version
──────────────────────────────────────────────────────────────────────────────
The previous version targeted DeepSeek-V2-Lite (MLA architecture), where K
and V both derive from a shared low-rank latent via a single kv_b_proj weight.
That made N exact by construction (zero approximation error).

This version targets standard GQA (gpt-oss-20b-BF16), where k_proj and
v_proj are separate, independently-trained Linear layers. N is now an
approximation, and its quality must be validated empirically.

Key structural changes:
  ┌─────────────────────────────────┬──────────────────────────────────────┐
  │ MLA (DeepSeek)                  │ Standard GQA (gpt-oss)               │
  ├─────────────────────────────────┼──────────────────────────────────────┤
  │ Shared kv_b_proj weight         │ Separate k_proj, v_proj weights      │
  │ Fat weight: [v_dim × lora_rank] │ Tall weight: [d_model × head_dim]    │
  │ Interleaved K/V layout per head │ Independent row slices per head      │
  │ N shape: [H, v_dim, qk_nope]   │ N shape: [H, head_dim, head_dim]     │
  │ Zero approximation error        │ Error depends on subspace alignment  │
  └─────────────────────────────────┴──────────────────────────────────────┘

Target model
──────────────────────────────────────────────────────────────────────────────
  gpt-oss-20b-BF16

Weight layout:
  self_attn.k_proj.weight  ∈  R^{num_kv_heads * head_dim  ×  d_model}
  self_attn.v_proj.weight  ∈  R^{num_kv_heads * head_dim  ×  d_model}

  Per-head slice (head h, D = head_dim):
    W_K_h  =  k_proj.weight[h*D : (h+1)*D, :].T   ∈  R^{d_model × D}
    W_V_h  =  v_proj.weight[h*D : (h+1)*D, :].T   ∈  R^{d_model × D}
    N_h    =  pinv(W_V_h) @ W_K_h                  ∈  R^{D × D}

Computed in FP64 for numerical precision of the pseudo-inverse.
Stored and used at runtime in BF16.
"""

import torch
import torch.nn as nn


# ── Attribute helpers ─────────────────────────────────────────────────────────

def _get_num_kv_heads(attn: nn.Module) -> int:
    """
    Return the number of KV heads (g), e.g. 8 for a 20B GQA model.

    Tries a broad set of attribute names used by different model families.
    Falls back to inspecting attn.config if present.
    """
    # Direct attribute — covers Llama, Mistral, GPT-style variants
    for attr in (
        "num_key_value_heads",   # HF standard (Llama 3, Mistral, …)
        "num_kv_heads",          # some custom models
        "num_kv_attention_heads",
        "kv_heads",
        "n_kv_heads",
    ):
        if hasattr(attn, attr):
            return int(getattr(attn, attr))

    # Config object fallback (some models store dims only on config, not attn)
    if hasattr(attn, "config"):
        for attr in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
            if hasattr(attn.config, attr):
                return int(getattr(attn.config, attr))

    # Nothing matched — print available numeric-looking attributes to help diagnose
    numeric_attrs = {
        k: getattr(attn, k)
        for k in dir(attn)
        if not k.startswith("_")
        and isinstance(getattr(attn, k, None), int)
    }
    raise AttributeError(
        f"Cannot find KV head count on {type(attn).__name__}.\n"
        f"Available int attributes: {numeric_attrs}\n"
        f"Add the correct attribute name to the list in _get_num_kv_heads()."
    )


def _get_head_dim(attn: nn.Module) -> int:
    """
    Return head_dim (D), the per-head projection dimension.

    Try the direct attribute first; fall back to hidden_size // num_heads
    for models that don't expose it explicitly.
    """
    if hasattr(attn, "head_dim"):
        return int(attn.head_dim)
    # Fallback: infer from hidden_size and query head count
    for num_heads_attr in ("num_heads", "num_attention_heads"):
        if hasattr(attn, num_heads_attr):
            return int(attn.hidden_size // getattr(attn, num_heads_attr))
    raise AttributeError(
        f"Cannot determine head_dim on {type(attn).__name__}. "
        f"Expected attribute 'head_dim', or 'hidden_size' + 'num_heads'."
    )


# ── Per-head weight extraction ────────────────────────────────────────────────

def _get_per_head_weights(
    attn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract W_K and W_V as tall matrices [d_model × head_dim] per KV head.

    k_proj.weight has shape [num_kv_heads * head_dim, d_model]  (nn.Linear convention:
    weight is [out_features, in_features], and K = x @ weight.T).

    We slice along the output (row) dimension to isolate each head, then
    transpose to get the natural projection shape [d_model × head_dim].

        W_K_h = k_proj.weight[h*D : (h+1)*D, :].T    [d_model, head_dim]
        W_V_h = v_proj.weight[h*D : (h+1)*D, :].T    [d_model, head_dim]

    Converted to float64 here so the SVD and pseudo-inverse are computed at
    full precision regardless of the model's storage dtype.

    Returns:
        W_K: [num_kv_heads, d_model, head_dim]  float64
        W_V: [num_kv_heads, d_model, head_dim]  float64
    """
    if not hasattr(attn, "k_proj") or not hasattr(attn, "v_proj"):
        raise AttributeError(
            f"Cannot find k_proj or v_proj on {type(attn).__name__}. "
            f"Expected standard GQA Linear projections."
        )

    num_kv_heads = _get_num_kv_heads(attn)
    head_dim     = _get_head_dim(attn)

    # weight shape: [num_kv_heads * head_dim, d_model]
    W_K_flat = attn.k_proj.weight.double()   # [H*D, d_model]
    W_V_flat = attn.v_proj.weight.double()   # [H*D, d_model]

    # Validate shape before slicing
    expected_rows = num_kv_heads * head_dim
    for name, W in (("k_proj", W_K_flat), ("v_proj", W_V_flat)):
        if W.shape[0] != expected_rows:
            raise ValueError(
                f"{name}.weight row count mismatch: got {W.shape[0]}, "
                f"expected num_kv_heads({num_kv_heads}) × head_dim({head_dim}) "
                f"= {expected_rows}."
            )

    # Reshape to [num_kv_heads, head_dim, d_model], then transpose last two dims
    # → [num_kv_heads, d_model, head_dim] (tall matrix per head, natural projection form).
    W_K = W_K_flat.view(num_kv_heads, head_dim, -1).transpose(1, 2)  # [H, d_model, D]
    W_V = W_V_flat.view(num_kv_heads, head_dim, -1).transpose(1, 2)  # [H, d_model, D]

    return W_K, W_V


# ── Per-layer N computation ───────────────────────────────────────────────────

def compute_N_matrix(
    attn: nn.Module,
    out_dtype: torch.dtype = torch.bfloat16,
    svd_threshold: float = 1e-5,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Compute the KV relation matrix N for a single GQA attention module.

    For each KV head h, solves:
        min_{N_h}  ‖ W_V_h @ N_h  −  W_K_h ‖_F

    via truncated SVD of W_V_h (a tall matrix [d_model × head_dim]):

        W_V_h = U @ diag(s) @ Vh.T
            U:   [d_model, head_dim]   left singular vectors
            s:   [head_dim]            singular values
            Vh:  [head_dim, head_dim]  right singular vectors

        pinv(W_V_h) = Vh @ diag(s_inv) @ U.T   shape [head_dim, d_model]

        N_h = pinv(W_V_h) @ W_K_h              shape [head_dim, head_dim]

    SVD note — tall vs fat matrix compared to MLA version:
      MLA used a FAT matrix [v_dim × lora_rank] where v_dim < lora_rank,
      so full_matrices=False gave U:[v_dim,v_dim], Vh:[v_dim,lora_rank].
      Here W_V_h is TALL [d_model × head_dim] where d_model >> head_dim,
      so full_matrices=False gives U:[d_model,head_dim], Vh:[head_dim,head_dim].
      The pseudo-inverse formula changes accordingly — see inline comments.

    svd_threshold: relative singular value cutoff. Values below
        svd_threshold * s.max() are treated as zero to avoid amplifying
        numerical noise from near-zero singular values (ill-conditioned heads).
        Default 1e-5 is conservative; raise to 1e-3 if N shows large values.

    Returns:
        N: [num_kv_heads, head_dim, head_dim], dtype=out_dtype
    """
    num_kv_heads = _get_num_kv_heads(attn)
    head_dim     = _get_head_dim(attn)
    W_K, W_V     = _get_per_head_weights(attn)  # [H, d_model, head_dim], float64

    N_heads = []
    for h in range(num_kv_heads):
        # W_V[h]:  [d_model, head_dim]  — tall matrix, d_model >> head_dim
        # W_K[h]:  [d_model, head_dim]
        #
        # SVD of W_V[h] with full_matrices=False (economy SVD):
        #   U:  [d_model, head_dim]   ← only head_dim left singular vectors kept
        #   s:  [head_dim]
        #   Vh: [head_dim, head_dim]
        #
        # W_V[h] = U @ diag(s) @ Vh
        #
        # pinv(W_V[h]) = Vh.T @ diag(s_inv) @ U.T   shape [head_dim, d_model]
        #
        # N_h = pinv(W_V[h]) @ W_K[h]               shape [head_dim, head_dim]

        U, s, Vh = torch.linalg.svd(W_V[h], full_matrices=False)
        # U:  [d_model, head_dim],  s: [head_dim],  Vh: [head_dim, head_dim]

        # Zero out singular values below the relative threshold to avoid
        # amplifying noise from near-zero directions (numerical regularisation).
        s_inv = torch.where(
            s > svd_threshold * s.max(),
            1.0 / s,
            torch.zeros_like(s),
        )

        # pinv(W_V_h) = Vh.T @ diag(s_inv) @ U.T
        # Note: compared to MLA, Vh is now square [head_dim × head_dim] and
        # U is tall [d_model × head_dim], so the contraction order changes.
        pinv_WV = Vh.T @ torch.diag(s_inv) @ U.T   # [head_dim, d_model]

        # N_h = pinv(W_V_h) @ W_K_h
        N_h = pinv_WV @ W_K[h]                     # [head_dim, head_dim]

        N_heads.append(N_h)

        if verbose:
            n_kept   = (s > svd_threshold * s.max()).sum().item()
            # Reconstruction quality: how well does W_V_h @ N_h recover W_K_h?
            K_rec    = W_V[h] @ N_h
            rel_err  = ((W_K[h] - K_rec).norm() / W_K[h].norm()).item()
            cond     = (s.max() / s[s > 0].min()).item()
            print(
                f"    head {h:2d}: "
                f"max_s={s.max():.4f}  min_s={s.min():.4f}  "
                f"kept={n_kept}/{len(s)}  cond={cond:.1f}  "
                f"rel_err={rel_err:.4f}"
            )

    # Stack heads and cast to runtime dtype (BF16 for gpt-oss)
    N = torch.stack(N_heads, dim=0).to(dtype=out_dtype)  # [H, head_dim, head_dim]
    return N


# ── Model-level helper ────────────────────────────────────────────────────────

def compute_all_N_matrices(
    model: nn.Module,
    out_dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    verbose: bool = False,
) -> list[torch.Tensor]:
    """
    Compute N matrices for every decoder layer in the model.

    Iterates over model.model.layers, calling compute_N_matrix on each
    layer's self_attn module. N matrices are computed in FP64 (inside
    compute_N_matrix) and stored at out_dtype (default BF16 for gpt-oss).

    Returns:
        List of N tensors, one per layer.
        Each tensor: [num_kv_heads, head_dim, head_dim], dtype=out_dtype.

    Memory note: for gpt-oss-20b with typical GQA dims (e.g. 8 KV heads,
    head_dim=128, 40 layers, BF16):
        8 × 128 × 128 × 2 bytes × 40 layers ≈ 10 MB total — negligible.
    """
    if device is None:
        device = next(model.parameters()).device

    layers = model.model.layers
    print(f"Computing N matrices for {len(layers)} layers on device={device}...")

    N_matrices = []
    for i, layer in enumerate(layers):
        N = compute_N_matrix(
            layer.self_attn,
            out_dtype=out_dtype,
            verbose=verbose,
        )
        # Move to target device non-blocking; detach from autograd graph
        # (N is a static inference-time constant, not a trainable parameter).
        N_target = N.to(device=device, non_blocking=True).detach()
        N_matrices.append(N_target)

        # Always print layer 0 so the user can confirm shapes look right
        # before waiting for all layers to finish.
        if i == 0 or verbose:
            print(
                f"  Layer {i:3d}: N shape={tuple(N_target.shape)}  "
                f"dtype={N_target.dtype}  device={N_target.device}"
            )

    if not verbose:
        print(f"  ... ({len(N_matrices)} layers total)")

    print("Done.")
    return N_matrices
    