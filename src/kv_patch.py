"""
kv_patch.py — Patch GQA attention to use VOnlyCache.

Target model: gpt-oss-20b-BF16  (GptOssAttention, standard GQA)

What this file does
──────────────────────────────────────────────────────────────────────────────
For each decoder layer, replaces attn.forward so that:
  • Only V is written to VOnlyCache — K is never stored
  • At each decode step, K is reconstructed from the accumulated V via:
        K_pre_h = V_h @ N_h          (per KV head, pre-RoPE)
        K_h     = RoPE(K_pre_h, pos) (same RoPE path as original)
  • N matrices are precomputed offline (compute_n_matrix.py) and injected
    as closed-over constants at patch time — zero runtime overhead to compute

Difference from MLA version (DeepSeek-V2-Lite)
──────────────────────────────────────────────────────────────────────────────
  ┌──────────────────────────────────────┬──────────────────────────────────────┐
  │ MLA (DeepSeek)                       │ Standard GQA (gpt-oss)               │
  ├──────────────────────────────────────┼──────────────────────────────────────┤
  │ Cache: kv_a_norm + k_pe_roped        │ Cache: V only                        │
  │ K exact via kv_b_proj  (no error)    │ K approximate via V @ N (has error)  │
  │ K split into nope+rope parts         │ K is a single head_dim vector        │
  │ RoPE applied to k_pe (pre-cache)     │ RoPE applied to K_pre AFTER cache    │
  │ kv_b_proj expansion at attn time     │ repeat_kv expansion at attn time     │
  │ No N matrices needed                 │ N: [G, head_dim, head_dim] per layer │
  └──────────────────────────────────────┴──────────────────────────────────────┘

RoPE handling — the critical change
──────────────────────────────────────────────────────────────────────────────
In MLA, k_pe was cached post-RoPE so positions were baked in at write time.
Here K is reconstructed from V at attention time, so RoPE must be applied
to the reconstructed K over the FULL accumulated context:

    full_pos_ids = [0, 1, ..., S_total-1]
    K_acc = RoPE(V_acc @ N, full_pos_ids)

This requires recomputing cos/sin for the full sequence at each decode step,
and passing full_pos_ids rather than the current-token position_ids.

Cache layout
──────────────────────────────────────────────────────────────────────────────
  VOnlyCache.value_cache[layer] = V  [B, G, S, head_dim]   (accumulated)
  VOnlyCache.key_cache          = [] (empty — K never stored)

GQA expansion
──────────────────────────────────────────────────────────────────────────────
Standard GQA (e.g. Llama 3 8B: 32 Q heads, 8 KV heads) requires expanding
reconstructed K and cached V from G heads to H heads before attention:

    K_exp = repeat_kv(K_acc, H // G)   [B, H, S, D]
    V_exp = repeat_kv(V_acc, H // G)   [B, H, S, D]

Usage
──────────────────────────────────────────────────────────────────────────────
    from compute_n_matrix import compute_all_N_matrices
    from kv_patch import patch_kv_model
    from ops.v_only_cache import VOnlyCache

    N_matrices = compute_all_N_matrices(model, out_dtype=torch.bfloat16)
    model = patch_kv_model(model, N_matrices=N_matrices)

    output = model.generate(
        **inputs,
        past_key_values=VOnlyCache(),
        use_cache=True,
    )
"""

import math
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── GQA utility ───────────────────────────────────────────────────────────────

def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand K or V from G KV heads to H query heads for GQA attention.

    Each KV head is repeated n_rep = H // G times so that every query head
    has a matching K and V to attend to.

    Args:
        hidden_states: [B, G, S, head_dim]
        n_rep:         H // G  (group size — how many Q heads share each KV head)

    Returns:
        [B, H, S, head_dim]  where H = G * n_rep
    """
    if n_rep == 1:
        # MHA or MQA with a single head — no expansion needed
        return hidden_states
    B, G, S, D = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]          # [B, G, 1, S, D]
        .expand(B, G, n_rep, S, D)               # [B, G, n_rep, S, D]
        .reshape(B, G * n_rep, S, D)             # [B, H, S, D]
    )


# ── Model-difference helpers (resolved once at patch time) ───────────────────

def _resolve_rope_fn(attn: nn.Module):
    """
    Return the module-level apply_rotary_pos_emb function for this model.
    It lives in the same Python module as the attention class.
    """
    mod = importlib.import_module(type(attn).__module__)
    fn  = getattr(mod, "apply_rotary_pos_emb", None)
    if fn is None:
        raise AttributeError(
            f"Could not find apply_rotary_pos_emb in {type(attn).__module__}."
        )
    return fn


_ROPE_ATTRS = ("rotary_emb", "rotary_pos_emb", "rope", "rotary_embedding", "rotary")


class _FunctionalRoPE(nn.Module):
    """
    Fallback RoPE module for models that compute positional embeddings
    purely functionally (no stored rotary_emb sub-module anywhere).

    Implements standard RoPE: cos/sin tables built from position_ids
    using the base frequency from model.config (rope_theta or 10000.0).

    This produces the same result as the standard HF LLaMA-style rotary_emb
    and is compatible with apply_rotary_pos_emb.
    """
    def __init__(self, head_dim: int, rope_theta: float = 10000.0, full_dim: bool = False):
        super().__init__()
        self.head_dim   = head_dim
        self.rope_theta = rope_theta
        # full_dim=True  → LLaMA/rotate_half style: cos/sin shape [1,1,S,head_dim]
        # full_dim=False → half-split style (GPT-OSS): cos/sin shape [1,1,S,head_dim//2]
        self.full_dim = full_dim
        # inv_freq: [head_dim//2] — precomputed, registered as buffer (not a param)
        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, ref: torch.Tensor, seq_len: int, **kwargs):
        """
        Returns (cos, sin) for positions 0..seq_len-1.
        ref is used only for device/dtype — same convention as HF rotary_emb.

        Shape convention:
          - LLaMA-style apply_rotary_pos_emb uses rotate_half internally and
            expects cos/sin of shape [1, 1, S, head_dim] (full, duplicated freqs).
          - GPT-OSS style splits the head in half and expects cos/sin of shape
            [1, 1, S, head_dim//2] (raw freqs only — no duplication).

        We detect which convention the model uses by inspecting the apply_rotary_pos_emb
        source: if it references 'rotate_half' we use full head_dim; otherwise half.
        Stored in self.full_dim after first resolution.
        """
        inv_freq = self.inv_freq.to(device=ref.device)
        t     = torch.arange(seq_len, device=ref.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)                # [S, head_dim//2]

        if self.full_dim:
            # LLaMA / rotate_half convention:
            # apply_rotary_pos_emb does NOT index by position_ids internally.
            # It expects cos/sin pre-broadcast as [1, 1, S, head_dim].
            emb = torch.cat([freqs, freqs], dim=-1)     # [S, head_dim]
            cos = emb.cos()[None, None, :, :]           # [1, 1, S, head_dim]
            sin = emb.sin()[None, None, :, :]
        else:
            # Half-split convention (GPT-OSS):
            # apply_rotary_pos_emb indexes cos/sin internally via position_ids:
            #   cos = cos[position_ids].unsqueeze(1)  → [B, 1, S, head_dim//2]
            # So we must return raw 2D [S, head_dim//2] — NOT [1, 1, S, head_dim//2].
            # Returning [1, 1, S, D//2] would make cos[position_ids] produce
            # a 5D tensor ([B, S, 1, S, D//2]) and break the attention matmul.
            cos = freqs.cos()                           # [S, head_dim//2]
            sin = freqs.sin()                           # [S, head_dim//2]

        return cos.to(ref.dtype), sin.to(ref.dtype)


def _resolve_rotary_emb(
    attn:     nn.Module,
    parent:   nn.Module | None = None,
    head_dim: int = 0,
    config    = None,
):
    """
    Return a callable that produces (cos, sin) for this model.

    Search order:
      1. attn itself (LLaMA/Mistral style)
      2. parent decoder layer (some custom models)
      3. model.config rotary_emb at model root (passed via config)
      4. Fallback: build a standard functional RoPE from config params
    """
    # 1. Try the attention module
    for attr in _ROPE_ATTRS:
        if hasattr(attn, attr):
            return getattr(attn, attr)

    # 2. Try the parent decoder layer
    if parent is not None:
        for attr in _ROPE_ATTRS:
            if hasattr(parent, attr):
                return getattr(parent, attr)
        for attr in _ROPE_ATTRS:
            if attr in parent._modules:
                return parent._modules[attr]

    # 3. Fallback: build standard RoPE from config
    # rope_theta defaults to 10000.0 (standard RoPE); many models override it.
    rope_theta = 10000.0
    if config is not None:
        rope_theta = float(getattr(config, "rope_theta", rope_theta))
    if head_dim <= 0:
        raise ValueError("head_dim must be > 0 to build fallback RoPE")

    # Detect cos/sin shape convention from the model's apply_rotary_pos_emb source.
    # LLaMA/rotate_half style: function body calls rotate_half() → expects full head_dim.
    # GPT-OSS/half-split style: splits tensor in two → expects head_dim//2.
    full_dim = False
    try:
        import inspect
        mod = importlib.import_module(type(attn).__module__)
        fn  = getattr(mod, "apply_rotary_pos_emb", None)
        if fn is not None and "rotate_half" in inspect.getsource(fn):
            full_dim = True
    except Exception:
        pass  # if inspection fails, default to half-split (GPT-OSS style)

    if not _resolve_rotary_emb._warned:
        print(
            f"  [kv_patch] No rotary_emb module found on attention or decoder layer. "
            f"Using _FunctionalRoPE(head_dim={head_dim}, rope_theta={rope_theta}, "
            f"full_dim={full_dim}) for all layers."
        )
        _resolve_rotary_emb._warned = True
    return _FunctionalRoPE(head_dim=head_dim, rope_theta=rope_theta, full_dim=full_dim)

_resolve_rotary_emb._warned = False


def _get_cos_sin(
    rotary_emb_module,
    ref: torch.Tensor,
    seq_len: int,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Call the rotary embedding module with whatever signature it expects.

    Tries the most common calling conventions in order:
      1. rotary_emb(ref, seq_len=N)       — LLaMA 2 / Mistral style
      2. rotary_emb(ref, position_ids)    — some custom models
      3. rotary_emb(seq_len=N)            — no ref tensor required
      4. rotary_emb(position_ids)         — position-ids only style
    """
    # Convention 1: standard HF Llama/Mistral rotary_emb(value, seq_len=N)
    try:
        return rotary_emb_module(ref, seq_len=seq_len)
    except TypeError:
        pass
    # Convention 2: rotary_emb(ref, position_ids)
    try:
        return rotary_emb_module(ref, position_ids)
    except TypeError:
        pass
    # Convention 3: no ref tensor needed
    try:
        return rotary_emb_module(seq_len=seq_len)
    except TypeError:
        pass
    # Convention 4: position_ids only
    try:
        result = rotary_emb_module(position_ids)
        # Some models return (cos, sin), others return a single tensor
        if isinstance(result, (tuple, list)) and len(result) == 2:
            return result
        raise TypeError("unexpected return type")
    except TypeError:
        pass
    raise RuntimeError(
        f"Could not call {type(rotary_emb_module).__name__} with any known signature.\n"
        f"Inspect the module and update _get_cos_sin() with the correct calling convention."
    )


def _resolve_scale(attn: nn.Module) -> float:
    """
    Return the attention softmax scale.
    Standard GQA: 1/sqrt(head_dim). Some models store a custom value.
    """
    for attr in ("softmax_scale", "scaling", "scale"):
        if hasattr(attn, attr):
            return float(getattr(attn, attr))
    # Compute from head_dim
    if hasattr(attn, "head_dim"):
        return 1.0 / math.sqrt(attn.head_dim)
    # Last resort: infer from q_proj output
    num_heads = int(getattr(attn, "num_heads", getattr(attn, "num_attention_heads", 1)))
    head_dim  = attn.q_proj.weight.shape[0] // num_heads
    return 1.0 / math.sqrt(head_dim)


def _resolve_dims(attn: nn.Module) -> tuple[int, int, int]:
    """
    Return (num_heads, num_kv_heads, head_dim) for the attention module.

    num_heads:    number of query heads (H)
    num_kv_heads: number of KV heads (G), G ≤ H
    head_dim:     per-head projection dimension (D)
    """
    # ── Query heads ───────────────────────────────────────────────────────────
    _Q_ATTRS = (
        "num_heads",
        "num_attention_heads",
        "num_q_heads",
        "num_query_heads",
        "n_heads",
        "n_attention_heads",
    )
    num_heads = None
    for attr in _Q_ATTRS:
        if hasattr(attn, attr):
            num_heads = int(getattr(attn, attr))
            break
    if num_heads is None and hasattr(attn, "config"):
        for attr in _Q_ATTRS:
            if hasattr(attn.config, attr):
                num_heads = int(getattr(attn.config, attr))
                break
    if num_heads is None:
        # Infer from q_proj output dimension and head_dim (if available)
        if hasattr(attn, "head_dim") and hasattr(attn, "q_proj"):
            num_heads = attn.q_proj.weight.shape[0] // int(attn.head_dim)
        else:
            numeric_attrs = {
                k: getattr(attn, k)
                for k in dir(attn)
                if not k.startswith("_") and isinstance(getattr(attn, k, None), int)
            }
            raise AttributeError(
                f"Cannot find num_heads on {type(attn).__name__}.\n"
                f"Available int attributes: {numeric_attrs}\n"
                f"Add the correct attribute name to _Q_ATTRS in _resolve_dims()."
            )

    # ── KV heads ──────────────────────────────────────────────────────────────
    _KV_ATTRS = (
        "num_key_value_heads",
        "num_kv_heads",
        "num_kv_attention_heads",
        "kv_heads",
        "n_kv_heads",
        "num_query_groups",   # some models use "query groups" for GQA groups
    )
    num_kv_heads = None
    for attr in _KV_ATTRS:
        if hasattr(attn, attr):
            num_kv_heads = int(getattr(attn, attr))
            break
    if num_kv_heads is None and hasattr(attn, "config"):
        for attr in _KV_ATTRS:
            if hasattr(attn.config, attr):
                num_kv_heads = int(getattr(attn.config, attr))
                break
    if num_kv_heads is None:
        num_kv_heads = num_heads  # fallback: treat as MHA (no GQA)

    # ── Head dimension ────────────────────────────────────────────────────────
    if hasattr(attn, "head_dim"):
        head_dim = int(attn.head_dim)
    elif hasattr(attn, "q_proj"):
        head_dim = attn.q_proj.weight.shape[0] // num_heads
    else:
        raise AttributeError(
            f"Cannot determine head_dim on {type(attn).__name__}. "
            f"Expected 'head_dim' attribute or 'q_proj' Linear layer."
        )

    return num_heads, num_kv_heads, head_dim


# ── K reconstruction ──────────────────────────────────────────────────────────

def _reconstruct_k_pre(
    v_acc: torch.Tensor,
    N_layer: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct the pre-RoPE K from accumulated V using per-head N matrices.

    The approximation per head h:
        K_pre_h  ≈  V_h @ N_h

    Vectorised over all KV heads via einsum:
        K_pre[b, g, s, d]  =  sum_i  V[b, g, s, i] * N[g, i, d]

    Args:
        v_acc:   [B, G, S, head_dim]  accumulated V from VOnlyCache
        N_layer: [G, head_dim, head_dim]  precomputed N for this layer

    Returns:
        K_pre: [B, G, S, head_dim]  reconstructed pre-RoPE K
    """
    # Cast to float32 for the matmul to avoid BF16 precision loss in N,
    # then cast back to match v_acc dtype.
    return torch.einsum(
        "bgsi,gid->bgsd",
        v_acc.float(),
        N_layer.float(),
    ).to(v_acc.dtype)


# ── Per-layer patching ────────────────────────────────────────────────────────

def _patch_attention_forward(
    attn: nn.Module,
    N_layer: torch.Tensor,
    parent: nn.Module | None = None,
    model_config = None,
    debug: bool = False,
) -> None:
    """
    Replace attn.forward with a V-only-cache forward for standard GQA.

    All model-specific values are resolved once here and captured in the
    closure — zero attribute lookup overhead at inference time.

    Args:
        attn:         GptOssAttention (or any standard GQA attention module)
        N_layer:      [G, head_dim, head_dim]  precomputed N for this layer
        parent:       parent decoder layer — needed when rotary_emb lives there
        model_config: model.config — used as fallback to build functional RoPE
        debug:        print resolved dims/scale on the first layer for sanity check
    """
    _apply_rope  = _resolve_rope_fn(attn)
    _scale       = _resolve_scale(attn)
    _num_heads, _num_kv_heads, _head_dim = _resolve_dims(attn)
    _rotary_emb  = _resolve_rotary_emb(
        attn, parent=parent, head_dim=_head_dim, config=model_config
    )
    _n_rep       = _num_heads // _num_kv_heads   # group size for repeat_kv
    _layer_idx   = getattr(attn, "layer_idx", None)

    # N_layer is a constant for this layer — captured in closure, not recomputed
    _N = N_layer  # [G, head_dim, head_dim]

    if debug:
        print(f"  [patch debug] layer_idx={_layer_idx}:")
        print(f"    num_heads={_num_heads}  num_kv_heads={_num_kv_heads}  "
              f"head_dim={_head_dim}  n_rep={_n_rep}  scale={_scale:.6f}")
        print(f"    N shape={tuple(_N.shape)}  dtype={_N.dtype}")

    def patched_forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        past_key_value=None,    # some model versions use singular form
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        # Normalise cache — accept both singular and plural forms
        cache = past_key_values if past_key_values is not None else past_key_value
        B, S_q, _ = hidden_states.shape
        device = hidden_states.device

        # ── Step 1: Q projection ──────────────────────────────────────────────
        # Standard single-stage q_proj (gpt-oss does not use two-stage Q like MLA).
        q = attn.q_proj(hidden_states)                          # [B, S_q, H*D]
        q = q.view(B, S_q, _num_heads, _head_dim).transpose(1, 2)  # [B, H, S_q, D]

        # ── Step 2: V projection for current tokens only ──────────────────────
        # We compute V for the new tokens and immediately write to cache.
        # K is NOT computed here — it will be reconstructed from V later.
        v = attn.v_proj(hidden_states)                              # [B, S_q, G*D]
        v = v.view(B, S_q, _num_kv_heads, _head_dim).transpose(1, 2)  # [B, G, S_q, D]

        # ── Step 3: Determine full context length for RoPE ────────────────────
        # RoPE must be computed for positions 0..S_total-1 so that the
        # reconstructed K gets the correct rotation for every cached position.
        S_cache = 0
        if cache is not None and _layer_idx is not None:
            S_cache = cache.get_seq_length(_layer_idx)
        S_total = S_cache + S_q

        # ── Step 4: Compute cos/sin for full context length ───────────────────
        # _rotary_emb is resolved once at patch time via _resolve_rotary_emb().
        # We must pass S_total (not just S_q) so that indices for all cached
        # positions are available when we apply RoPE to the reconstructed K.
        # _get_cos_sin tries multiple calling conventions automatically.
        cos, sin = _get_cos_sin(_rotary_emb, v, S_total, position_ids)

        # ── Step 5: Apply RoPE to Q (current token positions only) ───────────
        # position_ids here is [B, S_q] — the positions of the new tokens.
        # We call apply_rotary_pos_emb with Q twice and discard the second
        # output (there is no pre-cache K to rotate at this stage).
        q_roped, _ = _apply_rope(q, q, cos, sin, position_ids)
        # q_roped: [B, H, S_q, D]

        # ── Step 6: Cache V, get full accumulated V ───────────────────────────
        # VOnlyCache.update() appends v along the sequence dimension and
        # returns the full accumulated tensor [B, G, S_total, D].
        # K is never written to the cache.
        if cache is not None:
            v_acc = cache.update(v, _layer_idx)   # [B, G, S_total, D]
        else:
            v_acc = v                              # prefill without cache

        # ── Step 7: Reconstruct pre-RoPE K from accumulated V ─────────────────
        # K_pre_h ≈ V_h @ N_h  for each KV head h.
        # This is the approximation step — quality depends on subspace alignment
        # between col(W_K_h) and col(W_V_h) (measured in Test 1 and Test 2).
        K_pre_acc = _reconstruct_k_pre(v_acc, _N)  # [B, G, S_total, D]

        # ── Step 8: Apply RoPE to reconstructed K for ALL accumulated positions ──
        # This is the key difference from the original forward:
        #   - Original: RoPE is applied to K for current tokens only, then cached
        #   - Here:     RoPE is applied to reconstructed K for ALL positions
        #               (0..S_total-1) at every decode step
        #
        # full_pos_ids covers positions for the entire accumulated context,
        # NOT just the current token. This is necessary because K_pre_acc
        # spans all S_total positions and each needs its own rotation angle.
        full_pos_ids = (
            torch.arange(S_total, device=device)
            .unsqueeze(0)
            .expand(B, -1)   # [B, S_total]
        )
        _, K_acc = _apply_rope(K_pre_acc, K_pre_acc, cos, sin, full_pos_ids)
        # K_acc: [B, G, S_total, D]
        # (We pass K_pre_acc as both q and k and discard the first output.)

        # ── Step 9: GQA head expansion ────────────────────────────────────────
        # Each KV head is shared by n_rep = H // G query heads.
        # Expand K and V from G heads to H heads so the attention matmul shapes align.
        K_exp = _repeat_kv(K_acc, _n_rep)   # [B, H, S_total, D]
        V_exp = _repeat_kv(v_acc, _n_rep)   # [B, H, S_total, D]

        # ── Step 10: Scaled dot-product attention ─────────────────────────────
        # Manual matmul + float32 softmax, matching the original model's
        # numerical behaviour. Using F.scaled_dot_product_attention would
        # give different BF16 numerics due to its fused kernel precision.
        attn_weights = torch.matmul(q_roped, K_exp.transpose(2, 3)) * _scale
        # attn_weights: [B, H, S_q, S_total]

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q_roped.dtype)

        attn_output = torch.matmul(attn_weights, V_exp)
        # attn_output: [B, H, S_q, D]

        # ── Step 11: Output projection ────────────────────────────────────────
        attn_output = (
            attn_output
            .transpose(1, 2)                     # [B, S_q, H, D]
            .reshape(B, S_q, _num_heads * _head_dim)  # [B, S_q, H*D]
            .contiguous()
        )
        attn_output = attn.o_proj(attn_output)   # [B, S_q, d_model]

        return attn_output, None, cache

    attn.forward = patched_forward


# ── Public API ────────────────────────────────────────────────────────────────

def patch_kv_model(
    model: nn.Module,
    N_matrices: list[torch.Tensor],
    device: torch.device | None = None,
) -> nn.Module:
    """
    Patch all decoder layers in a GQA model to use VOnlyCache.

    For each layer, replaces attn.forward with a closure that:
      1. Projects V for the current token and caches it in VOnlyCache
      2. Reconstructs K_pre from accumulated V via K_pre_h = V_h @ N_h
      3. Applies RoPE to Q (current positions) and K_pre (all positions)
      4. Expands K and V from G KV heads to H query heads (repeat_kv)
      5. Runs standard scaled dot-product attention

    K is never stored in the cache. The 50% memory saving comes entirely
    from eliminating the K cache.

    Args:
        model:      GQA causal LM (e.g. gpt-oss-20b-BF16)
        N_matrices: List of N tensors from compute_all_N_matrices().
                    One per layer: [num_kv_heads, head_dim, head_dim].
        device:     Target device (defaults to model's current device).

    Returns:
        The patched model (modified in-place).

    Example:
        N_matrices = compute_all_N_matrices(model, out_dtype=torch.bfloat16)
        model = patch_kv_model(model, N_matrices=N_matrices)
        output = model.generate(
            **inputs,
            past_key_values=VOnlyCache(),
            use_cache=True,
        )
    """
    if device is None:
        device = next(model.parameters()).device

    layers = model.model.layers

    if len(N_matrices) != len(layers):
        raise ValueError(
            f"N_matrices length mismatch: got {len(N_matrices)}, "
            f"expected {len(layers)} (one per decoder layer)."
        )

    print(f"Patching {len(layers)} layers for V-only KV cache (K = V @ N)...")

    for i, layer in enumerate(layers):
        # Move this layer's N to the correct device — compute_n_matrix.py may
        # have placed it on CPU if device was not specified there.
        N_layer = N_matrices[i].to(device=device, non_blocking=True)

        _patch_attention_forward(
            layer.self_attn,
            N_layer=N_layer,
            parent=layer,              # pass decoder layer so rotary_emb can be found there
            model_config=getattr(model, "config", None),  # fallback for functional RoPE
            debug=(i == 0),            # print resolved dims for first layer only
        )

        if i == 0:
            print(f"  Layer 0: patched  (V cached, K reconstructed via K_pre = V @ N)")

    print(f"Done. {len(layers)} layers patched.")
    print()
    print("Usage:")
    print("  from ops.v_only_cache import VOnlyCache")
    print("  output = model.generate(**inputs, past_key_values=VOnlyCache(), use_cache=True)")

    return model