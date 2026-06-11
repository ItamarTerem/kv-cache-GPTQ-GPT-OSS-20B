"""
kv_relation_module.py — Runtime module for k_nope reconstruction.

Role in the pipeline
──────────────────────────────────────────────────────────────────────────────
compute_n_matrix.py   →  computes N offline (FP64 pinv, returned in BF16)
kv_relation_module.py →  holds N as a buffer, exposes forward(v) → k_nope
kv_patch.py           →  attaches one KVRelationModule per layer, patches forward

Target model: DeepSeek-V2-Lite (deepseek-ai/DeepSeek-V2-Lite)

What this module does at runtime
──────────────────────────────────────────────────────────────────────────────
Given the cached value tensor v and the pre-computed static matrix N:

    k_nope  =  v  @  N

Shapes:
    v      [batch, num_kv_heads, seq_len, v_head_dim]
    N      [num_kv_heads, v_head_dim, qk_nope_head_dim]      ← registered buffer
    output [batch, num_kv_heads, seq_len, qk_nope_head_dim]

torch.matmul broadcasts N's head dimension over the batch dimension:
    [B, H, S, v_dim]  @  [H, v_dim, k_dim]  →  [B, H, S, k_dim]

N is registered as a buffer (not a parameter) — it is static, never updated
by an optimiser, but moves with the module when .to(device) or .cuda() is called.
"""

import torch
import torch.nn as nn


class KVRelationModule(nn.Module):
    """
    Holds the static KV relation matrix N and reconstructs k_nope from v.

    One instance is attached to each attention layer by kv_patch.py as:
        attn.kv_relation = KVRelationModule(N)

    Forward call (inside the patched attention forward):
        k_nope = attn.kv_relation(v_cached)
    """

    def __init__(self, N: torch.Tensor) -> None:
        """
        Args:
            N: KV relation matrix, shape [num_kv_heads, v_head_dim, qk_nope_head_dim].
               Produced by compute_n_matrix.compute_N_matrix().
               Typically BF16; moves to device automatically with the module.
        """
        super().__init__()

        if N.dim() != 3:
            raise ValueError(
                f"N must be 3-D [num_kv_heads, v_head_dim, qk_nope_head_dim], "
                f"got shape {tuple(N.shape)}"
            )

        # Register as buffer: persistent (saved in state_dict), non-trainable,
        # moves to device automatically when module.to(device) is called.
        self.register_buffer("N", N)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_kv_heads(self) -> int:
        return self.N.shape[0]

    @property
    def v_head_dim(self) -> int:
        return self.N.shape[1]

    @property
    def qk_nope_head_dim(self) -> int:
        return self.N.shape[2]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct k_nope from v using the static relation k_nope = v @ N.

        Args:
            v: value tensor, shape [batch, num_kv_heads, seq_len, v_head_dim].
               Typically the accumulated v from KVNopelessCache.

        Returns:
            k_nope: shape [batch, num_kv_heads, seq_len, qk_nope_head_dim].

        Matmul broadcast:
            v  [B, H, S, v_dim]  @  N  [H, v_dim, k_dim]  →  [B, H, S, k_dim]
            N's H dimension broadcasts over B automatically.
        """
        # Upcast to FP32 for the matmul — N is FP32, and doing the dot product
        # in FP32 avoids BF16 rounding accumulation over the v_head_dim elements.
        # Cast result back to v's original dtype before returning.
        return torch.matmul(v.float(), self.N).to(v.dtype)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def reconstruction_error(
        self,
        v: torch.Tensor,
        k_nope_ref: torch.Tensor,
    ) -> dict[str, float]:
        """
        Measure how accurately k_nope = v @ N reconstructs the reference k_nope.

        Useful during verification to confirm N is well-conditioned and the
        reconstruction is numerically correct.

        Args:
            v:          value tensor  [batch, heads, seq, v_head_dim]
            k_nope_ref: reference k_nope computed by the original projection
                        [batch, heads, seq, qk_nope_head_dim]

        Returns:
            dict with:
                'max_abs_diff'  — max absolute difference over all elements
                'mean_abs_diff' — mean absolute difference
                'top1_match'    — True if argmax over last dim matches everywhere
        """
        with torch.no_grad():
            k_nope_hat = self.forward(v)
            diff       = (k_nope_hat - k_nope_ref).abs()

        return {
            "max_abs_diff":  diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "top1_match":    (
                k_nope_hat.argmax(dim=-1) == k_nope_ref.argmax(dim=-1)
            ).all().item(),
        }

    def extra_repr(self) -> str:
        return (
            f"num_kv_heads={self.num_kv_heads}, "
            f"v_head_dim={self.v_head_dim}, "
            f"qk_nope_head_dim={self.qk_nope_head_dim}, "
            f"dtype={self.N.dtype}"
        )
