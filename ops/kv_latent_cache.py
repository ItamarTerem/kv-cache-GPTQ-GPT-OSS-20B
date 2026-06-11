"""
kv_latent_cache.py — KV cache storing (kv_a_norm, k_pe_roped) per layer.

Target model: DeepSeek-V2-Lite (deepseek-ai/DeepSeek-V2-Lite)

Why this is better than the v@N approach
──────────────────────────────────────────────────────────────────────────────
The previous KVNopelessCache tried to reconstruct k_nope from v via a
static matrix N = pinv(W_V.T) @ W_K.T. This is fundamentally approximate:
v = kv_a_norm @ W_V.T maps 512D → 128D, permanently discarding 384 dimensions.
k_nope depends on those discarded dimensions too, so the reconstruction
is lossy regardless of precision.

This cache instead stores kv_a_norm — the shared latent vector that BOTH
k_nope and v are derived from. Reconstruction is then exact:

    kv      = kv_b_proj(kv_a_norm_acc)
    k_nope  = kv[..., :qk_nope_head_dim]
    v       = kv[..., qk_nope_head_dim:]

Memory comparison (per token, per layer, BF16, batch=1):
──────────────────────────────────────────────────────────────────────────────
  Standard DynamicCache:  k_full [16 × 192] + v [16 × 128]  = 5120 dims
  KVLatentCache (this):   kv_a_norm [512]   + k_pe [64]     =  576 dims
  Saving: ~8.9×

Storage layout:
  key_cache[layer]   = kv_a_norm:    [B, 1, S, kv_lora_rank]
  value_cache[layer] = k_pe_roped:   [B, 1, S, qk_rope_head_dim]

Both use 1 "head" — kv_a_norm is shared across all KV heads, and k_pe is
broadcast to all heads at attention time. This fits DynamicCache's (key, value)
storage without modification.

The patched attention forward (kv_patch.py) is responsible for:
  1. Writing  (kv_a_norm, k_pe_roped) to this cache
  2. Reading  (kv_a_norm_acc, k_pe_acc) from this cache
  3. Computing kv = kv_b_proj(kv_a_norm_acc) → splitting k_nope_acc, v_acc
  4. Reconstructing k = cat([k_nope_acc, k_pe_acc.expand(num_heads)])
"""

import torch
from transformers import DynamicCache


class KVLatentCache(DynamicCache):
    """
    DynamicCache variant storing (kv_a_norm, k_pe_roped) instead of (k_full, v).

    key_cache[i]   — kv_a_norm:   [B, 1, seq_len, kv_lora_rank]
    value_cache[i] — k_pe_roped:  [B, 1, seq_len, qk_rope_head_dim]

    k_nope and v are recomputed exactly at attention time via kv_b_proj.
    """

    def __init__(self) -> None:
        super().__init__()
        # Always initialise our own lists — transformers 4.47+ refactored
        # DynamicCache to use a different internal representation, so we can
        # no longer rely on the parent's key_cache/value_cache attributes.
        # Overriding update() and get_seq_length() below ensures our lists
        # are always the source of truth.
        self.key_cache: list   = []
        self.value_cache: list = []

    # ── Core update (overrides DynamicCache) ─────────────────────────────────

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        """
        Concatenate (kv_a_norm, k_pe_roped) along the sequence dimension
        and return the accumulated tensors for this layer.

        We manage our own key_cache / value_cache lists directly so that
        get_seq_length() and the diagnostic helpers always see the real data,
        regardless of which internal representation the parent DynamicCache
        version uses.
        """
        # Grow the lists to cover layer_idx
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.tensor([]))
            self.value_cache.append(torch.tensor([]))

        if self.key_cache[layer_idx].numel() == 0:
            self.key_cache[layer_idx]   = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=-2
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    # ── Sequence-length query (overrides DynamicCache) ────────────────────────

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for the given layer."""
        if layer_idx >= len(self.key_cache):
            return 0
        t = self.key_cache[layer_idx]
        if t is None or t.numel() == 0:
            return 0
        return t.shape[-2]

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_dynamic_cache(cls, cache: DynamicCache) -> "KVLatentCache":
        """
        Wrap an existing DynamicCache as a KVLatentCache.
        Only meaningful at the start of a new generation.
        """
        new = cls()
        new.key_cache   = cache.key_cache
        new.value_cache = cache.value_cache
        return new

    # ── Compatibility shim ────────────────────────────────────────────────────

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Alias for get_seq_length() — compatibility with transformers < 4.40."""
        return self.get_seq_length(layer_idx)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def latent_dim(self, layer_idx: int = 0) -> int | None:
        """Return the stored kv_a_norm dimension (kv_lora_rank). Should be 512."""
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[-1]
        return None

    def kpe_dim(self, layer_idx: int = 0) -> int | None:
        """Return the stored k_pe dimension (qk_rope_head_dim). Should be 64."""
        if layer_idx < len(self.value_cache) and self.value_cache[layer_idx] is not None:
            return self.value_cache[layer_idx].shape[-1]
        return None

    def cache_size_bytes(self) -> dict[str, int]:
        """Return current memory footprint in bytes."""
        latent_bytes = sum(t.nbytes for t in self.key_cache   if t is not None)
        kpe_bytes    = sum(t.nbytes for t in self.value_cache if t is not None)
        return {
            "latent_bytes": latent_bytes,
            "kpe_bytes":    kpe_bytes,
            "total_bytes":  latent_bytes + kpe_bytes,
        }

    def report(self) -> str:
        """Human-readable summary of cache contents."""
        n = len([t for t in self.key_cache if t is not None])
        lines = [f"KVLatentCache | {n} layers populated"]
        if n > 0:
            idx = next(i for i, t in enumerate(self.key_cache) if t is not None)
            lat_shape = tuple(self.key_cache[idx].shape)
            kpe_shape = tuple(self.value_cache[idx].shape)
            sizes = self.cache_size_bytes()
            lines += [
                f"  kv_a_norm : {lat_shape}  dtype={self.key_cache[idx].dtype}",
                f"  k_pe      : {kpe_shape}  dtype={self.value_cache[idx].dtype}",
                f"  latent mem: {sizes['latent_bytes'] / 1e6:7.2f} MB",
                f"  k_pe mem  : {sizes['kpe_bytes']    / 1e6:7.2f} MB",
                f"  total mem : {sizes['total_bytes']  / 1e6:7.2f} MB",
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.report()


# ── Comparison helper ─────────────────────────────────────────────────────────

def compare_cache_sizes(
    latent: KVLatentCache,
    standard: DynamicCache,
) -> None:
    """Print a side-by-side memory comparison."""
    latent_sizes  = latent.cache_size_bytes()
    standard_key  = sum(t.nbytes for t in standard.key_cache   if t is not None)
    standard_val  = sum(t.nbytes for t in standard.value_cache if t is not None)
    standard_total = standard_key + standard_val

    latent_total = latent_sizes["total_bytes"]
    saving_pct   = 100.0 * (1.0 - latent_total / standard_total) if standard_total else 0.0

    print("── Cache size comparison ───────────────────────────────────────")
    print(f"  Standard cache  (k_full + v):       {standard_total / 1e6:.2f} MB")
    print(f"  Latent cache    (kv_a_norm + k_pe): {latent_total   / 1e6:.2f} MB")
    print(f"  Saving                            : {saving_pct:.1f}%")
