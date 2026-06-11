"""
kv_nopeless_cache.py — KV cache that omits k_nope, storing only (k_pe, v).

Target model: DeepSeek-V2-Lite (deepseek-ai/DeepSeek-V2-Lite)

Why this class exists
──────────────────────────────────────────────────────────────────────────────
HuggingFace's DynamicCache stores full key tensors:

    k_full  =  cat([k_nope, k_pe], dim=-1)
               └── non-RoPE part ─┘└─ RoPE part ─┘

In MLA attention, k_nope can always be reconstructed from v:

    k_nope  =  v  @  N          (N computed offline once per layer)

So there is no reason to keep k_nope in the cache across decode steps.
This class stores only (k_pe, v), cutting the key-side cache memory by:

    saved = qk_nope_head_dim / (qk_nope_head_dim + qk_rope_head_dim)

For DeepSeek-V2-Lite (qk_nope=128, qk_rope=64):
    saved = 128 / 192  ≈  67% of the key cache

The patched attention forward (kv_patch.py) is responsible for:
    1. Writing  (k_pe, v)    to this cache   — NOT k_full
    2. Reading  (k_pe, v)    from this cache
    3. Recomputing  k_nope = v @ N
    4. Reconstructing  k = cat([k_nope, k_pe], dim=-1)  before attention

Interface
──────────────────────────────────────────────────────────────────────────────
Identical to DynamicCache: .update(), .get_seq_length(), etc.
HF's generate() loop passes past_key_values through without inspecting tensor
shapes inside it, so this is a drop-in replacement.
"""

import torch
from transformers import DynamicCache


class KVNopelessCache(DynamicCache):
    """
    DynamicCache variant that stores (k_pe, v) instead of (k_full, v).

    The key tensors stored here have shape:
        [batch, num_kv_heads, seq_len, qk_rope_head_dim]   ← k_pe only

    instead of the standard:
        [batch, num_kv_heads, seq_len, qk_nope_head_dim + qk_rope_head_dim]

    v tensors are unchanged:
        [batch, num_kv_heads, seq_len, v_head_dim]

    Usage (from within the patched attention forward):

        # at write time — pass k_pe, not k_full:
        k_pe_acc, v_acc = past_key_values.update(
            k_pe, v, self.layer_idx, cache_kwargs
        )

        # at read time — reconstruct k_nope then k:
        k_nope = torch.matmul(v_acc, self.kv_relation_N)
        k_rope = k_pe_acc.expand(-1, num_heads, -1, -1)
        k = torch.cat([k_nope, k_rope], dim=-1)
    """

    def __init__(self) -> None:
        super().__init__()
        # Ensure these exist immediately — newer transformers versions only
        # create them lazily on the first update() call, which causes
        # AttributeError when diagnostic methods are called on an empty cache.
        if not hasattr(self, "key_cache"):
            self.key_cache: list = []
        if not hasattr(self, "value_cache"):
            self.value_cache: list = []

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_dynamic_cache(cls, cache: DynamicCache) -> "KVNopelessCache":
        """
        Wrap an existing DynamicCache as a KVNopelessCache.
        Useful when HF initialises the cache before the first forward pass.

        Note: the existing cache contents (if any) are NOT converted —
        this is only meaningful at the start of a new generation.
        """
        new = cls()
        new.key_cache   = cache.key_cache
        new.value_cache = cache.value_cache
        return new

    # ── Core update — unchanged from DynamicCache ─────────────────────────────
    #
    # DynamicCache.update() concatenates whatever tensors you give it along the
    # sequence dimension. Since we pass k_pe (not k_full), it accumulates k_pe.
    # No override is needed — the base implementation does exactly what we want.
    #
    # Keeping this comment block here so it is obvious to future readers that
    # the omission of an override() is intentional, not an oversight.

    # ── Compatibility shim ────────────────────────────────────────────────────
    #
    # DeepSeek-V2-Lite with older transformers versions calls
    # past_key_values.get_usable_length(seq_len) instead of get_seq_length().
    # DynamicCache added this method in transformers >= 4.40; older installs
    # lack it.

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """
        Return the number of cached tokens usable for the given layer.
        Alias for get_seq_length() — added for compatibility with
        transformers < 4.40.
        """
        return self.get_seq_length(layer_idx)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def key_head_dim(self, layer_idx: int = 0) -> int | None:
        """
        Return the stored key head dimension for a given layer.
        For a correctly used KVNopelessCache this equals qk_rope_head_dim (64),
        NOT qk_nope_head_dim + qk_rope_head_dim (192).
        Returns None if the layer has not been populated yet.
        """
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[-1]
        return None

    def cache_size_bytes(self) -> dict[str, int]:
        """
        Return the current cache memory footprint in bytes.

        Returns a dict with:
            'key_bytes'   — bytes used by all stored k_pe tensors
            'value_bytes' — bytes used by all stored v tensors
            'total_bytes' — sum
        """
        key_bytes   = sum(t.nbytes for t in self.key_cache   if t is not None)
        value_bytes = sum(t.nbytes for t in self.value_cache if t is not None)
        return {
            "key_bytes":   key_bytes,
            "value_bytes": value_bytes,
            "total_bytes": key_bytes + value_bytes,
        }

    def report(self) -> str:
        """
        Human-readable summary of what is stored in the cache.

        Example output:
            KVNopelessCache | 27 layers populated
              key   (k_pe only):  [1, 1, 128, 64]   dtype=bfloat16
              value:              [1, 1, 128, 128]  dtype=bfloat16
              key memory  :   0.50 MB
              value memory:   1.00 MB
              total memory:   1.50 MB
        """
        n = len([t for t in self.key_cache if t is not None])
        lines = [f"KVNopelessCache | {n} layers populated"]

        if n > 0:
            idx = next(i for i, t in enumerate(self.key_cache) if t is not None)
            k_shape = tuple(self.key_cache[idx].shape)
            v_shape = tuple(self.value_cache[idx].shape)
            k_dtype = self.key_cache[idx].dtype
            v_dtype = self.value_cache[idx].dtype

            sizes = self.cache_size_bytes()
            lines += [
                f"  key   (k_pe only):  {k_shape}  dtype={k_dtype}",
                f"  value:              {v_shape}  dtype={v_dtype}",
                f"  key memory  : {sizes['key_bytes']   / 1e6:7.2f} MB",
                f"  value memory: {sizes['value_bytes'] / 1e6:7.2f} MB",
                f"  total memory: {sizes['total_bytes'] / 1e6:7.2f} MB",
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.report()


# ── Comparison helper ─────────────────────────────────────────────────────────

def compare_cache_sizes(
    nopeless: KVNopelessCache,
    standard: DynamicCache,
) -> None:
    """
    Print a side-by-side comparison of nopeless vs standard cache memory.
    """
    nopeless_sizes = nopeless.cache_size_bytes()
    standard_key   = sum(t.nbytes for t in standard.key_cache   if t is not None)
    standard_val   = sum(t.nbytes for t in standard.value_cache if t is not None)
    standard_total = standard_key + standard_val

    nopeless_total = nopeless_sizes["total_bytes"]
    saving_pct     = 100.0 * (1.0 - nopeless_total / standard_total) if standard_total else 0.0

    print("── Cache size comparison ───────────────────────────────────────")
    print(f"  Standard cache  (k_full + v): {standard_total / 1e6:.2f} MB")
    print(f"  Nopeless cache  (k_pe   + v): {nopeless_total / 1e6:.2f} MB")
    print(f"  Saving                      : {saving_pct:.1f}%")
    print(f"  (key only: {standard_key/1e6:.2f} MB → {nopeless_sizes['key_bytes']/1e6:.2f} MB)")
