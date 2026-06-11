"""
v_only_cache.py — KV cache that stores only V, never K.

Background
──────────────────────────────────────────────────────────────────────────────
In standard GQA with the K = V @ N approximation, K is never cached.
Instead, at each attention step:

    K_pre = V_accumulated @ N_h   (per KV head, pre-RoPE)
    K     = RoPE(K_pre, positions)

Only V accumulates across decode steps. This cache holds exactly that —
a list of V tensors (one per decoder layer), grown by one token each step.

Cache layout
──────────────────────────────────────────────────────────────────────────────
    value_cache[layer_idx]  =  V  [B, G, S, head_dim]   dtype = model dtype
    key_cache               =  []  (always empty — K is never stored)

    B        = batch size
    G        = num_kv_heads (e.g. 8 for Llama 3 8B)
    S        = accumulated sequence length (grows by 1 each decode step)
    head_dim = per-head dimension (e.g. 128)

Relationship to DynamicCache
──────────────────────────────────────────────────────────────────────────────
DynamicCache stores both key_cache and value_cache. VOnlyCache is the same
structure but with key_cache permanently empty. The update() signature is
different: it takes only v (not k, v) and returns only v_accumulated.

Usage
──────────────────────────────────────────────────────────────────────────────
    from ops.v_only_cache import VOnlyCache

    past_kv = VOnlyCache()
    output  = model.generate(**inputs, past_key_values=past_kv, use_cache=True)

    # Inspect after generation:
    print(past_kv.cache_size_bytes())        # {"total_bytes": N, "total_mb": N}
    print(past_kv.get_seq_length(layer_idx=0))  # accumulated sequence length
"""

import torch
from transformers import DynamicCache


class VOnlyCache(DynamicCache):
    """
    KV cache that stores only V tensors (K is never written).

    Inherits DynamicCache so the model's cache-handling code paths
    (past_key_values.update(), get_seq_length(), etc.) work without
    modification. The key difference: update() accepts only v, and
    key_cache is always an empty list.

    Why inherit DynamicCache?
        The HuggingFace generation loop calls methods like get_seq_length()
        and get_usable_length() on the cache object. Inheriting gives us
        those for free and ensures compatibility with any model that checks
        isinstance(cache, DynamicCache).
    """

    def __init__(self) -> None:
        super().__init__()
        # key_cache is inherited from DynamicCache but we never write to it.
        # Explicitly reset to empty to be safe (parent __init__ sets it to []).
        self.key_cache:   list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    # ── Core write/read ───────────────────────────────────────────────────────

    def update(
        self,
        v: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Append the current step's V to the cache and return accumulated V.

        Called once per layer per forward step. On the first call for a layer
        (prefill), the full prompt V is stored. On subsequent calls (decode),
        one token's V is appended.

        Args:
            v:         [B, G, S_new, head_dim]  — V for new token(s)
            layer_idx: which decoder layer this V belongs to

        Returns:
            v_acc: [B, G, S_total, head_dim]  — all V seen so far for this layer

        Note: this intentionally does NOT accept a k argument. If kv_patch.py
        ever accidentally passes k here, Python will raise a TypeError — which
        is the correct behaviour (K must never be stored).
        """
        if layer_idx >= len(self.value_cache):
            # First write for this layer (prefill): store as-is
            self.value_cache.append(v)
        else:
            # Subsequent writes (decode): concatenate along sequence dimension (dim=2)
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], v], dim=2
            )
        return self.value_cache[layer_idx]

    # ── Sequence length queries (used by kv_patch.py and HF generation loop) ──

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Return the number of tokens currently cached for layer_idx.

        Called by kv_patch.py to compute S_total = S_cache + S_new
        before building full_pos_ids for RoPE reconstruction.

        Returns 0 if the layer has not been written to yet (e.g. before
        the first forward pass, or when use_cache=False).
        """
        if layer_idx >= len(self.value_cache):
            return 0
        return self.value_cache[layer_idx].shape[2]   # dim 2 = sequence

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """
        Return the number of already-cached tokens for layer_idx.

        HuggingFace's modeling code calls this to compute the total KV
        sequence length when building the attention mask. Equivalent to
        get_seq_length() for our purposes.
        """
        return self.get_seq_length(layer_idx)

    # ── Shape accessors (used by verify_kv_relation.py Test 4) ───────────────

    def value_dim_kv_heads(self, layer_idx: int = 0) -> int:
        """
        Return the number of KV heads (G) stored for layer_idx.

        Shape of stored V: [B, G, S, head_dim] — G is dim 1.
        Used in Test 4 to verify G matches the model's num_key_value_heads.
        """
        if layer_idx >= len(self.value_cache):
            raise IndexError(f"Layer {layer_idx} not yet written to cache.")
        return self.value_cache[layer_idx].shape[1]

    def value_dim_head(self, layer_idx: int = 0) -> int:
        """
        Return the per-head dimension (head_dim) stored for layer_idx.

        Shape of stored V: [B, G, S, head_dim] — head_dim is dim 3.
        Used in Test 4 to verify head_dim matches the model's head_dim.
        """
        if layer_idx >= len(self.value_cache):
            raise IndexError(f"Layer {layer_idx} not yet written to cache.")
        return self.value_cache[layer_idx].shape[3]

    def key_cache_empty(self) -> bool:
        """
        Return True if no K has been stored (expected: always True).

        Used in Test 4 to confirm K is never written. If this returns False
        it means kv_patch.py has a bug where K is being cached.
        """
        return len(self.key_cache) == 0

    # ── Memory measurement (used by verify_kv_relation.py Test 5) ────────────

    def cache_size_bytes(self) -> dict:
        """
        Return total bytes consumed by the V cache.

        Only value_cache is counted — key_cache is always empty.
        Returns a dict so callers can access both raw bytes and MB.

        Example output:
            {"total_bytes": 134217728, "total_mb": 128.0}
        """
        total = sum(
            t.nbytes
            for t in self.value_cache
            if t is not None
        )
        return {
            "total_bytes": total,
            "total_mb":    total / 1e6,
        }

    # ── Compatibility with HF generation utilities ────────────────────────────

    def __len__(self) -> int:
        """Number of layers currently cached."""
        return len(self.value_cache)

    def __repr__(self) -> str:
        n_layers = len(self.value_cache)
        if n_layers == 0:
            return "VOnlyCache(empty)"
        S = self.value_cache[0].shape[2]
        G = self.value_cache[0].shape[1]
        D = self.value_cache[0].shape[3]
        mb = self.cache_size_bytes()["total_mb"]
        return (
            f"VOnlyCache("
            f"layers={n_layers}, seq_len={S}, "
            f"kv_heads={G}, head_dim={D}, "
            f"{mb:.1f} MB)"
        )