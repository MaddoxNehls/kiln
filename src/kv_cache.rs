//! Key/value cache for incremental decoding.
//!
//! Without a cache, generating token `n` recomputes K and V for all `n - 1`
//! previous tokens: O(n²) work per sequence just in projections. Since K and V
//! of past positions never change (causal attention: a token only sees its
//! past), they are computed once and appended here.
//!
//! Layout per layer: `[kv_heads, capacity, head_dim]`, i.e. *head-major*.
//! Attention for one head scans every cached position of that head, so making
//! that scan a single contiguous stream matters more than making the append
//! (a few hundred floats per token) contiguous. Memory is allocated up front
//! for `capacity` positions, so decoding never reallocates.

use crate::config::Config;

pub struct KvCache {
    layers: Vec<LayerCache>,
    len: usize,
    capacity: usize,
    kv_heads: usize,
    head_dim: usize,
}

struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Read-only view of one layer's cache, as consumed by attention.
pub(crate) struct LayerView<'a> {
    k: &'a [f32],
    v: &'a [f32],
    capacity: usize,
    head_dim: usize,
}

impl KvCache {
    pub fn new(config: &Config, capacity: usize) -> Self {
        let (kv_heads, head_dim) = (config.num_kv_heads(), config.head_dim());
        let size = kv_heads * capacity * head_dim;
        let layers = (0..config.num_hidden_layers)
            .map(|_| LayerCache {
                k: vec![0.0; size],
                v: vec![0.0; size],
            })
            .collect();
        Self {
            layers,
            len: 0,
            capacity,
            kv_heads,
            head_dim,
        }
    }

    /// Number of positions currently cached.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Forget everything after the first `len` positions. Cheap: entries past
    /// `len` are simply overwritten by the next forward pass.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// Store K/V for positions `start..start + n`. `k`, `v` are `[n, kv_heads,
    /// head_dim]` (the projection output layout) and are scattered head-major.
    pub(crate) fn write(&mut self, layer: usize, start: usize, k: &[f32], v: &[f32]) {
        let hd = self.head_dim;
        let cache = &mut self.layers[layer];
        for (t, (kt, vt)) in k
            .chunks_exact(self.kv_heads * hd)
            .zip(v.chunks_exact(self.kv_heads * hd))
            .enumerate()
        {
            for h in 0..self.kv_heads {
                let dst = (h * self.capacity + start + t) * hd;
                cache.k[dst..dst + hd].copy_from_slice(&kt[h * hd..(h + 1) * hd]);
                cache.v[dst..dst + hd].copy_from_slice(&vt[h * hd..(h + 1) * hd]);
            }
        }
    }

    /// Mark `n` freshly written positions as valid (after all layers ran).
    pub(crate) fn advance(&mut self, n: usize) {
        self.len += n;
        debug_assert!(self.len <= self.capacity);
    }

    pub(crate) fn layer(&self, layer: usize) -> LayerView<'_> {
        let cache = &self.layers[layer];
        LayerView {
            k: &cache.k,
            v: &cache.v,
            capacity: self.capacity,
            head_dim: self.head_dim,
        }
    }
}

impl<'a> LayerView<'a> {
    /// Keys of `kv_head` for positions `0..len`, as `[len, head_dim]`.
    pub(crate) fn keys(&self, kv_head: usize, len: usize) -> &'a [f32] {
        let start = kv_head * self.capacity * self.head_dim;
        &self.k[start..start + len * self.head_dim]
    }

    /// Values of `kv_head` for positions `0..len`, as `[len, head_dim]`.
    pub(crate) fn values(&self, kv_head: usize, len: usize) -> &'a [f32] {
        let start = kv_head * self.capacity * self.head_dim;
        &self.v[start..start + len * self.head_dim]
    }
}
