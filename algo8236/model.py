"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, NamedTuple, Tuple, Optional, Union, Dict


PREDICT_RULE_RARE_FIDS = (5, 6, 7, 8, 10, 12, 16, 84, 85)
PREDICT_RULE_MATCH_PAIRS = (
    (5, 'a', 42),
    (5, 'b', 70),
    (5, 'b', 78),
    (5, 'c', 30),
    (5, 'd', 21),
    (6, 'b', 71),
    (6, 'b', 78),
    (83, 'b', 68),
    (83, 'b', 75),
    (83, 'd', 21),
    (84, 'b', 70),
    (84, 'd', 21),
    (85, 'b', 78),
    (85, 'b', 79),
)
PREDICT_RULE_PROTECT_ITEM56 = (
    (40, 0), (40, 775), (49, 645), (51, 328), (57, 0),
    (102, 156), (110, 0), (167, 11), (191, 0), (214, 711),
    (221, 172), (263, 172), (274, 0),
)
SEMANTIC_RULE_FEATURE_DIM = 20


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    seq_cyclic_time: dict = None  # {domain: tensor [B, L, 4/7/12]} time features
    action_type: torch.Tensor = None
    timestamp: torch.Tensor = None


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        rope_on_q: Optional[bool] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            rope_on_q: Optional override for applying RoPE to the Q side.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            apply_rope_on_q = self.rope_on_q if rope_on_q is None else rope_on_q
            if apply_rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert key_padding_mask to SDPA format
        sdpa_attn_mask = None
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = padding
            # SDPA expects (B, 1, 1, Lk) bool mask, True = attend
            sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Lk)
            sdpa_attn_mask = sdpa_attn_mask.expand(B, self.num_heads, Lq, Lk)

        if attn_mask is not None:
            # attn_mask: additive float mask (Lq, Lk), -inf means do not attend
            # Convert to bool: positions that are not -inf are True
            bool_attn = (attn_mask == 0)  # (Lq, Lk)
            bool_attn = bool_attn.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            if sdpa_attn_mask is not None:
                sdpa_attn_mask = sdpa_attn_mask & bool_attn
            else:
                sdpa_attn_mask = bool_attn

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Only zero attention rows that have no valid keys. Do not blanket
        # replace NaNs from normal rows, otherwise real numerical divergence
        # would be silently hidden.
        if sdpa_attn_mask is not None:
            empty_attn = ~sdpa_attn_mask.any(dim=-1, keepdim=True)
            out = out.masked_fill(empty_attn, 0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model, num_heads=num_heads, dropout=dropout, rope_on_q=True
        )

        self.ffn = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None,
                rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None,
                rope_on_q: Optional[bool] = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        kv_norm = self.norm1(kv) if kv is not None else x_norm
        
        x_attn, _ = self.self_attn(
            query=x_norm, key=kv_norm, value=kv_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope_cos=rope_cos, rope_sin=rope_sin,
            rope_on_q=rope_on_q,
        )
        x = residual + self.dropout(x_attn)

        residual = x
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = residual + self.dropout(x_ffn)
        return x


class CrossNetV2(nn.Module):
    """DCN-V2 Cross Network: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l"""
    def __init__(self, in_features: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.empty(in_features, in_features))
            for _ in range(num_layers)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.empty(in_features))
            for _ in range(num_layers)
        ])
        for i in range(num_layers):
            nn.init.xavier_normal_(self.kernels[i])
            nn.init.zeros_(self.bias[i])

    def forward(self, x_0: torch.Tensor) -> torch.Tensor:
        x_l = x_0
        for i in range(self.num_layers):
            xl_w = torch.matmul(x_l, self.kernels[i])
            x_l = x_0 * xl_w + self.bias[i] + x_l
        return x_l


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        ids = int_feats[:, offset].long()
                        if self.training:
                            ns_mask = torch.rand(ids.shape, device=ids.device) < 0.2
                            ids = ids.masked_fill(ns_mask, 0)
                        fid_emb = emb_layer(ids)  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        if self.training:
                            ns_mask = torch.rand(vals.shape, device=vals.device) < 0.2
                            vals = vals.masked_fill(ns_mask, 0)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for vs, offset, length in feature_specs:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                vs, offset, length = self.feature_specs[fid_idx]
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        ids = int_feats[:, offset].long()
                        # NS feature mask: randomly zero-out 20% of feature
                        # values during training to simulate OOB/new-ID
                        # scenarios in the test set.
                        if self.training:
                            ns_mask = torch.rand(ids.shape, device=ids.device) < 0.2
                            ids = ids.masked_fill(ns_mask, 0)
                        fid_emb = emb_layer(ids)
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        if self.training:
                            ns_mask = torch.rand(vals.shape, device=vals.device) < 0.2
                            vals = vals.masked_fill(ns_mask, 0)
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


DENSE_INT_PAIR_FIDS = {62, 63, 64, 65, 66}


class DenseIntPairProjector(nn.Module):
    """Use aligned user dense stats to gate sparse user/item NS tokens.

    The projector intentionally uses only high-signal user profile/stat fids
    instead of the whole dense soup. Its role is a light residual interaction:
    dense statistics provide a sample-wise context that modulates int-token
    representations.
    """

    def __init__(self, source_dim: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.source_proj = nn.Sequential(
            nn.LayerNorm(self.source_dim),
            nn.Linear(self.source_dim, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.user_gate = nn.Linear(d_model, 2 * d_model)
        self.item_gate = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.user_gate.weight)
        nn.init.zeros_(self.user_gate.bias)
        nn.init.zeros_(self.item_gate.weight)
        nn.init.zeros_(self.item_gate.bias)

    def forward(
        self,
        dense_part: torch.Tensor,
        user_tokens: torch.Tensor,
        item_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = self.source_proj(dense_part)

        user_gamma, user_beta = self.user_gate(ctx).chunk(2, dim=-1)
        item_gamma, item_beta = self.item_gate(ctx).chunk(2, dim=-1)

        user_tokens = (
            user_tokens * (1.0 + torch.tanh(user_gamma).unsqueeze(1))
            + user_beta.unsqueeze(1)
        )
        item_tokens = (
            item_tokens * (1.0 + torch.tanh(item_gamma).unsqueeze(1))
            + item_beta.unsqueeze(1)
        )
        return user_tokens, item_tokens


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        seq_time_feature_dim: int = 4,
        user_dense_fids_info: Optional[List[Tuple[int, int, int]]] = None,
        use_dense_int_pair: bool = False,
        seq_fid_lists: Optional[Dict[str, List[int]]] = None,
        item_rule_fids_info: Optional[List[Tuple[int, int, int, int]]] = None,
        use_predict_rule_calibrator: bool = False,
        predict_rule_scale: float = 0.02,
        time_zone_offset_hours: float = 8.0,
        use_semantic_rule_features: bool = False,
        semantic_rule_direct_scale: float = 1.0,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.seq_encoder_type = seq_encoder_type
        _time_dim = int(seq_time_feature_dim)
        self.seq_time_feature_dim = _time_dim if _time_dim in (4, 7, 12) else 4
        self.use_dense_int_pair = bool(use_dense_int_pair)
        self.seq_slot_by_fid = {
            domain: {
                int(fid): idx
                for idx, fid in enumerate(seq_fid_lists.get(domain, []))
            }
            for domain in self.seq_domains
        } if seq_fid_lists else {}
        self.item_rule_fids_info = {
            int(fid): (int(offset), int(length), int(vs))
            for fid, offset, length, vs in (item_rule_fids_info or [])
        }
        self.use_predict_rule_calibrator = bool(use_predict_rule_calibrator)
        self.predict_rule_scale = float(predict_rule_scale)
        self.time_zone_offset_seconds = float(time_zone_offset_hours) * 3600.0
        self.use_semantic_rule_features = bool(use_semantic_rule_features)
        self.semantic_rule_direct_scale = float(semantic_rule_direct_scale)

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection (if available)
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        if self.use_semantic_rule_features:
            self.semantic_rule_proj = nn.Sequential(
                nn.Linear(SEMANTIC_RULE_FEATURE_DIM, d_model),
                nn.LayerNorm(d_model),
            )
            rule_hidden = max(16, d_model // 2)
            self.semantic_rule_logit = nn.Sequential(
                nn.Linear(SEMANTIC_RULE_FEATURE_DIM, rule_hidden),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(rule_hidden, action_num),
            )
            nn.init.zeros_(self.semantic_rule_logit[-1].weight)
            nn.init.zeros_(self.semantic_rule_logit[-1].bias)
            logging.info(
                "Semantic rule features enabled: dim=%d, direct_scale=%.4f",
                SEMANTIC_RULE_FEATURE_DIM,
                self.semantic_rule_direct_scale,
            )

        pair_indices: List[int] = []
        if self.use_dense_int_pair and user_dense_fids_info:
            for fid, offset, length in user_dense_fids_info:
                if int(fid) in DENSE_INT_PAIR_FIDS:
                    pair_indices.extend(
                        range(int(offset), int(offset) + int(length)))
        if self.use_dense_int_pair and pair_indices:
            self.register_buffer(
                'dense_int_pair_indices',
                torch.tensor(pair_indices, dtype=torch.long),
                persistent=False,
            )
            self.dense_int_pair = DenseIntPairProjector(
                source_dim=len(pair_indices),
                d_model=d_model,
                dropout=dropout_rate,
            )
            logging.info(
                "DenseIntPairProjector enabled: source_dim=%d, "
                "whitelist_fids=%s",
                len(pair_indices),
                sorted(DENSE_INT_PAIR_FIDS),
            )
        else:
            self.use_dense_int_pair = False

        # Total NS token count
        self.num_item_ns = num_item_ns
        self.num_user_ns = num_user_ns
        self.num_ns = (num_user_ns + (1 if self.has_user_dense else 0)
                       + num_item_ns + (1 if self.has_item_dense else 0)
                       + (1 if self.use_semantic_rule_features else 0))

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        # Removed RankMixer constraint since we use unified DecoderBlock


        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== Cyclic Time Projection (hour/weekday sin/cos) ==================
        # Borrowed from algo2025 finalist solution: absolute temporal
        # periodicity / recency context as 4/7/12-dim features -> d_model.
        self.time_cyclic_proj = nn.Linear(self.seq_time_feature_dim, d_model, bias=False)
        nn.init.normal_(self.time_cyclic_proj.weight, mean=0.0, std=0.02)

        # ================== FiLM Conditioning (User → Item/Seq Modulation) ==================
        # Borrowed from algo2025 finalist solution: user representation
        # dynamically modulates item/sequence tokens via FiLM (Feature-wise
        # Linear Modulation): x' = x * (1 + tanh(γ)) + β.
        self.film_conditioner = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.film_conditioner.weight)
        nn.init.zeros_(self.film_conditioner.bias)

        # ================== Decoder Blocks ==================

        self.seq_encoders = nn.ModuleDict()
        for domain in self.seq_domains:
            self.seq_encoders[domain] = DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
            )

        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
            )
            for _ in range(num_hyformer_blocks)
        ])
        self.final_norm = RMSNorm(d_model)

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Cross Network V2
        cross_dim = 2 * d_model
        self.cross_net = CrossNetV2(in_features=cross_dim, num_layers=2)

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(cross_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes only high-cardinality embeddings.

        Preserves low-cardinality and time feature embeddings.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, (vs, offset, length) in enumerate(specs):
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1

        logging.info(f"Re-initialized {reinit_count} high-cardinality Embeddings "
                     f"(vocab>{cardinality_threshold}), kept {skip_count}")
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
        cyclic_time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model.

        Args:
            cyclic_time: Optional (B, L, 4/7/12) tensor with cyclic or richer
                per-step time features.
        """
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)
        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        # Add cyclic time projection (hour/weekday sin/cos → d_model)
        if cyclic_time is not None:
            token_emb = token_emb + self.time_cyclic_proj(cyclic_time)

        return token_emb

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _run_decoder_blocks(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        apply_dropout: bool = True,
        user_ns_for_film: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run decoder blocks with optional FiLM conditioning.

        Args:
            user_ns_for_film: (B, num_user_ns, D) user NS tokens for FiLM
                conditioning.  When provided, the pooled user vector
                modulates item NS and sequence tokens before the decoder.
        """
        if apply_dropout:
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        # FiLM conditioning: user representation modulates item/seq tokens.
        # x' = x * (1 + tanh(γ)) + β  where (γ, β) = Linear(user_vec)
        if user_ns_for_film is not None:
            user_vec = user_ns_for_film.mean(dim=1)  # (B, D)
            film_params = self.film_conditioner(user_vec)  # (B, 2*D)
            gamma, beta = film_params.chunk(2, dim=-1)  # each (B, D)
            gamma = torch.tanh(gamma).unsqueeze(1)  # (B, 1, D)
            beta = beta.unsqueeze(1)  # (B, 1, D)
            # Modulate each sequence token list
            seq_tokens_list = [
                s * (1.0 + gamma) + beta for s in seq_tokens_list
            ]
            # Modulate the item portion of ns_tokens (last num_item_ns + dense)
            num_user = self.num_user_ns + (1 if self.has_user_dense else 0)
            if ns_tokens.shape[1] > num_user:
                user_part = ns_tokens[:, :num_user, :]
                item_part = ns_tokens[:, num_user:, :]
                item_part = item_part * (1.0 + gamma) + beta
                ns_tokens = torch.cat([user_part, item_part], dim=1)

        # 3.1 Decoupled Cross-Attention
        # Use NS Tokens as Query, Seq Tokens as Key/Value (Forced Target Attention)
        combined_seqs = seq_tokens_list
        B, num_ns, D = ns_tokens.shape
        device = ns_tokens.device

        curr_query = ns_tokens
        query_is_ns = True
        kv_is_seq = len(combined_seqs) > 0
        if kv_is_seq:
            curr_kv = torch.cat(combined_seqs, dim=1)  # (B, L_seq_total, D)
            curr_kv_mask = torch.cat(seq_masks_list, dim=1)  # (B, L_seq_total)
        else:
            curr_kv = curr_query
            curr_kv_mask = torch.zeros(B, num_ns, dtype=torch.bool, device=device)

        L_total = curr_kv.shape[1]

        # Precompute RoPE for KV
        rope_cos = None
        rope_sin = None
        if self.rotary_emb is not None and kv_is_seq:
            rope_cos, rope_sin = self.rotary_emb(L_total, device)

        for block in self.blocks:
            curr_query = block(
                x=curr_query,
                kv=curr_kv,
                key_padding_mask=curr_kv_mask,
                attn_mask=None,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                rope_on_q=not query_is_ns,
            )

        # Output for classification: Pool User and Item separately
        output = self.final_norm(curr_query)  # (B, num_ns, D)
        output = self.output_proj(output)     # (B, num_ns, D)
        
        num_user = self.num_user_ns + (1 if self.has_user_dense else 0)
        num_item = (
            self.num_item_ns
            + (1 if self.has_item_dense else 0)
            + (1 if self.use_semantic_rule_features else 0)
        )
        
        user_out = output[:, :num_user, :].mean(dim=1)  # (B, D)
        item_out = output[:, -num_item:, :].mean(dim=1)  # (B, D)
        
        output = torch.cat([user_out, item_out], dim=1)  # (B, 2*D)
        output = self.cross_net(output)                        # DCN-v2
        
        return output

    def _get_user_ns_tokens(
        self, user_ns: torch.Tensor, has_user_dense: bool,
        user_dense_tok: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Gather all user-side NS tokens (sparse + optional dense)."""
        parts = [user_ns]
        if has_user_dense and user_dense_tok is not None:
            parts.append(user_dense_tok)
        return torch.cat(parts, dim=1)  # (B, num_user_ns, D)

    @staticmethod
    def _domain_key(domain: str) -> str:
        d = str(domain).lower()
        for key in ('a', 'b', 'c', 'd'):
            if d.endswith(key) or f'_{key}' in d:
                return key
        return d

    def _item_scalar(
        self,
        item_int: torch.Tensor,
        fid: int,
        default: float = 0.0,
    ) -> torch.Tensor:
        spec = self.item_rule_fids_info.get(int(fid))
        if spec is None:
            return item_int.new_full((item_int.shape[0],), default, dtype=torch.float32)
        offset, _length, _vs = spec
        if item_int.shape[1] <= offset:
            return item_int.new_full((item_int.shape[0],), default, dtype=torch.float32)
        return item_int[:, offset].float()

    def _item_any_rare(self, item_int: torch.Tensor, fid: int) -> torch.Tensor:
        spec = self.item_rule_fids_info.get(int(fid))
        if spec is None:
            return torch.zeros(item_int.shape[0], dtype=torch.bool, device=item_int.device)
        offset, length, vs = spec
        if vs <= 0 or item_int.shape[1] <= offset:
            return torch.zeros(item_int.shape[0], dtype=torch.bool, device=item_int.device)
        vals = item_int[:, offset:offset + max(1, length)]
        return ((vals > 0) & (vals >= int(vs))).any(dim=1)

    def _protected_item56_pair(self, item_int: torch.Tensor) -> torch.Tensor:
        item5 = self._item_scalar(item_int, 5).long()
        item6 = self._item_scalar(item_int, 6).long()
        protected = torch.zeros(item_int.shape[0], dtype=torch.bool, device=item_int.device)
        for v5, v6 in PREDICT_RULE_PROTECT_ITEM56:
            protected = protected | ((item5 == int(v5)) & (item6 == int(v6)))
        return protected

    def _history_match_flags(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item_int = inputs.item_int_feats
        B = item_int.shape[0]
        device = item_int.device
        match_hits = torch.zeros(B, device=device, dtype=torch.float32)
        recent_match = torch.zeros(B, device=device, dtype=torch.bool)

        for item_fid, pair_domain, seq_fid in PREDICT_RULE_MATCH_PAIRS:
            spec = self.item_rule_fids_info.get(int(item_fid))
            if spec is None:
                continue
            item_offset, _item_length, item_vs = spec
            if item_int.shape[1] <= item_offset:
                continue
            target = item_int[:, item_offset].long()
            valid_target = target > 0
            if item_vs > 0:
                valid_target = valid_target & (target < int(item_vs))
            if not valid_target.any():
                continue

            for domain in self.seq_domains:
                if self._domain_key(domain) != pair_domain:
                    continue
                slot = self.seq_slot_by_fid.get(domain, {}).get(int(seq_fid))
                if slot is None:
                    continue
                seq = inputs.seq_data[domain]
                if seq.shape[1] <= slot:
                    continue
                seq_vals = seq[:, slot, :].long()
                hit = (seq_vals == target.unsqueeze(1)) & valid_target.unsqueeze(1)
                match_hits = match_hits + hit.sum(dim=1).float()
                recent_match = recent_match | hit[:, :20].any(dim=1)

        match_any = match_hits > 0
        return match_any, recent_match, match_hits

    @staticmethod
    def _pair_mask(values_a: torch.Tensor, values_b: torch.Tensor,
                   pairs: Tuple[Tuple[int, int], ...]) -> torch.Tensor:
        mask = torch.zeros_like(values_a, dtype=torch.bool)
        for va, vb in pairs:
            mask = mask | ((values_a == int(va)) & (values_b == int(vb)))
        return mask

    def _semantic_rule_components(
        self,
        inputs: ModelInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return rule feature vector, signed rule level, and hit mask.

        This is the train-time counterpart of the list-level reranker rules:
        the same semantic groups become learnable model inputs instead of only
        inference-time score edits.
        """
        item_int = inputs.item_int_feats
        B = item_int.shape[0]
        device = item_int.device

        item5 = self._item_scalar(item_int, 5).long()
        item6 = self._item_scalar(item_int, 6).long()
        item10 = self._item_scalar(item_int, 10).long()
        item12 = self._item_scalar(item_int, 12).long()
        tag_all_zero = (
            (self._item_scalar(item_int, 83) <= 0)
            & (self._item_scalar(item_int, 84) <= 0)
            & (self._item_scalar(item_int, 85) <= 0)
        )

        protected = self._protected_item56_pair(item_int)
        head_protect = tag_all_zero & self._pair_mask(
            item5, item6,
            (
                (83, 0), (267, 237), (109, 168), (259, 494), (138, 301),
                (246, 682), (300, 0), (167, 0), (167, 780), (49, 645),
            ),
        )

        match_any, recent_match, match_hits = self._history_match_flags(inputs)
        match0 = ~match_any

        item_rare_count = torch.zeros(B, device=device, dtype=torch.float32)
        for fid in PREDICT_RULE_RARE_FIDS:
            item_rare_count = item_rare_count + self._item_any_rare(item_int, fid).float()
        oov_item = item_rare_count > 0

        age1h = torch.zeros(B, device=device, dtype=torch.bool)
        age6h = torch.zeros(B, device=device, dtype=torch.bool)
        age1d = torch.zeros(B, device=device, dtype=torch.bool)
        for tb in inputs.seq_time_buckets.values():
            valid = tb > 0
            age1h = age1h | (valid & (tb <= 31)).any(dim=1)
            age6h = age6h | (valid & (tb <= 41)).any(dim=1)
            age1d = age1d | (valid & (tb <= 48)).any(dim=1)

        late_window = torch.zeros(B, device=device, dtype=torch.bool)
        if inputs.timestamp is not None:
            wall = torch.remainder(
                inputs.timestamp.float() + self.time_zone_offset_seconds,
                86400.0,
            )
            minute = wall / 60.0
            late_window = (minute >= 9 * 60) & (minute < 9 * 60 + 15)

        p4 = tag_all_zero & age1d & (
            ((item5 == 40) & ((item6 == 775) | (item6 == 0) | (item6 == 172))
             & ((item10 == 84) | (item12 == 0) | (item12 == 42) | (item12 == 509)))
            | (((item5 == 221) | (item5 == 263)) & (item6 == 172)
               & ((item10 == 84) | (item10 == 187) | (item12 == 42) | (item12 == 509)))
            | ((item5 == 51) & (item6 == 328)
               & ((item10 == 84) | (item10 == 22) | (item12 == 509)))
        )
        p3 = tag_all_zero & protected & age1d & ~p4 & (
            (item10 == 84) | (item10 == 187) | (item10 == 22)
            | (item12 == 0) | (item12 == 42) | (item12 == 509)
        )
        n4 = tag_all_zero & match0 & (
            ((item5 == 84) & ((item6 == 714) | (item6 == 661))
             & ((item10 == 182) | (item10 == 33)))
            | ((item5 == 84) & (item10 == 182)
               & ((item12 == 833) | (item12 == 230) | (item12 == 996)))
        )
        n3 = tag_all_zero & match0 & ~n4 & (
            ((item5 == 164) & ((item6 == 160) | (item6 == 724))
             & ((item10 == 203) | (item12 == 1869)))
            | ((item5 == 152) & ((item10 == 167) | (item10 == 203)))
        )
        p2 = (
            (((match_hits >= 2.0) & age1d)
             | (match_any & recent_match & protected & age1h))
            & ~oov_item & ~n4 & ~n3 & ~head_protect
        )
        p1 = (
            oov_item & match_any & (protected | recent_match)
            & ~n4 & ~n3
        )
        n2 = (
            oov_item & match0 & late_window & (age1h | age6h)
            & ~protected & ~n4 & ~n3
        )
        n1 = (
            (item_rare_count >= 3.0) & match0 & tag_all_zero
            & ~protected & ~n4 & ~n3 & ~n2
        )

        pos_level = (
            4.0 * p4.float() + 3.0 * p3.float()
            + 2.0 * p2.float() + p1.float()
        ) / 4.0
        neg_level = (
            4.0 * n4.float() + 3.0 * n3.float()
            + 2.0 * n2.float() + n1.float()
        ) / 4.0
        signed_level = pos_level - neg_level
        hit_mask = (pos_level > 0) | (neg_level > 0)

        rare_norm = torch.clamp(item_rare_count / float(len(PREDICT_RULE_RARE_FIDS)), 0.0, 1.0)
        match_norm = torch.clamp(torch.log1p(match_hits) / math.log(16.0), 0.0, 1.0)
        feats = torch.stack([
            p4.float(), p3.float(), p2.float(), p1.float(),
            n4.float(), n3.float(), n2.float(), n1.float(),
            pos_level, neg_level, signed_level,
            match_any.float(), recent_match.float(), match_norm,
            rare_norm, age1h.float(), age6h.float(), age1d.float(),
            tag_all_zero.float(), protected.float(),
        ], dim=1)
        return feats, signed_level, hit_mask

    def semantic_rule_signed_score(self, inputs: ModelInput) -> torch.Tensor:
        """Expose deterministic rule direction for optional training losses."""
        _feats, signed_level, _hit_mask = self._semantic_rule_components(inputs)
        return signed_level

    def _apply_predict_rule_calibrator(
        self,
        logits: torch.Tensor,
        inputs: ModelInput,
    ) -> torch.Tensor:
        if (
            not self.use_predict_rule_calibrator
            or self.training
            or inputs.item_int_feats is None
            or self.predict_rule_scale <= 0.0
        ):
            return logits

        item_int = inputs.item_int_feats
        B = item_int.shape[0]
        device = item_int.device
        scale = float(self.predict_rule_scale)

        prob = torch.sigmoid(logits.squeeze(-1)).detach()
        low_guard = prob < 0.06
        high_guard = prob > 0.90
        mid_band = (prob >= 0.08) & (prob <= 0.88)
        upper_band = (prob >= 0.12) & (prob <= 0.96)

        oov16 = self._item_any_rare(item_int, 16)
        item_rare_count = torch.zeros(B, device=device, dtype=torch.float32)
        for fid in PREDICT_RULE_RARE_FIDS:
            item_rare_count = item_rare_count + self._item_any_rare(item_int, fid).float()

        age1h = torch.zeros(B, device=device, dtype=torch.bool)
        age6h = torch.zeros(B, device=device, dtype=torch.bool)
        age1d = torch.zeros(B, device=device, dtype=torch.bool)
        for tb in inputs.seq_time_buckets.values():
            valid = tb > 0
            age1h = age1h | (valid & (tb <= 31)).any(dim=1)
            age6h = age6h | (valid & (tb <= 41)).any(dim=1)
            age1d = age1d | (valid & (tb <= 48)).any(dim=1)

        match_any, recent_match, match_hits = self._history_match_flags(inputs)
        match0 = ~match_any
        protected = self._protected_item56_pair(item_int)
        item5 = self._item_scalar(item_int, 5).long()
        item6 = self._item_scalar(item_int, 6).long()
        item10 = self._item_scalar(item_int, 10).long()
        item12 = self._item_scalar(item_int, 12).long()
        tag_all_zero = (
            (self._item_scalar(item_int, 83) <= 0)
            & (self._item_scalar(item_int, 84) <= 0)
            & (self._item_scalar(item_int, 85) <= 0)
        )

        # 126720 diagnostics showed the retrained checkpoint demoted many
        # high-base-score samples in this item family. Recover them only as a
        # bounded inference-time nudge; do not train a new dependency on it.
        recover_family = tag_all_zero & age1d & (
            ((item5 == 40) & ((item6 == 775) | (item6 == 0) | (item6 == 172)) & (item10 == 84))
            | ((item5 == 221) & (item6 == 172))
            | ((item5 == 263) & (item6 == 172))
            | ((item5 == 51) & (item6 == 328))
        )
        recover_family_broad = tag_all_zero & age1d & protected & ~recover_family & (
            (item10 == 84)
            | (item10 == 187)
            | (item10 == 22)
            | (item10 == 182)
            | (item12 == 0)
            | (item12 == 42)
            | (item12 == 509)
        )

        # The same diagnostics showed cold tag-empty samples in the 84/182
        # family being lifted from deep tail into the head. Let the backbone
        # keep genuinely strong ones, but suppress match0 mid/high scores.
        over_promoted_cold = tag_all_zero & match0 & (
            ((item5 == 84) & (item10 == 182))
            | ((item5 == 84) & ((item6 == 714) | (item6 == 661)))
            | ((item10 == 182) & ((item12 == 833) | (item12 == 230)))
        )
        over_promoted_cold_broad = tag_all_zero & match0 & ~protected & ~over_promoted_cold & (
            ((item5 == 84) & ((item10 == 33) | (item10 == 203) | (item12 == 833) | (item12 == 230)))
            | (((item6 == 714) | (item6 == 661)) & ((item10 == 33) | (item10 == 182) | (item12 == 833) | (item12 == 230)))
            | ((item5 == 164) & ((item10 == 203) | (item12 == 1869)))
            | ((item5 == 152) & ((item10 == 167) | (item10 == 203)))
        )

        p2 = (
            (match_hits >= 2.0)
            | (match_any & recent_match & age1h & ~oov16)
            | (match_any & protected & age1h)
            | (recover_family & (prob >= 0.03) & (prob <= 0.88))
        )
        p1 = (
            (match_any & age1d & ~oov16)
            | (oov16 & match_any & (protected | recent_match))
            | (recover_family & (prob > 0.88) & (prob <= 0.96))
        ) & ~p2

        n2 = (
            (oov16 & match0 & tag_all_zero & age1h & ~protected)
            | ((item_rare_count >= 3.0) & match0 & ~protected)
            | (over_promoted_cold & (prob >= 0.10) & (prob <= 0.92))
        )
        n1 = (
            (oov16 & match0 & ~protected)
            | ((item_rare_count >= 1.0) & match0 & tag_all_zero & ~protected)
            | (over_promoted_cold & (prob > 0.92) & (prob <= 0.98))
        ) & ~n2

        if inputs.timestamp is not None:
            wall = torch.remainder(
                inputs.timestamp.float() + self.time_zone_offset_seconds,
                86400.0,
            )
            minute = wall / 60.0
            late_window = (minute >= 9 * 60) & (minute < 9 * 60 + 15)
            early_window = (minute >= 7 * 60 + 40) & (minute < 8 * 60)
            n1 = n1 | (oov16 & match0 & late_window & ~protected)
            p1 = p1 | (match_any & early_window & ~oov16)

        delta = torch.zeros(B, device=device, dtype=logits.dtype)
        delta = delta + (0.60 * scale) * (p2 & mid_band).float()
        delta = delta + (0.30 * scale) * (p1 & mid_band).float()
        delta = delta - (0.80 * scale) * (n2 & upper_band).float()
        delta = delta - (0.35 * scale) * (n1 & upper_band).float()
        # Medium-strength family correction.  The previous safe rule barely
        # moved ranks; this version still avoids the very top, but gives enough
        # logit room for the two diagnosed families to change mid-rank order.
        delta = delta + (4.40 * scale) * (recover_family & (prob >= 0.03) & (prob <= 0.45)).float()
        delta = delta + (2.80 * scale) * (recover_family & (prob > 0.45) & (prob <= 0.82)).float()
        delta = delta - (4.70 * scale) * (over_promoted_cold & (prob >= 0.09) & (prob <= 0.78)).float()
        delta = delta - (3.00 * scale) * (over_promoted_cold & (prob > 0.78) & (prob <= 0.92)).float()
        # Broaden coverage to adjacent values seen in the profile, but at a
        # lower strength so this does not become a hand-written second model.
        delta = delta + (2.35 * scale) * (recover_family_broad & (prob >= 0.04) & (prob <= 0.42)).float()
        delta = delta + (1.45 * scale) * (recover_family_broad & (prob > 0.42) & (prob <= 0.70)).float()
        delta = delta - (2.75 * scale) * (over_promoted_cold_broad & (prob >= 0.08) & (prob <= 0.58)).float()
        delta = delta - (1.65 * scale) * (over_promoted_cold_broad & (prob > 0.58) & (prob <= 0.86)).float()

        # Guardrails: keep this as a rank nudge, not a second model.
        delta = torch.where(
            low_guard & (delta > 0) & ~(recover_family | recover_family_broad),
            delta * 0.20,
            delta,
        )
        delta = torch.where(high_guard & (delta < 0) & n2, delta * 0.25, delta)
        delta = torch.where(high_guard & (delta < 0) & ~n2, torch.zeros_like(delta), delta)
        delta = torch.where(high_guard & (delta > 0), torch.zeros_like(delta), delta)
        delta = torch.clamp(delta, min=-10.0 * scale, max=10.0 * scale)
        return logits + delta.unsqueeze(-1).to(logits.dtype)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens: grouped projection
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)   # (B, num_user_groups, D)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)   # (B, num_item_groups, D)

        if self.use_dense_int_pair:
            dense_part = inputs.user_dense_feats.index_select(
                dim=1, index=self.dense_int_pair_indices)
            user_ns, item_ns = self.dense_int_pair(dense_part, user_ns, item_ns)

        user_dense_tok = None
        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)  # (B, 1, D)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok_item = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)  # (B, 1, D)
            ns_parts.append(item_dense_tok_item)
        semantic_rule_feats = None
        if self.use_semantic_rule_features:
            semantic_rule_feats, _signed, _hit = self._semantic_rule_components(inputs)
            rule_tok = F.silu(self.semantic_rule_proj(semantic_rule_feats)).unsqueeze(1)
            ns_parts.append(rule_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

        # Gather user NS tokens for FiLM conditioning
        user_ns_for_film = self._get_user_ns_tokens(
            user_ns, self.has_user_dense, user_dense_tok)

        # 2. Embed each sequence domain (dynamic), with cyclic time
        seq_tokens_list = []
        seq_masks_list = []
        cyclic_dict = inputs.seq_cyclic_time or {}
        for domain in self.seq_domains:
            cyclic_time = cyclic_dict.get(domain, None)
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                cyclic_time=cyclic_time)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            
            if self.seq_encoder_type in ('transformer', 'swiglu'):
                encoded = self.seq_encoders[domain](
                    x=tokens, kv=None,
                    key_padding_mask=mask,
                    attn_mask=None, rope_cos=None, rope_sin=None)
            else:
                encoded = tokens
                
            seq_tokens_list.append(encoded)
            seq_masks_list.append(mask)

        # 3 & 4. Unified Decoder stack + output projection + FiLM
        output = self._run_decoder_blocks(
            ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=self.training,
            user_ns_for_film=user_ns_for_film,
        )

        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        if self.use_semantic_rule_features and semantic_rule_feats is not None:
            logits = logits + self.semantic_rule_direct_scale * self.semantic_rule_logit(semantic_rule_feats)
        return logits

    def predict(self, inputs: ModelInput) -> torch.Tensor:
        """Runs inference without dropout, returning logits."""
        # Reuses forward logic but without dropout
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        if self.use_dense_int_pair:
            dense_part = inputs.user_dense_feats.index_select(
                dim=1, index=self.dense_int_pair_indices)
            user_ns, item_ns = self.dense_int_pair(dense_part, user_ns, item_ns)

        user_dense_tok = None
        ns_parts = [user_ns]
        if self.has_user_dense:
            user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok_item = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok_item)
        semantic_rule_feats = None
        if self.use_semantic_rule_features:
            semantic_rule_feats, _signed, _hit = self._semantic_rule_components(inputs)
            rule_tok = F.silu(self.semantic_rule_proj(semantic_rule_feats)).unsqueeze(1)
            ns_parts.append(rule_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)

        # Gather user NS tokens for FiLM conditioning
        user_ns_for_film = self._get_user_ns_tokens(
            user_ns, self.has_user_dense, user_dense_tok)

        seq_tokens_list = []
        seq_masks_list = []
        cyclic_dict = inputs.seq_cyclic_time or {}
        for domain in self.seq_domains:
            cyclic_time = cyclic_dict.get(domain, None)
            tokens = self._embed_seq_domain(
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain],
                cyclic_time=cyclic_time)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            
            if self.seq_encoder_type in ('transformer', 'swiglu'):
                encoded = self.seq_encoders[domain](
                    x=tokens, kv=None,
                    key_padding_mask=mask,
                    attn_mask=None, rope_cos=None, rope_sin=None)
            else:
                encoded = tokens
                
            seq_tokens_list.append(encoded)
            seq_masks_list.append(mask)

        output = self._run_decoder_blocks(
            ns_tokens, seq_tokens_list, seq_masks_list,
            apply_dropout=False,
            user_ns_for_film=user_ns_for_film,
        )

        logits = self.clsfier(output)
        if self.use_semantic_rule_features and semantic_rule_feats is not None:
            logits = logits + self.semantic_rule_direct_scale * self.semantic_rule_logit(semantic_rule_feats)
        logits = self._apply_predict_rule_calibrator(logits, inputs)
        return logits
