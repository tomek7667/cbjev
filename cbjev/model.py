"""The network: a ModernBERT-style encoder plus a small decision head.

Written from scratch instead of wrapping `transformers.ModernBertModel`, for three reasons:

* the attention mask is ours to shape. cbjev's *packed* layout puts the state and every
  question into one row and needs "questions see the state, the state sees only itself,
  questions never see each other", which the stock module cannot express;
* the forward is a plain function of a handful of tensors, so it can be captured in a CUDA
  graph per shape bucket without fighting framework hooks;
* weights load straight from the same safetensors names Laya checkpoints use, so an existing
  checkpoint runs here unchanged ("classic" layout) and is the starting point for training.

Both mmBERT and ModernBERT share this architecture; only the sizes and RoPE bases differ.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderSpec:
    vocab: int
    dim: int
    layers: int
    heads: int
    ffn: int
    window: int              # half-width of the local attention band (tokens either side)
    every_global: int        # layer i is global when i % every_global == 0
    theta_global: float
    theta_local: float
    eps: float = 1e-5

    @classmethod
    def from_hf(cls, cfg: Dict) -> "EncoderSpec":
        rope = cfg.get("rope_parameters") or {}
        g = (rope.get("full_attention") or {}).get("rope_theta", cfg.get("global_rope_theta", 160000.0))
        l = (rope.get("sliding_attention") or {}).get("rope_theta", cfg.get("local_rope_theta", 10000.0))
        return cls(
            vocab=cfg["vocab_size"], dim=cfg["hidden_size"], layers=cfg["num_hidden_layers"],
            heads=cfg["num_attention_heads"], ffn=cfg["intermediate_size"],
            window=cfg.get("local_attention", 128) // 2,
            every_global=cfg.get("global_attn_every_n_layers", 3),
            theta_global=float(g), theta_local=float(l),
            eps=cfg.get("norm_eps", 1e-5),
        )


def _ln(dim: int, eps: float) -> nn.LayerNorm:
    return nn.LayerNorm(dim, eps=eps, bias=False)


class _Attn(nn.Module):
    def __init__(self, s: EncoderSpec):
        super().__init__()
        self.heads, self.hd = s.heads, s.dim // s.heads
        self.Wqkv = nn.Linear(s.dim, 3 * s.dim, bias=False)
        self.Wo = nn.Linear(s.dim, s.dim, bias=False)


class _MLP(nn.Module):
    def __init__(self, s: EncoderSpec):
        super().__init__()
        self.Wi = nn.Linear(s.dim, 2 * s.ffn, bias=False)
        self.Wo = nn.Linear(s.ffn, s.dim, bias=False)


class _Layer(nn.Module):
    def __init__(self, s: EncoderSpec, first: bool):
        super().__init__()
        self.attn_norm = nn.Identity() if first else _ln(s.dim, s.eps)
        self.attn = _Attn(s)
        self.mlp_norm = _ln(s.dim, s.eps)
        self.mlp = _MLP(s)


class _Embeddings(nn.Module):
    def __init__(self, s: EncoderSpec):
        super().__init__()
        self.tok_embeddings = nn.Embedding(s.vocab, s.dim)
        self.norm = _ln(s.dim, s.eps)


def _rope_tables(theta: float, hd: int, n: int) -> torch.Tensor:
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    f = torch.outer(torch.arange(n, dtype=torch.float64), inv)
    return torch.stack([f.cos(), f.sin()]).float()          # [2, n, hd/2]


def _lin(m: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Linear in the weight's dtype (bf16 on GPU); the caller keeps the residual in fp32."""
    return F.linear(x.to(m.weight.dtype), m.weight, m.bias)


def _compute_dtype(w: torch.Tensor) -> torch.dtype:
    """The dtype matmuls actually run in: the autocast dtype while training, else the weight's."""
    if w.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return w.dtype


def _norm(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return x if isinstance(m, nn.Identity) else F.layer_norm(x, m.normalized_shape, m.weight, m.bias, m.eps)


def _rotate(x: torch.Tensor, cs: torch.Tensor) -> torch.Tensor:
    # x: [B, H, L, hd]; cs: [2, B, 1, L, hd/2] already gathered at each token's position.
    # Same "rotate half" convention as ModernBERT, done in fp32 like the reference.
    x1, x2 = x.float().chunk(2, -1)
    c, s = cs[0], cs[1]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(x.dtype)


class Encoder(nn.Module):
    """ModernBERT forward over explicit positions and an explicit allow-mask."""

    def __init__(self, s: EncoderSpec, max_pos: int = 8192):
        super().__init__()
        self.spec = s
        self.embeddings = _Embeddings(s)
        self.layers = nn.ModuleList([_Layer(s, i == 0) for i in range(s.layers)])
        self.final_norm = _ln(s.dim, s.eps)
        hd = s.dim // s.heads
        self.register_buffer("rope_g", _rope_tables(s.theta_global, hd, max_pos), persistent=False)
        self.register_buffer("rope_l", _rope_tables(s.theta_local, hd, max_pos), persistent=False)
        self.checkpointing = False

    def _layer(self, lyr: _Layer, h, cs, bias):
        B, L, D = h.shape
        a = lyr.attn
        qkv = _lin(a.Wqkv, _norm(lyr.attn_norm, h)).view(B, L, 3, a.heads, a.hd).permute(2, 0, 3, 1, 4)
        q, k, v = _rotate(qkv[0], cs), _rotate(qkv[1], cs), qkv[2]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        h = h + _lin(a.Wo, o.transpose(1, 2).reshape(B, L, D)).float()
        x, gate = _lin(lyr.mlp.Wi, _norm(lyr.mlp_norm, h)).chunk(2, -1)
        return h + _lin(lyr.mlp.Wo, F.gelu(x) * gate).float()

    def forward(self, ids: torch.Tensor, pos: torch.Tensor, allow: torch.Tensor) -> torch.Tensor:
        """ids/pos: [B, L] long; allow: [B, L, L] bool (query row may read key column)."""
        s = self.spec
        h = _norm(self.embeddings.norm, self.embeddings.tok_embeddings(ids).float())
        near = (pos[:, :, None] - pos[:, None, :]).abs() <= s.window
        neg = -1e4
        wd = _compute_dtype(self.layers[0].attn.Wqkv.weight)
        zero = torch.zeros((), dtype=wd, device=h.device)
        bias_g = torch.where(allow, zero, neg)[:, None]
        bias_l = torch.where(allow & near, zero, neg)[:, None]
        cs_g = self.rope_g[:, pos][:, :, None]
        cs_l = self.rope_l[:, pos][:, :, None]
        for i, lyr in enumerate(self.layers):
            glob = i % s.every_global == 0
            args = (lyr, h, cs_g if glob else cs_l, bias_g if glob else bias_l)
            if self.checkpointing and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(self._layer, *args, use_reentrant=False)
            else:
                h = self._layer(*args)
        return self.final_norm(h)


class _HeadLayer(nn.Module):
    """Pre-norm transformer block with torch.nn.TransformerEncoderLayer's parameter names."""

    def __init__(self, d: int, heads: int, ffn: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.linear1 = nn.Linear(d, ffn)
        self.linear2 = nn.Linear(ffn, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.heads = heads

    def forward(self, h, bias, drop: float = 0.0):
        B, L, D = h.shape
        hd = D // self.heads
        w, b = self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
        qkv = F.linear(_norm(self.norm1, h).to(w.dtype), w, b)
        q, k, v = qkv.view(B, L, 3, self.heads, hd).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), dropout_p=drop)
        o = _lin(self.self_attn.out_proj, o.transpose(1, 2).reshape(B, L, D)).float()
        h = h + F.dropout(o, drop, self.training)
        x = F.dropout(F.relu(_lin(self.linear1, _norm(self.norm2, h))), drop, self.training)
        return h + F.dropout(_lin(self.linear2, x).float(), drop, self.training)


class _HeadStack(nn.Module):
    def __init__(self, d, heads, ffn, n):
        super().__init__()
        self.layers = nn.ModuleList([_HeadLayer(d, heads, ffn) for _ in range(n)])


class DecisionNet(nn.Module):
    """Encoder + typed head. Scores every option marker; the caller softmaxes per question.

    Inputs (all [B, L] unless noted):
      ids, pos      token ids and RoPE positions
      seg           0 for state tokens, q >= 1 for tokens of the q-th question in the row
      qtype         question type id per token (0 choice, 1 score, 2 noul), ignored for seg 0
      valid         False on padding
      markers       [B, M] long positions of option markers (row-local), -1 for none
    """

    def __init__(self, spec: EncoderSpec, head_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        d = spec.dim
        self.encoder = Encoder(spec)
        self.head = _HeadStack(d, max(1, d // 64), 4 * d, head_layers) if head_layers else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.dropout = dropout

    shared = False       # packed rows: may state tokens read the questions? (see layout.packed_row)

    def allow_mask(self, seg: torch.Tensor, valid: torch.Tensor, packed: bool) -> torch.Tensor:
        keys = valid[:, None, :]
        if not packed:
            return keys.expand(-1, seg.size(1), -1)
        # questions read the state and themselves, never each other
        same = seg[:, :, None] == seg[:, None, :]
        state_key = (seg == 0)[:, None, :]
        allow = same | state_key
        if self.shared:
            allow = allow | (seg == 0)[:, :, None]         # the state reads everything
        return keys & allow

    def type_embedding(self, qtype: torch.Tensor, seg: torch.Tensor, packed: bool) -> torch.Tensor:
        w = self.type_emb.weight.float()
        if not packed:
            return w[qtype]
        if not self.shared:
            return w[qtype.clamp(0, 2)] * (seg > 0)[..., None].float()
        # state tokens carry per-type question counts (layout.state_type_code): use their mean
        c = (qtype - 3).clamp(min=0)
        n = torch.stack([c % 64, (c // 64) % 64, c // 4096], -1).float()
        mix = n / n.sum(-1, keepdim=True).clamp(min=1)
        one = F.one_hot(qtype.clamp(0, 2), 3).float()
        return torch.where((qtype >= 3)[..., None], mix, one) @ w

    def forward(self, ids, pos, seg, qtype, valid, markers, packed: bool = True):
        allow = self.allow_mask(seg, valid, packed)
        h = self.encoder(ids, pos, allow)
        h = h + self.type_embedding(qtype, seg, packed)
        if self.head is not None:
            cd = _compute_dtype(self.head.layers[0].linear1.weight)
            bias = torch.where(allow, torch.zeros((), dtype=cd, device=h.device), -1e4)[:, None]
            drop = self.dropout if self.training else 0.0
            for lyr in self.head.layers:
                h = lyr(h, bias, drop)
        idx = markers.clamp(min=0)[..., None].expand(-1, -1, h.size(-1))
        sc = self.scorer
        m = _norm(sc[0], torch.gather(h, 1, idx))
        z = _lin(sc[3], F.gelu(_lin(sc[1], m))).squeeze(-1).float()
        return z.masked_fill(markers < 0, -1e4)

    def cast_linear(self, dtype: torch.dtype) -> "DecisionNet":
        """Matmul weights and embeddings to `dtype`; LayerNorms and the residual stay fp32."""
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                mod.to(dtype)
            elif isinstance(mod, nn.MultiheadAttention):
                mod.in_proj_weight.data = mod.in_proj_weight.data.to(dtype)
                mod.in_proj_bias.data = mod.in_proj_bias.data.to(dtype)
        self.type_emb.float()
        return self


def load_weights(net: DecisionNet, sd: Dict[str, torch.Tensor]) -> None:
    """Load a Laya- or cbjev-format state dict; the act head Laya ships is not used here."""
    own = net.state_dict()
    keep = {k: v for k, v in sd.items() if k in own}
    missing = [k for k in own if k not in keep and not k.startswith("encoder.rope_")]
    if missing:
        raise ValueError("checkpoint lacks %d tensors, e.g. %s" % (len(missing), missing[:3]))
    for k, v in keep.items():
        if tuple(v.shape) != tuple(own[k].shape):
            raise ValueError("shape mismatch for %s: %s vs %s" % (k, tuple(v.shape), tuple(own[k].shape)))
    net.load_state_dict(keep, strict=False)
