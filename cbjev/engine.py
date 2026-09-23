"""Running the network: batching, padding buckets and CUDA graph replay.

A decision call is small (tens to a few thousand tokens), so on a GPU its cost is dominated by
launching ~300 kernels from Python rather than by arithmetic. The engine pads each batch up to a
shape bucket and, the first time it sees a bucket, records the whole forward as a CUDA graph;
later calls with that bucket copy their inputs into the graph's buffers and replay it in one
launch. Buckets are coarse enough that a service sees a handful of them, and fine enough that
padding waste stays under ~25 %.

Everything else (CPU, MPS, or `graphs=False`) runs the same forward eagerly.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .model import DecisionNet


def _bucket(n: int, steps=(32, 64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024)) -> int:
    for s in steps:
        if n <= s:
            return s
    return (n + 255) // 256 * 256


def _rows_bucket(n: int) -> int:
    for s in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
        if n <= s:
            return s
    return (n + 31) // 32 * 32


class Batch:
    """Host-side arrays for one forward pass."""

    def __init__(self, B: int, L: int, M: int, pad: int):
        self.ids = np.full((B, L), pad, dtype=np.int64)
        self.pos = np.zeros((B, L), dtype=np.int64)
        self.seg = np.full((B, L), -1, dtype=np.int64)
        self.qtype = np.zeros((B, L), dtype=np.int64)
        self.valid = np.zeros((B, L), dtype=bool)
        self.markers = np.full((B, M), -1, dtype=np.int64)


class Engine:
    def __init__(self, net: DecisionNet, device: torch.device, dtype: torch.dtype, packed: bool,
                 graphs: bool = True, max_graphs: int = 48):
        self.net, self.device, self.dtype, self.packed = net, device, dtype, packed
        self.graphs = graphs and device.type == "cuda"
        self.max_graphs = max_graphs
        self._graphs: Dict[Tuple[int, int, int], Tuple] = {}
        self._lock = threading.Lock()
        self._pool = None

    def _eager(self, t):
        return self.net(*t, packed=self.packed)

    def _to_device(self, b: Batch):
        # one pinned host buffer, one copy, then views: fewer tiny H2D transfers per call
        host = np.concatenate([b.ids.ravel(), b.pos.ravel(), b.seg.ravel(), b.qtype.ravel(),
                               b.valid.ravel().astype(np.int64), b.markers.ravel()])
        buf = torch.from_numpy(host)
        if self.device.type == "cuda":
            buf = buf.pin_memory().to(self.device, non_blocking=True)
        else:
            buf = buf.to(self.device)
        B, L = b.ids.shape
        M = b.markers.shape[1]
        n = B * L
        ids, pos, seg, qt, valid = (buf[i * n:(i + 1) * n].view(B, L) for i in range(5))
        markers = buf[5 * n:].view(B, M)
        return ids, pos, seg, qt, valid.bool(), markers

    def _capture(self, key, tensors):
        static = [t.clone() for t in tensors]
        s = torch.cuda.Stream(self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            for _ in range(2):
                self._eager(static)
        torch.cuda.current_stream(self.device).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=self._pool):
            out = self._eager(static)
        if len(self._graphs) >= self.max_graphs:
            self._graphs.pop(next(iter(self._graphs)))
        self._graphs[key] = (g, static, out)
        return self._graphs[key]

    @torch.inference_mode()
    def run(self, b: Batch) -> np.ndarray:
        """Marker logits [B, M] as float32 numpy."""
        t = self._to_device(b)
        if not self.graphs:
            return self._eager(t).float().cpu().numpy()
        key = b.ids.shape + (b.markers.shape[1],)
        with self._lock:
            entry = self._graphs.get(key) or self._capture(key, t)
            g, static, out = entry
            for dst, src in zip(static, t):
                dst.copy_(src)
            g.replay()
            return out.float().cpu().numpy()

    def warm(self, shapes: List[Tuple[int, int, int]]):
        for B, L, M in shapes:
            b = Batch(B, L, M, 0)
            b.valid[:, :2] = True
            b.markers[:, 0] = 1
            b.seg[:, :] = 0
            self.run(b)

    def clear(self):
        with self._lock:
            self._graphs.clear()
            self._pool = None
