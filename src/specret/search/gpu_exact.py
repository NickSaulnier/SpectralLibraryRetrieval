"""Exact cosine top-k against an in-memory library.

With a few thousand library spectra, exact search is one matrix multiply: a full EMIT scene
(~1.5M pixels) takes well under a second on a modern GPU. Falls back to NumPy on the CPU
when PyTorch or CUDA is unavailable.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # the "gpu" extra is optional
    torch = None


def _pick_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ExactIndex:
    """Top-k by cosine similarity. Rows of `vectors` must be unit length."""

    def __init__(self, vectors: np.ndarray, ids: np.ndarray, device: str | None = None):
        self.ids = np.asarray(ids)
        self.device = _pick_device(device)
        self.use_torch = torch is not None
        if self.use_torch:
            self._lib = torch.as_tensor(np.asarray(vectors, dtype=np.float32), device=self.device)
        else:
            self._lib = np.asarray(vectors, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def search(
        self, queries: np.ndarray, k: int = 10, batch: int = 65536
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices), each (n_queries, k), best first."""
        q_all = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        k = min(k, len(self))
        scores = np.empty((q_all.shape[0], k), dtype=np.float32)
        idx = np.empty((q_all.shape[0], k), dtype=np.int64)
        for start in range(0, q_all.shape[0], batch):
            q = q_all[start : start + batch]
            if self.use_torch:
                with torch.no_grad():
                    sim = torch.as_tensor(q, device=self.device) @ self._lib.T
                    s, i = torch.topk(sim, k, dim=1)
                scores[start : start + len(q)] = s.cpu().numpy()
                idx[start : start + len(q)] = i.cpu().numpy()
            else:
                sim = q @ self._lib.T
                part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
                order = np.argsort(-np.take_along_axis(sim, part, axis=1), axis=1)
                top = np.take_along_axis(part, order, axis=1)
                scores[start : start + len(q)] = np.take_along_axis(sim, top, axis=1)
                idx[start : start + len(q)] = top
        return scores, idx
