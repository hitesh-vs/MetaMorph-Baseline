"""
metamorph/algos/ppo/gcn.py

MorphologyGCN — a small 2-layer GCN that converts per-limb node features
into structural embeddings. Its parameters are learned end-to-end by PPO
(no separate loss, no supervision — just policy gradient through the whole network).

Lives here rather than in the env because:
  - it has learnable parameters → it IS part of the model
  - the env only provides raw graph data (adjacency, features)
  - swapping encoding modes is a model-level decision, not an env-level one

Consumed by the Actor/Critic in actor_critic.py via:
    self.gcn = build_gcn_from_cfg(cfg)   # returns None if mode == "none"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metamorph.config import cfg


# ──────────────────────────────────────────────────────────────
# Layer
# ──────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """
    H' = σ( Â H W )
    where Â = D^{-1/2} A D^{-1/2} (passed in precomputed).

    Params: in_dim * out_dim + out_dim
    """
    def __init__(self, in_dim: int, out_dim: int, activation: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.use_activation = activation

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        X:      (..., N, in_dim)
        A_norm: (N, N)  or  (..., N, N)
        """
        out = A_norm @ X            # neighbor aggregation
        out = self.linear(out)      # linear transform
        if self.use_activation:
            out = F.relu(out)
        return out


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

class MorphologyGCN(nn.Module):
    """
    2-layer GCN producing per-limb structural embeddings.

    Defaults (topological input, hidden=16, out=8):
      Layer 1: 6×16 + 16  = 112 params
      Layer 2: 16×8  + 8  = 136 params
      Total  :               248 params  ← intentionally tiny

    The output embeddings are concatenated with proprioceptive observations
    per limb before the policy MLP head.
    """

    def __init__(self, node_feat_dim: int, hidden_dim: int = 16, out_dim: int = 8):
        super().__init__()
        self.out_dim = out_dim
        self.gcn1 = GCNLayer(node_feat_dim, hidden_dim, activation=True)
        self.gcn2 = GCNLayer(hidden_dim, out_dim, activation=False)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        X:      (batch, N, node_feat_dim)  or  (N, node_feat_dim)
        A_norm: (N, N)
        Returns (batch, N, out_dim)        or  (N, out_dim)
        """
        h = self.gcn1(X, A_norm)
        h = self.gcn2(h, A_norm)
        return h

    # ------------------------------------------------------------------
    # Numpy helpers (used at obs-assembly time, outside training loop)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_adjacency(A: np.ndarray) -> torch.Tensor:
        deg = A.sum(axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
        A_norm = (d_inv_sqrt @ A @ d_inv_sqrt).astype(np.float32)
        return torch.from_numpy(A_norm)

    @torch.no_grad()
    def embed(self, X_np: np.ndarray, A_np: np.ndarray) -> np.ndarray:
        """numpy in → numpy out, no grad. For obs pre-processing."""
        A_norm = self.normalize_adjacency(A_np)
        X = torch.from_numpy(X_np.astype(np.float32))
        return self.forward(X, A_norm).numpy()


# ──────────────────────────────────────────────────────────────
# Factory (reads cfg so callers don't need to know the details)
# ──────────────────────────────────────────────────────────────

def build_gcn_from_cfg(cfg) -> "MorphologyGCN | None":
    """
    Returns a MorphologyGCN if GRAPH_ENCODING is 'onehot' or 'topological',
    or None for the 'none' baseline.

    The caller (actor_critic.py) should:
        self.gcn = build_gcn_from_cfg(cfg)
        if self.gcn is not None:
            prop_dim += self.gcn.out_dim   # widen the MLP input
    """
    mode = cfg.MODEL.GRAPH_ENCODING   # "none" | "onehot" | "topological"

    if mode == "none":
        return None

    feat_dim = {
        "onehot":      7,   # fixed vocab, see graphs/parser.py::ONEHOT_CATEGORIES
        "topological": 6,   # depth, n_children, subtree_size, is_leaf, is_root, degree
    }[mode]

    return MorphologyGCN(
        node_feat_dim=feat_dim,
        hidden_dim=cfg.MODEL.GCN.HIDDEN_DIM,
        out_dim=cfg.MODEL.GCN.OUT_DIM,
    )