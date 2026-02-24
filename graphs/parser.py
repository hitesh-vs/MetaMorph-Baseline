"""
graphs/parser.py

Parses a MuJoCo XML into a graph and produces node features.
No GCN here — this is pure data extraction.
The env imports this to build graph metadata at init time.
The policy model imports this (or uses precomputed outputs) for GCN input.

Three feature modes driven by cfg.MODEL.GRAPH_ENCODING:
  "none"        — no graph, returns None (baseline)
  "onehot"      — name-heuristic one-hot labels  (hardcoded semantic roles)
  "topological" — topology-derived features only  (no name parsing)
"""

import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict


# All possible semantic categories for one-hot (fixed vocab so dim is consistent)
ONEHOT_CATEGORIES = ["torso", "hip", "knee", "ankle", "shoulder", "elbow", "other"]


class MujocoGraphParser:
    """
    Parses worldbody of a MuJoCo XML into nodes + edges.
    Call once per robot at env init time; results are static.
    """

    def __init__(self, xml_path: str):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        self.nodes = []   # list of body names in DFS order
        self.edges = []   # list of (parent_name, child_name) tuples
        self._parse_worldbody(root)

        # Precompute index map
        self.idx = {name: i for i, name in enumerate(self.nodes)}
        self.N = len(self.nodes)

    def _parse_worldbody(self, root):
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("No <worldbody> found in XML")
        for body in worldbody.findall("body"):
            self._traverse(body, parent=None)

    def _traverse(self, body, parent):
        name = body.attrib.get("name", f"body_{len(self.nodes)}")
        self.nodes.append(name)
        if parent is not None:
            self.edges.append((parent, name))
        for child in body.findall("body"):
            self._traverse(child, parent=name)

    # ------------------------------------------------------------------
    # Adjacency
    # ------------------------------------------------------------------

    def build_adjacency(self) -> np.ndarray:
        """
        Returns (N, N) undirected adjacency matrix with self-loops.
        Self-loops are required for GCN (each node attends to itself).
        """
        A = np.zeros((self.N, self.N), dtype=np.float32)
        for p, c in self.edges:
            i, j = self.idx[p], self.idx[c]
            A[i, j] = 1.0
            A[j, i] = 1.0
        A += np.eye(self.N, dtype=np.float32)
        return A

    def normalized_adjacency(self) -> np.ndarray:
        """
        D^{-1/2} A D^{-1/2} symmetric normalization.
        Ready to pass directly into GCN layers.
        """
        A = self.build_adjacency()
        deg = A.sum(axis=1)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
        return (d_inv_sqrt @ A @ d_inv_sqrt).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature modes
    # ------------------------------------------------------------------

    def features_onehot(self) -> np.ndarray:
        """
        (N, 7) one-hot based on name heuristics.
        Fixed vocab = ONEHOT_CATEGORIES so dim is always 7,
        even when a robot has no shoulders/elbows.
        This encodes hardcoded semantic roles — good baseline, not general.
        """
        cat_idx = {c: i for i, c in enumerate(ONEHOT_CATEGORIES)}
        X = np.zeros((self.N, len(ONEHOT_CATEGORIES)), dtype=np.float32)
        for i, name in enumerate(self.nodes):
            n = name.lower()
            if "torso" in n or "pelvis" in n:
                cat = "torso"
            elif "thigh" in n or "hip" in n:
                cat = "hip"
            elif "knee" in n:
                cat = "knee"
            elif "ankle" in n:
                cat = "ankle"
            elif "shoulder" in n:
                cat = "shoulder"
            elif "elbow" in n:
                cat = "elbow"
            else:
                cat = "other"
            X[i, cat_idx[cat]] = 1.0
        return X

    def features_topological(self) -> np.ndarray:
        """
        (N, 6) topology-derived features. Zero name parsing.
        Works identically on any robot regardless of naming.

        Columns:
          0  depth from root            (normalized 0-1)
          1  number of children         (normalized 0-1)
          2  subtree size               (normalized by N)
          3  is_leaf                    (binary)
          4  is_root                    (binary)
          5  degree                     (normalized 0-1)
        """
        parent_map = {name: None for name in self.nodes}
        children_map = defaultdict(list)
        for p, c in self.edges:
            parent_map[c] = p
            children_map[p].append(c)

        root = next(n for n in self.nodes if parent_map[n] is None)

        # Depth
        depth = {}
        def _depth(node, d):
            depth[node] = d
            for ch in children_map[node]:
                _depth(ch, d + 1)
        _depth(root, 0)

        # Subtree size
        subtree = {}
        def _subtree(node):
            s = 1 + sum(_subtree(ch) for ch in children_map[node])
            subtree[node] = s
            return s
        _subtree(root)

        # Degree (without self-loop)
        degree = defaultdict(int)
        for p, c in self.edges:
            degree[p] += 1
            degree[c] += 1

        max_d = max(depth.values()) or 1
        max_ch = max(len(children_map[n]) for n in self.nodes) or 1
        max_deg = max(degree.values()) or 1

        X = np.zeros((self.N, 6), dtype=np.float32)
        for i, name in enumerate(self.nodes):
            X[i, 0] = depth[name] / max_d
            X[i, 1] = len(children_map[name]) / max_ch
            X[i, 2] = subtree[name] / self.N
            X[i, 3] = float(len(children_map[name]) == 0)   # is_leaf
            X[i, 4] = float(parent_map[name] is None)        # is_root
            X[i, 5] = degree[name] / max_deg
        return X

    def get_features(self, mode: str) -> np.ndarray:
        """
        Convenience dispatcher.
        mode: "onehot" | "topological"
        """
        if mode == "onehot":
            return self.features_onehot()
        elif mode == "topological":
            return self.features_topological()
        else:
            raise ValueError(f"Unknown feature mode: {mode!r}. Use 'onehot' or 'topological'.")

    @property
    def onehot_dim(self) -> int:
        return len(ONEHOT_CATEGORIES)   # always 7

    @property
    def topological_dim(self) -> int:
        return 6