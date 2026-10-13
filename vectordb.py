"""VectorDB Core Engine in Python.

Implements:
1. VectorItem data representation
2. BruteForce search
3. KD-Tree spatial index
4. HNSW (Hierarchical Navigable Small World) multi-layer graph index
5. VectorDB coordinator combining all three indices with thread safety
"""

import math
import random
import heapq
import time
import threading
from typing import List, Tuple, Dict, Optional, Any
from distance import DistFn, get_dist_fn, cosine, euclidean


class VectorItem:
    """Represents a vector item stored in the database."""

    def __init__(self, item_id: int, metadata: str, category: str, emb: List[float]):
        self.id = item_id
        self.metadata = metadata
        self.category = category
        self.emb = list(emb)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "metadata": self.metadata,
            "category": self.category,
            "embedding": self.emb,
        }


# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    """Exact linear k-NN search across all vectors."""

    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        scored = [(dist(q, v.emb), v.id) for v in self.items]
        scored.sort(key=lambda x: x[0])
        return scored[:k]

    def remove(self, item_id: int) -> None:
        self.items = [v for v in self.items if v.id != item_id]


# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    """Node in the KD-Tree holding a VectorItem and split children."""

    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional[KDNode] = None
        self.right: Optional[KDNode] = None


class KDTree:
    """k-d tree for multidimensional spatial indexing and range/nearest neighbor search."""

    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    def _insert(self, node: Optional[KDNode], v: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(v)
        ax = depth % self.dims
        if v.emb[ax] < node.item.emb[ax]:
            node.left = self._insert(node.left, v, depth + 1)
        else:
            node.right = self._insert(node.right, v, depth + 1)
        return node

    def insert(self, v: VectorItem) -> None:
        self.root = self._insert(self.root, v, 0)

    def _knn(
        self,
        node: Optional[KDNode],
        q: List[float],
        k: int,
        depth: int,
        dist: DistFn,
        heap: List[Tuple[float, int]],
    ) -> None:
        if node is None:
            return

        dn = dist(q, node.item.emb)
        # We maintain a max-heap of size <= k using negative distances: (-dist, id)
        if len(heap) < k:
            heapq.heappush(heap, (-dn, node.item.id))
        elif dn < -heap[0][0]:
            heapq.heapreplace(heap, (-dn, node.item.id))

        ax = depth % self.dims
        diff = q[ax] - node.item.emb[ax]

        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist, heap)
        # If the hypersphere crosses the split plane or heap isn't full, check other subtree
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.root is None or k <= 0:
            return []
        heap: List[Tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist, heap)
        res = [(-d, item_id) for d, item_id in heap]
        res.sort(key=lambda x: x[0])
        return res

    def rebuild(self, items: List[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.insert(v)


# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class HNSWNode:
    """Node in the HNSW multilayer graph."""

    def __init__(self, item: VectorItem, max_lyr: int):
        self.item = item
        self.max_lyr = max_lyr
        # nbrs[lc] = list of neighbor IDs on layer lc
        self.nbrs: List[List[int]] = [[] for _ in range(max_lyr + 1)]


class HNSW:
    """Production-grade Hierarchical Navigable Small World Graph index.
    
    Provides approximate nearest neighbor (ANN) search with logarithmic complexity.
    """

    def __init__(self, m: int = 16, ef_build: int = 200, seed: int = 42):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_pt = -1
        self.G: Dict[int, HNSWNode] = {}
        self.rng = random.Random(seed)

    def _rand_level(self) -> int:
        u = self.rng.random()
        while u == 0.0:
            u = self.rng.random()
        return int(math.floor(-math.log(u) * self.mL))

    def search_layer(
        self, q: List[float], ep: int, ef: int, lyr: int, dist: DistFn
    ) -> List[Tuple[float, int]]:
        """Greedy best-first search within a single graph layer."""
        if ep not in self.G:
            return []

        vis = {ep: True}
        d0 = dist(q, self.G[ep].item.emb)

        # Min-heap of candidates to explore: (dist, id)
        cands: List[Tuple[float, int]] = [(d0, ep)]
        # Max-heap of nearest found so far: (-dist, id)
        found: List[Tuple[float, int]] = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            worst_found_dist = -found[0][0]

            if len(found) >= ef and cd > worst_found_dist:
                break

            if cid not in self.G or lyr >= len(self.G[cid].nbrs):
                continue

            for nid in self.G[cid].nbrs[lyr]:
                if nid in vis or nid not in self.G:
                    continue
                vis[nid] = True
                nd = dist(q, self.G[nid].item.emb)
                worst_found_dist = -found[0][0]

                if len(found) < ef or nd < worst_found_dist:
                    heapq.heappush(cands, (nd, nid))
                    if len(found) < ef:
                        heapq.heappush(found, (-nd, nid))
                    else:
                        heapq.heapreplace(found, (-nd, nid))

        res = [(-d, nid) for d, nid in found]
        res.sort(key=lambda x: x[0])
        return res

    def _select_nbrs(self, cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [cid for _, cid in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        item_id = item.id
        lvl = self._rand_level()
        self.G[item_id] = HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = lvl
            return

        ep = self.entry_pt

        # 1. Zoom in greedily from topLayer down to lvl + 1
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                w = self.search_layer(item.emb, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]

        # 2. From min(topLayer, lvl) down to layer 0, connect neighbors
        for lc in range(min(self.top_layer, lvl), -1, -1):
            w = self.search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(w, max_m)
            self.G[item_id].nbrs[lc] = list(sel)

            # Bidirectional connections
            for nid in sel:
                if nid not in self.G:
                    continue
                while len(self.G[nid].nbrs) <= lc:
                    self.G[nid].nbrs.append([])
                conn = self.G[nid].nbrs[lc]
                conn.append(item_id)
                if len(conn) > max_m:
                    ds = [
                        (dist(self.G[nid].item.emb, self.G[c].item.emb), c)
                        for c in conn
                        if c in self.G
                    ]
                    ds.sort(key=lambda x: x[0])
                    self.G[nid].nbrs[lc] = [c for _, c in ds[:max_m]]

            if w:
                ep = w[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = item_id

    def knn(self, q: List[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1 or not self.G:
            return []

        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                w = self.search_layer(q, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]

        w = self.search_layer(q, ep, max(ef, k), 0, dist)
        return w[:k]

    def remove(self, item_id: int) -> None:
        if item_id not in self.G:
            return

        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                while item_id in layer:
                    layer.remove(item_id)

        if self.entry_pt == item_id:
            self.entry_pt = -1
            for nid in self.G:
                if nid != item_id:
                    self.entry_pt = nid
                    break

        del self.G[item_id]
        if not self.G:
            self.top_layer = -1

    def get_info(self) -> Dict[str, Any]:
        top_layer = self.top_layer
        node_count = len(self.G)
        max_l = max(top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []

        for item_id, nd in self.G.items():
            nodes.append({
                "id": item_id,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr": nd.max_lyr,
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if item_id < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": item_id, "dst": nid, "lyr": lc})

        return {
            "topLayer": top_layer,
            "nodeCount": node_count,
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def __len__(self) -> int:
        return len(self.G)


# =====================================================================
#  VECTOR DATABASE COORDINATOR
# =====================================================================

class VectorDB:
    """Thread-safe vector database combining BruteForce, KD-Tree, and HNSW."""

    def __init__(self, dims: int):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(m=16, ef_build=200)
        self.mu = threading.Lock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: List[float], dist: Optional[DistFn] = None) -> int:
        if dist is None:
            dist = cosine
        with self.mu:
            v = VectorItem(self.next_id, meta, cat, emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.bf.remove(item_id)
            self.hnsw.remove(item_id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(
        self, q: List[float], k: int, metric: str = "cosine", algo: str = "hnsw"
    ) -> Dict[str, Any]:
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter_ns()

            algo_lower = (algo or "").lower().strip()
            if algo_lower == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo_lower == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, ef=50, dist=dfn)

            us = int((time.perf_counter_ns() - t0) / 1000)

            hits = []
            for d, item_id in raw:
                if item_id in self.store:
                    item = self.store[item_id]
                    hits.append({
                        "id": item.id,
                        "metadata": item.metadata,
                        "category": item.category,
                        "distance": round(d, 6),
                        "embedding": item.emb,
                    })

            return {
                "results": hits,
                "latencyUs": us,
                "algo": algo,
                "metric": metric,
            }

    def benchmark(self, q: List[float], k: int, metric: str = "cosine") -> Dict[str, Any]:
        with self.mu:
            dfn = get_dist_fn(metric)

            def time_fn(fn):
                t0 = time.perf_counter_ns()
                fn()
                return int((time.perf_counter_ns() - t0) / 1000)

            bf_us = time_fn(lambda: self.bf.knn(q, k, dfn))
            kd_us = time_fn(lambda: self.kdt.knn(q, k, dfn))
            hnsw_us = time_fn(lambda: self.hnsw.knn(q, k, ef=50, dist=dfn))

            return {
                "bruteforceUs": bf_us,
                "kdtreeUs": kd_us,
                "hnswUs": hnsw_us,
                "itemCount": len(self.store),
            }

    def all(self) -> List[Dict[str, Any]]:
        with self.mu:
            return [v.to_dict() for v in self.store.values()]

    def hnsw_info(self) -> Dict[str, Any]:
        with self.mu:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.mu:
            return len(self.store)
