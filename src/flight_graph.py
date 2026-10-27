"""
The TiranaFly flight graph  G = (V, E)  and its exact distance oracle.

Vertices V
    clients  : delivery points inside the OSM coverage polygon
    depots   : candidate depot sites on the outskirts
    vertices : corners of the (buffered) restricted-airspace polygons

Edges E   (every edge is a straight flight leg, weight = its length in m)
    depot  - client     if the segment avoids every restricted polygon
    depot  - vertex     "
    client - vertex     "
    vertex - vertex     "
    and only if the length is <= D_max  (graph pruning: a leg longer than
    the range can never be flown, so its weight is infinity).

Why this graph is exact (and why there are no client-client edges)
    Lozano-Perez & Wesley (1979): among polygonal obstacles the shortest
    path between two points is a polyline whose interior turning points are
    obstacle vertices.  Hence
      * if two points see each other, the shortest path is the segment
        itself and no graph search is needed;
      * otherwise the shortest path runs  p -> v1 -> ... -> vk -> q  with
        v1..vk obstacle vertices, and every one of those legs is an edge of
        G.  Dijkstra / A* on G therefore returns the exact optimum.
    Client-client visible pairs are evaluated directly as segments by the
    DistanceOracle (same optimality argument), which keeps |E| at
    O(n * m) instead of O(n^2) for n clients and m obstacle vertices.

Visibility tests use a Shapely STRtree over the restricted polygons shrunk
by 0.5 m, so that a leg running along a polygon boundary (e.g. between two
consecutive vertices) is allowed while a leg through the interior is not.
"""
import math
import time

import numpy as np
import shapely
from shapely.strtree import STRtree

from .shortest_paths import dijkstra, reconstruct

CLIENT, DEPOT, VERTEX = 0, 1, 2


class FlightGraph:
    def __init__(self, clients_xy, depots_xy, obstacles, vertices_xy, d_max):
        self.d_max = float(d_max)
        self.xy = np.array(list(clients_xy) + list(depots_xy) + list(vertices_xy), dtype=float)
        n_c, n_d, n_v = len(clients_xy), len(depots_xy), len(vertices_xy)
        self.kind = np.array([CLIENT] * n_c + [DEPOT] * n_d + [VERTEX] * n_v)
        self.clients = np.arange(0, n_c)
        self.depots = np.arange(n_c, n_c + n_d)
        self.vertices = np.arange(n_c + n_d, n_c + n_d + n_v)
        self.obstacles = list(obstacles)
        shrunk = [p.buffer(-0.5) for p in self.obstacles]
        self._tree = STRtree(shrunk) if shrunk else None
        self.adj = [[] for _ in range(len(self.xy))]
        self.n_edges = 0
        self.stats = {}
        self._build()

    # ------------------------------------------------------------------ geometry
    def visible(self, a, b):
        """Vectorised visibility of segments xy[a[k]] -> xy[b[k]].
        Returns a boolean array (True = no restricted polygon crossed)."""
        a, b = np.asarray(a), np.asarray(b)
        ok = np.ones(len(a), dtype=bool)
        if self._tree is None or len(a) == 0:
            return ok
        coords = np.stack([self.xy[a], self.xy[b]], axis=1)          # (k, 2, 2)
        lines = shapely.linestrings(coords)
        hit, _ = self._tree.query(lines, predicate="intersects")
        ok[np.unique(hit)] = False
        return ok

    def euclid(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        return np.hypot(*(self.xy[a] - self.xy[b]).T)

    # ------------------------------------------------------------------ build
    def _add_edges(self, A, B):
        """Candidate pairs (A[k], B[k]) -> keep visible ones within d_max."""
        A, B = np.asarray(A), np.asarray(B)
        if len(A) == 0:
            return 0
        w = self.euclid(A, B)
        keep = w <= self.d_max
        A, B, w = A[keep], B[keep], w[keep]
        vis = self.visible(A, B)
        A, B, w = A[vis], B[vis], w[vis]
        for u, v, d in zip(A.tolist(), B.tolist(), w.tolist()):
            self.adj[u].append((v, d))
            self.adj[v].append((u, d))
        return len(A)

    def _build(self):
        t0 = time.time()
        C, D, V = self.clients, self.depots, self.vertices
        pairs = {
            "depot-client": np.array(np.meshgrid(D, C)).reshape(2, -1),
            "depot-vertex": np.array(np.meshgrid(D, V)).reshape(2, -1),
            "client-vertex": np.array(np.meshgrid(C, V)).reshape(2, -1),
        }
        iu = np.triu_indices(len(V), k=1)
        pairs["vertex-vertex"] = np.stack([V[iu[0]], V[iu[1]]]) if len(V) > 1 else np.zeros((2, 0), int)
        for name, (A, B) in pairs.items():
            self.stats[name] = self._add_edges(A, B)
        self.n_edges = sum(self.stats.values())
        self.stats["build_seconds"] = round(time.time() - t0, 2)
        self.stats["n_vertices"] = len(self.xy)
        self.stats["n_edges"] = self.n_edges

    # ------------------------------------------------------------------ queries
    def path_xy(self, path):
        return [tuple(self.xy[u]) for u in path]


class DistanceOracle:
    """Exact shortest flight distances between a subset of graph nodes.

    d(i, j) = |ij|                     if i sees j and |ij| <= d_max
            = Dijkstra_G(i, j)         otherwise (via obstacle vertices)
            = inf                      if no feasible path within d_max
    Every answer is an exact shortest path by the theorem quoted in
    flight_graph.py.  Dijkstra is only run from nodes that have at least
    one blocked partner, and each run is bounded by d_max.
    """

    def __init__(self, G: FlightGraph):
        self.G = G
        self.stats = {"dijkstra_runs": 0, "blocked_pairs": 0, "direct_pairs": 0}
        self._prev = {}                 # source -> predecessor map (for paths)

    def matrix(self, nodes):
        G = self.G
        nodes = np.asarray(nodes)
        n = len(nodes)
        iu = np.triu_indices(n, k=1)
        A, B = nodes[iu[0]], nodes[iu[1]]
        w = G.euclid(A, B)
        vis = G.visible(A, B)
        direct = vis & (w <= G.d_max)
        M = np.full((n, n), math.inf)
        np.fill_diagonal(M, 0.0)
        M[iu[0][direct], iu[1][direct]] = w[direct]
        M[iu[1][direct], iu[0][direct]] = w[direct]
        self.stats["direct_pairs"] += int(direct.sum())
        blocked = ~direct & (w <= G.d_max)       # worth searching (bound on detour not known, so search)
        self.stats["blocked_pairs"] += int(blocked.sum())
        # Dijkstra from every node that has a blocked partner
        need = {}
        for k in np.flatnonzero(blocked):
            need.setdefault(iu[0][k], []).append(iu[1][k])
        pos = {int(v): i for i, v in enumerate(nodes)}
        for i_local, targets_local in need.items():
            src = int(nodes[i_local])
            targets = [int(nodes[t]) for t in targets_local]
            dist, prev = dijkstra(G.adj, src, max_dist=G.d_max, targets=targets)
            self.stats["dijkstra_runs"] += 1
            self._prev[src] = prev
            for t in targets:
                d = dist.get(t, math.inf)
                M[i_local, pos[t]] = M[pos[t], i_local] = d
        return M

    def path(self, i, j):
        """Coordinates of the shortest path i -> j (direct segment or via
        obstacle vertices)."""
        G = self.G
        if i == j:
            return [tuple(G.xy[i])]
        if G.visible([i], [j])[0]:
            return [tuple(G.xy[i]), tuple(G.xy[j])]
        prev = self._prev.get(i)
        if prev is None or j not in prev:
            dist, prev = dijkstra(G.adj, i, max_dist=G.d_max, targets=[j])
            self._prev[i] = prev
            if j not in dist:
                return None
        return G.path_xy(reconstruct(prev, j))
