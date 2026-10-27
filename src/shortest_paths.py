"""
Shortest-path algorithms on the TiranaFly flight graph.

The adjacency structure is a list `adj` where adj[u] is a list of (v, w)
pairs with w = Euclidean length of the straight flight leg u -> v in metres.

`dijkstra()` is the textbook binary-heap Dijkstra (CLRS ch. 24), extended
for the drone problem with
  * a distance bound (`max_dist`): vertices farther than the bound are
    never expanded.  Used for battery coverage (bound = usable range / 2),
  * an optional target set with early termination,
and complemented by two further algorithms built on the same relaxation
step:
  * `multi_source_dijkstra()`: all depots start at distance 0; one run
    labels every client with its nearest depot (a graph Voronoi diagram),
  * `astar()`: point-to-point search with the Euclidean heuristic, which
    is admissible and consistent because every edge weight is the
    Euclidean length of that edge, hence A* returns the exact shortest
    path while expanding a subset of the vertices Dijkstra would expand.

All three run in O((|V| + |E|) log |V|) with a binary heap.
"""
import heapq
import math


def dijkstra(adj, source, max_dist=math.inf, targets=None):
    """Single-source shortest paths.  Returns (dist, prev) dictionaries
    containing exactly the vertices settled within `max_dist`."""
    dist = {source: 0.0}
    prev = {source: None}
    settled = set()
    remaining = set(targets) if targets is not None else None
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in settled:
            continue                      # stale heap entry
        if d > max_dist:
            break                         # everything left is farther
        settled.add(u)
        if remaining is not None:
            remaining.discard(u)
            if not remaining:
                break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    # vertices that were relaxed but not settled within the bound are dropped
    dist = {v: d for v, d in dist.items() if v in settled}
    return dist, prev


def multi_source_dijkstra(adj, sources, max_dist=math.inf):
    """Shortest distance from the NEAREST of several sources.
    Returns (dist, prev, origin) where origin[v] is the source v is closest
    to.  Equivalent to Dijkstra from a virtual super-source joined to every
    source by a zero-weight edge."""
    dist, prev, origin = {}, {}, {}
    pq = []
    for s in sources:
        dist[s] = 0.0
        prev[s] = None
        origin[s] = s
        pq.append((0.0, s))
    heapq.heapify(pq)
    settled = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in settled:
            continue
        if d > max_dist:
            break
        settled.add(u)
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                origin[v] = origin[u]
                heapq.heappush(pq, (nd, v))
    dist = {v: d for v, d in dist.items() if v in settled}
    origin = {v: o for v, o in origin.items() if v in dist}
    return dist, prev, origin


def astar(adj, xy, source, target, max_dist=math.inf):
    """Point-to-point shortest path with the straight-line heuristic.
    Returns (distance, path_as_list_of_vertices) or (inf, None)."""
    tx, ty = xy[target]

    def h(u):
        x, y = xy[u]
        return math.hypot(x - tx, y - ty)

    g = {source: 0.0}
    prev = {source: None}
    closed = set()
    pq = [(h(source), 0.0, source)]
    expanded = 0
    while pq:
        f, d, u = heapq.heappop(pq)
        if u in closed:
            continue
        if u == target:
            return d, reconstruct(prev, target), expanded
        if f > max_dist:
            break
        closed.add(u)
        expanded += 1
        for v, w in adj[u]:
            nd = d + w
            if nd < g.get(v, math.inf):
                g[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd + h(v), nd, v))
    return math.inf, None, expanded


def reconstruct(prev, target):
    """Walk the predecessor map back from target to the source."""
    path, u = [], target
    while u is not None:
        path.append(u)
        u = prev[u]
    return path[::-1]
