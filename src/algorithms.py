"""
Route construction and improvement on a shortest-path distance matrix.

All functions work on ONE zone: `M` is the (n+1) x (n+1) matrix of exact
shortest flight distances returned by `DistanceOracle.matrix()`, index 0
is the depot and 1..n are the clients of the zone.  A route is a list of
client indices; its length is the closed cycle  0 -> r[0] -> ... -> r[-1] -> 0.
Because M already contains the detours around restricted airspace, every
heuristic below automatically produces obstacle-aware routes.

Construction
    nearest_neighbor()    greedy NN tour (kept as the baseline of the report)
    cheapest_insertion()  insert each client where it increases the cycle
                          least - O(n) per insertion, better than NN and the
                          natural primitive for the incremental sweep
Improvement (local search; each move only shortens the cycle, so a feasible
route stays feasible)
    two_opt()             best-improvement 2-opt with O(1) move evaluation
    or_opt()              move a segment of 1..3 clients elsewhere
    local_search()        2-opt + Or-opt to a local optimum
    relocate_between()    inter-route move: shift one client to another
                          route when that shortens the total and keeps both
                          routes within range and payload
"""
import math


def route_length(route, M):
    if not route:
        return 0.0
    total = M[0, route[0]] + M[route[-1], 0]
    for a, b in zip(route, route[1:]):
        total += M[a, b]
    return total


# ---------------------------------------------------------------- construction
def nearest_neighbor(clients, M):
    unvisited = list(clients)
    route, cur = [], 0
    while unvisited:
        nxt = min(unvisited, key=lambda c: M[cur, c])
        unvisited.remove(nxt)
        route.append(nxt)
        cur = nxt
    return route


def insertion_cost(route, c, M):
    """Cheapest position to insert client c into `route`.
    Returns (delta_length, position)."""
    if not route:
        return M[0, c] + M[c, 0], 0
    best, pos = math.inf, 0
    seq = [0] + route + [0]
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        d = M[a, c] + M[c, b] - M[a, b]
        if d < best:
            best, pos = d, i
    return best, pos


def cheapest_insertion(clients, M):
    route = []
    for c in clients:
        _, pos = insertion_cost(route, c, M)
        route.insert(pos, c)
    return route


# ---------------------------------------------------------------- improvement
def two_opt(route, M):
    """Best-improvement 2-opt: reverse route[i..j] if that shortens the
    cycle.  Each move is evaluated in O(1) from four matrix entries."""
    r = list(route)
    n = len(r)
    improved = True
    while improved and n >= 2:
        improved = False
        best_gain, best_ij = 1e-9, None
        seq = [0] + r + [0]
        for i in range(1, n):
            for j in range(i + 1, n + 1):
                a, b = seq[i - 1], seq[i]
                c, d = seq[j], seq[j + 1]
                gain = (M[a, b] + M[c, d]) - (M[a, c] + M[b, d])
                if gain > best_gain:
                    best_gain, best_ij = gain, (i, j)
        if best_ij:
            i, j = best_ij
            r[i - 1:j] = r[i - 1:j][::-1]
            improved = True
    return r


def or_opt(route, M, max_seg=3):
    """Move a segment of 1..max_seg consecutive clients to the best other
    position (possibly reversed) while that shortens the cycle."""
    r = list(route)
    improved = True
    while improved:
        improved = False
        n = len(r)
        seq = [0] + r + [0]
        best_gain, best_route = 1e-9, None
        for L in range(1, min(max_seg, n - 1) + 1):
            for i in range(0, n - L + 1):
                seg = r[i:i + L]
                rest = r[:i] + r[i + L:]
                before, after = seq[i], seq[i + L + 1]
                removal_gain = M[before, seg[0]] + M[seg[-1], after] - M[before, after]
                seqr = [0] + rest + [0]
                for rev in (False, True):
                    s = seg[::-1] if rev else seg
                    for k in range(len(rest) + 1):
                        a, b = seqr[k], seqr[k + 1]
                        ins = M[a, s[0]] + M[s[-1], b] - M[a, b]
                        gain = removal_gain - ins
                        if gain > best_gain:
                            best_gain, best_route = gain, rest[:k] + s + rest[k:]
        if best_route is not None:
            r = best_route
            improved = True
    return r


def local_search(route, M):
    r = list(route)
    while True:
        before = route_length(r, M)
        r = two_opt(r, M)
        r = or_opt(r, M)
        if route_length(r, M) >= before - 1e-9:
            return r


def relocate_between(routes, M, weight, d_max, w_max):
    """Inter-route relocate.  `routes` is a list of client lists, `weight`
    maps client -> kg.  Repeatedly move one client from route A to the
    cheapest position of route B when the total length decreases and B
    stays within range and payload.  Empty routes are removed, so this
    move can also reduce the number of trips."""
    routes = [list(r) for r in routes]
    load = [sum(weight[c] for c in r) for r in routes]
    length = [route_length(r, M) for r in routes]
    improved = True
    while improved:
        improved = False
        best_gain, best = 1e-6, None
        for a, A in enumerate(routes):
            for idx, c in enumerate(A):
                prev_ = A[idx - 1] if idx > 0 else 0
                next_ = A[idx + 1] if idx + 1 < len(A) else 0
                removal_gain = M[prev_, c] + M[c, next_] - M[prev_, next_]
                for b, B in enumerate(routes):
                    if a == b or load[b] + weight[c] > w_max + 1e-9:
                        continue
                    delta, pos = insertion_cost(B, c, M)
                    if length[b] + delta > d_max + 1e-9:
                        continue
                    gain = removal_gain - delta
                    if gain > best_gain:
                        best_gain, best = gain, (a, idx, b, pos)
        if best:
            a, idx, b, pos = best
            c = routes[a].pop(idx)
            routes[b].insert(pos, c)
            load[a] -= weight[c]
            load[b] += weight[c]
            length[a] = route_length(routes[a], M)
            length[b] = route_length(routes[b], M)
            if not routes[a]:
                del routes[a], load[a], length[a]
            improved = True
    return routes
