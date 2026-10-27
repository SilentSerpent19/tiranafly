"""
Petal (trip) construction for one zone.

The project's routing idea is the SWEEP heuristic (Gillett & Miller, 1974):
clients are taken in polar order around the depot and packed into petals
while the range and payload constraints hold.  It is strengthened here in
three ways:

  1. the metric is the exact shortest-flight-path matrix M (obstacle-aware),
  2. the petal is grown by cheapest insertion (O(n) per client) instead of
     recomputing a nearest-neighbour tour for every candidate, and
  3. the sweep is ROTATED: it is run from several start angles and the
     best result is kept.  A single start angle is arbitrary and can split
     a dense cluster across two petals; the multi-start removes that bias.

Every closed petal is improved by local search (2-opt + Or-opt), then an
inter-petal relocate pass is applied to the whole zone.

A Clarke-Wright SAVINGS construction (1964) is provided as a second,
independent heuristic and used as the baseline in the experiments.

Lower bounds (used to quantify the optimality gap):
  * trips    >= ceil( sum of payloads / W_max )
  * distance >= 1/2 * sum_c ( d1(c) + d2(c) )  where d1, d2 are the two
    smallest matrix entries of client c: every client has degree 2 in the
    cycle that serves it, so this is a valid bound on any set of cycles.
"""
import math

import numpy as np

from .algorithms import insertion_cost, local_search, relocate_between, route_length


def _finish(routes, M, weight, d_max, w_max):
    routes = [local_search(r, M) for r in routes]
    routes = relocate_between(routes, M, weight, d_max, w_max)
    routes = [local_search(r, M) for r in routes]
    for r in routes:
        assert route_length(r, M) <= d_max + 1e-6 and sum(weight[c] for c in r) <= w_max + 1e-9
    return routes


def sweep_once(order, M, weight, d_max, w_max):
    """One sweep over clients in the given angular `order`."""
    routes, cur, load, length = [], [], 0.0, 0.0
    for c in order:
        delta, pos = insertion_cost(cur, c, M)
        if cur and (load + weight[c] > w_max + 1e-9 or length + delta > d_max + 1e-9):
            routes.append(cur)
            cur, load, length = [], 0.0, 0.0
            delta, pos = insertion_cost(cur, c, M)
        cur.insert(pos, c)
        load += weight[c]
        length += delta
    if cur:
        routes.append(cur)
    return routes


def rotating_sweep(clients_by_angle, M, weight, d_max, w_max, n_starts=12):
    """Run the sweep from `n_starts` evenly spaced start positions of the
    angular order and keep the best (fewest trips, then shortest).
    Returns (best_routes, [(trips, total_m) per start])."""
    n = len(clients_by_angle)
    starts = min(n_starts, max(n, 1))
    best, best_key, trials = None, None, []
    for s in range(starts):
        k = int(s * n / starts)
        order = clients_by_angle[k:] + clients_by_angle[:k]
        routes = _finish(sweep_once(order, M, weight, d_max, w_max), M, weight, d_max, w_max)
        total = sum(route_length(r, M) for r in routes)
        key = (len(routes), total)
        trials.append(key)
        if best_key is None or key < best_key:
            best, best_key = routes, key
    return best, trials


def savings_petals(clients, M, weight, d_max, w_max):
    """Clarke-Wright savings with range and payload constraints (baseline)."""
    routes = {c: [c] for c in clients}
    owner = {c: c for c in clients}
    load = {c: weight[c] for c in clients}
    sav = sorted(((M[0, a] + M[0, b] - M[a, b], a, b) for a in clients for b in clients if a < b),
                 reverse=True)
    for s, a, b in sav:
        if s <= 0:
            break
        ra, rb = owner[a], owner[b]
        if ra == rb:
            continue
        A, B = routes[ra], routes[rb]
        if A[-1] == a and B[0] == b:
            merged = A + B
        elif B[-1] == b and A[0] == a:
            merged = B + A
        elif A[0] == a and B[0] == b:
            merged = A[::-1] + B
        elif A[-1] == a and B[-1] == b:
            merged = A + B[::-1]
        else:
            continue
        if load[ra] + load[rb] > w_max + 1e-9 or route_length(merged, M) > d_max + 1e-9:
            continue
        routes[ra] = merged
        load[ra] += load[rb]
        for c in B:
            owner[c] = ra
        del routes[rb], load[rb]
    return _finish(list(routes.values()), M, weight, d_max, w_max)


def single_client_trips(clients):
    """Naive baseline: one depot -> client -> depot trip per client."""
    return [[c] for c in clients]


def lower_bounds(clients, M, weight, w_max):
    """Lower bounds on trips and total distance for the zone.
    trips    >= ceil(total payload / W_max)
    distance >= max( degree bound, radial bound ) with
      degree bound : 1/2 sum_c (d1(c) + d2(c))             (every client has
                     degree 2 in its cycle)
      radial bound : 2/W_max * sum_c w_c * d(0, c)         (Haimovich &
                     Rinnooy Kan 1985: a trip of load L <= W_max is at least
                     2 * max_c d(0,c) >= 2/W_max * sum_{c in trip} w_c d(0,c))
    """
    trips_lb = math.ceil(sum(weight[c] for c in clients) / w_max - 1e-9)
    idx = [0] + list(clients)
    sub = M[np.ix_(idx, idx)].copy()
    np.fill_diagonal(sub, np.inf)
    two_smallest = np.sort(sub[1:], axis=1)[:, :2]
    degree_lb = 0.5 * float(two_smallest.sum())
    radial_lb = 2.0 / w_max * sum(weight[c] * M[0, c] for c in clients)
    return trips_lb, max(degree_lb, radial_lb)


def best_of(candidates, M):
    """Pick the best construction for a zone: fewest trips, then shortest.
    `candidates` is {name: routes}.  Returns (name, routes)."""
    key = lambda item: (len(item[1]), sum(route_length(r, M) for r in item[1]))
    return min(candidates.items(), key=key)


def summarize(routes, M, weight, k_pct_per_m):
    return [{"clients": list(r), "length_m": round(route_length(r, M), 1),
             "load_kg": round(sum(weight[c] for c in r), 2),
             "battery_pct": round(k_pct_per_m * route_length(r, M), 1)} for r in routes]
