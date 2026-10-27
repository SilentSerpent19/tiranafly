"""
End-to-end TiranaFly pipeline (shared by main.py and experiments.py).

    map  ->  restricted airspace  ->  clients  ->  candidate depot sites
         ->  flight graph (visibility graph, pruned at the usable range)
         ->  coverage sets (bounded Dijkstra)  ->  Set Cover (greedy + exact)
         ->  zones (multi-source Dijkstra)  ->  exact distance matrices
         ->  petals (rotating sweep | savings, local search, relocate)
         ->  fleet (FFD bin packing)  ->  results dict
"""
import math
import time

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

import config as C
from .clients import assign_payloads, compute_angles, generate_clients
from .depots import (assign_zones, best_cover_of_size, boundary_candidates, coverage_sets,
                     exact_set_cover, greedy_set_cover)
from .fleet import fleet_for_zone
from .flight_graph import DistanceOracle, FlightGraph
from .obstacles import build_obstacles
from .offline_map import setup_map
from .petals import (best_of, lower_bounds, rotating_sweep, savings_petals, single_client_trips,
                     summarize)
from .algorithms import route_length


def run(n_clients=None, d_max=None, w_max=None, seed=None, verbose=True, methods=("sweep", "savings"),
        sweep_starts=None):
    n_clients = n_clients or C.NUMBER_OF_CLIENTS
    d_max = d_max or C.D_MAX
    w_max = w_max or C.W_MAX
    seed = C.RANDOM_SEED if seed is None else seed
    sweep_starts = sweep_starts or C.SWEEP_STARTS
    d_use = d_max * (1 - C.BATTERY_RESERVE)
    radius = d_use / 2.0
    k_pct_per_m = 100.0 / d_max
    log = print if verbose else (lambda *a, **k: None)
    T = {}

    # ------------------------------------------------------------ map & airspace
    t = time.time()
    log("Setting up map...")
    _, _, polygon = setup_map(C.CITY)
    log("Loading restricted airspace...")
    obstacles, vertices = build_obstacles(C.CITY, polygon, C.SAFETY_MARGIN_M)
    T["map_s"] = time.time() - t

    # ------------------------------------------------------------ clients & sites
    t = time.time()
    log(f"Generating {n_clients} clients...")
    gdf_clients = assign_payloads(generate_clients(polygon, n_clients, seed, exclude=obstacles),
                                  seed, C.PAYLOAD_RANGE)
    log("Sampling candidate depot sites on the outskirts...")
    gdf_sites = boundary_candidates(polygon, C.CANDIDATE_SPACING_M, C.CITY, C.ROAD_SNAP_MAX_M, obstacles)
    T["clients_sites_s"] = time.time() - t

    # ------------------------------------------------------------ flight graph
    t = time.time()
    log("Building flight graph (visibility graph pruned at usable range)...")
    G = FlightGraph(gdf_clients[["x", "y"]].values, gdf_sites[["x", "y"]].values, obstacles, vertices, d_use)
    T["graph_s"] = time.time() - t
    edge_counts = {k: v for k, v in G.stats.items() if "-" in k}
    log(f"  |V| = {G.stats['n_vertices']}  |E| = {G.stats['n_edges']}  "
        f"({G.stats['build_seconds']} s)  edges: {edge_counts}")

    # ------------------------------------------------------------ depot placement
    t = time.time()
    log("Coverage sets by bounded Dijkstra from every candidate...")
    cover, dist_from = coverage_sets(G, radius)
    universe = set(int(c) for c in G.clients)
    greedy, uncovered = greedy_set_cover(universe, cover)
    exact, opt_value = exact_set_cover(universe, cover)
    servable = universe - uncovered
    tie_break, tb_cost = best_cover_of_size(opt_value, servable, cover, dist_from)
    depots = tie_break or (greedy if len(greedy) == opt_value else exact)
    greedy_cost = sum(min(dist_from[j].get(c, math.inf) for j in greedy) for c in servable)
    T["depots_s"] = time.time() - t
    log(f"  greedy Set Cover: {len(greedy)} depots, exact optimum: {opt_value}, "
        f"unservable clients: {len(uncovered)}; p-median tie-break: total client-depot distance "
        f"{(tb_cost or greedy_cost) / 1000:.0f} km vs greedy {greedy_cost / 1000:.0f} km")

    # ------------------------------------------------------------ zones
    t = time.time()
    zone_of, dist_to_depot = assign_zones(G, depots, radius)
    zone_index = {d: i for i, d in enumerate(depots)}
    gdf_clients["cluster"] = [zone_index.get(zone_of.get(int(c), None), -1) for c in G.clients]
    gdf_clients["node"] = [int(c) for c in G.clients]
    gdf_depots = gpd.GeoDataFrame(
        {"zone": list(range(len(depots))), "node": depots,
         "x": [G.xy[d][0] for d in depots], "y": [G.xy[d][1] for d in depots]},
        geometry=[Point(*G.xy[d]) for d in depots], crs="EPSG:32634")
    by_zone = compute_angles(gdf_clients[gdf_clients["cluster"] >= 0],
                             {i: tuple(G.xy[d]) for i, d in enumerate(depots)})
    T["zones_s"] = time.time() - t

    # ------------------------------------------------------------ routing
    oracle = DistanceOracle(G)
    zones_out, petals_by_zone, M_by_zone, nodes_by_zone = [], {}, {}, {}
    totals = {"naive": [0, 0.0, 0], "best": [0, 0.0, 0]}
    t = time.time()
    for z, df in by_zone.items():
        depot = depots[z]
        nodes = [depot] + [int(n) for n in df["node"]]          # angular order
        local = list(range(1, len(nodes)))
        M = oracle.matrix(nodes)
        weight = {i: float(w) for i, w in zip(local, df["payload"])}
        cand = {}
        if "sweep" in methods:
            cand["sweep"], trials = rotating_sweep(local, M, weight, d_use, w_max, sweep_starts)
        if "savings" in methods:
            cand["savings"] = savings_petals(local, M, weight, d_use, w_max)
        name, best = best_of(cand, M)
        naive = single_client_trips(local)
        trips_lb, dist_lb = lower_bounds(local, M, weight, w_max)
        euclid = np.hypot(*(G.xy[nodes][:, None, :] - G.xy[nodes][None, :, :]).T).T
        finite = np.isfinite(M) & (euclid > 0)
        detour = float((M[finite] / euclid[finite]).max()) if finite.any() else 1.0
        mean_detour = float((M[finite] / euclid[finite]).mean()) if finite.any() else 1.0

        def stats(routes):
            s = summarize(routes, M, weight, k_pct_per_m)
            f = fleet_for_zone(s, C.SPEED_KMH, C.SERVICE_MIN, C.TURNAROUND_MIN, C.WORK_HOURS)
            return {"trips": len(routes), "km": round(sum(r["length_m"] for r in s) / 1000, 2),
                    "drones": f["drones"], "drones_lb": f["lower_bound"],
                    "fleet_minutes": f["total_minutes"], "utilisation": f["utilisation"],
                    "max_clients_per_trip": max(len(r) for r in routes) if routes else 0,
                    "routes": s, "schedule": f["schedule"]}

        zinfo = {"zone": z, "depot_node": depot, "depot_xy": [float(v) for v in G.xy[depot]],
                 "n_clients": len(local),
                 "farthest_client_m": round(max(M[0, c] for c in local), 1),
                 "detour_factor_max": round(detour, 3), "detour_factor_mean": round(mean_detour, 4),
                 "bounds": {"trips": trips_lb, "km": round(dist_lb / 1000, 2)},
                 "chosen_method": name,
                 "methods": {m: {"trips": len(r), "km": round(sum(route_length(x, M) for x in r) / 1000, 2)}
                             for m, r in cand.items()},
                 "naive": stats(naive), "best": stats(best)}
        if "sweep" in methods:
            zinfo["sweep_trials"] = [(int(a), round(float(b) / 1000, 2)) for a, b in trials]
        zones_out.append(zinfo)
        petals_by_zone[z] = [[nodes[i] for i in r] for r in best]      # graph node ids
        M_by_zone[z], nodes_by_zone[z] = M, nodes
        for key in ("naive", "best"):
            totals[key][0] += zinfo[key]["trips"]; totals[key][1] += zinfo[key]["km"]; totals[key][2] += zinfo[key]["drones"]
        log(f"  Z{z}: {len(local):4d} clients | naive {zinfo['naive']['trips']} trips {zinfo['naive']['km']} km "
            f"{zinfo['naive']['drones']} drones | {name} {zinfo['best']['trips']} trips {zinfo['best']['km']} km "
            f"{zinfo['best']['drones']} drones (LB {trips_lb} trips, {dist_lb / 1000:.1f} km) | detour max {detour:.2f}")
    T["routing_s"] = time.time() - t

    results = {
        "config": {"n_clients": n_clients, "d_max_m": d_max, "reserve": C.BATTERY_RESERVE, "d_use_m": d_use,
                   "w_max_kg": w_max, "payload_range": C.PAYLOAD_RANGE, "speed_kmh": C.SPEED_KMH,
                   "service_min": C.SERVICE_MIN, "turnaround_min": C.TURNAROUND_MIN, "work_hours": C.WORK_HOURS,
                   "safety_margin_m": C.SAFETY_MARGIN_M, "seed": seed, "sweep_starts": sweep_starts},
        "map": {"coverage_km2": round(polygon.area / 1e6, 2), "n_obstacles": len(obstacles),
                "n_obstacle_vertices": len(vertices)},
        "graph": {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in G.stats.items()},
        "oracle": dict(oracle.stats),
        "depots": {"n_candidates": len(gdf_sites), "greedy": len(greedy), "exact": opt_value,
                   "chosen": depots, "unservable": len(uncovered),
                   "greedy_total_km": round(greedy_cost / 1000, 1),
                   "tiebreak_total_km": round((tb_cost or greedy_cost) / 1000, 1),
                   "coverage_sizes": {int(j): len(s) for j, s in cover.items()}},
        "zones": zones_out,
        "totals": {k: {"trips": v[0], "km": round(v[1], 2), "drones": v[2]} for k, v in totals.items()},
        "timings_s": {k: round(v, 2) for k, v in T.items()},
    }
    objects = {"polygon": polygon, "obstacles": obstacles, "G": G, "oracle": oracle,
               "gdf_clients": gdf_clients, "gdf_depots": gdf_depots, "gdf_sites": gdf_sites,
               "petals": petals_by_zone, "M": M_by_zone, "nodes": nodes_by_zone}
    return results, objects
