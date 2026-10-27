"""
Experimental evaluation for the report.

    python experiments.py        (about 10 minutes; writes output/experiments.json
                                  and output/exp_*.png)

A  range sensitivity      D_max in {12, 16, 20} km   -> depots, trips, km, drones
B  payload sensitivity    W_max in {4, 6, 10} kg     -> trips, km, drones, bounds
C  scaling                N in {250, 500, 1000}      -> graph size, build / Dijkstra /
                                                        routing time
D  local-search ablation  on the largest zone: sweep -> +NN -> +2-opt -> +Or-opt ->
                          rotating sweep(+relocate) / savings(+relocate), with bounds
E  depot strategy         K-Means centroids (original) vs Set Cover on the
                          outskirts: number of depots, mean client-depot
                          distance, depots inside the built-up area?
F  Dijkstra vs A*         vertices expanded on random blocked pairs
"""
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from src.algorithms import nearest_neighbor, or_opt, route_length, two_opt
from src.depots import kmeans_benchmark
from src.petals import lower_bounds, rotating_sweep, savings_petals, sweep_once
from src.pipeline import run
from src.shortest_paths import astar, dijkstra

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
E = {}


def tot(res, key):
    return res["totals"]["best"][key]


# ---------------------------------------------------------------- A: range
rows = []
for d in (12_000, 16_000, 20_000):
    t = time.time()
    r, _ = run(d_max=d, verbose=False, sweep_starts=6)
    rows.append({"d_max_km": d / 1000, "depots": r["depots"]["exact"], "greedy": r["depots"]["greedy"],
                 "zones": [z["n_clients"] for z in r["zones"]],
                 "trips": tot(r, "trips"), "km": tot(r, "km"), "drones": tot(r, "drones"),
                 "trips_lb": sum(z["bounds"]["trips"] for z in r["zones"]),
                 "km_lb": round(sum(z["bounds"]["km"] for z in r["zones"]), 1),
                 "seconds": round(time.time() - t, 1)})
    print("A", rows[-1], flush=True)
E["A_range"] = rows

# ---------------------------------------------------------------- B: payload
rows = []
for w in (4.0, 6.0, 10.0):
    r, _ = run(w_max=w, verbose=False, sweep_starts=6)
    rows.append({"w_max_kg": w, "trips": tot(r, "trips"), "km": tot(r, "km"), "drones": tot(r, "drones"),
                 "trips_lb": sum(z["bounds"]["trips"] for z in r["zones"]),
                 "km_lb": round(sum(z["bounds"]["km"] for z in r["zones"]), 1),
                 "max_clients_per_trip": max(z["best"]["max_clients_per_trip"] for z in r["zones"]),
                 "method_per_zone": [z["chosen_method"] for z in r["zones"]],
                 "sweep_vs_savings": [(z["methods"].get("sweep"), z["methods"].get("savings")) for z in r["zones"]]})
    print("B", rows[-1], flush=True)
E["B_payload"] = rows

# ---------------------------------------------------------------- C: scaling
rows = []
main_res = main_obj = None
for n in (250, 500, 1000):
    r, obj = run(n_clients=n, verbose=False, sweep_starts=6)
    rows.append({"n_clients": n, "V": r["graph"]["n_vertices"], "E": r["graph"]["n_edges"],
                 "graph_s": r["timings_s"]["graph_s"], "depots_s": r["timings_s"]["depots_s"],
                 "routing_s": r["timings_s"]["routing_s"],
                 "dijkstra_runs": r["oracle"]["dijkstra_runs"], "blocked_pairs": r["oracle"]["blocked_pairs"],
                 "direct_pairs": r["oracle"]["direct_pairs"], "depots": r["depots"]["exact"]})
    print("C", rows[-1], flush=True)
    if n == 1000:
        main_res, main_obj = r, obj
E["C_scaling"] = rows

# ---------------------------------------------------------------- D: ablation on the largest zone
z = max(main_res["zones"], key=lambda z: z["n_clients"])
zid = z["zone"]
M, nodes = main_obj["M"][zid], main_obj["nodes"][zid]
clients = list(range(1, len(nodes)))
df = main_obj["gdf_clients"].set_index("node")
weight = {i: float(df.loc[nodes[i], "payload"]) for i in clients}
d_use, w_max = main_res["config"]["d_use_m"], main_res["config"]["w_max_kg"]


def total(routes):
    return round(sum(route_length(r, M) for r in routes) / 1000, 2)


base = sweep_once(clients, M, weight, d_use, w_max)        # plain sweep, one start angle
abl = {"zone": zid, "n_clients": len(clients),
       "sweep raw (cheapest insertion)": {"trips": len(base), "km": total(base)},
       "sweep + NN reorder": {"trips": len(base), "km": total([nearest_neighbor(r, M) for r in base])},
       "sweep + NN + 2-opt": {"trips": len(base), "km": total([two_opt(nearest_neighbor(r, M), M) for r in base])},
       "sweep + NN + 2-opt + Or-opt": {"trips": len(base),
                                       "km": total([or_opt(two_opt(nearest_neighbor(r, M), M), M) for r in base])}}
t = time.time()
sw, trials = rotating_sweep(clients, M, weight, d_use, w_max, C.SWEEP_STARTS)
abl["rotating sweep + local search + relocate"] = {"trips": len(sw), "km": total(sw), "seconds": round(time.time() - t, 1),
                                                   "trials": [(a, round(b / 1000, 1)) for a, b in trials]}
t = time.time()
sv = savings_petals(clients, M, weight, d_use, w_max)
abl["savings + local search + relocate"] = {"trips": len(sv), "km": total(sv), "seconds": round(time.time() - t, 1)}
lb_t, lb_d = lower_bounds(clients, M, weight, w_max)
abl["lower bound"] = {"trips": lb_t, "km": round(lb_d / 1000, 2)}
print("D", abl, flush=True)
E["D_ablation"] = abl

# ---------------------------------------------------------------- E: depot strategy
G = main_obj["G"]
gdf_clients = main_obj["gdf_clients"]
polygon = main_obj["polygon"]
radius = d_use / 2
gk, dk, k = kmeans_benchmark(gdf_clients, C.NUMBER_OF_ZONES, C.RANDOM_SEED, C.CITY, radius)
km_mean = float(np.mean([math.hypot(r.x - dk.iloc[r.cluster].x, r.y - dk.iloc[r.cluster].y) for r in gk.itertuples()]))
km_inside = int(sum(polygon.buffer(200).contains(p) for p in dk.geometry))
ours_mean = float(np.mean([main_obj["M"][zz["zone"]][0, i] for zz in main_res["zones"]
                           for i in range(1, zz["n_clients"] + 1)]))
E["E_depots"] = {"kmeans": {"depots": int(k), "mean_client_depot_m": round(km_mean),
                            "depots_inside_builtup": km_inside, "note": "Euclidean, ignores restricted airspace"},
                 "set_cover": {"depots": main_res["depots"]["exact"], "mean_client_depot_m": round(ours_mean),
                               "depots_inside_builtup": 0, "note": "shortest flight path, outskirts only"}}
print("E", E["E_depots"], flush=True)

# ---------------------------------------------------------------- F: Dijkstra vs A*
random.seed(1)
pairs, exp_d, exp_a, t_d, t_a = 0, 0, 0, 0.0, 0.0
cl = [int(c) for c in G.clients]
while pairs < 60:
    u, v = random.sample(cl, 2)
    if G.visible([u], [v])[0]:
        continue
    t = time.time(); dist, _ = dijkstra(G.adj, u, max_dist=d_use, targets=[v]); t_d += time.time() - t
    t = time.time(); da, _, ea = astar(G.adj, G.xy, u, v, max_dist=d_use); t_a += time.time() - t
    if v not in dist:
        continue
    assert abs(da - dist[v]) < 1e-6
    exp_d += len(dist); exp_a += ea; pairs += 1
E["F_astar"] = {"blocked_pairs_tested": pairs, "dijkstra_settled_avg": round(exp_d / pairs, 1),
                "astar_expanded_avg": round(exp_a / pairs, 1), "dijkstra_ms_avg": round(1000 * t_d / pairs, 2),
                "astar_ms_avg": round(1000 * t_a / pairs, 2), "same_distances": True}
print("F", E["F_astar"], flush=True)

with open(os.path.join(OUT, "experiments.json"), "w") as f:
    json.dump(E, f, indent=1, default=float)

# ---------------------------------------------------------------- figures
A = E["A_range"]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
axs[0].bar([str(a["d_max_km"]) for a in A], [a["depots"] for a in A], color="steelblue", edgecolor="k")
axs[0].set_title("Depots needed vs D_max (km)")
axs[1].plot([a["d_max_km"] for a in A], [a["trips"] for a in A], "o-", label="trips")
axs[1].plot([a["d_max_km"] for a in A], [a["trips_lb"] for a in A], "r_", ms=15, label="lower bound")
axs[1].set_title("Trips vs D_max (km)"); axs[1].legend()
axs[2].plot([a["d_max_km"] for a in A], [a["drones"] for a in A], "o-", color="seagreen")
axs[2].set_title("Drones vs D_max (km)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "exp_A_range.png"), dpi=150); plt.close(fig)

B = E["B_payload"]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].plot([b["w_max_kg"] for b in B], [b["trips"] for b in B], "o-", label="trips")
axs[0].plot([b["w_max_kg"] for b in B], [b["trips_lb"] for b in B], "r_", ms=15, label="payload lower bound")
axs[0].set_title("Trips vs payload (kg)"); axs[0].legend()
axs[1].plot([b["w_max_kg"] for b in B], [b["km"] for b in B], "o-", label="km")
axs[1].plot([b["w_max_kg"] for b in B], [b["km_lb"] for b in B], "r_", ms=15, label="lower bound")
axs[1].set_title("Distance vs payload (kg)"); axs[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "exp_B_payload.png"), dpi=150); plt.close(fig)

Cc = E["C_scaling"]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].plot([c["n_clients"] for c in Cc], [c["E"] for c in Cc], "o-"); axs[0].set_title("|E| of the flight graph vs clients")
for key in ("graph_s", "depots_s", "routing_s"):
    axs[1].plot([c["n_clients"] for c in Cc], [c[key] for c in Cc], "o-", label=key)
axs[1].set_title("Run time (s) vs clients"); axs[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "exp_C_scaling.png"), dpi=150); plt.close(fig)

names = [k for k in abl if k not in ("zone", "n_clients", "lower bound")]
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.bar(range(len(names)), [abl[k]["km"] for k in names], color="steelblue", edgecolor="k")
ax.axhline(abl["lower bound"]["km"], color="red", linestyle="--", label="distance lower bound")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8); ax.set_ylabel("km")
ax.set_title(f"Local-search ablation on zone Z{zid} ({len(clients)} clients)"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "exp_D_ablation.png"), dpi=150); plt.close(fig)
print("written", os.path.join(OUT, "experiments.json"))
