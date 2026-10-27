"""
Depot placement and zone partitioning.

Main method (Set Cover on the outskirts)
    1. candidate sites are sampled along the outer boundary of the OSM
       coverage polygon and snapped to the nearest primary road (resupply
       by truck), see `boundary_candidates()`;
    2. for every candidate a bounded Dijkstra on the flight graph gives its
       coverage set: clients whose round trip  depot -> c -> depot  fits the
       usable range, i.e. dist(depot, c) <= D_use / 2;
    3. the fewest depots that cover every client is a minimum Set Cover
       (NP-hard).  We run Chvatal's greedy algorithm (ln n approximation)
       AND solve the instance exactly with a small integer program
       (scipy.optimize.milp, HiGHS).  The exact value certifies the greedy
       result and reveals its optimality gap;
    4. zones are the cells of the graph Voronoi diagram of the chosen
       depots: one multi-source Dijkstra labels each client with its
       nearest depot by shortest flight path.

Benchmark method (original approach, kept for the experiments)
    K-Means centroids snapped to main roads, k increased until the largest
    Euclidean centroid-client distance is <= D/2.  `cluster_depots()`,
    `snap_to_roads()`, `kmeans_benchmark()`.
"""
import math
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
from scipy.optimize import Bounds, LinearConstraint, milp
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from sklearn.cluster import KMeans

from .shortest_paths import dijkstra, multi_source_dijkstra

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "offline_osmx"
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================================
# shared: road network
# ============================================================================
def load_roads(city):
    graph_file = os.path.join(CACHE_DIR, f"{city.replace(', ', '_')}_roads.graphml")
    if os.path.exists(graph_file):
        G_roads = ox.load_graphml(graph_file)
        print("Local... OK")
    else:
        print("Online...")
        G_roads = ox.graph_from_place(city, network_type="drive")
        ox.save_graphml(G_roads, graph_file)
        print("...OK")
    return G_roads


_MAIN_ROADS = {}


def main_roads(city):
    """GeoDataFrame of motorway/trunk/primary/secondary edges (EPSG:32634), cached."""
    if city not in _MAIN_ROADS:
        G_roads = ox.project_graph(load_roads(city), to_crs="EPSG:32634")
        edges = ox.graph_to_gdfs(G_roads, nodes=False, edges=True)
        primary_types = ["motorway", "trunk", "primary", "secondary"]
        _MAIN_ROADS[city] = edges[edges["highway"].apply(
            lambda h: any(t in primary_types for t in (h if isinstance(h, list) else [h])))]
    return _MAIN_ROADS[city]


def snap_to_roads(gdf_depots, city, max_dist=None):
    """Snap depot points to the nearest primary road (EPSG:32634).  If
    `max_dist` is given, points farther than that from any main road are
    left where they are."""
    roads = main_roads(city)
    snapped = []
    for depot in gdf_depots.geometry:
        dists = roads.geometry.distance(depot)
        idx = dists.idxmin()
        if max_dist is not None and dists[idx] > max_dist:
            snapped.append(depot)
        else:
            snapped.append(nearest_points(depot, roads.loc[idx].geometry)[1])
    gdf = gdf_depots.copy()
    gdf["geometry"] = snapped
    return gdf


# ============================================================================
# main method: candidate sites on the outskirts
# ============================================================================
def boundary_candidates(coverage_polygon, spacing_m, city=None, snap_max_m=None,
                        obstacles=(), close_m=400.0):
    """Candidate depot sites every `spacing_m` along the outer boundary of
    the delivery area.  The (multi)polygon is first 'closed' (buffer out and
    back in) so that small gaps between land-use patches do not count as
    outskirts, and only the exterior ring of the largest component is used."""
    closed = coverage_polygon.buffer(close_m).buffer(-close_m)
    parts = list(closed.geoms) if hasattr(closed, "geoms") else [closed]
    main = max(parts, key=lambda p: p.area)
    ring = Polygon(main.exterior).exterior
    n = max(3, int(ring.length // spacing_m))
    pts = [ring.interpolate(i * ring.length / n) for i in range(n)]
    gdf = gpd.GeoDataFrame(geometry=pts, crs="EPSG:32634")
    if city is not None:
        gdf = snap_to_roads(gdf, city, max_dist=snap_max_m)
    # a depot cannot sit inside restricted airspace
    if obstacles:
        keep = [not any(o.contains(p) for o in obstacles) for p in gdf.geometry]
        gdf = gdf[keep].reset_index(drop=True)
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf


def coverage_sets(G, radius):
    """{depot_node: set(client nodes with shortest-path distance <= radius)}
    using one bounded Dijkstra per candidate depot."""
    client_set = set(int(c) for c in G.clients)
    cover, dist_from = {}, {}
    for j in G.depots:
        j = int(j)
        dist, _ = dijkstra(G.adj, j, max_dist=radius)
        cover[j] = {v for v in dist if v in client_set}
        dist_from[j] = dist
    return cover, dist_from


def greedy_set_cover(universe, sets):
    """Chvatal (1979): pick the set with the most uncovered elements until
    everything is covered.  Approximation ratio H(max |S_j|) <= ln n + 1."""
    universe = set(universe)
    uncovered = set(universe)
    chosen = []
    while uncovered:
        best = max(sets, key=lambda j: (len(sets[j] & uncovered), -j))
        gain = sets[best] & uncovered
        if not gain:
            break
        chosen.append(best)
        uncovered -= gain
    covered = universe - uncovered
    # drop depots made redundant by later choices
    for j in sorted(chosen, key=lambda j: len(sets[j])):
        if len(chosen) == 1:
            break
        others = set().union(*(sets[o] for o in chosen if o != j))
        if covered <= others:
            chosen.remove(j)
    return chosen, uncovered


def exact_set_cover(universe, sets):
    """Minimum Set Cover solved exactly as a 0/1 integer program:
        min sum_j x_j   s.t.  sum_{j : c in S_j} x_j >= 1  for every client c.
    Used to certify the greedy solution (optimality gap)."""
    keys = list(sets)
    universe = [c for c in universe if any(c in sets[j] for j in keys)]
    if not universe:
        return [], 0
    A = np.zeros((len(universe), len(keys)))
    for col, j in enumerate(keys):
        for row, c in enumerate(universe):
            if c in sets[j]:
                A[row, col] = 1
    res = milp(c=np.ones(len(keys)), integrality=np.ones(len(keys)),
               bounds=Bounds(0, 1), constraints=LinearConstraint(A, lb=1, ub=np.inf))
    chosen = [keys[i] for i, x in enumerate(res.x) if x > 0.5]
    return chosen, int(round(res.fun))


def best_cover_of_size(k, universe, sets, dist_from, max_combinations=300_000):
    """Secondary objective.  Several depot sets of the minimum size k may
    cover everybody; among them choose the one minimising the total
    shortest-path distance from each client to its nearest chosen depot
    (the p-median criterion), which balances the zones and shortens every
    single-client trip.  Enumerates C(n, k) combinations when that is
    affordable, otherwise returns None."""
    from itertools import combinations
    from math import comb
    keys = list(sets)
    if comb(len(keys), k) > max_combinations:
        return None, None
    universe = list(universe)
    best, best_cost = None, math.inf
    for combo in combinations(keys, k):
        covered = set().union(*(sets[j] for j in combo))
        if not all(c in covered for c in universe):
            continue
        cost = sum(min(dist_from[j].get(c, math.inf) for j in combo) for c in universe)
        if cost < best_cost:
            best, best_cost = list(combo), cost
    return best, best_cost


def assign_zones(G, depots, radius):
    """Graph Voronoi: one multi-source Dijkstra from all chosen depots.
    Returns {client_node: depot_node} for clients within `radius` of their
    nearest depot, plus the distance map."""
    dist, prev, origin = multi_source_dijkstra(G.adj, [int(d) for d in depots], max_dist=radius)
    zone_of = {int(c): origin[int(c)] for c in G.clients if int(c) in origin}
    return zone_of, dist


# ============================================================================
# benchmark method: K-Means (original approach)
# ============================================================================
def cluster_depots(gdf_clients, number_of_zones, seed):
    coords = gdf_clients[["x", "y"]]
    kmeans = KMeans(n_clusters=number_of_zones, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(coords)
    gdf = gdf_clients.copy()
    gdf["cluster"] = labels
    centers = kmeans.cluster_centers_
    gdf_depots = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in centers], crs="EPSG:32634")
    return gdf, gdf_depots, centers, labels


def kmeans_benchmark(gdf_clients, k0, seed, city, radius, max_k=12):
    """Original strategy: increase k until every client is within `radius`
    (Euclidean) of its centroid.  Iterative instead of recursive, and the
    road snapping is applied once at the end."""
    for k in range(k0, max_k + 1):
        gdf, gdf_depots, centers, labels = cluster_depots(gdf_clients, k, seed)
        d = np.linalg.norm(gdf[["x", "y"]].values - centers[labels], axis=1)
        if d.max() <= radius:
            break
    gdf_depots = snap_to_roads(gdf_depots, city)
    gdf_depots["x"] = gdf_depots.geometry.x
    gdf_depots["y"] = gdf_depots.geometry.y
    return gdf, gdf_depots, k
