import random

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree


def generate_clients(polygon, number_of_clients, seed=40, exclude=()):
    """Generate random client points inside `polygon` (EPSG:32634) by
    Monte-Carlo acceptance.  Points inside any `exclude` polygon (restricted
    airspace) are rejected as well: a drone cannot land there."""
    minx, miny, maxx, maxy = polygon.bounds
    tree = STRtree(list(exclude)) if len(exclude) else None
    clients = []
    random.seed(seed)
    while len(clients) < number_of_clients:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if not polygon.contains(p):
            continue
        if tree is not None and len(tree.query(p, predicate="within")):
            continue
        clients.append(p)
    gdf = gpd.GeoDataFrame(geometry=clients, crs="EPSG:32634")
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    return gdf


def assign_payloads(gdf_clients, seed=40, payload_range=(0.5, 2.3)):
    """Add a 'payload' column (kg, uniform in `payload_range`)."""
    gdf = gdf_clients.copy()
    random.seed(seed)
    lo, hi = payload_range
    gdf["payload"] = [round(random.uniform(lo, hi), 3) for _ in range(len(gdf))]
    return gdf


def compute_angles(gdf_clients, depot_xy_by_zone):
    """Polar angle of every client around the depot of its zone (column
    'cluster').  Returns {zone_id: DataFrame sorted by angle} - the input of
    the sweep heuristic."""
    gdf = gdf_clients.copy()
    gdf["angle"] = 0.0
    for zone_id, (dx0, dy0) in depot_xy_by_zone.items():
        mask = gdf["cluster"] == zone_id
        gdf.loc[mask, "angle"] = np.arctan2(gdf.loc[mask, "y"] - dy0, gdf.loc[mask, "x"] - dx0)
    return {z: gdf[gdf["cluster"] == z].sort_values("angle").copy() for z in depot_xy_by_zone}
