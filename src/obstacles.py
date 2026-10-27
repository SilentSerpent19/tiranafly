"""
Restricted airspace (no-fly zones) for the TiranaFly flight graph.

Polygons are fetched ONCE from OpenStreetMap (hospitals, military areas,
stadiums, aerodromes/heliports, government buildings) and cached in
src/offline_osmx/<city>_restricted.gpkg, exactly like the other map layers.

Each polygon is buffered by a safety margin (horizontal separation a drone
must keep from the site), simplified, and dissolved.  The vertices of the
resulting polygons become nodes of the flight graph (see flight_graph.py):
by the Lozano-Perez & Wesley (1979) theorem, the shortest Euclidean path
between two points that avoids polygonal obstacles only turns at obstacle
vertices, so these are the only intermediate nodes a shortest flight path
can ever need.
"""
import os
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "offline_osmx"
CACHE_DIR.mkdir(exist_ok=True)

RESTRICTED_TAGS = {
    "amenity": ["hospital"],
    "landuse": ["military"],
    "leisure": ["stadium"],
    "aeroway": ["aerodrome", "heliport"],
    "building": ["government"],
    "office": ["government"],
}


def load_restricted(city):
    """Raw restricted polygons in EPSG:32634 (cached, else fetched from OSM)."""
    path = os.path.join(CACHE_DIR, f"{city.replace(', ', '_')}_restricted.gpkg")
    if os.path.exists(path):
        gdf = gpd.read_file(path)
        print("Local... OK")
    else:
        print("Online...")
        gdf = ox.features_from_place(city, tags=RESTRICTED_TAGS)
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        cols = [c for c in ["name", *RESTRICTED_TAGS] if c in gdf.columns]
        gdf = gdf[cols + ["geometry"]].reset_index(drop=True)
        gdf.to_file(path, driver="GPKG")
        print("...OK")
    return gdf.to_crs(epsg=32634)


def build_obstacles(city, coverage_polygon, safety_margin_m=40.0, min_area_m2=1500.0,
                    simplify_m=8.0):
    """
    Returns (obstacles, vertices):
      obstacles : list of shapely Polygons (buffered, simplified, dissolved),
                  restricted to those near the delivery area
      vertices  : list of (x, y) exterior vertices of those polygons
    """
    gdf = load_restricted(city)
    area_of_interest = coverage_polygon.buffer(2000)
    geoms = [g for g in gdf.geometry if g.is_valid and g.area >= min_area_m2
             and g.intersects(area_of_interest)]
    if not geoms:
        return [], []
    # mitre joins keep corners sharp -> few vertices; simplify removes jitter
    buffered = [g.buffer(safety_margin_m, join_style="mitre", mitre_limit=2.0) for g in geoms]
    dissolved = unary_union(buffered).simplify(simplify_m, preserve_topology=True)
    if isinstance(dissolved, Polygon):
        polys = [dissolved]
    elif isinstance(dissolved, MultiPolygon):
        polys = list(dissolved.geoms)
    else:
        polys = [g for g in getattr(dissolved, "geoms", []) if isinstance(g, Polygon)]
    # drop holes (a drone never needs to fly inside a restricted area)
    polys = [Polygon(p.exterior) for p in polys]
    vertices = [(x, y) for p in polys for (x, y) in list(p.exterior.coords)[:-1]]
    return polys, vertices
