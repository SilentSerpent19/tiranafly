import os
import osmnx as ox
import geopandas as gpd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "offline_osmx"

CACHE_DIR.mkdir(exist_ok=True)


def setup_map(city):
    """Returns (city_gdf, residential_gdf, polygon) in EPSG:32634,
    fetching from local folder if available."""
    
    city_file = os.path.join(CACHE_DIR, f"{city.replace(', ', '_')}_city.gpkg")
    residential_file = os.path.join(CACHE_DIR, f"{city.replace(', ', '_')}_residential.gpkg")

    # Load from local if exists
    if os.path.exists(city_file) and os.path.exists(residential_file):
        city_gdf = gpd.read_file(city_file)
        residential = gpd.read_file(residential_file)
        print("Local... OK")
    else:
        # Otherwise fetch from OSM
        print("Online...")
        city_gdf = ox.geocode_to_gdf(city)
        residential = ox.features_from_place(
            city,
            tags={"landuse": ["residential", "business", "commercial"]},
        )
        # Save locally for next time
        city_gdf.to_file(city_file, driver="GPKG")
        residential.to_file(residential_file, driver="GPKG")
        print("...0K")

    residential = residential.to_crs(epsg=32634)
    polygon = residential.geometry.union_all()
    return city_gdf, residential, polygon