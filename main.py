"""
TiranaFly - drone delivery network for Tirana.

    python main.py              run the pipeline, export figures, open the viewer
    python main.py --no-show    run and export only (headless)
"""
import json
import os
import sys

from config import AUTO_INTERVAL_MS, CITY
from src.pipeline import run
from src.offline_viz import build_routes_gdf, export_figures, interactive_plot

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main():
    os.makedirs(OUT, exist_ok=True)
    results, obj = run()

    print("\nSummary")
    print(f"  graph      : |V| = {results['graph']['n_vertices']}, |E| = {results['graph']['n_edges']}")
    print(f"  depots     : greedy {results['depots']['greedy']} / exact {results['depots']['exact']} "
          f"of {results['depots']['n_candidates']} candidates")
    for z in results["zones"]:
        b, n = z["best"], z["naive"]
        print(f"  Z{z['zone']}: {z['n_clients']:4d} clients | {z['chosen_method']:8s} {b['trips']:3d} trips "
              f"{b['km']:7.1f} km {b['drones']} drones | naive {n['trips']} trips {n['km']:.1f} km {n['drones']} drones")
    t = results["totals"]
    print(f"  TOTAL: naive {t['naive']['trips']} trips / {t['naive']['km']} km / {t['naive']['drones']} drones "
          f"->  optimised {t['best']['trips']} trips / {t['best']['km']} km / {t['best']['drones']} drones")
    print(f"  timings (s): {results['timings_s']}")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)

    print("\nBuilding route GeoDataFrame (legs follow shortest flight paths)...")
    gdf_routes = build_routes_gdf(obj["petals"], obj["gdf_depots"], obj["oracle"])
    gdf_routes.to_file(os.path.join(OUT, "routes.gpkg"), driver="GPKG")

    print("Exporting static figures...")
    export_figures(CITY, obj, gdf_routes, results, OUT)

    if "--no-show" not in sys.argv:
        print("Launching interactive visualisation...")
        interactive_plot(CITY, obj, gdf_routes, AUTO_INTERVAL_MS)


if __name__ == "__main__":
    main()
