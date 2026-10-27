# TiranaFly: a drone delivery network for Tirana

Case study in geospatial optimisation: where to put drone depots, how to split
a city into delivery zones, and how to route multi-parcel flights under battery
and payload limits, on real OpenStreetMap data for Tirana, Albania.

**Highlights**
- Flight graph = visibility graph over real restricted airspace (hospitals,
  military areas, stadiums, ministries from OSM), pruned at the usable range.
  Dijkstra on it gives provably exact shortest flight paths (Lozano-Pérez & Wesley).
- Dijkstra, bounded Dijkstra, multi-source Dijkstra and A* implemented from scratch.
- Depots by Set Cover: greedy (Chvátal) plus an exact integer program that
  certifies optimality, then a p-median tie-break among optimal covers.
- Zones = graph Voronoi cells (one multi-source Dijkstra).
- Routing: rotating sweep and Clarke-Wright savings on the exact metric, with
  2-opt, Or-opt and inter-trip relocate; results reported against lower bounds.
- Fleet sizing by First-Fit-Decreasing bin packing.
- Full experimental evaluation (range, payload, scaling, ablation, depot
  strategy, Dijkstra vs A*) and an interactive petal-by-petal viewer on a basemap.

Result for 1,000 clients and a 16 km / 6 kg drone: 2 depots (certified optimal),
241 trips, 1,285 km and 12 drones per day, versus 1,000 trips, 4,892 km and
34 drones when every parcel flies alone; within 5% / 16% / 0% of the lower bounds.

## Run
```bash
uv venv --python 3.12 .venv            # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python osmnx geopandas shapely scikit-learn contextily rasterio scipy matplotlib networkx pyogrio reportlab python-docx
.venv/bin/python main.py               # pipeline + figures + interactive viewer   (~2 min)
.venv/bin/python main.py --no-show     # headless: output/results.json, output/*.png, output/routes.gpkg
.venv/bin/python experiments.py        # sensitivity, ablation, scaling, depot strategy, A* (~10 min)
.venv/bin/python build_report.py       # TiranaFly_Report.pdf / .docx from the two JSON files
```
All OSM layers (land use, roads, restricted areas) and the basemap are cached in
`src/offline_osmx` and `src/offline_ctx`; the run is offline after the first fetch.

## Layout
| file | stage |
|---|---|
| `config.py` | drone spec, airspace margin, candidate spacing, routing/viewer settings |
| `src/offline_map.py` | OSM coverage polygon (residential/business/commercial) |
| `src/obstacles.py` | restricted airspace from OSM, buffered + dissolved; obstacle vertices |
| `src/clients.py` | Monte-Carlo clients, payloads, polar angles |
| `src/flight_graph.py` | visibility graph G=(V,E) pruned at the usable range; exact distance oracle |
| `src/shortest_paths.py` | Dijkstra (bounded), multi-source Dijkstra, A*, path reconstruction |
| `src/depots.py` | boundary candidate sites, coverage sets, greedy + exact Set Cover, p-median tie-break, zones; K-Means benchmark |
| `src/algorithms.py` | NN, cheapest insertion, 2-opt, Or-opt, inter-route relocate |
| `src/petals.py` | rotating sweep, savings, lower bounds |
| `src/fleet.py` | trip durations, First-Fit-Decreasing fleet sizing |
| `src/pipeline.py` | end-to-end run shared by `main.py` and `experiments.py` |
| `src/offline_viz.py` | static figures and the interactive petal viewer |

Data: © OpenStreetMap contributors; basemap tiles © CARTO.
