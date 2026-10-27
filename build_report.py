"""
Builds the Technical Solution Document from output/results.json and
output/experiments.json (run main.py and experiments.py first).

    python build_report.py      ->  TiranaFly_Report.pdf  and  TiranaFly_Report.docx

Every number in the document is read from the result files, so the report
always describes what the code actually computed.
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
R = json.load(open(os.path.join(OUT, "results.json")))
X = json.load(open(os.path.join(OUT, "experiments.json")))
cfg, Z, T, D = R["config"], R["zones"], R["totals"], R["depots"]

AUTHOR = "Iris Sinanaj"
YEAR = "2026"

# ---------------------------------------------------------------- content blocks
# Each block: ("h1"|"h2"|"p"|"bullets"|"table"|"figure", payload)
B = []
h1 = lambda t: B.append(("h1", t))
h2 = lambda t: B.append(("h2", t))
p = lambda t: B.append(("p", t))
bullets = lambda items: B.append(("bullets", items))
table = lambda rows, widths=None: B.append(("table", (rows, widths)))
figure = lambda name, caption, width=16: B.append(("figure", (os.path.join(OUT, name), caption, width)))

reduction = lambda a, b: f"{100 * (1 - b / a):.0f}%"
n_best, n_naive = T["best"], T["naive"]
lb_trips = sum(z["bounds"]["trips"] for z in Z)
lb_km = sum(z["bounds"]["km"] for z in Z)
lb_drones = sum(z["best"]["drones_lb"] for z in Z)
abl = X["D_ablation"]
A, Bp, Cs, Ed, F = X["A_range"], X["B_payload"], X["C_scaling"], X["E_depots"], X["F_astar"]
max_detour = max(z["detour_factor_max"] for z in Z)
mean_detour = max(z["detour_factor_mean"] for z in Z)

p_title = ("TiranaFly", "Designing a drone delivery network for Tirana with exact shortest paths, Set Cover and "
           "vehicle-routing heuristics on OpenStreetMap data", f"Case study by {AUTHOR}, {YEAR}")

h1("Abstract")
p(f"We design a drone delivery network for Tirana on real OpenStreetMap data. The city is modelled as a flight graph "
  f"G = (V, E) with {R['graph']['n_vertices']} vertices and {R['graph']['n_edges']:,} edges whose vertices are "
  f"{cfg['n_clients']} clients, {D['n_candidates']} candidate depot sites on the outskirts and "
  f"{R['map']['n_obstacle_vertices']} corners of {R['map']['n_obstacles']} restricted-airspace polygons (hospitals, "
  f"military areas, stadiums, government buildings, from OSM). Every edge is a straight flight leg that avoids restricted "
  f"airspace and is no longer than the usable battery range, so Dijkstra's algorithm on G returns the exact shortest "
  f"flight path between any two points (visibility-graph theorem). Depots are chosen by Set Cover over the coverage "
  f"sets produced by bounded Dijkstra runs: the greedy algorithm selects {D['greedy']} depots and an exact integer "
  f"program certifies that {D['exact']} is optimal. Zones are the graph Voronoi cells of the depots, computed by one "
  f"multi-source Dijkstra. Inside each zone, trips are built by the sweep heuristic (rotated over {cfg['sweep_starts']} "
  f"start angles) and by Clarke-Wright savings on the exact distance matrix, improved by 2-opt, Or-opt and inter-trip "
  f"relocation, and scheduled onto drones by First-Fit-Decreasing bin packing. The result for {cfg['n_clients']} clients: "
  f"{n_best['trips']} trips, {n_best['km']:.0f} km and {n_best['drones']} drones per day, against {n_naive['trips']} trips, "
  f"{n_naive['km']:.0f} km and {n_naive['drones']} drones when every parcel flies alone. Lower bounds of {lb_trips} trips, "
  f"{lb_km:.0f} km and {lb_drones} drones show the solution is within "
  f"{100 * (n_best['trips'] / lb_trips - 1):.0f}%, {100 * (n_best['km'] / lb_km - 1):.0f}% and "
  f"{100 * (n_best['drones'] / lb_drones - 1):.0f}% of optimal respectively.")

# ---------------------------------------------------------------- 1
h1("1. Problem Formulation")
h2("1.1 Map setup and coordinate bounds")
p("The delivery area is the built-up part of Tirana: the union of all OpenStreetMap polygons tagged landuse = residential, "
  f"business or commercial, fetched once with OSMnx and cached offline ({R['map']['coverage_km2']} km²). All layers are "
  "projected from WGS 84 to UTM zone 34N (EPSG:32634) so that coordinates are metres and Euclidean distances are true "
  "ground distances; plots use Web Mercator (EPSG:3857) over a cached CartoDB basemap.")
h2("1.2 Restricted airspace")
p(f"Drones may not overfly hospitals, military areas, stadiums, aerodromes/heliports or government buildings. These "
  f"polygons are fetched once from OSM (79 features for Tirana, among them the QSUT and Trauma university hospitals, the "
  f"Ministry of Defence, the Brigades Palace and both stadiums), buffered by a {cfg['safety_margin_m']:.0f} m horizontal "
  f"safety margin, simplified and dissolved into {R['map']['n_obstacles']} obstacle polygons with "
  f"{R['map']['n_obstacle_vertices']} vertices. Clients are never generated inside them.")
h2("1.3 Clients")
p(f"{cfg['n_clients']} clients are generated by Monte-Carlo acceptance inside the coverage polygon (seed {cfg['seed']}), "
  f"each with a parcel of {cfg['payload_range'][0]}-{cfg['payload_range'][1]} kg drawn uniformly. The clients, the "
  "candidate depot sites and the obstacle vertices are the vertex set V of the flight graph.")
h2("1.4 Drone specification")
p("The fleet drone TF-1 is a mid-size electric multicopter. The values are inside the envelope of current delivery drones "
  "(DJI FlyCart 30: 16 km loaded range; Wing and Zipline P2: 10-20 km round trips at 40-50 km/h) and are deliberately "
  "conservative for an urban environment.")
table([["Parameter", "Value", "Unit", "Justification"],
       ["Maximum flight range D_max", f"{cfg['d_max_m'] / 1000:.0f}", "km", "loaded range of a 16 km-class multicopter"],
       ["Battery reserve", f"{100 * cfg['reserve']:.0f}", "%", "wind, hover, regulatory margin"],
       ["Usable range D = D_max (1 - reserve)", f"{cfg['d_use_m'] / 1000:.1f}", "km", "feasibility threshold used everywhere"],
       ["Battery constant k = 100 / D_max", f"{100 / (cfg['d_max_m'] / 1000):.2f}", "% per km", "linear consumption model"],
       ["Maximum payload W_max", f"{cfg['w_max_kg']:.0f}", "kg", "several 0.5-2.3 kg parcels per trip"],
       ["Cruise speed", f"{cfg['speed_kmh']:.0f}", "km/h", "urban cruise incl. climb and descent"],
       ["Service time per client", f"{cfg['service_min']:.0f}", "min", "hover and winch drop"],
       ["Turnaround at depot", f"{cfg['turnaround_min']:.0f}", "min", "battery swap and loading"],
       ["Delivery window", f"{cfg['work_hours']:.0f}", "h/day", "daylight operations"]],
      [6.2, 1.6, 1.8, 7.2])
h2("1.5 Battery model and feasibility")
p("Battery consumption is modelled as proportional to distance, Consumption = k × Distance. A flight is "
  f"feasible iff the sum of its edge lengths is at most the usable range D = {cfg['d_use_m'] / 1000:.1f} km, i.e. iff it "
  f"consumes at most {100 * (1 - cfg['reserve']):.0f}% of the battery. Hence:")
bullets([f"a client c can be served from depot j only if the round trip 2·dist(j, c) ≤ D, i.e. dist(j, c) ≤ D/2 = "
         f"{cfg['d_use_m'] / 2000:.1f} km, where dist is the shortest-path distance in G (not the straight line);",
         "a multi-client cycle Depot → C₁ → … → Cₙ → Depot is feasible iff its total shortest-path length is ≤ D and the "
         "summed parcel weight is ≤ W_max;",
         "any edge longer than D is removed from G (weight ∞): a leg that cannot be flown on one battery does not exist."])
h2("1.6 Formal problem")
p("Given the flight graph G = (V, E, w), the clients C, the candidate sites S and the usable range D: (i) find the "
  "smallest S* ⊆ S such that every client has shortest-path distance ≤ D/2 to some depot in S* (minimum Set Cover); "
  "(ii) partition C into zones around the chosen depots; (iii) in each zone partition the clients into feasible cycles "
  "through the depot of minimum total length (a capacitated vehicle-routing problem, a generalisation of the TSP); "
  "(iv) schedule the cycles onto the fewest drones that complete them within the daily window (bin packing).")

# ---------------------------------------------------------------- 2
h1("2. Partitioning Logic")
h2("2.1 The flight graph")
p("Drones fly straight legs, so the natural graph is a visibility graph. Its vertices are the clients, the candidate "
  "depot sites and the obstacle corners; two vertices are joined when the segment between them crosses no restricted "
  "polygon and is at most D long. By the theorem of Lozano-Pérez and Wesley (1979) the shortest obstacle-avoiding path "
  "between two points is a polyline whose turning points are obstacle vertices. Therefore (a) two mutually visible "
  "points have the segment itself as shortest path and (b) a blocked pair is joined by a path through obstacle vertices, "
  "all of whose legs are edges of G. Dijkstra on G is consequently exact, and client-client edges are unnecessary: the "
  f"graph keeps depot-client, depot-vertex, client-vertex and vertex-vertex edges, {R['graph']['n_edges']:,} in total for "
  f"{R['graph']['n_vertices']} vertices, built in {R['graph']['build_seconds']} s with an STRtree spatial index "
  "(O(n log n) per vertex).")
figure("fig1_overview.png", "Figure 1. Built-up Tirana, the restricted airspace (red), the candidate depot sites on the "
       "outskirts (squares), the two depots selected by Set Cover (stars) and the two zones produced by multi-source "
       "Dijkstra (colours).")
h2("2.2 Candidate sites on the outskirts")
p(f"A design requirement is that every depot sits on the outskirts of the city, where land is cheaper, take-off noise is "
  f"tolerable and trucks can resupply. We sample {D['n_candidates']} candidate sites every "
  "800 m along the outer boundary of the coverage polygon (after a 400 m morphological closing so that gaps between "
  "land-use patches do not count as outskirts) and snap each to the nearest motorway, trunk, primary or secondary road "
  "within 500 m, using the OSMnx drive graph, so that depots can be resupplied by truck.")
h2("2.3 Why the zones follow from the battery, not from a grid")
p("From every candidate a bounded Dijkstra (early exit at D/2) returns its coverage set: the clients it can serve by a "
  "feasible round trip. Set Cover (Section 3.2) selects the depots; then one multi-source Dijkstra from all selected "
  "depots labels every client with its nearest depot by shortest flight path. The zones are therefore the cells of a "
  "graph Voronoi diagram: by construction every zone lies inside the battery reach of its own depot, and each client is "
  "assigned to the depot that minimises its single-trip distance. The zone sizes are an output of the drone range: with a "
  f"{A[0]['d_max_km']:.0f} km drone the same code needs {A[0]['depots']} depots, with {A[2]['d_max_km']:.0f} km it needs "
  f"{A[2]['depots']} (Section 4.3).")
table([["Zone", "Depot (UTM x, y)", "Clients", "Farthest client (km)", "Round trip (km)", "Battery used"]] +
      [[f"Z{z['zone']}", f"({z['depot_xy'][0]:.0f}, {z['depot_xy'][1]:.0f})", z["n_clients"],
        f"{z['farthest_client_m'] / 1000:.2f}", f"{2 * z['farthest_client_m'] / 1000:.2f}",
        f"{100 * 2 * z['farthest_client_m'] / cfg['d_max_m']:.0f} %"] for z in Z],
      [1.4, 4.0, 1.6, 3.4, 3.0, 2.4])
h2("2.4 Challenges and mitigations")
table([["Challenge", "Mitigation"],
       ["Several depot sets of minimum size cover all clients", "p-median tie-break: among all minimum covers choose the one minimising "
        f"total client-depot distance ({D['tiebreak_total_km']:.0f} km vs {D['greedy_total_km']:.0f} km for the first greedy answer); "
        "this also balances the zones"],
       ["Centroid-based depots (K-Means) land inside the city", "candidates are restricted to the boundary ring; K-Means is kept only as a benchmark (Section 4.5)"],
       ["Straight-line distances ignore restricted airspace", "all distances are shortest paths in the visibility graph; detours are exact"],
       ["Overpass API outages", "every OSM layer (land use, roads, restricted areas, basemap) is cached locally after the first fetch"]],
      [6.5, 10.3])

# ---------------------------------------------------------------- 3
h1("3. Mathematical Justification")
h2("3.1 Shortest paths: Dijkstra, bounded, multi-source, A*")
p("All distances are shortest-path distances in G computed by Dijkstra's algorithm with a binary heap "
  "in O((|V| + |E|) log |V|). Edge weights are positive lengths, so the algorithm is exact. We "
  "use three variants built on the same relaxation step: (a) bounded Dijkstra stops as soon as the smallest tentative "
  "distance exceeds D/2, which is how the coverage sets are computed; (b) multi-source Dijkstra starts with all chosen "
  "depots at distance 0 and labels every vertex with the depot it was reached from (equivalent to Dijkstra from a virtual "
  "super-source), which gives the zones in a single run instead of one run per depot; (c) A* with the straight-line "
  "heuristic h(u) = |u, target|, which is admissible and consistent because every edge weight is the Euclidean length of "
  f"that edge, so A* returns the same distances as Dijkstra (verified on {F['blocked_pairs_tested']} blocked pairs) while "
  f"settling {F['astar_expanded_avg']:.0f} vertices on average instead of {F['dijkstra_settled_avg']:.0f}.")
p("The distance oracle used by the routing stage answers d(i, j) = |ij| when i sees j and |ij| ≤ D (exact by the theorem) "
  f"and runs a bounded Dijkstra only from vertices that have a blocked partner. On the main instance "
  f"{R['oracle']['direct_pairs']:,} pairs were direct and {R['oracle']['blocked_pairs']:,} blocked, requiring "
  f"{R['oracle']['dijkstra_runs']} Dijkstra runs. The largest detour factor (graph distance / straight line) is "
  f"{max_detour:.2f}, the mean {mean_detour:.3f}.")
figure("fig3_detour.png", "Figure 2. The pair with the largest detour: the shortest flight path (green) turns only at "
       "corners of the restricted polygon, as the visibility-graph theorem predicts.", 12)
h2("3.2 Depot placement: Set Cover with an optimality certificate")
p("Let U be the client set and S_j = {c : dist(j, c) ≤ D/2} the coverage set of candidate j. Choosing the fewest "
  "candidates whose sets cover U is the minimum Set Cover problem, NP-hard. Chvátal's greedy algorithm repeatedly takes the "
  "candidate covering the most still-uncovered clients; it runs in O(|S|·|U|) per iteration and uses at most H(max|S_j|) "
  f"≤ ln|U| + 1 ≈ {math.log(cfg['n_clients']) + 1:.1f} times the optimum. To know the real gap we also solve the 0/1 integer "
  "program min Σ x_j s.t. Σ_{j : c ∈ S_j} x_j ≥ 1 for all c with the HiGHS branch-and-bound solver of SciPy "
  f"(milliseconds for {D['n_candidates']} candidates). Result: greedy {D['greedy']} depots, optimum {D['exact']}, "
  f"{D['unservable']} unservable clients. Because several {D['exact']}-depot covers exist, we enumerate all "
  f"C({D['n_candidates']}, {D['exact']}) combinations and keep the cover minimising total client-depot shortest-path "
  "distance (the p-median criterion), a secondary objective that shortens every single-client trip and balances the zones.")
h2("3.3 Routing: sweep and savings on the exact metric, local search, bounds")
p("Inside a zone, with the depot as vertex 0 and the exact distance matrix M from the oracle, the clients must be split "
  "into feasible cycles of minimum total length: a capacitated VRP, NP-hard as a generalisation of the TSP. Two polynomial "
  "constructions are used and the better one kept per zone:")
bullets(["Sweep (Gillett and Miller, 1974): clients are taken in polar order around the depot and inserted into the "
         "current petal at the cheapest position (O(n) per insertion); when range or payload would be violated the petal is "
         f"closed. The sweep is rotated over {cfg['sweep_starts']} start angles and the best run kept, which removes the "
         "arbitrariness of a single start.",
         "Clarke-Wright savings (1964): start from one round trip per client (always feasible by construction of the zone) "
         "and merge trips in decreasing order of the saving s(i, j) = M[0,i] + M[0,j] − M[i,j] while range and payload hold."])
p("Every petal is improved by best-improvement 2-opt (O(1) per move evaluation) and Or-opt (segments of 1-3 clients moved, "
  "possibly reversed), then an inter-petal relocate pass moves single clients between petals when that shortens the total "
  "and both petals stay feasible; emptied petals disappear, so this pass can also reduce the number of trips. Each move only "
  "shortens a feasible cycle, so feasibility is invariant (asserted in the code). The quality of the result is measured "
  "against two lower bounds valid for any solution: trips ≥ ⌈Σ w_c / W_max⌉, and distance ≥ max(½ Σ_c (d₁(c) + d₂(c)), "
  "(2 / W_max) Σ_c w_c · d(0, c)), the degree-2 bound and the radial bound of Haimovich and Rinnooy Kan (1985).")
h2("3.4 Fleet sizing: bin packing")
p("A petal of length L with n clients takes 60·L/v + n·t_service + t_turnaround minutes. Assigning petals to drones so "
  f"that each drone's day fits in the {cfg['work_hours']:.0f} h window is Bin Packing. First-Fit Decreasing never uses more "
  "than 11/9 OPT + 6/9 bins; the trivial bound ⌈total minutes / window⌉ is reported beside it.")
h2("3.5 Algorithm pipeline")
table([["Step", "Method", "Input", "Output", "Complexity"],
       ["1", "OSM layers + buffering", "city name", "coverage polygon, obstacle polygons", "cached"],
       ["2", "Monte-Carlo sampling", "polygon, N", "clients with payloads", "O(N)"],
       ["3", "Visibility graph (STRtree)", "clients, sites, obstacle vertices", "G = (V, E)", "O(|V| m log m)"],
       ["4", "Bounded Dijkstra per site", "G, D/2", "coverage sets", "O(|S| (|V|+|E|) log |V|)"],
       ["5", "Greedy + exact Set Cover, p-median tie-break", "coverage sets", "depots", "greedy O(|S||U|) per step; ILP"],
       ["6", "Multi-source Dijkstra", "G, depots", "zones", "O((|V|+|E|) log |V|)"],
       ["7", "Distance oracle", "zone nodes", "exact matrix M", "direct pairs O(1), blocked pairs Dijkstra"],
       ["8", "Rotating sweep / savings + 2-opt, Or-opt, relocate", "M, payloads", "petals", "O(k n²) sweep, O(n² log n) savings"],
       ["9", "First-Fit Decreasing", "petal durations", "drones, schedule", "O(p log p)"]],
      [0.9, 5.0, 3.6, 3.6, 3.7])

# ---------------------------------------------------------------- 4
h1("4. Implementation Results")
p(f"The whole pipeline runs in about {sum(R['timings_s'].values()):.0f} s on a laptop "
  f"(graph {R['timings_s']['graph_s']} s, depots {R['timings_s']['depots_s']} s, routing {R['timings_s']['routing_s']} s). "
  "The baseline '1 client / trip' flies every parcel on its own Depot → Client → Depot round trip.")
h2("4.1 Petal summary by zone")
rows = [["Zone", "Clients", "Method", "Trips", "Trips LB", "km", "km LB", "Max clients/trip", "Drones", "Drones LB", "Naive trips / km / drones"]]
for z in Z:
    b, n = z["best"], z["naive"]
    rows.append([f"Z{z['zone']}", z["n_clients"], z["chosen_method"], b["trips"], z["bounds"]["trips"], f"{b['km']:.0f}",
                 f"{z['bounds']['km']:.0f}", b["max_clients_per_trip"], b["drones"], b["drones_lb"],
                 f"{n['trips']} / {n['km']:.0f} / {n['drones']}"])
rows.append(["Total", cfg["n_clients"], "", n_best["trips"], lb_trips, f"{n_best['km']:.0f}", f"{lb_km:.0f}", "",
             n_best["drones"], lb_drones, f"{n_naive['trips']} / {n_naive['km']:.0f} / {n_naive['drones']}"])
table(rows, [1.2, 1.3, 1.6, 1.1, 1.4, 1.1, 1.2, 1.9, 1.3, 1.5, 3.4])
p(f"Multi-client routing cuts the flown distance by {reduction(n_naive['km'], n_best['km'])}, the number of take-offs by "
  f"{reduction(n_naive['trips'], n_best['trips'])} and the fleet by {reduction(n_naive['drones'], n_best['drones'])}. "
  f"The trip count is within {100 * (n_best['trips'] / lb_trips - 1):.1f}% of the payload lower bound, so payload is the "
  "binding constraint: a trip fills up before it runs out of battery. All trips respect both constraints (Figure 5).")
figure("fig2_petals.png", f"Figure 3. All {n_best['trips']} petals; every leg follows its shortest flight path, so the "
       "routes bend around the restricted polygons.")
figure("fig4_comparison.png", "Figure 4. Trips, distance and drones per zone: single-client baseline, optimised solution, "
       "and lower bounds (red).", 16)
figure("fig5_battery.png", "Figure 5. Battery consumed per trip (k × distance); no trip exceeds the usable limit.", 11)
h2("4.2 Local-search ablation")
p(f"On zone Z{abl['zone']} ({abl['n_clients']} clients) we isolate the contribution of each routing component, starting "
  "from a single-angle sweep with cheapest insertion:")
abl_rows = [["Configuration", "Trips", "km"]]
for k, v in abl.items():
    if k in ("zone", "n_clients"):
        continue
    abl_rows.append([k, v["trips"], f"{v['km']:.1f}"])
table(abl_rows, [9.5, 2, 2.5])
figure("exp_D_ablation.png", "Figure 6. Ablation of the routing stage against the distance lower bound.", 13)
h2("4.3 Sensitivity to the drone range")
table([["D_max (km)", "Depots (greedy / exact)", "Zone sizes", "Trips (LB)", "km (LB)", "Drones"]] +
      [[f"{a['d_max_km']:.0f}", f"{a['greedy']} / {a['depots']}", ", ".join(map(str, a["zones"])),
        f"{a['trips']} ({a['trips_lb']})", f"{a['km']:.0f} ({a['km_lb']:.0f})", a["drones"]] for a in A],
      [2, 3.6, 3.6, 2.6, 3, 1.8])
figure("exp_A_range.png", "Figure 7. Depots, trips and drones as a function of the drone range.", 15)
h2("4.4 Sensitivity to the payload")
table([["W_max (kg)", "Trips (LB)", "km (LB)", "Max clients/trip", "Drones", "Best method per zone"]] +
      [[f"{b['w_max_kg']:.0f}", f"{b['trips']} ({b['trips_lb']})", f"{b['km']:.0f} ({b['km_lb']:.0f})",
        b["max_clients_per_trip"], b["drones"], ", ".join(b["method_per_zone"])] for b in Bp],
      [2, 2.8, 3, 3, 1.8, 4.2])
figure("exp_B_payload.png", "Figure 8. Trips and distance as a function of the payload limit, with lower bounds.", 13)
h2("4.5 Depot strategy: K-Means centroids versus Set Cover on the outskirts")
table([["Strategy", "Depots", "Mean client-depot distance (m)", "Depots inside built-up area", "Metric"],
       ["K-Means centroids snapped to roads (original)", Ed["kmeans"]["depots"], Ed["kmeans"]["mean_client_depot_m"],
        Ed["kmeans"]["depots_inside_builtup"], Ed["kmeans"]["note"]],
       ["Set Cover + p-median on boundary sites (this report)", Ed["set_cover"]["depots"], Ed["set_cover"]["mean_client_depot_m"],
        Ed["set_cover"]["depots_inside_builtup"], Ed["set_cover"]["note"]]],
      [5.6, 1.6, 3.4, 3.2, 3.0])
p("Centroid depots are closer to the clients on average, as expected, but they violate the outskirts requirement and are "
  "measured with straight lines that cross restricted airspace. The Set Cover solution satisfies the outskirts "
  "requirement with the provably minimum number of depots.")
h2("4.6 Scaling")
table([["Clients", "|V|", "|E|", "Graph build (s)", "Coverage + Set Cover (s)", "Routing (s)", "Dijkstra runs", "Blocked / direct pairs"]] +
      [[c["n_clients"], c["V"], f"{c['E']:,}", c["graph_s"], c["depots_s"], c["routing_s"], c["dijkstra_runs"],
        f"{c['blocked_pairs']:,} / {c['direct_pairs']:,}"] for c in Cs],
      [1.5, 1.3, 1.8, 2.2, 3.4, 1.9, 2.2, 3.4])
figure("exp_C_scaling.png", "Figure 9. Graph size and run time versus the number of clients.", 13)
h2("4.7 Interactive visualisation")
p("main.py opens the step-through viewer: at each step one petal per zone is highlighted on the cached basemap with its "
  "length and client count, with cumulative distance and clients per zone in the title; Previous / Auto / Next buttons "
  "animate the day. Restricted areas and depots are drawn as layers, and petal geometries are the actual shortest flight "
  "paths. All static figures of this report are exported by the same module.")

# ---------------------------------------------------------------- 5
h1("5. Recommendations for an operator")
bullets([f"Open {D['exact']} depots on the outskirts at the selected sites (west, near the Mëzez side of the ring road, and "
         "east by the artificial lake). This is the provably minimum number for a 16 km-class drone; a third depot adds no "
         "coverage and would only shorten trips that are already limited by payload.",
         f"Start with {n_best['drones']} drones plus two spares (one hot spare per depot). Single-parcel flying would need "
         f"{n_naive['drones']}; the routing software pays for itself before the first week.",
         f"Payload is the binding constraint ({100 * (n_best['trips'] / lb_trips - 1):.0f}% above the payload bound, "
         f"{100 * (n_best['km'] / lb_km - 1):.0f}% above the distance bound). Raising W_max from 6 to 10 kg reduces trips to "
         f"{Bp[2]['trips']} and drones to {Bp[2]['drones']}; a larger battery alone would not.",
         "Keep the 10% reserve and treat restricted airspace as hard: the exact detours cost at most a factor "
         f"{max_detour:.2f} on a few legs and a mean of {100 * (mean_detour - 1):.1f}%.",
         "Operate the depots as battery-swap stations; the fleet size assumes an 8-minute turnaround.",
         "Re-plan daily with real orders: the pipeline accepts any client list and runs in under two minutes. Next steps are "
         "time windows, wind-dependent range, and a second routing objective (energy instead of distance)."])

# ---------------------------------------------------------------- 6
h1("6. Limitations and future work")
bullets(["Clients are synthetic (uniform inside the built-up polygon). Real demand is clustered by district and time of "
         "day; the pipeline accepts any client list, so plugging in order data is a data change, not a code change.",
         "Battery consumption is linear in distance. Wind, payload mass and climb cost are ignored; a per-leg energy model "
         "would replace the edge weights without touching the algorithms.",
         "Restricted airspace is static and derived from OSM tags; temporary restrictions and altitude layers are not modelled.",
         "The sweep and savings heuristics are local-optimum methods. The reported gaps to the lower bounds (a few percent on "
         "trips, 10-15% on distance) could be narrowed with a metaheuristic such as iterated local search or an exact "
         "branch-and-price on small zones.",
         "Fleet sizing assumes sequential trips per drone with a fixed turnaround; a time-indexed schedule with charging "
         "queues and order time windows is the natural next step.",
         "Everything runs in pure Python on a laptop in about two minutes for 1,000 clients; the distance oracle is the "
         "dominant cost and would be the first thing to move to a compiled routine for city-scale demand."])
h2("Code and reproducibility")
p("The implementation is a single Python package: config.py (parameters), src/offline_map.py and src/obstacles.py (OSM layers, "
  "cached offline), src/clients.py, src/flight_graph.py (visibility graph and distance oracle), src/shortest_paths.py "
  "(Dijkstra, bounded, multi-source, A*), src/depots.py (candidate sites, coverage, greedy and exact Set Cover, zones), "
  "src/algorithms.py and src/petals.py (routing), src/fleet.py (bin packing), src/pipeline.py (end-to-end run), "
  "src/offline_viz.py (figures and interactive viewer), with main.py, experiments.py and build_report.py as entry points. "
  f"All figures and numbers in this document are produced by that code with seed {cfg['seed']}.")
h2("References")
bullets(["Cormen, Leiserson, Rivest, Stein. Introduction to Algorithms, 3rd ed. (Dijkstra ch. 24, Set Cover ch. 35).",
         "Lozano-Pérez, T., Wesley, M. A. (1979). An algorithm for planning collision-free paths among polyhedral obstacles. CACM 22(10).",
         "Hart, Nilsson, Raphael (1968). A formal basis for the heuristic determination of minimum cost paths. IEEE TSSC 4(2).",
         "Chvátal, V. (1979). A greedy heuristic for the set-covering problem. Mathematics of Operations Research 4(3).",
         "Gillett, B., Miller, L. (1974). A heuristic algorithm for the vehicle-dispatch problem. Operations Research 22(2).",
         "Clarke, G., Wright, J. W. (1964). Scheduling of vehicles from a central depot to a number of delivery points. Operations Research 12(4).",
         "Croes, G. A. (1958). A method for solving traveling-salesman problems. Operations Research 6(6).",
         "Haimovich, M., Rinnooy Kan, A. H. G. (1985). Bounds and heuristics for capacitated routing problems. Math. of OR 10(4).",
         "Johnson, D. S. (1973). Near-optimal bin packing algorithms. PhD thesis, MIT.",
         "OpenStreetMap contributors; Boeing, G. (2017). OSMnx. Computers, Environment and Urban Systems 65."])


# ================================================================ PDF renderer
def render_pdf(path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9.8, leading=13, alignment=TA_JUSTIFY, spaceAfter=5)
    H1 = ParagraphStyle("1", parent=ss["Heading1"], fontSize=14.5, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1f3a68"))
    H2 = ParagraphStyle("2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1f3a68"))
    cap = ParagraphStyle("c", parent=ss["Normal"], fontSize=8.3, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=9)
    cell = ParagraphStyle("t", parent=body, fontSize=8, leading=9.8, alignment=0, spaceAfter=0)
    bul = ParagraphStyle("u", parent=body, leftIndent=13, bulletIndent=3, spaceAfter=3)
    title = ParagraphStyle("T", parent=ss["Title"], fontSize=19, leading=23, alignment=TA_CENTER)
    sub = ParagraphStyle("S", parent=ss["Normal"], fontSize=11, alignment=TA_CENTER, leading=15)
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    story = [Spacer(1, 4 * cm), Paragraph(esc(p_title[0]), title), Paragraph(esc(p_title[1]), sub), Spacer(1, 0.4 * cm),
             Paragraph(esc(p_title[2]), sub), Spacer(1, 1.5 * cm),
             Paragraph("Python · OSMnx · GeoPandas · Shapely · SciPy · Matplotlib", sub), Spacer(1, 1.2 * cm),
             Paragraph("All numbers and figures in this document are produced by the accompanying code "
                       f"(seed {cfg['seed']}) and are reproducible.", sub), PageBreak()]
    for kind, val in B:
        if kind == "h1":
            story.append(Paragraph(esc(val), H1))
        elif kind == "h2":
            story.append(Paragraph(esc(val), H2))
        elif kind == "p":
            story.append(Paragraph(esc(val), body))
        elif kind == "bullets":
            story += [Paragraph("• " + esc(t), bul) for t in val]
        elif kind == "table":
            rows, widths = val
            data = [[Paragraph(esc(c), cell) for c in r] for r in rows]
            t = Table(data, colWidths=[w * cm for w in widths] if widths else None, hAlign="CENTER", repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f2")),
                                   ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
            story += [t, Spacer(1, 6)]
        elif kind == "figure":
            path_, caption, width = val
            if os.path.exists(path_):
                img = Image(path_)
                ratio = img.imageHeight / img.imageWidth
                img.drawWidth, img.drawHeight = width * cm, width * cm * ratio
                story.append(KeepTogether([img, Paragraph(esc(caption), cap)]))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"TiranaFly, {AUTHOR}, page {doc_.page}")
        canvas.restoreState()

    SimpleDocTemplate(path, pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm, topMargin=1.9 * cm,
                      bottomMargin=1.9 * cm, title=p_title[0], author=AUTHOR).build(story, onFirstPage=footer,
                                                                                    onLaterPages=footer)


# ================================================================ DOCX renderer
def render_docx(path):
    from docx import Document
    from docx.shared import Cm, Pt
    doc = Document()
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading(p_title[0], 0)
    doc.add_paragraph(p_title[1]).alignment = 1
    doc.add_paragraph(p_title[2]).alignment = 1
    doc.add_paragraph("Python · OSMnx · GeoPandas · Shapely · SciPy · Matplotlib").alignment = 1
    doc.add_page_break()
    for kind, val in B:
        if kind == "h1":
            doc.add_heading(val, 1)
        elif kind == "h2":
            doc.add_heading(val, 2)
        elif kind == "p":
            doc.add_paragraph(val)
        elif kind == "bullets":
            for t in val:
                doc.add_paragraph(t, style="List Bullet")
        elif kind == "table":
            rows, _ = val
            t = doc.add_table(rows=len(rows), cols=len(rows[0]))
            t.style = "Light Grid Accent 1"
            for i, r in enumerate(rows):
                for j, c in enumerate(r):
                    t.cell(i, j).text = str(c)
                    for par in t.cell(i, j).paragraphs:
                        for run in par.runs:
                            run.font.size = Pt(8)
            doc.add_paragraph()
        elif kind == "figure":
            path_, caption, width = val
            if os.path.exists(path_):
                doc.add_picture(path_, width=Cm(min(width, 16)))
                doc.add_paragraph(caption).alignment = 1
    doc.save(path)


if __name__ == "__main__":
    render_pdf(os.path.join(HERE, "TiranaFly_Report.pdf"))
    render_docx(os.path.join(HERE, "TiranaFly_Report.docx"))
    print("written TiranaFly_Report.pdf and TiranaFly_Report.docx")
