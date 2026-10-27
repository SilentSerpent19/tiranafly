"""
Visualisation on the cached CartoDB basemap.

    build_routes_gdf()  petals as LineStrings whose legs follow the exact
                        shortest flight paths (they bend around restricted
                        airspace)
    export_figures()    static PNGs for the report: overview (zones, depots,
                        candidates, restricted areas), all petals, detour
                        example, naive-vs-optimised bars, battery histogram
    interactive_plot()  the step-through viewer (Previous / Auto / Next)
"""
import os
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button as MplButton
from shapely.geometry import LineString

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "offline_ctx"
CACHE_DIR.mkdir(exist_ok=True)


def ensure_offline_basemap(city, gdf_plot, zoom=15):
    raster_path = os.path.join(CACHE_DIR, f"{city.replace(', ', '_')}_basemap_z{zoom}.tif")
    if not os.path.exists(raster_path):
        print("Online...")
        west, south, east, north = gdf_plot.total_bounds
        ctx.bounds2raster(west, south, east, north, path=raster_path, zoom=zoom,
                          source=ctx.providers.CartoDB.Positron)
        print("...OK")
    else:
        print("Local... OK")
    return raster_path


def build_routes_gdf(petals_by_zone, gdf_depots, oracle):
    """One row per petal.  Geometry = depot -> c1 -> ... -> depot along the
    shortest flight paths returned by the DistanceOracle."""
    rows, lines = [], []
    petal_id = 0
    for zone, petals in petals_by_zone.items():
        depot_node = int(gdf_depots.loc[gdf_depots["zone"] == zone, "node"].iloc[0])
        for route in petals:
            seq = [depot_node] + list(route) + [depot_node]
            coords = []
            for a, b in zip(seq, seq[1:]):
                p = oracle.path(a, b)
                coords.extend(p if not coords else p[1:])
            lines.append(LineString(coords))
            rows.append({"cluster": zone, "petal_id": petal_id, "n_clients": len(route)})
            petal_id += 1
    gdf = gpd.GeoDataFrame(rows, geometry=lines, crs="EPSG:32634")
    gdf["distance_m"] = gdf.geometry.length
    gdf["cluster_step"] = gdf.groupby("cluster").cumcount()
    gdf["cum_distance_m"] = gdf.groupby("cluster")["distance_m"].cumsum()
    gdf["cum_clients"] = gdf.groupby("cluster")["n_clients"].cumsum()
    return gdf


def _layers(obj):
    """Project everything to Web Mercator for plotting."""
    return {"clients": obj["gdf_clients"].to_crs(3857), "depots": obj["gdf_depots"].to_crs(3857),
            "sites": obj["gdf_sites"].to_crs(3857),
            "obstacles": gpd.GeoDataFrame(geometry=obj["obstacles"], crs="EPSG:32634").to_crs(3857)}


def _base(ax, L, basemap, title):
    L["obstacles"].plot(ax=ax, color="red", alpha=0.25, edgecolor="darkred", linewidth=0.6, zorder=3)
    ctx.add_basemap(ax, source=basemap)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)


def export_figures(city, obj, gdf_routes, results, out_dir):
    L = _layers(obj)
    routes = gdf_routes.to_crs(3857)
    basemap = ensure_offline_basemap(city, L["clients"], zoom=15)
    zones = sorted(L["depots"]["zone"])
    col = {z: plt.cm.tab10(i % 10) for i, z in enumerate(zones)}

    # 1. overview: candidates, chosen depots, zones, restricted areas
    fig, ax = plt.subplots(figsize=(13, 9))
    for z in zones:
        L["clients"][L["clients"]["cluster"] == z].plot(ax=ax, color=col[z], markersize=9, zorder=5,
                                                        label=f"Zone {z}")
    L["sites"].plot(ax=ax, marker="s", facecolor="white", edgecolor="grey", markersize=45, zorder=6,
                    label=f"candidate sites ({len(L['sites'])})")
    L["depots"].plot(ax=ax, marker="*", color="black", markersize=320, zorder=7, label="selected depots")
    for _, d in L["depots"].iterrows():
        ax.annotate(f"D{d['zone']}", (d.geometry.x, d.geometry.y), xytext=(6, 6), textcoords="offset points",
                    fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    _base(ax, L, basemap, f"Zones from multi-source Dijkstra, depots by Set Cover "
                          f"({results['depots']['greedy']} greedy / {results['depots']['exact']} exact), "
                          f"red = restricted airspace")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig1_overview.png"), dpi=150); plt.close(fig)

    # 2. all petals along shortest flight paths
    fig, ax = plt.subplots(figsize=(13, 9))
    for z in zones:
        routes[routes["cluster"] == z].plot(ax=ax, color=col[z], linewidth=0.9, alpha=0.85, zorder=4)
    L["clients"].plot(ax=ax, color="black", markersize=3, zorder=5)
    L["depots"].plot(ax=ax, marker="*", color="black", markersize=320, zorder=7)
    t = results["totals"]["best"]
    _base(ax, L, basemap, f"All petals ({t['trips']} trips, {t['km']} km, {t['drones']} drones); "
                          f"legs follow exact shortest flight paths around restricted airspace")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig2_petals.png"), dpi=150); plt.close(fig)

    # 3. detour example: the pair of zone nodes with the largest detour factor
    G, oracle = obj["G"], obj["oracle"]
    best = None
    for z, nodes in obj["nodes"].items():
        M = obj["M"][z]
        xy = G.xy[nodes]
        E = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1) ** 0.5
        ok = (E > 0) & (M < float("inf"))
        ratio = M[ok] / E[ok]
        k = int(ratio.argmax())
        i, j = [int(v) for v in (list(zip(*ok.nonzero()))[k])]
        if best is None or ratio[k] > best[0]:
            best = (float(ratio[k]), nodes[i], nodes[j], float(M[i, j]), float(E[i, j]))
    if best:
        ratio, d, c, dm, em = best
        path = oracle.path(d, c)
        pl = gpd.GeoSeries([LineString(path), LineString([tuple(G.xy[d]), tuple(G.xy[c])])],
                           crs="EPSG:32634").to_crs(3857)
        fig, ax = plt.subplots(figsize=(10, 8))
        pl.iloc[[1]].plot(ax=ax, color="orange", linestyle="--", linewidth=2, zorder=5, label="straight line")
        pl.iloc[[0]].plot(ax=ax, color="green", linewidth=3, zorder=6, label="shortest flight path (Dijkstra)")
        minx, miny, maxx, maxy = pl.total_bounds
        pad = 600
        ax.set_xlim(minx - pad, maxx + pad); ax.set_ylim(miny - pad, maxy + pad)
        ax.legend(loc="upper left")
        _base(ax, L, basemap, f"Largest detour: {dm / 1000:.2f} km along the graph vs {em / 1000:.2f} km straight "
                              f"(factor {ratio:.2f})")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig3_detour.png"), dpi=150); plt.close(fig)

    # 4. naive vs optimised, with lower bounds
    Z = results["zones"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    xs = list(range(len(Z)))
    for ax, key, ttl in zip(axs, ["trips", "km", "drones"], ["Trips per day", "km flown per day", "Drones"]):
        ax.bar([x - 0.2 for x in xs], [z["naive"][key] for z in Z], 0.4, color="lightgrey", edgecolor="k",
               label="1 client / trip")
        ax.bar([x + 0.2 for x in xs], [z["best"][key] for z in Z], 0.4, color="seagreen", edgecolor="k",
               label="optimised")
        lb = [z["bounds"][key] for z in Z] if key in ("trips", "km") else [z["best"]["drones_lb"] for z in Z]
        ax.scatter(xs, lb, marker="_", s=400, color="red", zorder=5, label="lower bound")
        ax.set_xticks(xs); ax.set_xticklabels([f"Z{z['zone']}" for z in Z]); ax.set_title(ttl)
    axs[0].legend(fontsize=8)
    fig.suptitle("Effect of multi-client routing per zone, with lower bounds")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig4_comparison.png"), dpi=150); plt.close(fig)

    # 5. battery histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    bp = [r["battery_pct"] for z in Z for r in z["best"]["routes"]]
    ax.hist(bp, bins=20, color="steelblue", edgecolor="k")
    lim = 100 * (1 - results["config"]["reserve"])
    ax.axvline(lim, color="red", linestyle="--", label=f"usable limit {lim:.0f} %")
    ax.set_xlabel("battery used per trip (%) = k x distance"); ax.set_ylabel("trips"); ax.legend()
    ax.set_title("Battery consumption of all optimised trips")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig5_battery.png"), dpi=150); plt.close(fig)
    print(f"  figures written to {out_dir}")


def interactive_plot(city, obj, gdf_routes, auto_interval_ms=2000):
    """Step-through viewer: Previous / Auto / Next buttons; one petal per
    zone is highlighted at each step."""
    L = _layers(obj)
    gdf_routes_plot = gdf_routes.to_crs(3857)
    offline_basemap = ensure_offline_basemap(city, gdf_routes_plot, zoom=15)

    cluster_ids = sorted(gdf_routes["cluster"].unique())
    max_step = max(int(gdf_routes[gdf_routes["cluster"] == c]["cluster_step"].max()) for c in cluster_ids)
    cluster_colors = {c: plt.cm.tab10(i % 10) for i, c in enumerate(cluster_ids)}
    state = {"step": 0, "auto_running": False}

    fig, ax = plt.subplots(figsize=(13, 8))
    plt.subplots_adjust(bottom=0.08)
    ax_prev = fig.add_axes([0.15, 0.02, 0.18, 0.04])
    ax_auto = fig.add_axes([0.41, 0.02, 0.18, 0.04])
    ax_next = fig.add_axes([0.67, 0.02, 0.18, 0.04])
    btn_prev = MplButton(ax_prev, "◀  Previous")
    btn_auto = MplButton(ax_auto, "Auto  ▶▶")
    btn_next = MplButton(ax_next, "Next  ▶")
    timer = fig.canvas.new_timer(interval=auto_interval_ms)

    def draw(step):
        ax.cla()
        L["clients"].plot(ax=ax, column="cluster", cmap="Pastel2", markersize=15, zorder=20)
        L["clients"].plot(ax=ax, color="black", markersize=3, zorder=22)
        gdf_routes_plot.plot(ax=ax, color="gray", linewidth=1, alpha=0.35, zorder=21)
        parts = []
        ymin, ymax = ax.get_ylim()
        offset = (ymax - ymin) * 0.1
        for cluster in cluster_ids:
            rc_plot = gdf_routes_plot[gdf_routes_plot["cluster"] == cluster]
            rc_orig = gdf_routes[gdf_routes["cluster"] == cluster]
            active_step = min(step, int(rc_plot["cluster_step"].max()))
            active_plot = rc_plot[rc_plot["cluster_step"] == active_step]
            active_orig = rc_orig[rc_orig["cluster_step"] == active_step]
            active_plot.plot(ax=ax, color=cluster_colors[cluster], linewidth=4, zorder=25)
            parts.append(f"Z{cluster}: {active_orig['cum_distance_m'].iloc[0] / 1000:.2f} km | "
                         f"{active_orig['cum_clients'].iloc[0]} clients")
            centroid = active_plot.geometry.iloc[0].centroid
            ax.text(centroid.x, centroid.y + offset,
                    f"Z{cluster}\nP{active_step}\n{active_orig['distance_m'].iloc[0] / 1000:.2f} km\n"
                    f"{active_orig['n_clients'].iloc[0]} clients",
                    fontsize=9, fontweight="bold", color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"), zorder=30)
        L["depots"].plot(ax=ax, color="black", marker="X", markersize=140, zorder=26)
        _base(ax, L, offline_basemap, f"Petal step {step + 1}  |  " + "   ".join(parts))
        fig.canvas.draw_idle()

    def _stop_auto():
        timer.stop()
        state["auto_running"] = False
        btn_auto.label.set_text("Auto  ▶▶")
        fig.canvas.draw_idle()

    def auto_tick():
        if state["step"] < max_step:
            state["step"] += 1
            draw(state["step"])
        else:
            _stop_auto()

    timer.add_callback(auto_tick)

    def prev_clicked(_):
        if state["auto_running"]:
            _stop_auto()
        state["step"] = max(0, state["step"] - 1)
        draw(state["step"])

    def next_clicked(_):
        if state["auto_running"]:
            _stop_auto()
        state["step"] = min(max_step, state["step"] + 1)
        draw(state["step"])

    def auto_clicked(_):
        if state["auto_running"]:
            _stop_auto()
        else:
            if state["step"] >= max_step:
                state["step"] = 0
            state["auto_running"] = True
            btn_auto.label.set_text("Stop  ■")
            timer.start()

    btn_prev.on_clicked(prev_clicked)
    btn_auto.on_clicked(auto_clicked)
    btn_next.on_clicked(next_clicked)
    draw(state["step"])
    plt.show()
