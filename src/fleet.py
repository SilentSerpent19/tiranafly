"""
Fleet sizing: how many drones fly all petals of a zone within the daily
window?

Each petal is a job with duration
    t = 60 * length / speed  +  n_clients * service  +  turnaround   (min)
and the drones are bins of capacity `window` minutes.  Minimising the number
of bins is Bin Packing (NP-hard).  We use First-Fit Decreasing (Johnson,
1973), whose solution never exceeds  11/9 * OPT + 6/9  bins, and report the
trivial lower bound  ceil(total time / window)  next to it so the gap is
visible.
"""
import math


def petal_minutes(length_m, n_clients, speed_kmh, service_min, turnaround_min):
    return 60.0 * (length_m / 1000.0) / speed_kmh + n_clients * service_min + turnaround_min


def ffd_schedule(durations, window_min):
    """First-Fit Decreasing.  Returns list of bins, each a list of job ids."""
    order = sorted(range(len(durations)), key=lambda i: -durations[i])
    bins, slack = [], []
    for i in order:
        d = durations[i]
        for b, s in enumerate(slack):
            if d <= s + 1e-9:
                bins[b].append(i)
                slack[b] -= d
                break
        else:
            bins.append([i])
            slack.append(window_min - d)
    return bins


def fleet_for_zone(petals, speed_kmh, service_min, turnaround_min, work_hours):
    window = work_hours * 60.0
    dur = [petal_minutes(p["length_m"], len(p["clients"]), speed_kmh, service_min, turnaround_min)
           for p in petals]
    bins = ffd_schedule(dur, window)
    return {"drones": len(bins), "lower_bound": math.ceil(sum(dur) / window - 1e-9),
            "total_minutes": round(sum(dur), 1), "schedule": bins,
            "utilisation": round(sum(dur) / (len(bins) * window), 3) if bins else 0.0}
