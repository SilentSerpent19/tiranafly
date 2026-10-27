CITY = "Tirana, Albania"
NUMBER_OF_CLIENTS = 1000
RANDOM_SEED = 40

# ---- Drone specification (TiranaFly TF-1, mid-size electric multicopter) ----
# Values are inside the envelope of current delivery drones (DJI FlyCart 30:
# 16 km loaded range, 30 kg class; Wing / Zipline P2: 10-20 km round trips,
# 2-3 kg).  We take a 16 km range but keep a 10 % battery reserve, and a
# 6 kg payload so that a trip can carry several 0.5-2.3 kg parcels.
D_MAX = 16_000          # metres, maximum flight distance on one battery
BATTERY_RESERVE = 0.10  # fraction of the battery kept for safety
D_USE = D_MAX * (1 - BATTERY_RESERVE)   # usable range (14 400 m)
K_PCT_PER_M = 100.0 / D_MAX             # battery % per metre (linear model)
W_MAX = 6.0             # kg, maximum payload
PAYLOAD_RANGE = (0.5, 2.3)              # kg per parcel (uniform)
SPEED_KMH = 50.0        # average cruise speed
SERVICE_MIN = 2.0       # minutes hovering per delivery
TURNAROUND_MIN = 8.0    # minutes at depot (battery swap + loading)
WORK_HOURS = 8.0        # daily delivery window

# ---- Airspace ----
SAFETY_MARGIN_M = 40.0  # horizontal clearance kept from restricted sites

# ---- Depot placement ----
CANDIDATE_SPACING_M = 800     # spacing of candidate sites along the city outskirts
ROAD_SNAP_MAX_M = 500         # snap a site to a primary road if one is this close

# ---- Routing ----
SWEEP_STARTS = 12             # start angles tried by the rotating sweep

# ---- Visualisation ----
AUTO_INTERVAL_MS = 1000       # milliseconds between auto-play steps

# Benchmark mode (the original approach, kept for the experimental comparison)
NUMBER_OF_ZONES = 4
