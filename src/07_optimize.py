import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
import folium

K, T_MIN, LAMBDA = 10, 10, 0.5
MIN_REACH = 5          # ignore sites that reach fewer hexagons (dead-end edge sites)
RANDOM_RUNS = 1000
MATCH_KM = 1.5         # two sites within 1.5 km count as "the same area"

grid = gpd.read_file("data/processed/grid_scores.geojson").reset_index(drop=True)
cov = pd.read_csv("data/processed/coverage.csv")
POP = grid["pop"].values.astype(float)
TOTAL_POP = POP.sum()


def make_weights(lam=LAMBDA, pop_only=False):
    if pop_only:
        return POP.copy()
    return POP * grid["demand"].values * (1 - lam * grid["supply"].values)


def eligible(t_min):
    c = cov[cov["secs"] <= t_min * 60]
    reach = c.groupby("cand")["hex"].nunique()
    ok = [i for i in reach.index if reach[i] >= MIN_REACH and POP[i] > 0]
    return c[c["cand"].isin(ok)], ok


def solve(w, k, t_min):
    c, cands = eligible(t_min)
    cands = [int(i) for i in cands]
    hexes = np.sort(c["hex"].unique())
    cidx = {cnd: a for a, cnd in enumerate(cands)}
    hidx = {int(h): b for b, h in enumerate(hexes)}
    n, m = len(cands), len(hexes)

    r = c["hex"].astype(int).map(hidx).values           # coverage row of each pair
    col = c["cand"].astype(int).map(cidx).values        # site column of each pair

    # variables: [x_0..x_{n-1} (site chosen), y_0..y_{m-1} (hexagon covered)]
    # constraint per hexagon j:  y_j - sum(x_i that reach j) <= 0
    # last constraint:           sum(x_i) = k
    rows = np.concatenate([r, np.arange(m), np.full(n, m)])
    cols = np.concatenate([col, n + np.arange(m), np.arange(n)])
    vals = np.concatenate([-np.ones(len(r)), np.ones(m), np.ones(n)])
    A = coo_matrix((vals, (rows, cols)), shape=(m + 1, n + m)).tocsr()

    lb = np.concatenate([np.full(m, -np.inf), [k]])
    ub = np.concatenate([np.zeros(m), [k]])
    obj = np.concatenate([np.zeros(n), -w[hexes]])       # maximize value = minimize -value
    integrality = np.concatenate([np.ones(n), np.zeros(m)])

    res = milp(obj, constraints=LinearConstraint(A, lb, ub),
               integrality=integrality, bounds=Bounds(0, 1),
               options={"time_limit": 120, "mip_rel_gap": 0.001})
    if res.x is None:
        raise RuntimeError(f"Solver found no solution: {res.message}")
    return [cands[a] for a in range(n) if res.x[a] > 0.5]

def covered(sites, t_min):
    c = cov[(cov["secs"] <= t_min * 60) & (cov["cand"].isin(list(sites)))]
    return np.unique(c["hex"].values).astype(int)


def score(sites, t_min, w):
    h = covered(sites, t_min)
    return w[h].sum(), POP[h].sum()


def psum(hexset):
    idx = np.array(sorted(hexset), dtype=int)
    return POP[idx].sum()


def km_between(i, j):
    la1, lo1, la2, lo2 = map(np.radians, [grid.at[i, "lat"], grid.at[i, "lng"],
                                          grid.at[j, "lat"], grid.at[j, "lng"]])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def stability(a, b):
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    hit = sum(any(km_between(s, l) <= MATCH_KM for l in large) for s in small)
    return hit / len(small)


# ---------- 1. Base solution ----------
w_base = make_weights()
W_TOTAL = w_base.sum()
print("Solving base case...")
sites = solve(w_base, K, T_MIN)
val, pop = score(sites, T_MIN, w_base)

# ---------- 2. Baselines ----------
_, cands = eligible(T_MIN)
rng = np.random.default_rng(42)
rv, rp = [], []
for _ in range(RANDOM_RUNS):
    s = rng.choice(cands, K, replace=False)
    v, p = score(s, T_MIN, w_base)
    rv.append(v)
    rp.append(p)
top = sorted(cands, key=lambda i: -grid.at[i, "opportunity"])[:K]
tv, tp = score(top, T_MIN, w_base)

print("\n=== RESULTS (k=%d stores, %d-minute delivery) ===" % (K, T_MIN))
print(f"{'Method':<32}{'Value covered':>15}{'People covered':>18}{'% of population':>17}")
print(f"{'Optimized':<32}{val / W_TOTAL:>14.1%}{int(pop):>18,}{pop / TOTAL_POP:>17.1%}")
print(f"{'Random (avg of 1000 runs)':<32}{np.mean(rv) / W_TOTAL:>14.1%}{int(np.mean(rp)):>18,}{np.mean(rp) / TOTAL_POP:>17.1%}")
print(f"{'Random (best of 1000 runs)':<32}{np.max(rv) / W_TOTAL:>14.1%}{int(rp[int(np.argmax(rv))]):>18,}{rp[int(np.argmax(rv))] / TOTAL_POP:>17.1%}")
print(f"{'Top-10 by opportunity score':<32}{tv / W_TOTAL:>14.1%}{int(tp):>18,}{tp / TOTAL_POP:>17.1%}")
print(f"\nOptimized beats random average by {val / np.mean(rv) - 1:.0%} (value) "
      f"and the naive top-10 by {val / tv - 1:.0%}")

# ---------- 3. Site table ----------
sets = {s: set(covered([s], T_MIN)) for s in sites}
rows = []
for s in sites:
    others = set().union(*[sets[o] for o in sites if o != s])
    excl = sets[s] - others
    rows.append(dict(site=s, h3=grid.at[s, "h3"], lat=round(grid.at[s, "lat"], 5),
                     lng=round(grid.at[s, "lng"], 5), pop_in_10min=int(psum(sets[s])),
                     exclusive_pop=int(psum(excl)), hexagons=len(sets[s])))
tab = pd.DataFrame(rows).sort_values("exclusive_pop", ascending=False).reset_index(drop=True)
tab.insert(0, "rank", tab.index + 1)
tab.drop(columns="site").to_csv("data/processed/optimal_sites.csv", index=False)
print("\nRecommended sites (check each lat,lng on Google Maps):")
print(tab[["rank", "lat", "lng", "pop_in_10min", "exclusive_pop"]].to_string(index=False))

# ---------- 4. Sensitivity ----------
scen = [
    ("Base (10 stores, 10 min, penalty 0.5)", K, 10, make_weights(0.5)),
    ("8-minute delivery", K, 8, w_base),
    ("12-minute delivery", K, 12, w_base),
    ("5 stores", 5, 10, w_base),
    ("15 stores", 15, 10, w_base),
    ("No competition penalty", K, 10, make_weights(0.0)),
    ("Strong competition penalty (1.0)", K, 10, make_weights(1.0)),
    ("Population only (control)", K, 10, make_weights(pop_only=True)),
]
print("\nRunning sensitivity scenarios (a few minutes)...")
srows = []
for name, k, t, w in scen:
    s = solve(w, k, t)
    _, p = score(s, t, w)
    srows.append(dict(scenario=name, stores=k, minutes=t,
                      pct_population_covered=round(100 * p / TOTAL_POP, 1),
                      sites_same_area_as_base=round(stability(s, sites), 2)))
sens = pd.DataFrame(srows)
sens.to_csv("data/processed/sensitivity.csv", index=False)
print(sens.to_string(index=False))

# ---------- 5. Map ----------
palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#42d4f4", "#f032e6", "#9a6324", "#808000", "#008080"]
rank_of = {int(r.site): int(r.rank) for r in tab.itertuples()}
c = cov[(cov["secs"] <= T_MIN * 60) & (cov["cand"].isin(sites))]
nearest = c.sort_values("secs").drop_duplicates("hex").set_index("hex")["cand"]
grid["site"] = [rank_of.get(int(nearest[i]), 0) if i in nearest.index else 0 for i in grid.index]
grid["color"] = grid["site"].map(lambda r: palette[(r - 1) % len(palette)] if r > 0 else "#cccccc")

tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
m = folium.Map(location=[grid["lat"].mean(), grid["lng"].mean()], zoom_start=11, tiles=None)
folium.TileLayer(tiles=tiles, attr="Esri, HERE, Garmin, OpenStreetMap contributors", name="Streets").add_to(m)
folium.GeoJson(
    grid[["h3", "pop", "site", "color", "geometry"]],
    style_function=lambda f: {"fillColor": f["properties"]["color"], "color": "#555", "weight": 0.3,
                              "fillOpacity": 0.55 if f["properties"]["site"] > 0 else 0.1},
    tooltip=folium.GeoJsonTooltip(fields=["pop", "site"], aliases=["Population", "Served by site #"]),
).add_to(m)
for r in tab.itertuples():
    style = ("background:#111;color:#fff;border-radius:50%;width:26px;height:26px;"
             "line-height:26px;text-align:center;font-weight:bold;border:2px solid #fff")
    label = f'<div style="{style}">{r.rank}</div>'
    tip = f"Site {r.rank}: {r.pop_in_10min:,} people within 10 min"
    icon = folium.DivIcon(html=label, icon_size=(26, 26), icon_anchor=(13, 13))
    folium.Marker([r.lat, r.lng], tooltip=tip, icon=icon).add_to(m)
m.save("data/processed/optimal_map.html")
print("\nSaved optimal_sites.csv, sensitivity.csv, optimal_map.html")