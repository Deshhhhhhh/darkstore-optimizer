import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import folium
import h3
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

st.set_page_config(page_title="Dark Store Optimizer - Jaipur", layout="wide")

RS = "\u20b9"
DATA = Path(__file__).resolve().parent.parent / "data" / "processed"
MIN_REACH = 5
TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
ATTR = "Esri, HERE, Garmin, OpenStreetMap contributors"
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
           "#9a6324", "#808000", "#008080", "#bcf60c", "#fabebe", "#e6beff", "#800000",
           "#aaffc3", "#000075", "#ffd8b1", "#469990", "#a9a9a9", "#ffe119"]


@st.cache_data
def load():
    grid = pd.read_csv(DATA / "grid_scores.csv")
    cov = pd.read_csv(DATA / "coverage.csv")
    return grid, cov


@st.cache_data(show_spinner="Optimizing store locations...")
def optimize(k, t_min, lam, pop_only):
    grid, cov = load()
    pop = grid["pop"].to_numpy(float)
    if pop_only:
        w = pop.copy()
    else:
        w = pop * grid["demand"].to_numpy(float) * (1 - lam * grid["supply"].to_numpy(float))

    c = cov[cov["secs"] <= t_min * 60]
    reach = c.groupby("cand")["hex"].nunique()
    cands = [int(i) for i in reach.index if reach[i] >= MIN_REACH and pop[int(i)] > 0]
    c = c[c["cand"].isin(cands)]
    hexes = np.sort(c["hex"].unique())
    n, m = len(cands), len(hexes)
    k = min(k, n)

    cidx = {cd: a for a, cd in enumerate(cands)}
    hidx = {int(h): b for b, h in enumerate(hexes)}
    r = c["hex"].astype(int).map(hidx).to_numpy()
    col = c["cand"].astype(int).map(cidx).to_numpy()

    rows = np.concatenate([r, np.arange(m), np.full(n, m)])
    cols = np.concatenate([col, n + np.arange(m), np.arange(n)])
    vals = np.concatenate([-np.ones(len(r)), np.ones(m), np.ones(n)])
    A = coo_matrix((vals, (rows, cols)), shape=(m + 1, n + m)).tocsr()
    lb = np.concatenate([np.full(m, -np.inf), [k]])
    ub = np.concatenate([np.zeros(m), [k]])
    obj = np.concatenate([np.zeros(n), -w[hexes]])
    integrality = np.concatenate([np.ones(n), np.zeros(m)])

    res = milp(obj, constraints=LinearConstraint(A, lb, ub), integrality=integrality,
               bounds=Bounds(0, 1), options={"time_limit": 30, "mip_rel_gap": 0.002})
    if res.x is None:
        return [], np.array([], dtype=int), np.array([], dtype=int)

    sites = [cands[a] for a in range(n) if res.x[a] > 0.5]
    sel = c[c["cand"].isin(sites)].sort_values("secs").drop_duplicates("hex")
    return sites, sel["hex"].astype(int).to_numpy(), sel["cand"].astype(int).to_numpy()


def build_map(grid, owner, rank_of, sites_df):
    m = folium.Map(location=[grid["lat"].mean(), grid["lng"].mean()], zoom_start=11, tiles=None)
    folium.TileLayer(tiles=TILES, attr=ATTR, name="Streets").add_to(m)
    for i, (hx, p) in enumerate(zip(grid["h3"], grid["pop"])):
        s = owner.get(i)
        rk = rank_of.get(s, 0) if s is not None else 0
        color = PALETTE[(rk - 1) % len(PALETTE)] if rk > 0 else "#999999"
        tip = f"Population {int(p):,} | " + (f"served by site {rk}" if rk else "not covered")
        folium.Polygon(locations=list(h3.cell_to_boundary(hx)), color="#555555", weight=0.3,
                       fill=True, fill_color=color, fill_opacity=0.55 if rk else 0.08,
                       tooltip=tip).add_to(m)
    for r in sites_df.itertuples():
        style = ("background:#111;color:#fff;border-radius:50%;width:26px;height:26px;"
                 "line-height:26px;text-align:center;font-weight:bold;border:2px solid #fff")
        icon = folium.DivIcon(html=f'<div style="{style}">{r.Site}</div>',
                              icon_size=(26, 26), icon_anchor=(13, 13))
        folium.Marker([r.Lat, r.Lng], icon=icon,
                      tooltip=f"Site {r.Site}: {int(r.People_served):,} people served").add_to(m)
    return m


# ---------------- Sidebar ----------------
st.sidebar.header("Scenario controls")
k = st.sidebar.slider("Number of dark stores", 3, 20, 10)
t_min = st.sidebar.slider("Delivery time promise (minutes)", 6, 15, 10)
lam = st.sidebar.slider("Competition penalty", 0.0, 1.0, 0.5, 0.1,
                        help="How strongly existing shops reduce the value of an area.")
pop_only = st.sidebar.checkbox("Population only (ignore demand and competition scores)", value=False)
st.sidebar.caption("Rider speed is assumed constant at 20 km/h on the OpenStreetMap road network. "
                   "Study area: 12 km radius around central Jaipur.")

# ---------------- Compute ----------------
grid, cov = load()
sites, ah, asite = optimize(k, t_min, round(lam, 2), pop_only)
if not sites:
    st.error("The solver found no solution for these settings. Try different values.")
    st.stop()

pop = grid["pop"].to_numpy(float)
served = pd.DataFrame({"hex": ah, "site": asite})
served["pop"] = pop[served["hex"].to_numpy()]
per_site = served.groupby("site")["pop"].sum().reindex(sites).fillna(0).sort_values(ascending=False)
rank_of = {int(s): i + 1 for i, s in enumerate(per_site.index)}
owner = dict(zip(ah.tolist(), asite.tolist()))

sites_df = pd.DataFrame({
    "Site": [rank_of[int(s)] for s in per_site.index],
    "Lat": [round(float(grid.at[int(s), "lat"]), 5) for s in per_site.index],
    "Lng": [round(float(grid.at[int(s), "lng"]), 5) for s in per_site.index],
    "People_served": per_site.round().astype(int).to_numpy(),
})

covered = float(served["pop"].sum())
total = float(pop.sum())

# ---------------- Header ----------------
st.title("Quick-Commerce Dark Store Location Optimizer: Jaipur")
st.caption("Where should a quick-commerce company open its next dark stores so that the most "
           "people get delivery within the promised time? Move the sliders and the model re-optimizes.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stores", len(sites))
c2.metric("Delivery promise", f"{t_min} min")
c3.metric("People covered", f"{covered:,.0f}", f"{covered / total:.1%} of study area")
c4.metric("Avg people per store", f"{covered / len(sites):,.0f}")

tab1, tab2, tab3 = st.tabs(["Map and sites", "Business case", "Robustness and method"])

# ---------------- Tab 1 ----------------
with tab1:
    st.caption("Each colour is the area one store can reach within the delivery time over real roads. "
               "Grey hexagons are not covered. Hover over a hexagon for details.")
    fmap = build_map(grid, owner, rank_of, sites_df)
    components.html(fmap.get_root().render(), height=620)
    st.subheader("Recommended sites")
    st.dataframe(sites_df.rename(columns={"People_served": "People served"}), hide_index=True)
    st.download_button("Download site list (CSV)", sites_df.to_csv(index=False),
                       "recommended_sites.csv", "text/csv")

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Break-even calculator")
    st.warning("The defaults below are illustrative placeholders, not market data. "
               "Edit them to test your own assumptions.")
    a, b, c = st.columns(3)
    adopt = a.number_input("Monthly active customers (% of people covered)", 0.5, 20.0, 3.0, 0.5) / 100
    opm = a.number_input("Orders per active customer per month", 1.0, 20.0, 4.0, 0.5)
    aov = b.number_input(f"Average order value ({RS})", 100, 2000, 450, 10)
    margin = b.number_input("Contribution margin per order (%)", 1.0, 40.0, 12.0, 0.5) / 100
    fixed = c.number_input(f"Fixed cost per store per month ({RS} lakh)", 1.0, 30.0, 5.0, 0.5) * 100000

    biz = sites_df.copy()
    biz["Customers"] = biz["People_served"] * adopt
    biz["Orders_day"] = biz["Customers"] * opm / 30
    be = fixed / (aov * margin * 30)
    biz["Contribution_lakh"] = (biz["Orders_day"] * 30 * aov * margin - fixed) / 100000
    biz["Status"] = np.where(biz["Orders_day"] >= be, "Above break-even", "Below break-even")

    m1, m2, m3 = st.columns(3)
    m1.metric("Break-even orders per store per day", f"{be:,.0f}")
    m2.metric("Stores above break-even", f"{(biz['Orders_day'] >= be).sum()} of {len(biz)}")
    m3.metric(f"Network monthly contribution ({RS} lakh)", f"{biz['Contribution_lakh'].sum():,.1f}")

    show = biz[["Site", "People_served", "Orders_day", "Contribution_lakh", "Status"]].copy()
    show.columns = ["Site", "People served", "Est. orders/day", f"Monthly contribution ({RS} lakh)", "Status"]
    show["Est. orders/day"] = show["Est. orders/day"].round(0).astype(int)
    show[f"Monthly contribution ({RS} lakh)"] = show[f"Monthly contribution ({RS} lakh)"].round(1)
    st.dataframe(show, hide_index=True)
    st.bar_chart(biz.set_index("Site")["Orders_day"])
    st.caption(f"Bars show estimated orders per day per store. Break-even is about {be:,.0f} orders/day.")

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader("Sensitivity analysis")
    sens_file = DATA / "sensitivity.csv"
    if sens_file.exists():
        sens = pd.read_csv(sens_file)
        sens.columns = ["Scenario", "Stores", "Minutes", "% population covered",
                        "Share of sites in same area as base case"]
        st.dataframe(sens, hide_index=True)
        st.caption("Computed offline with 10 stores, 10 minutes and a 0.5 penalty as the base case. "
                   "Two sites count as the same area if they are within 1.5 km.")
    st.subheader("Method")
    st.markdown("""
1. **Grid:** Jaipur's 12 km study area split into H3 hexagons (resolution 8, about 0.74 km2 each).
2. **Demand:** WorldPop 2024 population (100 m), plus colleges, hostels, offices and malls from OpenStreetMap,
   and shop and amenity density from Overture Maps as a spending proxy.
3. **Competition:** grocery and chain-store counts from Overture Maps and OpenStreetMap.
4. **Reach:** travel-time zones computed on the OpenStreetMap road network at a constant 20 km/h.
5. **Optimization:** maximum-coverage location problem solved as a mixed-integer program (SciPy / HiGHS).
    """)
    st.subheader("Limitations")
    st.markdown("""
- WorldPop is a modelled estimate, not a census count.
- Open data cannot see existing dark stores, so competition is a proxy.
- Constant rider speed ignores traffic and time of day.
- Stores near the edge of the study area cannot see demand beyond it.
- The break-even inputs are assumptions and not real company data.
    """)