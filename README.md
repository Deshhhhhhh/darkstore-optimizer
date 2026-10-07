# Quick-Commerce Dark Store Location Optimizer: Jaipur

**Live dashboard:** https://darkstore-optimizer-zzbz7wrmfxqugs34zrknym.streamlit.app/
## Business problem
A quick-commerce company wants to open new dark stores in Jaipur. Each store costs lakhs in
rent, fit-out and inventory, and a wrong location means low orders and missed 10-minute
promises. **Where should the stores go to reach the most people, with the least overlap?**

## Key findings (10 stores, 10-minute delivery, 12 km study area)
- The optimized layout reaches **62.6% of the study-area population (about 1.81 million people)**.
  Random placement averages **27.0%** (best of 1,000 random layouts: 41.9%), and simply picking
  the 10 highest-scoring hexagons reaches only **23.1%**, because they cluster together.
- **Delivery time matters as much as store count.** 10 stores at 12 minutes cover 77.1%, almost
  the same as 15 stores at 10 minutes (77.6%).
- **Diminishing returns:** going from 5 to 10 stores adds about 27 points of coverage; 10 to 15 adds about 15.
- **Site choice is driven mainly by where people live and what the road network allows.**
  Removing the competition penalty leaves all 10 sites in the same areas; a population-only model
  keeps 70% of them.

## Method
1. **Grid:** H3 hexagons (resolution 8, about 0.74 km2) over a 12 km radius around central Jaipur.
2. **Demand:** WorldPop 2024 population (100 m), colleges, hostels, offices and malls (OpenStreetMap),
   and cafe, gym and salon density (Overture Maps) as a spending-power proxy.
3. **Competition:** grocery and chain-store counts from Overture Maps and OpenStreetMap.
4. **Reach:** travel-time zones on the OpenStreetMap road network at a constant 20 km/h.
5. **Optimization:** maximum-coverage location problem solved as a mixed-integer program
   (SciPy / HiGHS). A hexagon counts once even if several stores reach it.
6. **Hexagon value:** population x demand score x (1 - 0.5 x competition score).
   The weights are my own judgment calls, which is why the sensitivity analysis exists.
7. **Sensitivity:** 8 scenarios varying delivery time, number of stores and competition penalty.

## Limitations
- WorldPop is a modelled estimate, not a census count.
- Open data cannot see existing dark stores, so competition is a proxy from shops and chains.
- OpenStreetMap is under-mapped for Indian grocery shops, which is why Overture Maps was added.
- Constant rider speed ignores traffic and time of day.
- Stores near the edge of the study area cannot see demand beyond it.
- Dashboard break-even inputs are illustrative assumptions, not company data.

## Data sources
WorldPop, OpenStreetMap (OSMnx), Overture Maps Places, H3.
Contains data from OpenStreetMap contributors (ODbL), WorldPop (CC BY 4.0) and Overture Maps Foundation.

## Run it yourself
Dashboard only:

    pip install -r requirements.txt
    streamlit run app/app.py

Rebuild the full pipeline (needs `pip install -r requirements-pipeline.txt` and the WorldPop
India population file in `data/raw/`): run `src/01_build_grid.py`, `02_osm_features.py`,
`03_population.py`, `04_places.py`, `05_scores.py`, `06_coverage.py`, `07_optimize.py` in order.