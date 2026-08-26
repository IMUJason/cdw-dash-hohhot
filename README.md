# cdw-dash-hohhot

Minimal reproducible package for the paper *"Robust capacity phasing for construction and demolition waste recycling networks under demand drift and facility retirement: a small-sample-robust optimization study of Hohhot, China"* (submitted to *Journal of Cleaner Production*).

## What this package contains

- **`data/`** — all model inputs in documented CSV form (8 files, see below). All are derived from official public sources of Hohhot, China (ecological-environment annual reports 2023–2025; urban-management facility bulletin 2025; statistical yearbook 2025; seventh national census), and from driving distances on the municipal road network.
- **`src/`** — the six solver scripts implementing the full verification chain: deterministic model (CPLEX), two-stage SAA, drift-augmented support hedging (DASH), hybrid ALNS with exact LP recourse, the experiment runner, and the statistics module.
- **`results/`** — the archived deterministic base-case solution and `VERIFICATION_REPORT.md` documenting the V1–V5 cross-validation protocol (hand-computed tiny instance; conservation residuals; SAA/DASH degeneracy checks; ALNS gap).

## Model inputs (`data/`)

| File | Content | Provenance |
|---|---|---|
| `demand_districts.csv` | District-level generation estimates (2024/2025, ±30% bounds) | 2025 statistical yearbook (Tables 22-1 to 22-9) + 7th census |
| `facilities.csv` | Facility capacities and actual loads, 2023–2025 | Ecological-environment annual reports |
| `facility_locations.csv` | Facility coordinates (lng, lat, confidence) | Official addresses geocoded |
| `generation_points.csv` | Nine district seat coordinates | Administrative centers |
| `distance_matrix_km.csv` | 29×29 driving distance matrix | Municipal road network (rebuildable via OSRM from the released coordinates) |
| `parameters.csv` | Cost parameters with literature/EIA provenance and sensitivity ranges | See paper Table 2 |
| `annual_totals.csv` | Official annual generation totals 2023–2025 | Ecological-environment annual reports |
| `construction_activity.csv` | Construction-activity series used in the decomposition | Yearbook + district statistical communiqués |

## Reproducing the results

Python 3.10+ with `docplex` + IBM CPLEX (exact benchmarks) and `scipy` (HiGHS, for the LP recourse and the ALNS).

```bash
# 1. deterministic base case + V1/V2 consistency checks (requires CPLEX)
python src/cplex_d0.py

# 2. two-stage SAA and V3 degeneracy check
python src/cplex_saa.py

# 3. DASH and V5 check (eps = 0 reduces to SAA exactly)
python src/cplex_dro.py

# 4. hybrid ALNS and V4 gap check against the CPLEX optimum
python src/alns.py

# 5. main experiments (methods × 10 randomized replicas × 200 shared scenarios)
python src/run_experiments.py --stage all

# 6. statistical tests and figures
python src/run_stats.py
```

The raw official documents (annual reports, bulletins, yearbook pages) are public records; they are not redistributed here. Coordinates in `facility_locations.csv` and `generation_points.csv` are sufficient to rebuild the distance matrix with any open routing engine.

## Citation

If you use this package, please cite the paper.

## License

Code: MIT. Data: compiled from official public records of Hohhot; use subject to the source terms.
