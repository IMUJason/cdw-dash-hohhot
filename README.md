# cdw-dash-hohhot

Minimal reproducible package for the paper *"Robust capacity phasing for construction and demolition waste recycling networks under demand drift and facility retirement: a small-sample-robust optimization study of Hohhot, China"* (submitted to *Journal of Cleaner Production*).

## What this package contains

- **Input data (CSV, repository root)** — the two external inputs of the case, both derived from official public sources of Hohhot, China (ecological-environment annual reports 2023–2025; urban-management facility bulletin 2025; statistical yearbook 2025; seventh national census) and from driving distances on the municipal road network:
  - `cdw_generation_by_district_estimates_v3.csv` — district-level generation estimates and the 2023–2025 annual anchors
  - `distance_matrix_wide_km.csv` — 29-node driving-distance matrix (rebuildable via any open routing engine from the released facility coordinates, which are embedded in `src/build_instance.py`)
- **`src/`** — the core algorithm scripts: instance builder, deterministic model (CPLEX), two-stage SAA, drift-augmented support hedging (DASH), hybrid ALNS with exact LP recourse, carbon accounting, experiment runner, and statistics.
- **`results/`** — the archived deterministic base-case solution.
- **`VERIFICATION_REPORT.md`** — the V1–V5 cross-validation protocol (hand-computed tiny instance; conservation residuals; SAA/DASH degeneracy checks; ALNS gap).

## Reproducing the results

Python 3.10+ with `docplex` + IBM CPLEX (exact benchmarks) and `scipy` (HiGHS, for the LP recourse and the ALNS).

```bash
# 0. build the instance JSON from the two input CSVs
python src/build_instance.py

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

The raw official documents (annual reports, bulletins, yearbook pages) are public records; they are not redistributed here. Facility coordinates are embedded in `src/build_instance.py` and are sufficient to rebuild the distance matrix with any open routing engine.

## Citation

If you use this package, please cite the paper.

## License

Code: MIT. Data: compiled from official public records of Hohhot; use subject to the source terms.
