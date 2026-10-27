# cdw-dash-hohhot

Reproducible package for the paper *"A disposal-capacity certificate for robust CDW network design under demand drift and facility retirement: an application to Hohhot, China"*.

## What this package contains

- **Input data (CSV, repository root)** — the two external inputs of the case, both derived from official public sources of Hohhot, China (ecological-environment annual reports 2023–2025; urban-management facility bulletin 2025; statistical yearbook 2025; seventh national census) and from driving distances on the municipal road network:
  - `cdw_generation_by_district_estimates_v3.csv` — district-level generation estimates and the 2023–2025 annual anchors
  - `distance_matrix_wide_km.csv` — 29-node driving-distance matrix (rebuildable via any open routing engine from the released facility coordinates, which are embedded in `src/build_instance.py`)
- **`src/`** — the model and the experiments: instance builder, deterministic program (CPLEX), two-stage sample-average program, drift-augmented support hedging (DASH), the service-level coverage constraint, the disposal-capacity certificate scripts, carbon accounting, experiment runners, verification scripts, and the open-source evaluation heuristic.
- **`results/`** — one result file per study block: the main comparison across the eleven instances, the isolated re-solves of the slow cells, the penalty sweep, the leave-one-year-out backtests, the recovery and cost grids, the robustification ablation, the tightened reference optima used to certify the heuristic, and the certificate predictions.
- **`src/VERIFICATION_REPORT.md`** — the cross-validation protocol (hand-computed tiny instance; conservation residuals; SAA and DASH degeneracy checks; heuristic gap against proven optima).
- **`src/AUDIT_V23.md`, `src/AUDIT_V24.md`, `src/AUDIT_V25.md`** — the model-defect audit trail: the missing sorting-expansion variable, the terminal-stock fix, and the derivation and numerical verification of the disposal-capacity certificate.

## The model in one paragraph

A multi-period location, capacity, and allocation program for a four-echelon construction and demolition waste network. Sorting centres and recycling plants expand in 0.1 Mt modules, sorting centres hold buffers, inert spoil moves by direct haul, and landfill retirement is an endogenous, irreversible decision. Service level enters as a constraint rather than as a price, so a scenario-aware plan either serves the demand it was built for or is infeasible. The uncertainty treatment mixes the empirical measure of the three observed years with lifted demand scenarios, calibrated from official records rather than estimated from data.

## Reproducing the results

Python 3.10+ with `docplex` + IBM CPLEX (exact benchmarks) and `scipy` (HiGHS, for the LP recourse and the evaluation heuristic).

```bash
# 0. build the instance JSON from the two input CSVs
python src/build_instance.py

# 1. deterministic base case and consistency checks (requires CPLEX)
python src/cplex_d0.py

# 2. two-stage sample-average program and degeneracy check
python src/cplex_saa.py

# 3. DASH and the check that epsilon = 0 reduces to the sample average
python src/cplex_dro.py

# 4. main experiments (methods x 10 randomized replicas x 200 shared scenarios)
python src/run_experiments.py --stage all

# 5. verification and cross-checks
python src/rev2_scripts/audit_crosscheck.py
python src/rev2_scripts/audit_fixcheck.py

# 6. disposal-capacity certificate: prediction and numerical verification
python src/eps_formula_v2.py
python src/eps_formula_perturb.py

# 7. statistical tests (replica-level Wilcoxon with Holm correction, Cliff's delta)
python src/rev2_scripts/stats_v22.py
```

The raw official documents (annual reports, bulletins, yearbook pages) are public records; they are not redistributed here. Facility coordinates are embedded in `src/build_instance.py` and are sufficient to rebuild the distance matrix with any open routing engine.

## Citation

If you use this package, please cite the paper.

## License

Code: MIT. Data: compiled from official public records of Hohhot; use subject to the source terms.
