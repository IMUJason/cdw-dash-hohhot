# cdw-dash-hohhot

Reproducible package for the paper *"A disposal-capacity certificate for robust CDW network design under demand drift and facility retirement: an application to Hohhot, China"*.

## What is included

- **Official-data inputs (repository root)**
  - `cdw_generation_by_district_estimates_v3.csv`: district generation estimates and annual anchors
  - `distance_matrix_wide_km.csv`: 29-node road-distance matrix
- **`src/`**: model builders, CPLEX programs, DASH, service-level coverage, the open-source evaluation heuristic, carbon accounting, audit scripts and experiment runners.
- **`src/experiments/`**: archived result CSVs, certificate predictions, reference optima and first-stage configurations used by the analysis scripts.
- **`src/VERIFICATION_REPORT.md` and `src/AUDIT_V23.md`–`AUDIT_V25.md`**: consistency checks, model-defect fixes and certificate validation records.

The model uses a multi-period location, capacity and allocation formulation for a four-echelon construction and demolition waste network. Sorting centres and recycling plants expand in modules, sorting centres hold buffers, inert spoil can be hauled directly, and landfill retirement is endogenous and irreversible. Service level is a constraint: a scenario-aware plan either serves all demand or is infeasible.

The disposal-capacity certificate uses the effective chain capacity `A = min(S, K/rho)` and the horizon-total condition `F_min(D) = D - (1-kappa) min(A,D) <= R`. The DASH ceiling is obtained from `D_max/D0 - 1`, where `D_max = R/kappa` when `R/kappa <= A`, and `R + (1-kappa)A` otherwise. This is a necessary condition for every configuration; it is tight for the instance family reported in the paper.

## Environment

- Python 3.10 (tested with 3.10.19)
- `numpy==1.26.4`, `scipy==1.14.1`, `pandas`, `docplex`, IBM CPLEX 22.1.1 for exact benchmarks
- scipy/HiGHS is used for the LP recourse and the open-source evaluation heuristic

Do not use NumPy 2.x with the pinned docplex version: `docplex 2.24.231` refers to `numpy.float_`, which was removed in NumPy 2.0.

## Reproduce the main checks

Run these commands from the **repository root**. All paths are relative to the checkout; no local absolute path is required.

```bash
# Build src/instance_v1.json and src/instance_report.md from the input CSVs
python src/build_instance.py

# Deterministic and two-stage consistency checks (requires CPLEX)
python src/cplex_d0.py
python src/cplex_saa.py
python src/cplex_dro.py

# Replica-level statistics (requires the archived main CSV)
python src/rev2_scripts/stats_v22.py

# Certificate prediction and capacity-perturbation checks
python src/rev2_scripts/eps_formula_v2.py
python src/rev2_scripts/eps_formula_perturb.py

# Carbon accounting with explicit raw/direct-haul counterfactual columns
python src/rev2_scripts/run_co2_v24.py

# Point-mass reference: a lower bound on the exact Wasserstein worst case
python src/rev2_scripts/worstcase_reference.py
```

`build_instance.py` writes `src/instance_v1.json`. The analysis scripts read that file and the archived files under `src/experiments/`; they do not require a paper directory or a user-specific path.

For the small-instance heuristic benchmark, `src/rev2_scripts/gri_small_cplex.py` can be run after the main consistency checks. It uses CPLEX with `mip_gap=1e-9` and writes the reference results to `src/experiments/results_gri_small_cplex_tight.csv`.

## Result map

- `results_main_v22.csv`: main comparison, 11 instances × 5 methods (DASH dial 0.16)
- `results_main_exactfix.csv`: isolated re-solves of slow main-table cells
- `results_penalty_v22.csv`: penalty sweep and the DET/SAA crossover near 270 CNY/t
- `results_loo_v22.csv`: leave-one-year-out backtests
- `results_grid_v22.csv`, `results_rhophi_v22.csv`: cost/revenue and recovery sweeps
- `results_ablation_v22.csv`: SAA/MV/DASH/ROB comparisons
- `results_co2_v24.csv`: raw and direct-haul counterfactual transport/carbon columns
- `results_worstcase_ref.csv`: point-mass lower-bound reference, not a CCG solution
- `eps_formula_v2_instances.csv`, `eps_formula_v2_rhophi.csv`: certificate predictions
- `eps_formula_perturb.csv`: landfill-volume perturbation verification
- `results_certificate_cross_structure.csv`: changed demand/sorting/landfill structures
- `results_certificate_blind_points.csv`: out-of-grid recovery blind tests
- `results_gri_small_cplex_tight.csv`, `results_gri_final_table.csv`: tightened reference optima and multi-seed heuristic results

Raw official documents are public records and are not redistributed. Facility coordinates are embedded in `src/build_instance.py`; the distance matrix can be rebuilt with an open routing engine.

## Citation

If you use this package, please cite the paper.

## License

Code: MIT. Data: compiled from official public records of Hohhot; use subject to the source terms.
