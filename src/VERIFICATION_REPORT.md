# Verification report

This report documents the verification protocol behind the results in the paper
*A disposal-capacity certificate for robust CDW network design under demand
drift and facility retirement: an application to Hohhot, China*.

## 1. Recourse implementations (V1-V3)

| Check | Content | Result | Evidence |
|---|---|---|---|
| V1 | Hand-computed small instance vs CPLEX (two variants: all direct haul / rho-recycling) | PASS | 3230.0 = 3230.0, 980.0 = 980.0, difference 0.00e+00 |
| V2 | Real 9-8-4-6 x 3-period instance: optimality plus line-by-line flow conservation | PASS | integer optimal; max conservation residual 1.46e-11; recomputed cost decomposition equals the CPLEX objective |
| V3 | SAA degeneracy: the single-scenario program must equal the deterministic program | PASS | 1,047,001,158.22 = 1,047,001,158.22 |

The open-source recourse LP and the CPLEX recourse return the same objective
to a relative difference of 5.7e-15 for a fixed first-stage configuration.
DASH at eps = 0 equals the sample-average program exactly.

## 2. Main comparison

- 55 cells (11 instances x 5 methods), each solved to a relative MIP gap of at most 1e-4.
- The sixteen longest solves were verified in isolated single-run re-solves; the isolated
  objectives are the values reported in the paper and archived in `results_main_v22.csv`
  (`results_main_verification.csv` holds the verification solves).
- The service-level counterfactual is archived in `results_coverage_counterfactual.csv`:
  without the coverage constraint the plan leaves 466,700 t of the 2024 peak anchor
  unserved; imposing it costs 0.21% more objective and 24.4 M CNY more fixed capital.

## 3. Disposal-capacity certificate

- Predicted ceiling eps* = 0.1647 on the real network; solver check: DASH(0.17) is
  infeasible (`results_eps017_check.csv`).
- Cross-structure checks (`results_certificate_cross_structure.csv`): sorting +20%
  gives 0.2075 (feasible at 0.198, infeasible at 0.218); demand -20% gives 0.4559
  (0.446 / 0.466); landfill volume -10% gives 0.1139 (0.104 / 0.124).
- Blind points outside the design grid (`results_certificate_blind_points.csv`) match
  their predicted boundaries.
- Registry-vintage sensitivity moves the ceiling between roughly 4% and 18%;
  the support-corner D0 gives 10.6% (`eps_formula_perturb.csv`).

## 4. Heuristic reference

- The heuristic is certified against CPLEX optima at a 1e-9 gap on the same
  ten-scenario programs (`results_gri_final_table.csv`): it attains the proven
  optimum on ten of eleven instances; per-instance mean and worst gaps are recorded.
- Six-fold-scale clone: the CPLEX incumbent after 1,829 s is 5,417,449,352 at a 17.5%
  gap; no optimality claim is made at that size (`results_symmetry_test_m6.csv`).

## 5. Worst-case reference

- `results_worstcase_ref.csv` holds the point-mass objective on the most costly
  lifted scenario for each dial. This is an upper stress reference over the lifted
  support; it brackets the calibrated mixture from above, is not feasible in the
  Wasserstein ball, and is not a column-and-constraint-generation solution.

## 6. Reproducibility

- `solution_d0_base.json` archives the base-case plan (objective 718,902,859 CNY at
  a 9.0e-05 gap), consistent with the 0.719 billion reported in the paper.
- Figures and the graphical abstract regenerate from the archived CSVs via
  `src/make_figures.py` and `src/make_graphical_abstract.py`.
