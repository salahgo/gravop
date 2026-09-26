#!/bin/bash
# GravOp — historical SIDM application rebuild. See scripts/reproduce.py for current targets.
#   bash scripts/run_all.sh
# Stages are independent; comment out completed ones when resuming.
# Requires external reference code and N-body inputs; current research validation is separate.
set -euo pipefail
PY=${PY:-.venv/bin/python}
if [ "${GRAVOP_REBUILD_HISTORICAL:-}" != "1" ]; then
  echo "Historical application rebuild can overwrite artifacts. Use scripts/reproduce.py for current verification."
  echo "To rebuild in a separate checkout: GRAVOP_REBUILD_HISTORICAL=1 bash scripts/run_all.sh"
  exit 2
fi
mkdir -p logs results

echo "== acceptance gates =="
$PY -m pytest tests/ -q                                             2>&1 | tee logs/tests.log

echo "== E1 solver verification =="
$PY scripts/e1_verify.py smoke                                      2>&1 | tee logs/e1_smoke.log
$PY scripts/e1_verify.py vdep                                       2>&1 | tee logs/e1_vdep.log
# explicit-scheme cross-check against GravothermalSIDM (slow: ~45 min,
# dominated by the reference implementation)
$PY scripts/e1_crosscheck.py                                        2>&1 | tee logs/e1_crosscheck.log

echo "== E2 N-body validation (requires data/d3_fischer2025; see DATA.md) =="
$PY scripts/e2_nbody.py profiles                                    2>&1 | tee logs/e2_profiles.log
$PY scripts/e2_nbody.py compare                                     2>&1 | tee logs/e2_compare.log

echo "== datasets (seeds fixed; ~10 min on 20 cores) =="
$PY scripts/gen_dataset.py train     1300 --seed 20260827 --workers 20 --set train     2>&1 | tee logs/gen_train.log
$PY scripts/gen_dataset.py ood_param  100 --seed 30260827 --workers 20 --set ood_param 2>&1 | tee logs/gen_oodp.log
$PY scripts/gen_dataset.py ood_ic     100 --seed 40260827 --workers 20 --set ood_ic    2>&1 | tee logs/gen_oodic.log

echo "== training: 5-seed ensemble + baselines (~15 min) =="
for s in 0 1 2 3 4; do
  $PY scripts/train.py --model gravop --seed $s --out results/runs/gravop_s$s 2>&1 | tee logs/train_gravop_s$s.log
done
$PY scripts/train.py --model fno_nophys --seed 0 --out results/runs/fno_nophys 2>&1 | tee logs/train_nophys.log
$PY scripts/train.py --model deeponet   --seed 0 --out results/runs/deeponet   2>&1 | tee logs/train_deeponet.log
$PY scripts/baselines_pcamlp.py --seed 0 --out results/runs_pca/pcamlp         2>&1 | tee logs/train_pcamlp.log
# P9: the analytic (untrained) domain baseline; needs the gravop_s* runs above
$PY scripts/baselines_parametric.py                                            2>&1 | tee logs/baselines_parametric.log

echo "== ablations: data efficiency of the physics losses =="
for n in 65 130 260 520 1040; do for s in 0 1; do
  $PY scripts/train.py --model gravop     --seed $s --n_train $n --out results/ablation/phys_n${n}_s$s   2>&1 | tee logs/abl_phys_n${n}_s$s.log
  $PY scripts/train.py --model fno_nophys --seed $s --n_train $n --out results/ablation/nophys_n${n}_s$s 2>&1 | tee logs/abl_nophys_n${n}_s$s.log
done; done

echo "== evaluation, benchmarks, transfer =="
$PY scripts/evaluate_all.py                                         2>&1 | tee logs/evaluate_all.log
$PY scripts/e8_bench.py                                             2>&1 | tee logs/e8_bench.log
$PY scripts/e10_resolution.py --n_runs 60                           2>&1 | tee logs/e10_resolution.log
$PY scripts/e11_emulator_vs_nbody.py                                2>&1 | tee logs/e11_emulator_vs_nbody.log

echo "== Bayesian inference =="
$PY scripts/e9_inference.py mock                                    2>&1 | tee logs/e9_mock.log
$PY scripts/e9_inference.py emcee                                   2>&1 | tee logs/e9_emcee.log
$PY scripts/e9_inference.py grid --ngrid 31 --workers 20            2>&1 | tee logs/e9_grid.log
$PY scripts/e9_inference.py summary                                 2>&1 | tee logs/e9_summary.log
$PY scripts/e9_inference.py coverage --tau_obs 0.7                  2>&1 | tee logs/e9_coverage.log
$PY scripts/e9_inference.py coverage --tau_obs 0.4 --tag _tau04     2>&1 | tee logs/e9_coverage_tau04.log

echo "== figures =="
$PY scripts/make_figures.py                                         2>&1 | tee logs/figures.log
echo "DONE — see results/figures/"
