# interpretable-dr-me-test

Code for the finite-location causal witness experiments in the paper
*Semiparametrically Efficient Local Testing for Interpretable Distributional Treatment Effects*.

The repo is organized around a small set of runnable scripts. The scalar-outcome
core lives in `exp1_calibration_dr_me_plugin.py`, and later experiments reuse
that implementation.

## What is here

- `contour_estimated_oracle.py`
  Causal contour diagnostic comparing the feasible DR objective to the oracle one.
- `exp1_calibration_dr_me_plugin.py`
  Calibration under a confounded observational null.
- `exp2_power_exp1_extension.py`
  Power study and DR-xKTE comparison.
- `exp3_hd_whitening_ablation.py`
  High-dimensional whitening ablation.
- `exp4_local_chisq.py`
  Local noncentral chi-square validation.
- `exp5_runtime_drkte_compare.py`
  Runtime comparison against DR-xKTE.
- `octmnist_causal_vstar.py`
  OCTMNIST causal-location illustration.
- `octmnist_mean_matched_vstar.py`
  OCTMNIST mean-matched residual illustration.
- `octmnist_utils.py`
  Shared OCTMNIST download, loading, normalization, and image helpers.
- `repo_paths.py`
  Canonical output folders for all experiments.

## Output layout

Every script now defaults to a subfolder inside:

```text
results/
```

The main destinations are:

```text
results/
  contour/
  exp1/
  exp2/
  exp3/
  exp4/
  exp5/
  octmnist_causal/
  octmnist_mean_matched/
```

You can still override `--outdir` manually, but the defaults are meant to keep
the repo root clean.

## Dependencies

The scripts use a lightweight scientific Python stack:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `scikit-learn`

No heavy causal inference package is required. The OCTMNIST scripts download the
dataset directly and cache it under `octmnist_cache/`.

## Reproducing the main runs

### Contour diagnostic

```bash
python contour_estimated_oracle.py
```

This writes to `results/contour/` by default.

### Experiment 1: calibration

```bash
python exp1_calibration_dr_me_plugin.py \
  --n-reps 200 \
  --n-grid 300 600 1200 3000 \
  --J 2 \
  --M 80
```

### Experiment 2: power and DR-xKTE

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python exp2_power_exp1_extension.py \
  --n-reps 200 \
  --n-grid 300 600 1200 3000 \
  --settings null mean_shift variance_shift localized_bump \
  --n-jobs 12
```

### Experiment 3: high-dimensional whitening ablation

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python exp3_hd_whitening_ablation.py \
  --mode dim_sweep \
  --n-reps 200 \
  --n-fixed 3000 \
  --d-grid 5 10 25 50 \
  --settings two_bump \
  --J 3 \
  --M 300 \
  --effect 4.0 \
  --bump-prob 0.04 \
  --bandwidth-scale 0.6 \
  --n-jobs 8
```

### Experiment 4: local chi-square validation

```bash
python exp4_local_chisq.py \
  --n-reps 2000 \
  --n-values 500 1000 3000 \
  --h-values 0 1 2 3 4 6 8 \
  --pilot-n 100000 \
  --pilot-M 100 \
  --J 2
```

### Experiment 5: runtime comparison

```bash
python exp5_runtime_drkte_compare.py \
  --reps 5 \
  --n-grid 300 600 1200 3000 5000 10000 \
  --fixed-J 2 \
  --fixed-M 80 \
  --grad-steps 3
```

### OCTMNIST figure scripts

```bash
python octmnist_mean_matched_vstar.py
```

## Notes on the implementation

- `exp1` is the scalar-outcome reference implementation:
  propensity fitting, plug-in kernel regressions, DR pseudo-features, finite-location learning, and Hotelling testing all live there.
- `exp2` and `exp3` deliberately build on `exp1` so the power and ablation studies use the same core DR machinery.
- `exp5` keeps a separate runtime-oriented implementation because the point of that script is timing the end-to-end methods, not only reproducing the same rejection decisions.
- The OCTMNIST scripts share their dataset and image I/O code through `octmnist_utils.py`.

## Quick smoke runs

If you just want to check that everything works, lower the repetition count:

```bash
python exp1_calibration_dr_me_plugin.py --n-reps 10 --n-grid 300 600
python exp2_power_exp1_extension.py --n-reps 10 --n-grid 300 600 --n-jobs 1
python exp3_hd_whitening_ablation.py --mode dim_sweep --n-reps 10 --n-fixed 1200 --d-grid 5 10 --n-jobs 1
python exp4_local_chisq.py --n-reps 200 --n-values 500 --h-values 0 2 4
python exp5_runtime_drkte_compare.py --reps 2 --n-grid 300 600
```

## Typical artifacts

Depending on the script, you should expect:

- raw Monte Carlo CSVs
- summarized CSV tables
- paper-facing PNG figures
- metadata JSON files
- small packaged zip bundles for figure scripts
