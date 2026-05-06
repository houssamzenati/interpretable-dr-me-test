# interpretable-dr-me-test

This repository contains the experiment code for the paper
*Semiparametrically Efficient Local Testing for Interpretable Distributional Treatment Effects*.

The code is script-driven on purpose. `exp1_calibration_dr_me_plugin.py` is the
main scalar DR-ME implementation, and the later experiments either reuse it
directly or stay close to the same setup.

## Repository layout

The main scripts are:

- `exp1_calibration_dr_me_plugin.py`
  Confounded sharp-null calibration for DR-ME and its baselines.
- `exp2_power_exp1_extension.py`
  Power experiment with an equal-in-law null and DR-xKTE as the global
  comparison.
- `exp3_hd_whitening_ablation.py`
  High-dimensional outcome ablation for location learning.
- `exp4_local_chisq.py`
  Local noncentral chi-square validation for fixed locations.
- `exp5_runtime_drkte_compare.py`
  Runtime comparison between finite-location DR-ME variants and DR-xKTE.
- `contour_estimated_oracle.py`
  Contour diagnostic for the DR-ME training objective.
- `octmnist_location_learning.py`
  Residual-based OCTMNIST illustration.
- `octmnist_utils.py`
  Shared OCTMNIST loading and image helpers.
- `repo_paths.py`
  Default output folders.

## Outputs

By default, each script writes into a subfolder of `results/`. The main outputs
are figures, paper-ready `.tex` tables, and a few zip bundles for the OCTMNIST
figures. 

The paper-facing LaTeX tables are generated directly by the experiment scripts:

- `results/exp1/tab_exp1_calibration.tex`
- `results/exp2/tab_power_odrme_drxkte.tex`
- `results/exp2/tab_power_diagnostics_nmax.tex`
- `results/exp3/tab_exp_hd_null.tex`
- `results/exp3/tab_exp_hd_power.tex`
- `results/exp3/tab_exp_hd_selection.tex`
- `results/exp4/tab_exp4_local_power.tex`
- `results/exp5/tab_runtime_total_drkte.tex`
- `results/exp5/tab_runtime_core_drkte.tex`

You can still override `--outdir` if you want to send a run somewhere else.

## Dependencies

The code uses a small scientific Python stack:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `scikit-learn`

The OCTMNIST scripts download the dataset on demand and cache it under
`octmnist_cache/`.

## Main commands

### Contour figure

```bash
python contour_estimated_oracle.py
```

### Experiment 1

```bash
python exp1_calibration_dr_me_plugin.py \
  --n-reps 200 \
  --n-grid 300 600 1200 3000 \
  --J 2 \
  --M 80
```

### Experiment 2

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

### Experiment 3

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python exp3_hd_whitening_ablation.py \
  --mode dim_sweep \
  --n-reps 200 \
  --n-fixed 3000 \
  --d-grid 5 10 25 50 \
  --settings null two_bump \
  --J 3 \
  --M 300 \
  --effect 4.0 \
  --bump-prob 0.04 \
  --bandwidth-scale 0.6 \
  --n-jobs 8
```

### Experiment 4

```bash
python exp4_local_chisq.py \
  --n-reps 2000 \
  --n-values 500 1000 3000 \
  --h-values 0 1 2 3 4 6 8 \
  --pilot-n 100000 \
  --pilot-M 100 \
  --J 2
```

### Experiment 5

```bash
python exp5_runtime_drkte_compare.py \
  --reps 5 \
  --n-grid 300 600 1200 3000 5000 10000 \
  --fixed-J 2 \
  --fixed-M 80 \
  --grad-steps 3
```

### OCTMNIST figures

```bash
python octmnist_location_learning.py
```


