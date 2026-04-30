#!/usr/bin/env python3
"""
Experiment 3b: high-dimensional outcome whitening/location-learning ablation.

This is a high-dimensional extension of Experiment 3. It uses the same ODR-ME
split structure and the same final Hotelling/chi-square test for all variants.
Only the location-learning objective differs.

The point is to stress random finite locations in R^d_Y. Candidate locations are
observed outcomes from the training split. Under rare bump alternatives, only a
small fraction of candidates are near the informative treatment-induced bump;
full covariance whitening should use the training DR witness to find them,
whereas random locations should increasingly miss them as d_Y grows.

Recommended fixed-d_Y power run:
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  python exp3_hd_whitening_ablation.py \
    --mode power --n-reps 200 --n-grid 300 600 1200 3000 \
    --settings null local_bump two_bump --d-y 25 --J 5 --M 250 \
    --n-jobs -1 --outdir exp3_hd_power_d25_results

Recommended dimension-sweep run:
  python exp3_hd_whitening_ablation.py \
    --mode dim_sweep --n-reps 200 --n-fixed 3000 \
    --d-grid 2 5 10 25 50 --settings two_bump --J 5 --M 250 \
    --n-jobs -1 --outdir exp3_hd_dim_sweep_results
"""

import argparse
import math
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from repo_paths import results_dir

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import exp1_calibration_dr_me_plugin as exp1


METHOD_LABELS = {
    "full_whitened": "Full whitening (ODR-ME)",
    "raw": "Raw witness",
    "random": "Random locations",
}
PLOT_ORDER = list(METHOD_LABELS.values())

SETTING_LABELS = {
    "causal_null": "Null",
    "null": "Null",
    "local_bump": "Localized bump",
    "two_bump": "Two localized bumps",
}


# -----------------------------------------------------------------------------
# High-dimensional confounded DGP
# -----------------------------------------------------------------------------


def _fixed_loading_matrix(p: int, d_y: int):
    """Deterministic small loading matrix, independent of simulation seed."""
    j = np.arange(1, d_y + 1, dtype=float)
    W = np.zeros((p, d_y), dtype=float)
    W[0, :] = 0.22 * np.sin(0.37 * j)
    W[1, :] = 0.20 * np.cos(0.29 * j)
    W[2, :] = 0.16 * np.sin(0.17 * j + 0.3)
    W[3, :] = 0.12 * np.cos(0.41 * j - 0.1)
    W[4, :] = 0.10 * np.sin(0.23 * j + 0.7)
    return W[:p]


def outcome_mean(X: np.ndarray, d_y: int):
    """Moderate-dimensional prognostic signal shared by Y(0),Y(1)."""
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    W = _fixed_loading_matrix(p, d_y)
    base = X @ W
    # Add low-amplitude nonlinear prognostic variation. This creates real
    # confounding under the observed-data null, but remains learnable by the
    # rich nuisance basis used in the earlier experiments.
    j = np.arange(1, d_y + 1, dtype=float)
    nonlinear = (
        0.10 * np.sin(X[:, [0]] + 0.07 * j[None, :])
        + 0.08 * (X[:, [2]] ** 2 - 1.0) * np.cos(0.11 * j[None, :])
        + 0.06 * (X[:, [1]] * X[:, [3]]) * np.sin(0.13 * j[None, :])
    )
    return base + nonlinear


def simulate_hd_observational(
    n: int,
    seed: int,
    setting: str = "null",
    d_y: int = 25,
    sigma_y: float = 0.70,
    effect: float = 5.0,
    bump_prob: float = 0.06,
):
    """Simulate high-dimensional observational data.

    The interventional null has Y(0)=Y(1) in distribution. Under bump
    alternatives, Y(1) contains rare treatment-induced high-dimensional tail
    clusters. The observed training dictionary stores which locations came from
    observed bump units; this is only used for diagnostics, not by the methods.
    """
    rng = np.random.default_rng(seed)
    p = 5
    X = rng.normal(size=(n, p))
    logits = 0.90 * X[:, 0] - 0.75 * X[:, 1] + 0.55 * X[:, 2] - 0.40 * X[:, 3]
    pi0 = np.clip(expit(logits), 0.06, 0.94)
    A = rng.binomial(1, pi0).astype(int)

    mu = outcome_mean(X, d_y)
    eps0 = sigma_y * rng.normal(size=(n, d_y))
    eps1 = sigma_y * rng.normal(size=(n, d_y))

    # Sparse directions so that informative regions occupy a small part of R^d.
    v1 = np.zeros(d_y)
    v2 = np.zeros(d_y)
    block = min(4, d_y)
    v1[:block] = 1.0 / math.sqrt(block)
    if d_y >= 8:
        v2[4:8] = 1.0 / 2.0
    else:
        v2[-block:] = -1.0 / math.sqrt(block)

    # Bump labels: positive labels are treatment-arm bumps, negative labels are
    # control-arm bumps. The alternatives below are local swaps: both arms have
    # the same background mass, but their rare bump mass is placed in different
    # regions. This avoids making every background location informative.
    bump_type0 = np.zeros(n, dtype=int)
    bump_type1 = np.zeros(n, dtype=int)
    shift0 = np.zeros((n, d_y), dtype=float)
    shift1 = np.zeros((n, d_y), dtype=float)

    if setting in {"null", "causal_null"}:
        pass
    elif setting == "local_bump":
        b0 = rng.binomial(1, bump_prob, size=n).astype(bool)
        b1 = rng.binomial(1, bump_prob, size=n).astype(bool)
        bump_type0[b0] = -1
        bump_type1[b1] = 1
        shift0[b0] = -effect * v1
        shift1[b1] = effect * v1
    elif setting == "two_bump":
        u0 = rng.uniform(size=n)
        c01 = u0 < bump_prob
        c02 = (u0 >= bump_prob) & (u0 < 2.0 * bump_prob)
        bump_type0[c01] = -1
        bump_type0[c02] = -2
        shift0[c01] = -effect * v1
        shift0[c02] = -effect * v2

        u1 = rng.uniform(size=n)
        c11 = u1 < bump_prob
        c12 = (u1 >= bump_prob) & (u1 < 2.0 * bump_prob)
        bump_type1[c11] = 1
        bump_type1[c12] = 2
        shift1[c11] = effect * v1
        shift1[c12] = effect * v2
    else:
        raise ValueError(f"unknown setting: {setting}")

    Y0 = mu + eps0 + shift0
    Y1 = mu + eps1 + shift1
    Y = np.where(A[:, None] == 1, Y1, Y0)

    # Observed bump label for dictionary diagnostics. This label is not used by
    # any method. It only indicates whether a selected observed outcome came
    # from one of the sparse interventional bump regions.
    observed_bump_type = np.where(A == 1, bump_type1, bump_type0).astype(int)

    meta = {
        "d_y": int(d_y),
        "sigma_y": float(sigma_y),
        "effect": float(effect),
        "bump_prob": float(bump_prob),
        "true_pi_mean": float(pi0.mean()),
        "obs_bump_frac": float(np.mean(np.abs(observed_bump_type) > 0)),
        "obs_bump1_frac": float(np.mean(np.abs(observed_bump_type) == 1)),
        "obs_bump2_frac": float(np.mean(np.abs(observed_bump_type) == 2)),
        "observed_bump_type": observed_bump_type,
    }
    return X, A, Y, meta


# -----------------------------------------------------------------------------
# Dictionary and location-selection objectives
# -----------------------------------------------------------------------------


def make_observed_dictionary(
    Y_tr: np.ndarray,
    source_indices: np.ndarray,
    M: int,
    rng: np.random.Generator,
):
    """Use observed training outcomes as candidate locations."""
    n_tr = len(Y_tr)
    M_eff = min(int(M), n_tr)
    cols = rng.choice(n_tr, size=M_eff, replace=False)
    return np.asarray(Y_tr[cols], dtype=float), np.asarray(source_indices[cols], dtype=int)


def _mean_var(Z: np.ndarray):
    Z = np.asarray(Z, dtype=float)
    mu = Z.mean(axis=0)
    Zc = Z - mu
    var = np.sum(Zc * Zc, axis=0) / max(Z.shape[0] - 1, 1)
    return mu, var


def fast_full_greedy_select(Z_train: np.ndarray, J: int, tau: float):
    """Greedy maximization of mu^T(S+tau I)^(-1)mu via Schur gains."""
    Z = np.asarray(Z_train, dtype=float)
    n, M = Z.shape
    J = min(J, M)
    mu = Z.mean(axis=0)
    Zc = Z - mu
    S = (Zc.T @ Zc) / max(n - 1, 1)

    selected: List[int] = []
    available = np.ones(M, dtype=bool)

    scores = (mu * mu) / (np.diag(S) + tau)
    first = int(np.argmax(scores))
    selected.append(first)
    available[first] = False

    while len(selected) < J:
        cand = np.flatnonzero(available)
        A = S[np.ix_(selected, selected)] + tau * np.eye(len(selected))
        muA = mu[selected]
        try:
            Binv_mu = np.linalg.solve(A, muA)
            obj_A = float(muA @ Binv_mu)
            SAc = S[np.ix_(selected, cand)]
            Binv_SAc = np.linalg.solve(A, SAc)
        except np.linalg.LinAlgError:
            Ainv = np.linalg.pinv(A)
            Binv_mu = Ainv @ muA
            obj_A = float(muA @ Binv_mu)
            SAc = S[np.ix_(selected, cand)]
            Binv_SAc = Ainv @ SAc
        d = np.diag(S)[cand] + tau
        schur = np.maximum(d - np.sum(SAc * Binv_SAc, axis=0), 1e-12)
        residual_mean = mu[cand] - SAc.T @ Binv_mu
        obj = obj_A + (residual_mean * residual_mean) / schur
        best = int(cand[int(np.argmax(obj))])
        selected.append(best)
        available[best] = False
    return selected


def select_locations(Z_train: np.ndarray, J: int, tau: float, method: str, rng: np.random.Generator):
    M = Z_train.shape[1]
    J = min(int(J), M)
    if method == "full_whitened":
        return fast_full_greedy_select(Z_train, J, tau)
    mu, var = _mean_var(Z_train)
    if method == "raw":
        scores = mu * mu
        return [int(i) for i in np.argsort(scores)[-J:][::-1]]
    if method == "random":
        return [int(i) for i in rng.choice(M, size=J, replace=False)]
    raise ValueError(f"unknown method: {method}")


def selection_diagnostics(
    Z_train: np.ndarray,
    C: np.ndarray,
    cols: Sequence[int],
    dict_bump_type: np.ndarray,
):
    cols = list(cols)
    labels = np.asarray(dict_bump_type)[cols]
    out: Dict[str, float] = {
        "selected_bump_frac": float(np.mean(np.abs(labels) > 0)) if len(labels) else 0.0,
        "selected_bump_count": float(np.sum(np.abs(labels) > 0)),
        "selected_any_bump": float(np.any(np.abs(labels) > 0)),
        "selected_distinct_bump_types": float(len(set(int(x) for x in labels if int(x) != 0))),
        "selected_bump1_count": float(np.sum(np.abs(labels) == 1)),
        "selected_bump2_count": float(np.sum(np.abs(labels) == 2)),
    }
    if len(cols) <= 1:
        out.update({
            "avg_abs_corr_train": 0.0,
            "max_abs_corr_train": 0.0,
            "avg_pairwise_loc_dist": 0.0,
            "min_pairwise_loc_dist": 0.0,
        })
        return out

    Zs = np.asarray(Z_train[:, cols], dtype=float)
    Zs = Zs - Zs.mean(axis=0, keepdims=True)
    sd = Zs.std(axis=0, ddof=1)
    valid = sd > 1e-12
    if valid.sum() >= 2:
        R = np.corrcoef(Zs[:, valid], rowvar=False)
        R = np.atleast_2d(R)
        vals = np.abs(R[np.triu_indices(R.shape[0], k=1)])
        out["avg_abs_corr_train"] = float(vals.mean()) if len(vals) else 0.0
        out["max_abs_corr_train"] = float(vals.max()) if len(vals) else 0.0
    else:
        out["avg_abs_corr_train"] = 0.0
        out["max_abs_corr_train"] = 0.0

    locs = np.asarray(C[cols], dtype=float)
    D = np.sqrt(exp1.squared_distances(locs, locs))
    dvals = D[np.triu_indices(len(cols), k=1)]
    out["avg_pairwise_loc_dist"] = float(dvals.mean()) if len(dvals) else 0.0
    out["min_pairwise_loc_dist"] = float(dvals.min()) if len(dvals) else 0.0
    return out


# -----------------------------------------------------------------------------
# One run and summaries
# -----------------------------------------------------------------------------


def run_one(task):
    (
        n, seed, setting, d_y, J, M, sigma_y, effect, bump_prob,
        bandwidth_scale, tau, ridge, ridge_alpha, pi_l2, basis_mode, pvalue,
    ) = task
    rng = np.random.default_rng(seed)
    X, A, Y, meta = simulate_hd_observational(
        n=n,
        seed=seed,
        setting=setting,
        d_y=d_y,
        sigma_y=sigma_y,
        effect=effect,
        bump_prob=bump_prob,
    )
    obs_bump_type = meta.pop("observed_bump_type")

    perm = rng.permutation(n)
    n_eta = n // 3
    n_tr = n // 3
    idx_eta = perm[:n_eta]
    idx_tr = perm[n_eta:n_eta + n_tr]
    idx_te = perm[n_eta + n_tr:]

    C, C_source = make_observed_dictionary(Y[idx_tr], idx_tr, M, rng)
    dict_bump_type = obs_bump_type[C_source]
    ell_y = bandwidth_scale * exp1.median_bandwidth_y(Y[np.concatenate([idx_eta, idx_tr])], rng)
    ell_y = max(float(ell_y), 1e-3)

    nuis = exp1.PluginNuisance(
        ell_y=ell_y,
        ridge_alpha=ridge_alpha,
        pi_l2=pi_l2,
        basis_mode=basis_mode,
    ).fit(X[idx_eta], A[idx_eta], Y[idx_eta], C)

    Ztr_dr = exp1.pseudo_feature_matrix("dr", X[idx_tr], A[idx_tr], Y[idx_tr], C, ell_y, nuis)
    Zte_dr = exp1.pseudo_feature_matrix("dr", X[idx_te], A[idx_te], Y[idx_te], C, ell_y, nuis)

    out: Dict[str, object] = {
        "n": int(n),
        "seed": int(seed),
        "setting": "causal_null" if setting in {"null", "causal_null"} else setting,
        "d_y": int(d_y),
        "J": int(min(J, C.shape[0])),
        "M_requested": int(M),
        "M_eff": int(C.shape[0]),
        "n_eta": int(len(idx_eta)),
        "n_tr": int(len(idx_tr)),
        "n_te": int(len(idx_te)),
        "ell_y": float(ell_y),
        "dict_bump_frac": float(np.mean(np.abs(dict_bump_type) > 0)),
        "dict_bump1_frac": float(np.mean(np.abs(dict_bump_type) == 1)),
        "dict_bump2_frac": float(np.mean(np.abs(dict_bump_type) == 2)),
        "arm0_eta": int(nuis.arm_counts[0]),
        "arm1_eta": int(nuis.arm_counts[1]),
        **meta,
    }

    for method in METHOD_LABELS:
        cols = select_locations(Ztr_dr, J, tau, method, rng)
        stat, pval = exp1.hotelling_pvalue(Zte_dr, cols, ridge=ridge, pvalue=pvalue)
        out[f"p_{method}"] = float(pval)
        out[f"stat_{method}"] = float(stat)
        out[f"cols_{method}"] = ";".join(str(int(c)) for c in cols)
        diag = selection_diagnostics(Ztr_dr, C, cols, dict_bump_type)
        for k, v in diag.items():
            out[f"{k}_{method}"] = v
    return out


def summarize(df: pd.DataFrame, alpha: float):
    rows = []
    group_cols = ["setting", "d_y", "n"]
    for key, dfg in df.groupby(group_cols):
        setting, d_y, n = key
        for method, label in METHOD_LABELS.items():
            pvals = pd.to_numeric(dfg[f"p_{method}"], errors="coerce").dropna().to_numpy()
            rate = float(np.mean(pvals <= alpha)) if len(pvals) else np.nan
            se = math.sqrt(max(rate * (1.0 - rate), 1e-12) / max(len(pvals), 1)) if len(pvals) else np.nan
            row: Dict[str, object] = {
                "setting": setting,
                "d_y": int(d_y),
                "n": int(n),
                "method": label,
                "method_key": method,
                "n_reps": int(len(pvals)),
                "alpha": float(alpha),
                "rejection_rate": rate,
                "mc_se_rate": se,
                "ci95_low": rate - 1.96 * se,
                "ci95_high": rate + 1.96 * se,
                "mean_p": float(np.mean(pvals)) if len(pvals) else np.nan,
                "median_p": float(np.median(pvals)) if len(pvals) else np.nan,
            }
            for metric in [
                "avg_abs_corr_train", "max_abs_corr_train",
                "avg_pairwise_loc_dist", "min_pairwise_loc_dist",
                "selected_bump_frac", "selected_bump_count", "selected_any_bump",
                "selected_distinct_bump_types", "selected_bump1_count", "selected_bump2_count",
            ]:
                col = f"{metric}_{method}"
                if col in dfg.columns:
                    row[metric] = float(pd.to_numeric(dfg[col], errors="coerce").mean())
            row["dict_bump_frac"] = float(pd.to_numeric(dfg["dict_bump_frac"], errors="coerce").mean())
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_power(summary: pd.DataFrame, outpath: str, alpha: float):
    import matplotlib.pyplot as plt

    settings = [s for s in ["causal_null", "local_bump", "two_bump"] if s in set(summary["setting"])]
    dvals = sorted(summary["d_y"].unique())
    d_y = dvals[0] if len(dvals) == 1 else None
    ncols = len(settings)
    fig, axes = plt.subplots(1, ncols, figsize=(4.9 * ncols, 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, setting in zip(axes, settings):
        dfs = summary[summary["setting"] == setting]
        if d_y is not None:
            dfs = dfs[dfs["d_y"] == d_y]
        for method in PLOT_ORDER:
            dfm = dfs[dfs["method"] == method].sort_values("n")
            if dfm.empty:
                continue
            ax.errorbar(
                dfm["n"],
                dfm["rejection_rate"],
                yerr=1.96 * dfm["mc_se_rate"],
                marker="o",
                linewidth=1.5,
                capsize=2.5,
                label=method,
            )
        ax.axhline(alpha, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(SETTING_LABELS.get(setting, setting))
        ax.set_xlabel("sample size n")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(f"rejection rate at alpha={alpha}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center")
    title_d = f"d_y={d_y}" if d_y is not None else "multiple outcome dimensions"
    fig.suptitle(f"High-dimensional outcome whitening ablation ({title_d})", y=0.995)
    fig.tight_layout(rect=[0, 0.16, 1, 0.92])
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_bump_selection(summary: pd.DataFrame, outpath: str, setting: str = "two_bump"):
    import matplotlib.pyplot as plt

    if setting not in set(summary["setting"]):
        setting = "local_bump" if "local_bump" in set(summary["setting"]) else sorted(set(summary["setting"]))[-1]
    d = summary[summary["setting"] == setting].copy()
    if d.empty:
        return
    # If this is a power run, plot versus n at the first d_y. If a dimension sweep,
    # plot versus d_y at the fixed n.
    if d["d_y"].nunique() == 1:
        xvar = "n"
        xlabel = "sample size n"
        d = d.sort_values("n")
    else:
        xvar = "d_y"
        xlabel = "outcome dimension d_Y"
        max_n = int(d["n"].max())
        d = d[d["n"] == max_n].sort_values("d_y")

    metrics = [
        ("selected_any_bump", "Pr(select at least one bump location)"),
        ("selected_distinct_bump_types", "distinct bump types selected"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for method in PLOT_ORDER:
            dfm = d[d["method"] == method].sort_values(xvar)
            if dfm.empty or metric not in dfm.columns:
                continue
            ax.plot(dfm[xvar], dfm[metric], marker="o", linewidth=1.5, label=method)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xvar == "n":
            ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_title(SETTING_LABELS.get(setting, setting))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center")
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_dimension_sweep(summary: pd.DataFrame, outpath: str, alpha: float, setting: str = "two_bump"):
    import matplotlib.pyplot as plt

    d = summary[summary["setting"] == setting].copy()
    if d.empty:
        return
    max_n = int(d["n"].max())
    d = d[d["n"] == max_n].sort_values("d_y")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for method in PLOT_ORDER:
        dfm = d[d["method"] == method].sort_values("d_y")
        if dfm.empty:
            continue
        axes[0].errorbar(
            dfm["d_y"], dfm["rejection_rate"],
            yerr=1.96 * dfm["mc_se_rate"], marker="o", linewidth=1.5, capsize=2.5, label=method,
        )
        axes[1].plot(dfm["d_y"], dfm["selected_any_bump"], marker="o", linewidth=1.5, label=method)
    axes[0].axhline(alpha, linestyle="--", linewidth=1)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("outcome dimension d_Y")
    axes[0].set_ylabel(f"power at alpha={alpha}")
    axes[0].set_title(f"{SETTING_LABELS.get(setting, setting)}, n={max_n}")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("outcome dimension d_Y")
    axes[1].set_ylabel("Pr(select at least one bump location)")
    axes[1].set_title("Selection diagnostic")
    axes[1].grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center")
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Experiment driver
# -----------------------------------------------------------------------------


def build_tasks(args: argparse.Namespace):
    tasks = []
    setting_code = {"null": 0, "causal_null": 0, "local_bump": 1, "two_bump": 2}
    if args.mode == "power":
        for setting in args.settings:
            for n in args.n_grid:
                for r in range(args.n_reps):
                    seed = args.seed_offset + setting_code[setting] * 10_000_000 + int(args.d_y) * 1_000_000 + int(n) * 1000 + r
                    tasks.append((
                        int(n), int(seed), setting, int(args.d_y), int(args.J), int(args.M),
                        float(args.sigma_y), float(args.effect), float(args.bump_prob),
                        float(args.bandwidth_scale), float(args.tau), float(args.ridge),
                        float(args.ridge_alpha), float(args.pi_l2), args.basis_mode, args.pvalue,
                    ))
    elif args.mode == "dim_sweep":
        for setting in args.settings:
            for d_y in args.d_grid:
                for r in range(args.n_reps):
                    seed = args.seed_offset + 300_000_000 + setting_code[setting] * 10_000_000 + int(d_y) * 1_000_000 + int(args.n_fixed) * 1000 + r
                    tasks.append((
                        int(args.n_fixed), int(seed), setting, int(d_y), int(args.J), int(args.M),
                        float(args.sigma_y), float(args.effect), float(args.bump_prob),
                        float(args.bandwidth_scale), float(args.tau), float(args.ridge),
                        float(args.ridge_alpha), float(args.pi_l2), args.basis_mode, args.pvalue,
                    ))
    else:
        raise ValueError(args.mode)
    return tasks


def run_experiment(args: argparse.Namespace):
    os.makedirs(args.outdir, exist_ok=True)
    tasks = build_tasks(args)
    t0 = time.time()
    records = []
    if args.n_jobs == 1:
        for k, task in enumerate(tasks, start=1):
            records.append(run_one(task))
            if k % 25 == 0 or k == len(tasks):
                print(f"completed {k}/{len(tasks)} runs", flush=True)
    else:
        jobs = cpu_count() if args.n_jobs == -1 else max(1, int(args.n_jobs))
        with Pool(processes=jobs) as pool:
            for k, rec in enumerate(pool.imap_unordered(run_one, tasks), start=1):
                records.append(rec)
                if k % 25 == 0 or k == len(tasks):
                    print(f"completed {k}/{len(tasks)} runs", flush=True)

    df = pd.DataFrame(records)
    raw_path = os.path.join(args.outdir, "exp3_hd_raw_pvalues.csv")
    df.to_csv(raw_path, index=False)
    summary = summarize(df, args.alpha)
    summary_path = os.path.join(args.outdir, "exp3_hd_summary.csv")
    summary.to_csv(summary_path, index=False)

    if not getattr(args, "no_plots", False):
        if args.mode == "power":
            plot_power(summary, os.path.join(args.outdir, "exp3_hd_power_curves.png"), args.alpha)
            plot_bump_selection(summary, os.path.join(args.outdir, "exp3_hd_bump_selection.png"), setting="two_bump")
        else:
            plot_dimension_sweep(summary, os.path.join(args.outdir, "exp3_hd_dimension_sweep.png"), args.alpha, setting=args.settings[0])
            plot_bump_selection(summary, os.path.join(args.outdir, "exp3_hd_dim_bump_selection.png"), setting=args.settings[0])

    elapsed = time.time() - t0
    print(f"Wrote: {raw_path}")
    print(f"Wrote: {summary_path}")
    print(f"Elapsed: {elapsed:.1f} seconds")
    return df, summary


def make_zip(outdir: str, script_path: str):
    zip_path = os.path.join(os.path.dirname(outdir), os.path.basename(outdir.rstrip(os.sep)) + "_package.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in [script_path, os.path.join(THIS_DIR, "exp1_calibration_dr_me_plugin.py")]:
            if os.path.exists(p):
                zf.write(p, arcname=os.path.basename(p))
        for root, _, files in os.walk(outdir):
            for f in files:
                path = os.path.join(root, f)
                zf.write(path, arcname=os.path.join(os.path.basename(outdir), os.path.relpath(path, outdir)))
    return zip_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["power", "dim_sweep"], default="power")
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--n-grid", type=int, nargs="+", default=[300, 600, 1200, 3000])
    parser.add_argument("--n-fixed", type=int, default=3000)
    parser.add_argument("--d-y", type=int, default=25)
    parser.add_argument("--d-grid", type=int, nargs="+", default=[2, 5, 10, 25, 50])
    parser.add_argument("--settings", type=str, nargs="+", default=["null", "local_bump", "two_bump"], choices=["null", "local_bump", "two_bump"])
    parser.add_argument("--J", type=int, default=5)
    parser.add_argument("--M", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sigma-y", type=float, default=0.70)
    parser.add_argument("--effect", type=float, default=5.0)
    parser.add_argument("--bump-prob", type=float, default=0.06)
    parser.add_argument("--bandwidth-scale", type=float, default=0.70)
    parser.add_argument("--tau", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--pi-l2", type=float, default=1e-2)
    parser.add_argument("--basis-mode", choices=["rich", "linear"], default="rich")
    parser.add_argument("--pvalue", choices=["chi2", "hotelling_f"], default="chi2")
    parser.add_argument("--seed-offset", type=int, default=3530000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--outdir", type=str, default=str(results_dir("exp3")))
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    _, summary = run_experiment(args)
    shown_cols = [
        "setting", "d_y", "n", "method", "n_reps", "rejection_rate", "ci95_low", "ci95_high",
        "selected_any_bump", "selected_distinct_bump_types", "selected_bump_frac",
        "avg_abs_corr_train", "min_pairwise_loc_dist",
    ]
    shown_cols = [c for c in shown_cols if c in summary.columns]
    shown = summary[shown_cols].sort_values(["setting", "d_y", "n", "method"])
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.width", 260)
    print("\nSummary:")
    print(shown.to_string(index=False))
    if args.zip:
        zpath = make_zip(args.outdir, os.path.abspath(__file__))
        print(f"Wrote package: {zpath}")


if __name__ == "__main__":
    main()
