#!/usr/bin/env python3
"""
Experiment 2: power extension of Experiment 1, plus DR-xKTE.

This is deliberately the same design as Experiment 1:
  - same confounded observational DGP under the null;
  - same three-way split I_eta / I_tr / I_te;
  - same plug-in nuisances for ODR-ME, IPW-ME, and DM-ME;
  - same finite-location baselines.

The only change is that we add alternatives to the potential-outcome law and
report empirical rejection rates as power curves.  We also add DR-xKTE/AIPW-xKTE
as the global causal kernel baseline, using the same estimated propensity from
I_eta.  Fawkes is omitted because its permutation null is too slow for this
laptop-scale experiment.

The script expects exp1_calibration_dr_me_plugin.py to be in the same directory
or on PYTHONPATH.  This keeps Experiment 2 exactly tied to the Experiment 1 code.

Recommended run:
    python exp2_power_exp1_extension.py \
      --n-reps 200 \
      --n-grid 300 600 1200 3000 \
      --settings null mean_shift variance_shift localized_bump \
      --n-jobs -1 \
      --outdir exp2_power_exp1_extension_results

Fast check:
    python exp2_power_exp1_extension.py --n-reps 10 --n-grid 300 600 --n-jobs 1
"""

import argparse
import math
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.linalg as la
from scipy.special import expit
from sklearn.metrics import pairwise_distances, pairwise_kernels
from repo_paths import results_dir

# Import the exact Experiment-1 implementation.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import exp1_calibration_dr_me_plugin as exp1  # noqa: E402


# -----------------------------------------------------------------------------
# DGP: Experiment 1 null plus causal alternatives.
# -----------------------------------------------------------------------------

DEFAULT_EFFECTS = {
    "null": 0.0,
    "mean_shift": 0.35,
    "variance_shift": 0.45,
    "localized_bump": 1.00,
}

OUTPUT_SETTING = {
    "null": "causal_null",
    "mean_shift": "mean_shift",
    "variance_shift": "variance_shift",
    "localized_bump": "localized_bump",
}

PLOT_SETTINGS = ["causal_null", "mean_shift", "variance_shift", "localized_bump"]

SETTING_LABELS = {
    "causal_null": "Null",
    "null": "Null",
    "mean_shift": "Mean shift",
    "variance_shift": "Variance shift",
    "localized_bump": "Localized bump",
}


def simulate_observational(
    n: int,
    seed: int,
    setting: str,
    p: int = 5,
    sigma_y: float = 1.0,
    effect: Optional[float] = None,
    bump_prob: float = 0.12,
):
    """Generate Experiment-1 observational data under null or alternative.

    The null has equal interventional laws P_{Y(0)}=P_{Y(1)}, but observed
    treated/control outcomes differ because treatment is confounded by X.
    """
    if setting not in DEFAULT_EFFECTS:
        raise ValueError(f"Unknown setting: {setting}")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    pi = exp1.true_propensity(X)
    A = rng.binomial(1, pi).astype(np.int64)
    g = exp1.prognostic_g(X)
    delta = DEFAULT_EFFECTS[setting] if effect is None else float(effect)

    eps0 = rng.normal(scale=sigma_y, size=n)
    eps1 = rng.normal(scale=sigma_y, size=n)
    Y0 = g + eps0

    if setting == "null":
        Y1 = g + eps1
    elif setting == "mean_shift":
        Y1 = g + eps1 + delta
    elif setting == "variance_shift":
        Y1 = g + (1.0 + delta) * eps1
    elif setting == "localized_bump":
        # Mean-zero rare bump: a distributional/tail alternative, not a pure
        # average-effect benchmark.
        q = float(bump_prob)
        B = rng.binomial(1, q, size=n).astype(float)
        bump = (B - q) / math.sqrt(max(q * (1.0 - q), 1e-8))
        Y1 = g + eps1 + delta * bump
    else:
        raise AssertionError("unreachable")

    Y = np.where(A == 1, Y1, Y0).reshape(-1, 1)
    meta = {
        "effect": float(delta),
        "bump_prob": float(bump_prob),
        "mean_true_pi": float(pi.mean()),
    }
    return X, A, Y, meta


# -----------------------------------------------------------------------------
# DR-xKTE / AIPW-xKTE: user-provided statistic with numerical guards only.
# -----------------------------------------------------------------------------

def _as_2d(A: np.ndarray):
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    return A


def _safe_solve(A: np.ndarray, B: np.ndarray, jitter: float = 1e-8):
    # Fast dense solve for the KRR systems in DR-xKTE. We use the general
    # LAPACK solver instead of a pseudo-inverse fallback, since rare numerical
    # non-PD issues should not make a Monte Carlo repetition hang.
    try:
        return la.solve(A, B, assume_a="gen", check_finite=False)
    except Exception:
        Aj = A + jitter * np.eye(A.shape[0])
        try:
            return la.solve(Aj, B, assume_a="gen", check_finite=False)
        except Exception:
            return la.lstsq(Aj, B, check_finite=False)[0]


def xMMD2dr(XY, w, Xcov, T, kernel_function, **kwargs):
    """DR-xKTE^2 statistic from the supplied AIPW-xKTE code.

    Changes relative to the snippet: input coercion, propensity clipping, and
    degenerate-half safeguards.  The statistic and p-value convention are kept.
    """
    XY = _as_2d(XY)
    Xcov = _as_2d(Xcov)
    T = np.asarray(T).astype(int)
    w = np.asarray(w, dtype=float).squeeze()

    N = len(XY)
    if N % 2 == 1:
        XY, Xcov, T, w = XY[:-1], Xcov[:-1], T[:-1], w[:-1]
        N = len(XY)
    if N < 8:
        return 0.0
    N2 = N // 2

    Ta = T[:N2]
    Tb = T[N2:]
    Ya = XY[:N2]
    Yb = XY[N2:]
    Y0a = Ya[Ta == 0]
    Y1a = Ya[Ta == 1]
    Y0b = Yb[Tb == 0]
    Y1b = Yb[Tb == 1]
    if min(len(Y0a), len(Y1a), len(Y0b), len(Y1b)) < 2:
        return 0.0

    Y = np.vstack((Y0a, Y1a, Y0b, Y1b))
    m1 = len(Y0a)
    m = m1 + len(Y0b)
    n1 = len(Y1a)
    n = n1 + len(Y1b)

    Xa = Xcov[:N2, :]
    Xb = Xcov[N2:, :]
    X0a = Xa[Ta == 0, :]
    X0b = Xb[Tb == 0, :]
    X1a = Xa[Ta == 1, :]
    X1b = Xb[Tb == 1, :]
    X = np.vstack((X0a, X1a, X0b, X1b))

    w = np.clip(w.squeeze(), 0.03, 0.97)
    wa = w[:N2]
    wb = w[N2:]
    w0a = 1.0 / np.asarray(1 - wa[Ta == 0])
    w0b = 1.0 / np.asarray(1 - wb[Tb == 0])
    w1a = 1.0 / np.asarray(wa[Ta == 1])
    w1b = 1.0 / np.asarray(wb[Tb == 1])
    ww = np.concatenate((-w0a, w1a, -w0b, w1b))

    sigmaKX = np.median(pairwise_distances(X[N2:, :], X[N2:, :], metric="euclidean")) ** 2
    sigmaKX = float(max(sigmaKX, 1e-8))
    KX = pairwise_kernels(X, metric="rbf", gamma=1.0 / sigmaKX)
    gamma = sigmaKX

    mu0a = _safe_solve(KX[:m1, :m1] + gamma * np.eye(m1), KX[:m1, :m1 + n1])
    zeroed_mu0a = np.vstack((mu0a, np.zeros((n1, m1 + n1))))
    mu1a = _safe_solve(KX[m1:m1 + n1, m1:m1 + n1] + gamma * np.eye(n1), KX[m1:m1 + n1, :m1 + n1])
    zeroed_mu1a = np.vstack((np.zeros((m1, m1 + n1)), mu1a))
    muAa = np.hstack((zeroed_mu0a[:, :m1], zeroed_mu1a[:, m1:m1 + n1]))

    mu0b = _safe_solve(KX[m1 + n1:m + n1, m1 + n1:m + n1] + gamma * np.eye(m - m1), KX[m1 + n1:m + n1, m1 + n1:])
    zeroed_mu0b = np.vstack((mu0b, np.zeros((n - n1, m + n - m1 - n1))))
    mu1b = _safe_solve(KX[m + n1:, m + n1:] + gamma * np.eye(n - n1), KX[m + n1:, m1 + n1:])
    zeroed_mu1b = np.vstack((np.zeros((m - m1, m + n - m1 - n1)), mu1b))
    muAb = np.hstack((zeroed_mu0b[:, :m - m1], zeroed_mu1b[:, m - m1:]))

    left_side = zeroed_mu1a - zeroed_mu0a + ww[:N2].reshape(-1, 1) * (np.eye(m1 + n1) - muAa)
    right_side = zeroed_mu1b - zeroed_mu0b + ww[N2:].reshape(-1, 1) * (np.eye(m + n - m1 - n1) - muAb)

    KY = pairwise_kernels(Y[:N2], Y[N2:], metric=kernel_function, **kwargs)
    prod = left_side.T @ KY @ right_side
    U = prod.mean(1)
    sd = float(np.std(U))
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float(np.sqrt(len(U)) * U.mean() / sd)


def kernel_dr_two_sample_test_agnostic(Y, Xcov, T, w, kernel_function="rbf", **kwargs):
    stat = xMMD2dr(Y, w, Xcov, T, kernel_function, **kwargs)
    return float(stat), float(1.0 - st.norm.cdf(stat))


# -----------------------------------------------------------------------------
# One Monte Carlo repetition.
# -----------------------------------------------------------------------------

METHODS = {
    "p_dr_split": "ODR-ME",
    "p_dr_random": "ODR-ME-Rand",
    "p_ipw_plugin": "IPW-ME plug-in",
    "p_dm_plugin": "DM-ME plug-in",
    "p_naive_split": "Naive observed ME",
    "p_dr_nosplit": "ODR-ME-NoSplit",
    "p_dr_xkte": "DR-xKTE",
}

PLOT_ORDER_ALL = [
    "ODR-ME",
    "DR-xKTE",
    "ODR-ME-Rand",
    "IPW-ME plug-in",
    "DM-ME plug-in",
    "Naive observed ME",
    "ODR-ME-NoSplit",
]


def run_one(task):
    (
        n,
        seed,
        setting,
        J,
        M,
        sigma_y,
        effect,
        bump_prob,
        tau,
        ridge,
        ridge_alpha,
        pi_l2,
        basis_mode,
    ) = task

    rng = np.random.default_rng(seed)
    X, A, Y, meta = simulate_observational(
        n=n,
        seed=seed,
        setting=setting,
        sigma_y=sigma_y,
        effect=effect,
        bump_prob=bump_prob,
    )

    perm = rng.permutation(n)
    n_eta = n // 3
    n_tr = n // 3
    idx_eta = perm[:n_eta]
    idx_tr = perm[n_eta:n_eta + n_tr]
    idx_te = perm[n_eta + n_tr:]

    ell_y = exp1.median_bandwidth_y(Y[np.concatenate([idx_eta, idx_tr])], rng)
    C_tr = exp1.make_dictionary(Y[idx_tr], M)
    nuis = exp1.PluginNuisance(
        ell_y=ell_y,
        ridge_alpha=ridge_alpha,
        pi_l2=pi_l2,
        basis_mode=basis_mode,
    ).fit(X[idx_eta], A[idx_eta], Y[idx_eta], C_tr)

    p_global = float(A[idx_eta].mean())

    out: Dict[str, float] = {
        "n": int(n),
        "seed": int(seed),
        "setting": OUTPUT_SETTING.get(setting, setting),
        "J": int(J),
        "M": int(M),
        "n_eta": len(idx_eta),
        "n_tr": len(idx_tr),
        "n_te": len(idx_te),
        "ell_y": float(ell_y),
        "arm0_eta": nuis.arm_counts[0],
        "arm1_eta": nuis.arm_counts[1],
        **meta,
    }

    # Same Experiment-1 baselines.
    Ztr_dr = exp1.pseudo_feature_matrix("dr", X[idx_tr], A[idx_tr], Y[idx_tr], C_tr, ell_y, nuis)
    Zte_dr = exp1.pseudo_feature_matrix("dr", X[idx_te], A[idx_te], Y[idx_te], C_tr, ell_y, nuis)
    sel_dr = exp1.greedy_select(Ztr_dr, J, tau)
    stat, pval = exp1.hotelling_pvalue(Zte_dr, sel_dr, ridge=ridge, pvalue="chi2")
    out["p_dr_split"] = pval
    out["stat_dr_split"] = stat
    out["loc_dr_split"] = ";".join([f"{float(C_tr[c, 0]):.6g}" for c in sel_dr])

    sel_rand = list(rng.choice(M, size=J, replace=False))
    stat, pval = exp1.hotelling_pvalue(Zte_dr, sel_rand, ridge=ridge, pvalue="chi2")
    out["p_dr_random"] = pval
    out["stat_dr_random"] = stat

    Ztr_ipw = exp1.pseudo_feature_matrix("ipw", X[idx_tr], A[idx_tr], Y[idx_tr], C_tr, ell_y, nuis)
    Zte_ipw = exp1.pseudo_feature_matrix("ipw", X[idx_te], A[idx_te], Y[idx_te], C_tr, ell_y, nuis)
    sel_ipw = exp1.greedy_select(Ztr_ipw, J, tau)
    stat, pval = exp1.hotelling_pvalue(Zte_ipw, sel_ipw, ridge=ridge, pvalue="chi2")
    out["p_ipw_plugin"] = pval
    out["stat_ipw_plugin"] = stat

    Ztr_dm = exp1.pseudo_feature_matrix("dm", X[idx_tr], A[idx_tr], Y[idx_tr], C_tr, ell_y, nuis)
    Zte_dm = exp1.pseudo_feature_matrix("dm", X[idx_te], A[idx_te], Y[idx_te], C_tr, ell_y, nuis)
    sel_dm = exp1.greedy_select(Ztr_dm, J, tau)
    stat, pval = exp1.hotelling_pvalue(Zte_dm, sel_dm, ridge=ridge, pvalue="chi2")
    out["p_dm_plugin"] = pval
    out["stat_dm_plugin"] = stat

    Ztr_naive = exp1.pseudo_feature_matrix("naive", X[idx_tr], A[idx_tr], Y[idx_tr], C_tr, ell_y, nuis, p_global=p_global)
    Zte_naive = exp1.pseudo_feature_matrix("naive", X[idx_te], A[idx_te], Y[idx_te], C_tr, ell_y, nuis, p_global=p_global)
    sel_naive = exp1.greedy_select(Ztr_naive, J, tau)
    stat, pval = exp1.hotelling_pvalue(Zte_naive, sel_naive, ridge=ridge, pvalue="chi2")
    out["p_naive_split"] = pval
    out["stat_naive_split"] = stat

    # Invalid no-split ablation, exactly as in Experiment 1.
    C_te = exp1.make_dictionary(Y[idx_te], M)
    nuis_te = exp1.PluginNuisance(
        ell_y=ell_y,
        ridge_alpha=ridge_alpha,
        pi_l2=pi_l2,
        basis_mode=basis_mode,
    ).fit(X[idx_eta], A[idx_eta], Y[idx_eta], C_te)
    Zte_dr_nosplit = exp1.pseudo_feature_matrix("dr", X[idx_te], A[idx_te], Y[idx_te], C_te, ell_y, nuis_te)
    sel_nosplit = exp1.greedy_select(Zte_dr_nosplit, J, tau)
    stat, pval = exp1.hotelling_pvalue(Zte_dr_nosplit, sel_nosplit, ridge=ridge, pvalue="chi2")
    out["p_dr_nosplit"] = pval
    out["stat_dr_nosplit"] = stat

    # DR-xKTE on the same held-out I_te split, same estimated propensity.
    te_perm = rng.permutation(len(idx_te))
    idx_x = idx_te[te_perm]
    if len(idx_x) % 2 == 1:
        idx_x = idx_x[:-1]
    Yx = Y[idx_x]
    Xx = X[idx_x]
    Ax = A[idx_x]
    wx = nuis.predict_pi(Xx)
    gamma_y = 1.0 / (2.0 * ell_y * ell_y)
    stat, pval = kernel_dr_two_sample_test_agnostic(Yx, Xx, Ax, wx, kernel_function="rbf", gamma=gamma_y)
    out["p_dr_xkte"] = pval
    out["stat_dr_xkte"] = stat

    return out


# -----------------------------------------------------------------------------
# Summary and plotting.
# -----------------------------------------------------------------------------

def summarize(df: pd.DataFrame, alpha: float):
    rows = []
    for (setting, n), dfg in df.groupby(["setting", "n"]):
        R = len(dfg)
        nominal_se = math.sqrt(alpha * (1.0 - alpha) / R)
        for pcol, label in METHODS.items():
            if pcol not in dfg.columns:
                continue
            pvals = dfg[pcol].to_numpy(dtype=float)
            pvals = pvals[np.isfinite(pvals)]
            if len(pvals) == 0:
                continue
            rate = float(np.mean(pvals <= alpha))
            se = math.sqrt(max(rate * (1.0 - rate), 1e-12) / len(pvals))
            try:
                ks_p = float(st.kstest(pvals, "uniform").pvalue) if setting == "null" else np.nan
            except Exception:
                ks_p = np.nan
            rows.append({
                "setting": setting,
                "n": int(n),
                "method": label,
                "p_column": pcol,
                "n_reps": int(len(pvals)),
                "alpha": float(alpha),
                "rejection_rate": rate,
                "mc_se_rate": se,
                "ci95_low": rate - 1.96 * se,
                "ci95_high": rate + 1.96 * se,
                "nominal_mc95_low": alpha - 1.96 * nominal_se,
                "nominal_mc95_high": alpha + 1.96 * nominal_se,
                "mean_p": float(np.mean(pvals)),
                "median_p": float(np.median(pvals)),
                "ks_uniform_pvalue_if_null": ks_p,
            })
    return pd.DataFrame(rows)


def plot_panels(summary: pd.DataFrame, outpath: str, methods: Sequence[str], alpha: float, title: str):
    import matplotlib.pyplot as plt

    settings = [s for s in PLOT_SETTINGS if s in set(summary["setting"])]
    ncols = 2
    nrows = int(math.ceil(len(settings) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.8, 4.1 * nrows), sharey=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, setting in zip(axes, settings):
        dfs = summary[summary["setting"] == setting]
        for method in methods:
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
    for ax in axes[len(settings):]:
        ax.axis("off")
    axes[0].set_ylabel(f"rejection rate at alpha={alpha}")
    if len(axes) > 2:
        axes[2].set_ylabel(f"rejection rate at alpha={alpha}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=min(4, max(1, len(labels))), loc="lower center")
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_location_hist(df: pd.DataFrame, outpath: str):
    import matplotlib.pyplot as plt

    if "loc_dr_split" not in df.columns:
        return
    settings = [s for s in ["mean_shift", "variance_shift", "localized_bump"] if s in set(df["setting"])]
    if not settings:
        return
    fig, axes = plt.subplots(1, len(settings), figsize=(4.4 * len(settings), 3.4), sharey=False)
    if len(settings) == 1:
        axes = [axes]
    for ax, setting in zip(axes, settings):
        d = df[df["setting"] == setting]
        d = d[d["n"] == d["n"].max()]
        vals: List[float] = []
        for s in d["loc_dr_split"].dropna().astype(str):
            for token in s.split(";"):
                try:
                    vals.append(float(token))
                except ValueError:
                    pass
        if vals:
            ax.hist(vals, bins=30, density=True)
        ax.set_title(f"{SETTING_LABELS.get(setting, setting)}, largest n")
        ax.set_xlabel("selected ODR-ME location")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--n-grid", type=int, nargs="+", default=[300, 600, 1200, 3000])
    parser.add_argument("--settings", type=str, nargs="+", default=["null", "mean_shift", "variance_shift", "localized_bump"],
                        choices=list(DEFAULT_EFFECTS.keys()))
    parser.add_argument("--J", type=int, default=2)
    parser.add_argument("--M", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sigma-y", type=float, default=1.0)
    parser.add_argument("--effect", type=float, default=None)
    parser.add_argument("--bump-prob", type=float, default=0.12)
    parser.add_argument("--tau", type=float, default=1e-4)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--pi-l2", type=float, default=1e-2)
    parser.add_argument("--basis-mode", choices=["rich", "linear"], default="rich")
    parser.add_argument("--seed-offset", type=int, default=2027000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--outdir", type=str, default=str(results_dir("exp2")))
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tasks = []
    setting_code = {"null": 0, "mean_shift": 1, "variance_shift": 2, "localized_bump": 3}
    for setting in args.settings:
        for n in args.n_grid:
            for r in range(args.n_reps):
                seed = args.seed_offset + setting_code[setting] * 10_000_000 + int(n) * 100_000 + r
                tasks.append((
                    int(n), int(seed), setting, int(args.J), int(args.M), float(args.sigma_y),
                    args.effect, float(args.bump_prob), float(args.tau), float(args.ridge),
                    float(args.ridge_alpha), float(args.pi_l2), args.basis_mode,
                ))

    t0 = time.time()
    records = []
    if args.n_jobs == 1:
        for k, task in enumerate(tasks, start=1):
            records.append(run_one(task))
            if k % 25 == 0 or k == len(tasks):
                print(f"completed {k}/{len(tasks)} runs", flush=True)
    else:
        n_jobs = cpu_count() if args.n_jobs == -1 else max(1, int(args.n_jobs))
        with Pool(processes=n_jobs) as pool:
            for k, rec in enumerate(pool.imap_unordered(run_one, tasks), start=1):
                records.append(rec)
                if k % 25 == 0 or k == len(tasks):
                    print(f"completed {k}/{len(tasks)} runs", flush=True)

    df = pd.DataFrame(records)
    raw_path = os.path.join(args.outdir, "exp2_raw_pvalues.csv")
    df.to_csv(raw_path, index=False)

    summary = summarize(df, alpha=args.alpha)
    summary_path = os.path.join(args.outdir, "exp2_power_summary.csv")
    summary.to_csv(summary_path, index=False)

    all_plot = os.path.join(args.outdir, "exp2_all_exp1_baselines_plus_drxkte.png")
    main_plot = os.path.join(args.outdir, "exp2_odrme_vs_drxkte.png")
    loc_plot = os.path.join(args.outdir, "exp2_odrme_selected_locations.png")

    plot_panels(summary, all_plot, PLOT_ORDER_ALL, args.alpha, "Experiment 2: Experiment-1 baselines under alternatives")
    plot_panels(summary, main_plot, ["ODR-ME", "DR-xKTE"], args.alpha, "Experiment 2: ODR-ME versus DR-xKTE")
    plot_location_hist(df, loc_plot)

    shown = summary[["setting", "n", "method", "n_reps", "rejection_rate", "ci95_low", "ci95_high", "mean_p", "median_p"]]
    shown = shown.sort_values(["setting", "n", "method"])
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.width", 220)
    print("\nSummary:")
    print(shown.to_string(index=False))
    print(f"\nWrote: {raw_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {all_plot}")
    print(f"Wrote: {main_plot}")
    print(f"Wrote: {loc_plot}")
    print(f"Elapsed: {time.time() - t0:.1f} seconds")


if __name__ == "__main__":
    main()
