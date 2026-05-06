#!/usr/bin/env python3
"""
Experiment 1: calibration under a confounded observational sharp null.

Null DGP:
    X ~ N(0, I_p)
    A | X ~ Bernoulli(pi0(X))
    Y(0) = Y(1) = g(X) + eps

This is a sharp null: each unit has the same potential outcome under treatment
and control. The interventional laws are equal, but the observed
treated/control outcome laws differ because treatment is confounded by X.

Default methods:
    DR-ME split       : AIPW/DR finite-location ME with plug-in pi_hat and m_hat_a.
    DR-ME random      : Same DR pseudo-features, random locations independent of I_te.
    IPW-ME plug-in    : IPW finite-location ME using estimated pi_hat, not oracle pi.
    DM-ME plug-in     : Direct finite-location ME using estimated m_hat_1 - m_hat_0.
    Naive observed ME : Ordinary treated/control outcome ME ignoring X.
    DR-ME no split    : Invalid selector; learns locations on I_te and tests on I_te.

Three-way split:
    I_eta: fit pi_hat and m_hat_a(x;c)=E[k(c,Y)|A=a,X=x]
    I_tr : learn locations by maximizing a ridge-stabilized Hotelling proxy
    I_te : final fixed-location Hotelling/chi-square test

This script avoids heavy ML dependencies. The propensity is fit by a small
ridge-regularized logistic IRLS solver, and m_a(x;C) is fit by multi-output
ridge regression on a low-dimensional engineered feature basis.

Example:
    python exp1_calibration_dr_me_plugin.py --n-reps 200 --n-grid 300 600 1200 3000 --n-jobs 1
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2, f, kstest
from latex_tables import tex_float, write_booktabs_table
from repo_paths import results_dir


# ---------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------

def prognostic_g(X):
    """Nonlinear prognostic function g(X). X must have at least 5 columns."""
    return (
        0.90 * X[:, 0]
        + 0.60 * np.sin(X[:, 1])
        + 0.35 * (X[:, 2] ** 2 - 1.0)
        + 0.25 * X[:, 0] * X[:, 3]
        - 0.20 * np.cos(X[:, 4])
    )


def true_propensity(X):
    """True pi0(X)=P(A=1|X), clipped to enforce overlap."""
    lin = 0.90 * X[:, 0] - 0.75 * X[:, 1] + 0.55 * X[:, 2] - 0.40 * X[:, 3]
    return np.clip(expit(lin), 0.06, 0.94)


def simulate_confounded_null(
    n,
    seed,
    p=5,
    sigma_y=1.0,
):
    """Generate one observational sample under Y(0)=Y(1)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    pi = true_propensity(X)
    A = rng.binomial(1, pi).astype(np.int64)
    Y = prognostic_g(X) + rng.normal(scale=sigma_y, size=n)
    return X, A, Y.reshape(-1, 1)


# ---------------------------------------------------------------------
# Kernels and utility functions
# ---------------------------------------------------------------------

def squared_distances(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    AA = np.sum(A * A, axis=1)[:, None]
    BB = np.sum(B * B, axis=1)[None, :]
    D2 = AA + BB - 2.0 * (A @ B.T)
    return np.maximum(D2, 0.0)


def gaussian_kernel_y(Y, C, ell_y):
    gamma_y = 1.0 / (2.0 * ell_y * ell_y)
    return np.exp(-gamma_y * squared_distances(Y, C))


def median_bandwidth_y(
    Y,
    rng,
    max_points=512,
):
    Y = np.asarray(Y, dtype=float).reshape(len(Y), -1)
    n = len(Y)
    if n <= 1:
        return 1.0
    if n > max_points:
        Y = Y[rng.choice(n, size=max_points, replace=False)]
        n = len(Y)
    D = np.sqrt(squared_distances(Y, Y))
    vals = D[np.triu_indices(n, k=1)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    return float(max(np.median(vals), 1e-3))


def make_dictionary(Y, M, lo_q=0.02, hi_q=0.98):
    """Quantile-range grid dictionary for scalar outcomes."""
    y = np.asarray(Y).ravel()
    lo, hi = np.quantile(y, [lo_q, hi_q])
    span = max(hi - lo, 1e-3)
    lo -= 0.05 * span
    hi += 0.05 * span
    return np.linspace(lo, hi, M).reshape(-1, 1)


def basis_features(X, mode="rich"):
    """Feature map for laptop-scale multi-output regression of m_a(x;C)."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if mode == "linear":
        return X

    feats = [X, X ** 2 - 1.0, np.sin(X), np.cos(X)]
    interactions = []
    for j in range(p):
        for k in range(j + 1, p):
            interactions.append((X[:, j] * X[:, k]).reshape(n, 1))
    if interactions:
        feats.append(np.hstack(interactions))
    return np.hstack(feats)


def add_intercept_and_standardize(
    Phi,
    mean=None,
    scale=None,
):
    if mean is None:
        mean = Phi.mean(axis=0, keepdims=True)
    if scale is None:
        scale = Phi.std(axis=0, keepdims=True)
        scale = np.where(scale < 1e-8, 1.0, scale)
    Phis = (Phi - mean) / scale
    Phi1 = np.hstack([np.ones((Phi.shape[0], 1)), Phis])
    return Phi1, mean, scale


# ---------------------------------------------------------------------
# Plug-in nuisance models
# ---------------------------------------------------------------------

def fit_logistic_irls(
    X,
    y,
    l2=1e-2,
    max_iter=50,
    tol=1e-8,
):
    """Small ridge-logistic IRLS solver. Intercept is not penalized."""
    Phi0 = X.astype(float)
    Phi, mean, scale = add_intercept_and_standardize(Phi0)
    n, d = Phi.shape
    beta = np.zeros(d)
    penalty = l2 * np.eye(d)
    penalty[0, 0] = 0.0

    for _ in range(max_iter):
        eta = Phi @ beta
        p = np.clip(expit(eta), 1e-6, 1.0 - 1e-6)
        W = p * (1.0 - p)
        grad = Phi.T @ (y - p) - penalty @ beta
        H = Phi.T @ (Phi * W[:, None]) + penalty
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad
        beta_new = beta + step
        if np.linalg.norm(step) <= tol * (1.0 + np.linalg.norm(beta)):
            beta = beta_new
            break
        beta = beta_new
    return {"beta": beta, "mean": mean, "scale": scale}


def predict_logistic(model, X):
    Phi, _, _ = add_intercept_and_standardize(X.astype(float), model["mean"], model["scale"])
    return expit(Phi @ model["beta"])


@dataclass
class PluginNuisance:
    ell_y: float
    ridge_alpha: float = 1.0
    pi_l2: float = 1e-2
    pi_clip: float = 0.05
    basis_mode: str = "rich"

    def fit(self, X_eta, A_eta, Y_eta, C):
        self.C = np.asarray(C, dtype=float)

        self.pi_model = fit_logistic_irls(X_eta, A_eta.astype(float), l2=self.pi_l2)

        Phi_raw = basis_features(X_eta, mode=self.basis_mode)
        Phi_all, self.phi_mean, self.phi_scale = add_intercept_and_standardize(Phi_raw)

        self.m_coef = {}
        self.arm_counts = {}
        self.arm_mean_targets = {}

        for a in (0, 1):
            mask = A_eta == a
            self.arm_counts[a] = int(mask.sum())
            U = gaussian_kernel_y(Y_eta[mask], self.C, self.ell_y) if mask.sum() > 0 else np.zeros((0, len(self.C)))
            self.arm_mean_targets[a] = U.mean(axis=0, keepdims=True) if len(U) else np.zeros((1, len(self.C)))
            if mask.sum() < 8:
                self.m_coef[a] = np.zeros((Phi_all.shape[1], len(self.C)))
                self.m_coef[a][0, :] = self.arm_mean_targets[a]
                continue

            Phi = Phi_all[mask]
            G = Phi.T @ Phi
            rhs = Phi.T @ U
            penalty = self.ridge_alpha * np.eye(G.shape[0])
            penalty[0, 0] = 1e-10
            try:
                coef = np.linalg.solve(G + penalty, rhs)
            except np.linalg.LinAlgError:
                coef = np.linalg.pinv(G + penalty) @ rhs
            self.m_coef[a] = coef

        return self

    def predict_pi(self, X):
        p = predict_logistic(self.pi_model, X)
        return np.clip(p, self.pi_clip, 1.0 - self.pi_clip)

    def predict_m(self, X, a):
        Phi_raw = basis_features(X, mode=self.basis_mode)
        Phi, _, _ = add_intercept_and_standardize(Phi_raw, self.phi_mean, self.phi_scale)
        pred = Phi @ self.m_coef[a]
        # m_a(x;c)=E[k(c,Y)|A=a,X=x] lies in [0,1] for the Gaussian kernel.
        # Mild clipping prevents ridge extrapolation from dominating small n.
        return np.clip(pred, -0.05, 1.05)


# ---------------------------------------------------------------------
# Pseudo-feature matrices and tests
# ---------------------------------------------------------------------

def pseudo_feature_matrix(
    method,
    X,
    A,
    Y,
    C,
    ell_y,
    nuis,
    p_global=None,
):
    """Return n x M matrix of finite-location pseudo-features."""
    KYC = gaussian_kernel_y(Y, C, ell_y)
    Acol = A.reshape(-1, 1).astype(float)

    if method == "dr":
        p1 = nuis.predict_pi(X).reshape(-1, 1)
        m1 = nuis.predict_m(X, 1)
        m0 = nuis.predict_m(X, 0)
        phi1 = (Acol / p1) * (KYC - m1) + m1
        phi0 = ((1.0 - Acol) / (1.0 - p1)) * (KYC - m0) + m0
        return phi1 - phi0

    if method == "ipw":
        p1 = nuis.predict_pi(X).reshape(-1, 1)
        return (Acol / p1) * KYC - ((1.0 - Acol) / (1.0 - p1)) * KYC

    if method == "dm":
        return nuis.predict_m(X, 1) - nuis.predict_m(X, 0)

    if method == "naive":
        if p_global is None:
            p_global = float(np.mean(A))
        p_global = float(np.clip(p_global, 0.05, 0.95))
        return (Acol / p_global) * KYC - ((1.0 - Acol) / (1.0 - p_global)) * KYC

    raise ValueError(f"Unknown method: {method}")


def mean_cov(Z):
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = Z.shape[0]
    mu = Z.mean(axis=0)
    Zc = Z - mu
    S = (Zc.T @ Zc) / max(n - 1, 1)
    return mu, np.atleast_2d(S)


def hotelling_pvalue(
    Z,
    cols,
    ridge=1e-8,
    pvalue="chi2",
):
    Zs = Z[:, list(cols)]
    if Zs.ndim == 1:
        Zs = Zs.reshape(-1, 1)
    n, J = Zs.shape
    mu, S = mean_cov(Zs)
    inv = np.linalg.pinv(S + ridge * np.eye(J))
    stat = float(n * mu @ inv @ mu)

    if pvalue == "chi2":
        p = float(chi2.sf(stat, df=J))
    elif pvalue == "hotelling_f":
        if n > J:
            Fstat = (n - J) * stat / (J * (n - 1))
            p = float(f.sf(Fstat, J, n - J))
        else:
            p = float(chi2.sf(stat, df=J))
    else:
        raise ValueError(f"Unknown p-value mode: {pvalue}")

    return stat, p


def learning_criterion(Z, cols, tau):
    Zs = Z[:, list(cols)]
    if Zs.ndim == 1:
        Zs = Zs.reshape(-1, 1)
    _, J = Zs.shape
    mu, S = mean_cov(Zs)
    inv = np.linalg.pinv(S + tau * np.eye(J))
    return float(mu @ inv @ mu)


def greedy_select(Z_train, J, tau):
    """Greedy forward selection over dictionary columns.

    The default experiment uses J=2. For that case the second greedy step is
    vectorized, which keeps the 200-seed calibration run laptop-scale.
    """
    M = Z_train.shape[1]
    selected = []
    available = set(range(M))

    mu = Z_train.mean(axis=0)
    Zc = Z_train - mu
    var = np.sum(Zc * Zc, axis=0) / max(Z_train.shape[0] - 1, 1)

    # Fast first step.
    if J >= 1:
        crit1 = mu * mu / (var + tau)
        best_c = int(np.argmax(crit1))
        selected.append(best_c)
        available.remove(best_c)

    # Fully vectorized second step.
    if J == 2:
        s = selected[0]
        zcs = Zc[:, s]
        cov_s = (Zc.T @ zcs) / max(Z_train.shape[0] - 1, 1)
        a = var[s] + tau
        d = var + tau
        b = cov_s
        det = np.maximum(a * d - b * b, 1e-14)
        obj = (d * mu[s] * mu[s] - 2.0 * b * mu[s] * mu + a * mu * mu) / det
        obj[s] = -np.inf
        selected.append(int(np.argmax(obj)))
        return selected

    for _ in range(1, J):
        best_c = None
        best_val = -np.inf
        for c in available:
            val = learning_criterion(Z_train, selected + [c], tau)
            if val > best_val:
                best_val = val
                best_c = c
        assert best_c is not None
        selected.append(best_c)
        available.remove(best_c)

    return selected


# ---------------------------------------------------------------------
# One Monte Carlo repetition
# ---------------------------------------------------------------------

def run_one(
    n,
    seed,
    J,
    M,
    sigma_y,
    tau,
    ridge,
    ridge_alpha,
    pi_l2,
    pvalue,
    basis_mode,
):
    rng = np.random.default_rng(seed)
    X, A, Y = simulate_confounded_null(n=n, seed=seed, sigma_y=sigma_y)

    perm = rng.permutation(n)
    n_eta = n // 3
    n_tr = n // 3
    idx_eta = perm[:n_eta]
    idx_tr = perm[n_eta:n_eta + n_tr]
    idx_te = perm[n_eta + n_tr:]

    # Bandwidth uses nuisance+training splits, not final test split.
    ell_y = median_bandwidth_y(Y[np.concatenate([idx_eta, idx_tr])], rng)

    # Valid dictionary for split procedures: learned from training split only.
    C_tr = make_dictionary(Y[idx_tr], M)

    nuis = PluginNuisance(
        ell_y=ell_y,
        ridge_alpha=ridge_alpha,
        pi_l2=pi_l2,
        basis_mode=basis_mode,
    ).fit(X[idx_eta], A[idx_eta], Y[idx_eta], C_tr)

    p_global = float(A[idx_eta].mean())

    Ztr_dr = pseudo_feature_matrix("dr", X[idx_tr], A[idx_tr], Y[idx_tr], C_tr, ell_y, nuis)
    Zte_dr = pseudo_feature_matrix("dr", X[idx_te], A[idx_te], Y[idx_te], C_tr, ell_y, nuis)

    Ztr_ipw = pseudo_feature_matrix("ipw", X[idx_tr], A[idx_tr], Y[idx_tr], C_tr, ell_y, nuis)
    Zte_ipw = pseudo_feature_matrix("ipw", X[idx_te], A[idx_te], Y[idx_te], C_tr, ell_y, nuis)

    Ztr_dm = pseudo_feature_matrix("dm", X[idx_tr], A[idx_tr], Y[idx_tr], C_tr, ell_y, nuis)
    Zte_dm = pseudo_feature_matrix("dm", X[idx_te], A[idx_te], Y[idx_te], C_tr, ell_y, nuis)

    Ztr_naive = pseudo_feature_matrix("naive", X[idx_tr], A[idx_tr], Y[idx_tr], C_tr, ell_y, nuis, p_global=p_global)
    Zte_naive = pseudo_feature_matrix("naive", X[idx_te], A[idx_te], Y[idx_te], C_tr, ell_y, nuis, p_global=p_global)

    out = {
        "n": n,
        "seed": seed,
        "J": J,
        "M": M,
        "n_eta": len(idx_eta),
        "n_tr": len(idx_tr),
        "n_te": len(idx_te),
        "ell_y": ell_y,
        "basis_mode": basis_mode,
        "ridge_alpha": ridge_alpha,
        "pi_l2": pi_l2,
        "arm0_eta": nuis.arm_counts[0],
        "arm1_eta": nuis.arm_counts[1],
    }

    # Valid split DR-ME.
    sel_dr = greedy_select(Ztr_dr, J, tau)
    stat, pval = hotelling_pvalue(Zte_dr, sel_dr, ridge=ridge, pvalue=pvalue)
    out["p_dr_split"] = pval
    out["stat_dr_split"] = stat

    # Random-location DR-ME.
    sel_rand = list(rng.choice(M, size=J, replace=False))
    stat, pval = hotelling_pvalue(Zte_dr, sel_rand, ridge=ridge, pvalue=pvalue)
    out["p_dr_random"] = pval
    out["stat_dr_random"] = stat

    # IPW plug-in finite-location test: estimated propensity, no oracle pi.
    sel_ipw = greedy_select(Ztr_ipw, J, tau)
    stat, pval = hotelling_pvalue(Zte_ipw, sel_ipw, ridge=ridge, pvalue=pvalue)
    out["p_ipw_plugin"] = pval
    out["stat_ipw_plugin"] = stat

    # Direct-method plug-in finite-location test: estimated m_1-m_0.
    sel_dm = greedy_select(Ztr_dm, J, tau)
    stat, pval = hotelling_pvalue(Zte_dm, sel_dm, ridge=ridge, pvalue=pvalue)
    out["p_dm_plugin"] = pval
    out["stat_dm_plugin"] = stat

    # Ordinary observed-arm finite-location test ignoring X. Invalid under confounding.
    sel_naive = greedy_select(Ztr_naive, J, tau)
    stat, pval = hotelling_pvalue(Zte_naive, sel_naive, ridge=ridge, pvalue=pvalue)
    out["p_naive_split"] = pval
    out["stat_naive_split"] = stat

    # Invalid no-split DR selection: learn and test on I_te.
    # Refit nuisance targets on C_te because the location dictionary changed.
    C_te = make_dictionary(Y[idx_te], M)
    nuis_te = PluginNuisance(
        ell_y=ell_y,
        ridge_alpha=ridge_alpha,
        pi_l2=pi_l2,
        basis_mode=basis_mode,
    ).fit(X[idx_eta], A[idx_eta], Y[idx_eta], C_te)
    Zte_dr_nosplit = pseudo_feature_matrix("dr", X[idx_te], A[idx_te], Y[idx_te], C_te, ell_y, nuis_te)
    sel_nosplit = greedy_select(Zte_dr_nosplit, J, tau)
    stat, pval = hotelling_pvalue(Zte_dr_nosplit, sel_nosplit, ridge=ridge, pvalue=pvalue)
    out["p_dr_nosplit"] = pval
    out["stat_dr_nosplit"] = stat

    return out


# ---------------------------------------------------------------------
# Summary and plotting
# ---------------------------------------------------------------------

METHOD_LABELS = {
    "p_dr_split": "DR-ME split",
    "p_dr_random": "DR-ME random",
    "p_ipw_plugin": "IPW-ME plug-in",
    "p_dm_plugin": "DM-ME plug-in",
    "p_naive_split": "Naive observed ME",
    "p_dr_nosplit": "DR-ME no split",
}

PAPER_METHOD_LABELS = [
    ("DR-ME split", "ODR-ME"),
    ("DR-ME random", "ODR-ME-Rand"),
    ("IPW-ME plug-in", "IPW-ME plug-in"),
    ("DM-ME plug-in", "DM-ME plug-in"),
    ("Naive observed ME", "Naive observed ME"),
    ("DR-ME no split", "ODR-ME-NoSplit"),
]


def summarize(df, alpha):
    rows = []
    for n, dfn in df.groupby("n"):
        B = len(dfn)
        nominal_se = math.sqrt(alpha * (1.0 - alpha) / B)
        nominal_lo = alpha - 1.96 * nominal_se
        nominal_hi = alpha + 1.96 * nominal_se

        for col, label in METHOD_LABELS.items():
            pvals = dfn[col].to_numpy()
            reject = pvals <= alpha
            rate = float(reject.mean())
            se = math.sqrt(max(rate * (1.0 - rate), 1e-12) / B)
            try:
                ks_p = float(kstest(pvals, "uniform").pvalue)
            except Exception:
                ks_p = float("nan")
            rows.append({
                "n": int(n),
                "method": label,
                "p_column": col,
                "n_reps": int(B),
                "alpha": alpha,
                "rejection_rate": rate,
                "mc_se_rate": se,
                "ci95_low": rate - 1.96 * se,
                "ci95_high": rate + 1.96 * se,
                "nominal_mc95_low": nominal_lo,
                "nominal_mc95_high": nominal_hi,
                "mean_p": float(np.mean(pvals)),
                "median_p": float(np.median(pvals)),
                "ks_uniform_pvalue": ks_p,
                "within_nominal_mc95": bool(nominal_lo <= rate <= nominal_hi),
            })
    return pd.DataFrame(rows)


def plot_summary(summary, outpath, alpha):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for method, dfm in summary.groupby("method"):
        dfm = dfm.sort_values("n")
        ax.errorbar(
            dfm["n"],
            dfm["rejection_rate"],
            yerr=1.96 * dfm["mc_se_rate"],
            marker="o",
            capsize=3,
            label=method,
        )

    n_values = np.sort(summary["n"].unique())
    B = int(summary["n_reps"].iloc[0])
    nominal_se = math.sqrt(alpha * (1.0 - alpha) / B)
    ax.axhline(alpha, linestyle="--", linewidth=1)
    ax.fill_between(
        n_values,
        alpha - 1.96 * nominal_se,
        alpha + 1.96 * nominal_se,
        alpha=0.15,
    )
    ax.set_xscale("log")
    ax.set_xlabel("sample size n")
    ax.set_ylabel(f"empirical rejection rate at alpha={alpha}")
    ax.set_title("Experiment 1: calibration under confounded causal null")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def export_tex_tables(summary, outdir):
    n_values = sorted(int(n) for n in summary["n"].unique())
    by_method = {
        method: df.set_index("n")["rejection_rate"]
        for method, df in summary.groupby("method")
    }

    lines = [
        "Method & " + " & ".join(f"$n={n}$" for n in n_values) + r" \\",
        r"\midrule",
    ]
    for internal_name, paper_name in PAPER_METHOD_LABELS:
        row = [paper_name]
        series = by_method.get(internal_name)
        for n in n_values:
            value = series.get(n) if series is not None else None
            row.append(tex_float(value))
        lines.append(" & ".join(row) + r" \\")

    path = os.path.join(outdir, "tab_exp1_calibration.tex")
    write_booktabs_table(
        path,
        "l" + "c" * len(n_values),
        lines,
        "Empirical rejection rates under the confounded observational null. "
        "The target interventional laws are equal, $P_{Y(0)}=P_{Y(1)}$, but the observed "
        "treated and control outcome distributions differ because of confounding.",
        "tab:exp1_calibration",
    )
    return [path]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=200, help="Monte Carlo repetitions per n. Use >=200 for the paper run.")
    parser.add_argument("--n-grid", type=int, nargs="+", default=[300, 600, 1200, 3000])
    parser.add_argument("--J", type=int, default=2, help="number of selected locations")
    parser.add_argument("--M", type=int, default=80, help="number of candidate locations in the scalar grid dictionary")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sigma-y", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1e-4, help="ridge for location-learning criterion")
    parser.add_argument("--ridge", type=float, default=1e-8, help="ridge in final Hotelling statistic")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="ridge alpha for plug-in m_a regressions")
    parser.add_argument("--pi-l2", type=float, default=1e-2, help="L2 penalty for plug-in logistic propensity")
    parser.add_argument("--basis-mode", choices=["rich", "linear"], default="rich", help="feature basis for plug-in m_a regressions")
    parser.add_argument("--pvalue", choices=["chi2", "hotelling_f"], default="chi2")
    parser.add_argument("--seed-offset", type=int, default=2026000)
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel jobs; use -1 for all cores")
    parser.add_argument("--outdir", type=str, default=str(results_dir("exp1")))
    return parser.parse_args()


def _run_one_star(args_tuple):
    return run_one(*args_tuple)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tasks = []
    for n in args.n_grid:
        for r in range(args.n_reps):
            seed = args.seed_offset + 100000 * int(n) + r
            tasks.append((
                int(n),
                int(seed),
                args.J,
                args.M,
                args.sigma_y,
                args.tau,
                args.ridge,
                args.ridge_alpha,
                args.pi_l2,
                args.pvalue,
                args.basis_mode,
            ))

    t0 = time.time()

    if args.n_jobs == 1:
        records = []
        for k, task in enumerate(tasks, start=1):
            records.append(_run_one_star(task))
            if k % 25 == 0 or k == len(tasks):
                print(f"completed {k}/{len(tasks)} runs", flush=True)
    else:
        n_jobs = cpu_count() if args.n_jobs == -1 else max(1, int(args.n_jobs))
        with Pool(processes=n_jobs) as pool:
            records = []
            for k, rec in enumerate(pool.imap_unordered(_run_one_star, tasks), start=1):
                records.append(rec)
                if k % 25 == 0 or k == len(tasks):
                    print(f"completed {k}/{len(tasks)} runs", flush=True)

    df = pd.DataFrame(records)
    summary = summarize(df, alpha=args.alpha)
    plot_path = os.path.join(args.outdir, "exp1_calibration_plot.png")
    plot_summary(summary, plot_path, alpha=args.alpha)
    tex_paths = export_tex_tables(summary, args.outdir)

    print("\nCalibration summary:")
    shown = summary[
        [
            "n",
            "method",
            "n_reps",
            "rejection_rate",
            "ci95_low",
            "ci95_high",
            "nominal_mc95_low",
            "nominal_mc95_high",
            "within_nominal_mc95",
            "mean_p",
            "ks_uniform_pvalue",
        ]
    ].sort_values(["n", "method"])
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.width", 180)
    print(shown.to_string(index=False))
    print()
    print(f"Wrote: {plot_path}")
    for tex_path in tex_paths:
        print(f"Wrote: {tex_path}")
    print(f"Elapsed: {time.time() - t0:.1f} seconds")


if __name__ == "__main__":
    main()
