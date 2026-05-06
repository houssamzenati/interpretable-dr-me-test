#!/usr/bin/env python3
"""
Experiment 4: local noncentral chi-square validation for fixed/pilot-learned locations.

This experiment validates the fixed-location local asymptotic law

    n * zbar' Shat^{-1} zbar  ->  chi^2_J(noncentrality = h^2 lambda_V)

under local alternatives of size delta_n = h / sqrt(n).

Design:
    X ~ N(0,I_5)
    A | X ~ Bernoulli(pi0(X))
    Y(0) = g(X) + eps0
    Y(1) = g(X) + eps1 + delta_n

The treatment mechanism and X distribution are unchanged along the local path.
For the Gaussian outcome kernel and Gaussian residuals, the nuisance regressions
m_a(x;v)=E[k(v,Y)|A=a,X=x] are available in closed form. This isolates the
local chi-square approximation from nuisance-estimation error.

Locations:
    A large independent pilot sample under the null is used to choose a fixed
    pair of locations V by maximizing the local noncentrality proxy
        eta_V' Sigma_V^{-1} eta_V,
    where eta_V is the derivative of the finite-location signal at delta=0.
    These locations are then frozen for all Monte Carlo test samples.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from latex_tables import tex_float, write_booktabs_table
from repo_paths import results_dir


def prognostic_g(X):
    return (
        0.90 * X[:, 0]
        + 0.60 * np.sin(X[:, 1])
        + 0.35 * (X[:, 2] ** 2 - 1.0)
        + 0.25 * X[:, 0] * X[:, 3]
        - 0.20 * np.cos(X[:, 4])
    )


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def propensity(X):
    lin = 0.90 * X[:, 0] - 0.75 * X[:, 1] + 0.55 * X[:, 2] - 0.40 * X[:, 3]
    return np.clip(expit(lin), 0.06, 0.94)


def rbf_y(Y, V, ell_y):
    Y = np.asarray(Y).reshape(-1, 1)
    V = np.asarray(V).reshape(1, -1)
    return np.exp(-0.5 * ((Y - V) / ell_y) ** 2)


def conditional_kernel_mean(X, V, ell_y, sigma_y, delta):
    """
    E[ exp(-(v - Y)^2/(2 ell_y^2)) | X=x, treatment arm with shift delta ],
    when Y = g(x) + delta + eps, eps ~ N(0, sigma_y^2).
    """
    mu = prognostic_g(X).reshape(-1, 1) + float(delta)
    V = np.asarray(V).reshape(1, -1)
    denom = ell_y ** 2 + sigma_y ** 2
    const = math.sqrt(ell_y ** 2 / denom)
    return const * np.exp(-0.5 * ((V - mu) ** 2) / denom)


def derivative_mu_at_null(X, V, ell_y, sigma_y):
    """
    Derivative at delta=0 of E[k(v,g(X)+delta+eps)] - E[k(v,g(X)+eps)]
    with respect to delta.
    """
    mu = prognostic_g(X).reshape(-1, 1)
    V = np.asarray(V).reshape(1, -1)
    denom = ell_y ** 2 + sigma_y ** 2
    m0 = conditional_kernel_mean(X, V.ravel(), ell_y, sigma_y, delta=0.0)
    deriv = m0 * (V - mu) / denom
    return deriv.mean(axis=0)


def simulate_observed(n, rng, delta, sigma_y):
    X = rng.normal(size=(n, 5))
    pi = propensity(X)
    A = rng.binomial(1, pi).astype(int)
    eps0 = rng.normal(scale=sigma_y, size=n)
    eps1 = rng.normal(scale=sigma_y, size=n)
    base = prognostic_g(X)
    Y0 = base + eps0
    Y1 = base + float(delta) + eps1
    Y = np.where(A == 1, Y1, Y0)
    return X, A, Y.reshape(-1, 1)


def dr_feature_true_nuisance(X, A, Y, V, ell_y, sigma_y, delta):
    """
    DR finite-location pseudo-feature using the true propensity and true conditional kernel means.
    """
    pi = propensity(X).reshape(-1, 1)
    Acol = A.reshape(-1, 1).astype(float)
    K = rbf_y(Y, V, ell_y)
    m0 = conditional_kernel_mean(X, V, ell_y, sigma_y, delta=0.0)
    m1 = conditional_kernel_mean(X, V, ell_y, sigma_y, delta=delta)
    phi1 = (Acol / pi) * (K - m1) + m1
    phi0 = ((1.0 - Acol) / (1.0 - pi)) * (K - m0) + m0
    return phi1 - phi0


def hotelling_stat(Z, ridge):
    n, J = Z.shape
    zbar = Z.mean(axis=0)
    Zc = Z - zbar
    S = (Zc.T @ Zc) / max(n - 1, 1)
    return float(n * zbar @ np.linalg.pinv(S + ridge * np.eye(J)) @ zbar)


def choose_bandwidth_from_null(seed, n_pilot, sigma_y):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_pilot, 5))
    Y = prognostic_g(X) + rng.normal(scale=sigma_y, size=n_pilot)
    if n_pilot > 2500:
        idx = rng.choice(n_pilot, size=2500, replace=False)
        Y = Y[idx]
    D = np.abs(Y[:, None] - Y[None, :])
    vals = D[np.triu_indices_from(D, k=1)]
    vals = vals[vals > 1e-12]
    return float(max(np.median(vals), 1e-3))


def pilot_select_locations(seed, n_pilot, M, ell_y, sigma_y, J, ridge):
    """
    Select locations once on an independent pilot sample by maximizing the
    local noncentrality proxy eta' Sigma^{-1} eta.
    """
    rng = np.random.default_rng(seed)
    X, A, Y = simulate_observed(n_pilot, rng, delta=0.0, sigma_y=sigma_y)
    # dictionary over central + tail region of the null outcome law
    y = Y.ravel()
    lo, hi = np.quantile(y, [0.01, 0.99])
    pad = 0.10 * max(hi - lo, 1e-3)
    C = np.linspace(lo - pad, hi + pad, M)

    # Null DR features for all candidate locations.
    ZC = dr_feature_true_nuisance(X, A, Y, C, ell_y, sigma_y, delta=0.0)
    eta_C = derivative_mu_at_null(X, C, ell_y, sigma_y)

    # Precompute candidate covariance.
    Zc = ZC - ZC.mean(axis=0)
    Sigma_C = (Zc.T @ Zc) / max(n_pilot - 1, 1)

    selected = []
    available = set(range(M))
    for _ in range(J):
        best, best_val = None, -np.inf
        for c in available:
            cols = selected + [c]
            eta = eta_C[cols]
            S = Sigma_C[np.ix_(cols, cols)]
            val = float(eta @ np.linalg.pinv(S + ridge * np.eye(len(cols))) @ eta)
            if val > best_val:
                best_val = val
                best = c
        selected.append(int(best))
        available.remove(int(best))

    V = C[selected]
    eta_V = eta_C[selected]
    Sigma_V = Sigma_C[np.ix_(selected, selected)]
    lambda_V = float(eta_V @ np.linalg.pinv(Sigma_V + ridge * np.eye(J)) @ eta_V)
    return {
        "V": V,
        "candidate_grid": C,
        "selected_indices": selected,
        "eta_V": eta_V,
        "Sigma_V": Sigma_V,
        "lambda_V": lambda_V,
        "ell_y": ell_y,
    }


def run_mc(seed, n_values, h_values, n_reps, V, lambda_V, ell_y, sigma_y, ridge, alpha):
    rng = np.random.default_rng(seed)
    rows = []
    stat_rows = []
    J = len(V)
    crit = float(st.chi2.ppf(1.0 - alpha, df=J))
    for n in n_values:
        for h in h_values:
            delta_n = float(h) / math.sqrt(n)
            ncp = float(h * h * lambda_V)
            th_power = float(st.ncx2.sf(crit, df=J, nc=ncp))
            stats = np.empty(n_reps)
            for r in range(n_reps):
                X, A, Y = simulate_observed(n, rng, delta=delta_n, sigma_y=sigma_y)
                Z = dr_feature_true_nuisance(X, A, Y, V, ell_y, sigma_y, delta=delta_n)
                stats[r] = hotelling_stat(Z, ridge=ridge)
            emp_power = float(np.mean(stats > crit))
            emp_mean = float(np.mean(stats))
            emp_var = float(np.var(stats, ddof=1))
            th_mean = float(J + ncp)
            th_var = float(2.0 * (J + 2.0 * ncp))
            ks = st.kstest(stats, lambda x: st.ncx2.cdf(x, df=J, nc=ncp))
            rows.append({
                "n": n,
                "h": h,
                "delta_n": delta_n,
                "n_reps": n_reps,
                "df": J,
                "lambda_V": lambda_V,
                "ncp": ncp,
                "crit_alpha": crit,
                "emp_power": emp_power,
                "theory_power": th_power,
                "abs_power_error": abs(emp_power - th_power),
                "emp_mean": emp_mean,
                "theory_mean": th_mean,
                "emp_var": emp_var,
                "theory_var": th_var,
                "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            })
            # keep all stats for largest n and all h, plus null diagnostics for all n
            if n == max(n_values) or h == 0:
                for s in stats:
                    stat_rows.append({"n": n, "h": h, "stat": float(s), "ncp": ncp})
    return pd.DataFrame(rows), pd.DataFrame(stat_rows)


def make_figures(summary, stats, outdir, alpha):
    plt.figure(figsize=(8, 5))
    for n, sub in summary.groupby("n"):
        sub = sub.sort_values("h")
        plt.errorbar(
            sub["h"], sub["emp_power"],
            yerr=1.96 * np.sqrt(sub["emp_power"] * (1 - sub["emp_power"]) / sub["n_reps"]),
            marker="o", capsize=3, label=f"empirical n={n}"
        )
    th = summary.groupby("h", as_index=False).first().sort_values("h")
    plt.plot(th["h"], th["theory_power"], linestyle="--", linewidth=2, label="noncentral $\\chi^2$ theory")
    plt.xlabel("local alternative magnitude $h$")
    plt.ylabel(f"rejection probability at $\\alpha={alpha}$")
    plt.ylim(-0.02, 1.02)
    plt.title("Local power: empirical statistic vs noncentral $\\chi^2$ prediction")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "exp4_local_power_vs_theory.png", dpi=220)
    plt.close()

    # Mean/variance diagnostic.
    plt.figure(figsize=(8, 5))
    for n, sub in summary.groupby("n"):
        sub = sub.sort_values("h")
        plt.plot(sub["theory_mean"], sub["emp_mean"], marker="o", label=f"n={n}")
    lo = min(summary["theory_mean"].min(), summary["emp_mean"].min())
    hi = max(summary["theory_mean"].max(), summary["emp_mean"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    plt.xlabel("theoretical mean of $\\chi^2_J(h^2\\lambda_V)$")
    plt.ylabel("empirical mean of statistic")
    plt.title("Distribution check: first moment")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "exp4_mean_diagnostic.png", dpi=220)
    plt.close()

    # QQ plots for largest n, h=0 and largest h.
    n_max = int(stats["n"].max())
    h_values = sorted(stats.loc[stats["n"] == n_max, "h"].unique())
    h_show = [h_values[0], h_values[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, h in zip(axes, h_show):
        sub = stats[(stats["n"] == n_max) & (stats["h"] == h)].sort_values("stat")
        probs = (np.arange(1, len(sub) + 1) - 0.5) / len(sub)
        ncp = float(sub["ncp"].iloc[0])
        q_theory = st.ncx2.ppf(probs, df=2, nc=ncp)
        q_emp = sub["stat"].to_numpy()
        ax.scatter(q_theory, q_emp, s=12, alpha=0.6)
        lo = min(q_theory.min(), q_emp.min())
        hi = max(q_theory.max(), q_emp.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_title(f"QQ, n={n_max}, h={h:g}")
        ax.set_xlabel("theoretical quantile")
        ax.set_ylabel("empirical quantile")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "exp4_qq_largest_n.png", dpi=220)
    plt.close()

    # Null histogram for largest n.
    sub0 = stats[(stats["n"] == n_max) & (stats["h"] == 0)]
    xs = np.linspace(0, np.quantile(sub0["stat"], 0.995), 400)
    plt.figure(figsize=(7, 4))
    plt.hist(sub0["stat"], bins=35, density=True, alpha=0.65, label="empirical")
    plt.plot(xs, st.chi2.pdf(xs, df=2), linewidth=2, label="$\\chi^2_2$")
    plt.xlabel("test statistic")
    plt.ylabel("density")
    plt.title(f"Null distribution check, n={n_max}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "exp4_null_hist_largest_n.png", dpi=220)
    plt.close()


def export_tex_tables(summary, outdir):
    n_values = sorted(int(n) for n in summary["n"].unique())
    h_values = sorted(float(h) for h in summary["h"].unique())
    lines = [
        "$h$ & $h^2\\widehat\\lambda_V$ & Theory & "
        + " & ".join(f"$n={n}$" for n in n_values)
        + r" \\",
        r"\midrule",
    ]
    for h in h_values:
        block = summary[summary["h"] == h].set_index("n")
        first = block.iloc[0]
        row = [
            f"{int(h)}" if abs(h - int(h)) < 1e-12 else tex_float(h, digits=1),
            tex_float(first["ncp"]),
            tex_float(first["theory_power"]),
        ]
        for n in n_values:
            value = block.loc[n, "emp_power"] if n in block.index else None
            row.append(tex_float(value))
        lines.append(" & ".join(row) + r" \\")

    path = outdir / "tab_exp4_local_power.tex"
    write_booktabs_table(
        str(path),
        "ccc" + "c" * len(n_values),
        lines,
        "Experiment 4: empirical rejection rates under local alternatives, "
        "compared with the noncentral chi-square prediction. The theoretical "
        "power is computed from $\\chi^2_2(h^2\\widehat\\lambda_V)$.",
        "tab:exp4_local_power",
    )
    return [str(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=str(results_dir("exp4")))
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--n-values", type=int, nargs="+", default=[500, 1000, 3000])
    parser.add_argument("--h-values", type=float, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--pilot-n", type=int, default=100000)
    parser.add_argument("--pilot-M", type=int, default=100)
    parser.add_argument("--J", type=int, default=2)
    parser.add_argument("--sigma-y", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=94123)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ell_y = choose_bandwidth_from_null(seed=args.seed + 1, n_pilot=min(args.pilot_n, 50000), sigma_y=args.sigma_y)
    pilot = pilot_select_locations(
        seed=args.seed + 2,
        n_pilot=args.pilot_n,
        M=args.pilot_M,
        ell_y=ell_y,
        sigma_y=args.sigma_y,
        J=args.J,
        ridge=args.ridge,
    )
    V = np.asarray(pilot["V"], dtype=float)
    lambda_V = float(pilot["lambda_V"])

    summary, stat_draws = run_mc(
        seed=args.seed + 3,
        n_values=args.n_values,
        h_values=args.h_values,
        n_reps=args.n_reps,
        V=V,
        lambda_V=lambda_V,
        ell_y=ell_y,
        sigma_y=args.sigma_y,
        ridge=args.ridge,
        alpha=args.alpha,
    )

    make_figures(summary, stat_draws, outdir, alpha=args.alpha)
    tex_paths = export_tex_tables(summary, outdir)

    print("Selected V:", V)
    print("ell_y:", ell_y)
    print("lambda_V:", lambda_V)
    for tex_path in tex_paths:
        print("Wrote:", tex_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
