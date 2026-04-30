#!/usr/bin/env python3
"""
Standalone contour figure for the DR-ME location-learning diagnostic.

This script generates the requested main-text figure:
  - two side-by-side contour plots;
  - the contour surface is ALWAYS the estimated-nuisance DR-ME training objective;
  - the oracle-nuisance maximizer is displayed ONLY as an extra marker;
  - no oracle contour is plotted.

Default run:
    python contour_estimated_oracle_overlay_single.py

Outputs:
    contour_estimated_oracle_overlay/contour_estimated_with_oracle_v2_overlay_tight.png
    contour_estimated_oracle_overlay/contour_estimated_with_oracle_v2_overlay_tight.pdf
    contour_estimated_oracle_overlay/contour_estimated_with_oracle_v2_overlay_metadata.json
"""
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Keep BLAS thread usage controlled on laptops/servers.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from repo_paths import results_dir


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def sigmoid(x: np.ndarray):
    x = np.clip(np.asarray(x, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def covariance_matrix(z: np.ndarray):
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    if z.shape[0] <= 1:
        return np.eye(z.shape[1])
    centered = z - z.mean(axis=0, keepdims=True)
    return centered.T @ centered / (z.shape[0] - 1)


def hotelling_statistic(z: np.ndarray, ridge: float = 1e-6):
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    mean = z.mean(axis=0)
    cov = covariance_matrix(z)
    mat = cov + ridge * np.eye(z.shape[1])
    # pinv is slightly more stable for contour-plot sweeps.
    return float(z.shape[0] * mean @ np.linalg.pinv(mat) @ mean)


def split_indices(n: int, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n1 = int(round(frac * n))
    return idx[:n1], idx[n1:]


def split_folds(n: int, n_folds: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [fold for fold in np.array_split(idx, int(n_folds)) if len(fold) > 0]


class Sample:
    def __init__(self, x, a, y, y0, y1, propensity, mixture):
        self.x = np.asarray(x, dtype=float)
        self.a = np.asarray(a, dtype=float).reshape(-1)
        self.y = np.asarray(y, dtype=float)
        self.y0 = np.asarray(y0, dtype=float)
        self.y1 = np.asarray(y1, dtype=float)
        self.propensity = np.asarray(propensity, dtype=float).reshape(-1)
        self.mixture = mixture

    def subset(self, idx: np.ndarray):
        return Sample(
            self.x[idx],
            self.a[idx],
            self.y[idx],
            self.y0[idx],
            self.y1[idx],
            self.propensity[idx],
            self.mixture,
        )

    def split(self, train_fraction: float, seed: int):
        train_idx, test_idx = split_indices(self.x.shape[0], train_fraction, seed)
        return self.subset(train_idx), self.subset(test_idx)


# -----------------------------------------------------------------------------
# Synthetic interventional DGP
# -----------------------------------------------------------------------------


def make_bimodal_shift(
    n: int,
    seed: int,
    d: int = 5,
    propensity_intercept: float = 0.10,
    propensity_scale: float = 0.80,
    left_weight_y0: float = 0.44,
    center_weight_y0: float = 0.44,
    left_weight_y1: float = 0.08,
    center_weight_y1: float = 0.44,
    side_location: float = 1.80,
    left_sd: float = 0.34,
    center_sd: float = 0.25,
    right_sd: float = 0.34,
):
    """Two-dimensional potential-outcome mixture with confounded treatment."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    logits = propensity_intercept + propensity_scale * (0.9 * x[:, 0] - 0.6 * x[:, 1] + 0.3 * x[:, 2])
    propensity = np.clip(sigmoid(logits), 0.05, 0.95)
    a = rng.binomial(1, propensity)

    right_weight_y0 = max(0.0, 1.0 - left_weight_y0 - center_weight_y0)
    right_weight_y1 = max(0.0, 1.0 - left_weight_y1 - center_weight_y1)
    w0 = np.array([left_weight_y0, center_weight_y0, right_weight_y0], dtype=float)
    w1 = np.array([left_weight_y1, center_weight_y1, right_weight_y1], dtype=float)
    w0 /= w0.sum()
    w1 /= w1.sum()

    means = np.array([[-side_location, 0.0], [0.0, 0.0], [side_location, 0.0]], dtype=float)
    sds = np.array([left_sd, center_sd, right_sd], dtype=float)

    comp0 = rng.choice(3, size=n, p=w0)
    comp1 = rng.choice(3, size=n, p=w1)
    y0 = means[comp0] + rng.normal(size=(n, 2)) * sds[comp0, None]
    y1 = means[comp1] + rng.normal(size=(n, 2)) * sds[comp1, None]
    y = np.where(a[:, None] == 1, y1, y0)

    mixture = {
        "w0": w0.tolist(),
        "w1": w1.tolist(),
        "means": means.tolist(),
        "sds": sds.tolist(),
    }
    return Sample(x, a, y, y0, y1, propensity, mixture)


# -----------------------------------------------------------------------------
# Outcome kernel and oracle nuisance functions
# -----------------------------------------------------------------------------


def kernel_features(y: np.ndarray, locations: np.ndarray, kernel_scale: float):
    """Gaussian kernel evaluations exp(-||y-v||^2/(2*kernel_scale))."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    loc = np.asarray(locations, dtype=float)
    if loc.ndim == 1:
        loc = loc[None, :]
    d2 = np.sum((y[:, None, :] - loc[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * d2 / max(float(kernel_scale), 1e-8))


def mixture_kernel_mean(locations: np.ndarray, weights: np.ndarray, means: np.ndarray, sds: np.ndarray, kernel_scale: float):
    """E[k(v,Y)] for a Gaussian mixture in R^2 and Gaussian outcome kernel."""
    loc = np.asarray(locations, dtype=float)
    if loc.ndim == 1:
        loc = loc[None, :]
    out = np.zeros(loc.shape[0], dtype=float)
    d = 2
    for w, mu, sd in zip(weights, means, sds):
        denom = kernel_scale + sd * sd
        const = (kernel_scale / denom) ** (d / 2.0)
        d2 = np.sum((loc - mu[None, :]) ** 2, axis=1)
        out += w * const * np.exp(-0.5 * d2 / denom)
    return out


# -----------------------------------------------------------------------------
# Nuisance estimation for estimated-nuisance DR-ME
# -----------------------------------------------------------------------------


class BasisParams:
    def __init__(self, mean: np.ndarray, scale: np.ndarray, anchors: np.ndarray, anchor_bw: float, interaction_dim: int):
        self.mean = mean
        self.scale = scale
        self.anchors = anchors
        self.anchor_bw = anchor_bw
        self.interaction_dim = interaction_dim


def median_sq_distance(x: np.ndarray):
    if len(x) < 2:
        return 1.0
    d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    vals = d2[np.triu_indices(len(x), k=1)]
    vals = vals[vals > 0]
    return float(np.median(vals)) if len(vals) else 1.0


def fit_basis_parameters(x_train: np.ndarray, n_anchors: int, interaction_dim: int, seed: int):
    x_train = np.asarray(x_train, dtype=float)
    mean = x_train.mean(axis=0)
    scale = np.where(x_train.std(axis=0) < 1e-6, 1.0, x_train.std(axis=0))
    xs = (x_train - mean) / scale
    rng = np.random.default_rng(seed)
    n_anchor = min(int(n_anchors), len(x_train))
    if n_anchor > 0:
        anchors = xs[rng.choice(len(xs), size=n_anchor, replace=False)]
    else:
        anchors = np.empty((0, xs.shape[1]))
    anchor_bw = max(median_sq_distance(xs), 1.0)
    return BasisParams(mean, scale, anchors, anchor_bw, min(int(interaction_dim), xs.shape[1]))


def build_design_matrix(x: np.ndarray, bp: BasisParams):
    x = np.asarray(x, dtype=float)
    xs = (x - bp.mean) / bp.scale
    pieces = [np.ones((len(x), 1)), xs, xs ** 2]

    if bp.interaction_dim >= 2:
        interactions = []
        for i in range(bp.interaction_dim):
            for j in range(i + 1, bp.interaction_dim):
                interactions.append((xs[:, i] * xs[:, j])[:, None])
        if interactions:
            pieces.append(np.hstack(interactions))

    if bp.anchors.size > 0:
        d2 = np.sum((xs[:, None, :] - bp.anchors[None, :, :]) ** 2, axis=2)
        pieces.append(np.exp(-0.5 * d2 / bp.anchor_bw))

    return np.hstack(pieces)


def fit_logistic_ridge(design: np.ndarray, a: np.ndarray, l2: float, max_iter: int = 100):
    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.ones_like(beta)
    penalty[0] = 0.0
    a = np.asarray(a, dtype=float)
    for _ in range(max_iter):
        p = np.clip(sigmoid(design @ beta), 1e-8, 1 - 1e-8)
        w = p * (1 - p)
        grad = design.T @ (p - a) + l2 * penalty * beta
        hess = (design.T * w) @ design + l2 * np.diag(penalty)
        step = np.linalg.solve(hess + 1e-8 * np.eye(hess.shape[0]), grad)
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


def prepare_folds(sample: Sample, n_folds: int, seed: int, n_basis_anchors: int, interaction_dim: int, propensity_l2: float, regression_l2: float):
    """Prepare cross-fitting folds and precomputed ridge smoothers."""
    n = len(sample.a)
    folds = split_folds(n, n_folds, seed)
    prepared = []
    for fid, hold in enumerate(folds):
        mask = np.ones(n, dtype=bool)
        mask[hold] = False
        train = np.flatnonzero(mask)
        if len(np.unique(sample.a[train])) < 2:
            raise ValueError("A training fold has only one treatment arm.")

        bp = fit_basis_parameters(sample.x[train], n_basis_anchors, interaction_dim, seed + 31 * (fid + 1))
        d_train = build_design_matrix(sample.x[train], bp)
        d_hold = build_design_matrix(sample.x[hold], bp)
        beta_pi = fit_logistic_ridge(d_train, sample.a[train], propensity_l2)
        pi_hat = np.clip(sigmoid(d_hold @ beta_pi), 0.01, 0.99)

        fold = {"holdout_idx": hold, "pi_hat": pi_hat, "arms": {}}
        for arm in (0, 1):
            arm_idx = train[sample.a[train] == arm]
            d_arm = build_design_matrix(sample.x[arm_idx], bp)
            gram = d_arm.T @ d_arm + regression_l2 * np.eye(d_arm.shape[1])
            gram[0, 0] -= regression_l2
            smoother = d_hold @ np.linalg.solve(gram + 1e-8 * np.eye(gram.shape[0]), d_arm.T)
            fold["arms"][arm] = {"idx": arm_idx, "H": smoother}
        prepared.append(fold)
    return prepared


def crossfit_dr_all_locations(sample: Sample, locations: np.ndarray, kernel_scale: float, folds: List[Dict]):
    targets = kernel_features(sample.y, locations, kernel_scale)
    z = np.zeros((len(sample.a), targets.shape[1]), dtype=float)
    for fold in folds:
        hold = fold["holdout_idx"]
        k_hold = targets[hold]
        a = sample.a[hold][:, None]
        pi = fold["pi_hat"][:, None]
        mu = {}
        for arm in (0, 1):
            mu[arm] = fold["arms"][arm]["H"] @ targets[fold["arms"][arm]["idx"]]
        z[hold] = (a / pi) * (k_hold - mu[1]) + mu[1] - ((1 - a) / (1 - pi)) * (k_hold - mu[0]) - mu[0]
    return z


def oracle_dr_all_locations(sample: Sample, locations: np.ndarray, kernel_scale: float):
    targets = kernel_features(sample.y, locations, kernel_scale)
    pi = sample.propensity[:, None]
    a = sample.a[:, None]
    mix = sample.mixture
    weights0 = np.asarray(mix["w0"], dtype=float)
    weights1 = np.asarray(mix["w1"], dtype=float)
    means = np.asarray(mix["means"], dtype=float)
    sds = np.asarray(mix["sds"], dtype=float)
    m0 = mixture_kernel_mean(locations, weights0, means, sds, kernel_scale)[None, :]
    m1 = mixture_kernel_mean(locations, weights1, means, sds, kernel_scale)[None, :]
    return (a / pi) * (targets - m1) + m1 - ((1 - a) / (1 - pi)) * (targets - m0) - m0


# -----------------------------------------------------------------------------
# Contour score evaluation
# -----------------------------------------------------------------------------


def score_grid_from_z(z_all: np.ndarray, n_grid: int, ridge: float):
    fixed = z_all[:, 0]
    scores = np.empty(n_grid, dtype=float)
    best_idx = 0
    best_value = -np.inf
    for j in range(n_grid):
        val = hotelling_statistic(np.column_stack([fixed, z_all[:, j + 1]]), ridge)
        scores[j] = val
        if val > best_value:
            best_value = val
            best_idx = j
    return scores, int(best_idx), float(best_value)


def compute_panel_grid(train: Sample, fixed_v1: np.ndarray, grid_points: np.ndarray, kernel_scale: float, nuisance: str, folds: List[Dict], ridge: float):
    locations = np.vstack([fixed_v1[None, :], grid_points])
    if nuisance == "estimated":
        z_all = crossfit_dr_all_locations(train, locations, kernel_scale, folds)
    elif nuisance == "oracle":
        z_all = oracle_dr_all_locations(train, locations, kernel_scale)
    else:
        raise ValueError("nuisance must be 'estimated' or 'oracle'.")
    scores, best_idx, best_value = score_grid_from_z(z_all, len(grid_points), ridge)
    return {"scores": scores, "best_idx": best_idx, "best_value": best_value}


# -----------------------------------------------------------------------------
# Figure generation
# -----------------------------------------------------------------------------


def make_figure(args: argparse.Namespace):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sample = make_bimodal_shift(
        n=args.n,
        seed=args.seed,
        left_weight_y0=args.left_weight_y0,
        center_weight_y0=args.center_weight_y0,
        left_weight_y1=args.left_weight_y1,
        center_weight_y1=args.center_weight_y1,
        side_location=args.side_location,
        left_sd=args.side_sd,
        center_sd=args.center_sd,
        right_sd=args.side_sd,
    )
    train, _ = sample.split(args.train_fraction, args.seed + 11)

    folds = prepare_folds(
        train,
        args.n_folds,
        args.seed + 19,
        args.n_basis_anchors,
        args.interaction_dim,
        args.propensity_l2,
        args.regression_l2,
    )

    x_values = np.linspace(args.xmin, args.xmax, args.grid_nx)
    y_values = np.linspace(args.ymin, args.ymax, args.grid_ny)
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    grid_points = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

    panels = [
        {"title": r"A. $v_1$ fixed near central overlap", "fixed_v1": np.array([args.center_fixed_x, args.center_fixed_y])},
        {"title": r"B. $v_1$ fixed in the left discrepancy region", "fixed_v1": np.array([-args.side_location + args.left_fixed_dx, args.left_fixed_y])},
    ]

    results = []
    all_est_scores = []
    for panel in panels:
        est_res = compute_panel_grid(
            train=train,
            fixed_v1=panel["fixed_v1"],
            grid_points=grid_points,
            kernel_scale=args.kernel_scale,
            nuisance="estimated",
            folds=folds,
            ridge=args.ridge,
        )
        oracle_res = compute_panel_grid(
            train=train,
            fixed_v1=panel["fixed_v1"],
            grid_points=grid_points,
            kernel_scale=args.kernel_scale,
            nuisance="oracle",
            folds=folds,
            ridge=args.ridge,
        )

        est_grid = est_res["scores"].reshape(mesh_x.shape)
        est_best_v2 = grid_points[est_res["best_idx"]]
        oracle_best_v2 = grid_points[oracle_res["best_idx"]]

        results.append(
            {
                "title": panel["title"],
                "fixed_v1": panel["fixed_v1"],
                "estimated_grid": est_grid,
                "estimated_best_v2": est_best_v2,
                "oracle_best_v2": oracle_best_v2,
                "estimated_best_objective": float(est_res["best_value"]),
                "oracle_best_objective": float(oracle_res["best_value"]),
            }
        )
        all_est_scores.append(est_res["scores"])

    all_est_scores = np.concatenate(all_est_scores)
    if args.shared_color_scale:
        vmin = float(np.quantile(all_est_scores, args.color_quantile_low))
        vmax = float(np.quantile(all_est_scores, args.color_quantile_high))
        levels = np.linspace(vmin, vmax, args.n_levels)
    else:
        vmin = vmax = None
        levels = args.n_levels

    rng = np.random.default_rng(args.seed + 101)
    n_points = min(args.oracle_points, sample.y0.shape[0])
    idx0 = rng.choice(sample.y0.shape[0], size=n_points, replace=False)
    idx1 = rng.choice(sample.y1.shape[0], size=n_points, replace=False)
    y0_points = sample.y0[idx0]
    y1_points = sample.y1[idx1]

    fig, axes = plt.subplots(1, 2, figsize=(args.fig_width, args.fig_height), sharex=True, sharey=True)
    contour_obj = None

    for ax, res in zip(axes, results):
        contour_obj = ax.contourf(
            mesh_x,
            mesh_y,
            res["estimated_grid"],
            levels=levels,
            cmap=args.cmap,
            extend="both",
        )

        if args.show_contour_lines:
            ax.contour(mesh_x, mesh_y, res["estimated_grid"], levels=8, colors="k", linewidths=0.25, alpha=0.30)

        ax.scatter(
            y0_points[:, 0],
            y0_points[:, 1],
            s=args.point_size,
            alpha=args.point_alpha,
            marker="o",
            linewidths=0.0,
            color="#2c7fb8",
            rasterized=True,
        )
        ax.scatter(
            y1_points[:, 0],
            y1_points[:, 1],
            s=args.point_size,
            alpha=args.point_alpha,
            marker="^",
            linewidths=0.0,
            color="#d95f0e",
            rasterized=True,
        )

        fx, fy = res["fixed_v1"]
        ex, ey = res["estimated_best_v2"]
        ox, oy = res["oracle_best_v2"]

        ax.scatter([fx], [fy], marker="P", s=args.fixed_marker_size, c="#ffcc00", edgecolors="black", linewidths=1.0, zorder=5)
        ax.scatter([ex], [ey], marker="*", s=args.estimated_marker_size, c="white", edgecolors="black", linewidths=1.0, zorder=7)
        ax.scatter([ox], [oy], marker="D", s=args.oracle_marker_size, facecolors="none", edgecolors="#7b3294", linewidths=2.0, zorder=8)

        ax.set_title(res["title"], fontsize=args.title_fontsize, pad=1.0)
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_ylim(args.ymin, args.ymax)
        ax.set_xlabel(r"$v_{2,1}$", fontsize=args.label_fontsize)
        ax.tick_params(labelsize=args.tick_fontsize)
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel(r"$v_{2,2}$", fontsize=args.label_fontsize)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#2c7fb8", markeredgecolor="none", alpha=0.55, label=r"$P_{Y(0)}$ samples", markersize=6.5),
        Line2D([0], [0], marker="^", linestyle="", markerfacecolor="#d95f0e", markeredgecolor="none", alpha=0.55, label=r"$P_{Y(1)}$ samples", markersize=6.5),
        Line2D([0], [0], marker="P", linestyle="", markerfacecolor="#ffcc00", markeredgecolor="black", color="black", label=r"Fixed $v_1$", markersize=8),
        Line2D([0], [0], marker="*", linestyle="", markerfacecolor="white", markeredgecolor="black", color="black", label=r"Estimated $\hat v_2^\star$", markersize=12),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor="none", markeredgecolor="#7b3294", color="#7b3294", label=r"Oracle $\hat v_{2,or}^\star$", markersize=7),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=True,
        fontsize=args.legend_fontsize,
        bbox_to_anchor=(0.5, 0.962),
        handletextpad=0.35,
        columnspacing=0.78,
        borderpad=0.22,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.12, top=0.915, wspace=0.08)
    cax = fig.add_axes([0.905, 0.14, 0.018, 0.69])
    cb = fig.colorbar(contour_obj, cax=cax)
    cb.set_label(r"Training objective $\widehat{\lambda}^{tr}(v_1,v_2)$", fontsize=args.colorbar_fontsize)
    cb.ax.tick_params(labelsize=args.tick_fontsize)

    png_path = outdir / args.output_png
    pdf_path = outdir / args.output_pdf
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    meta = {
        "parameters": vars(args),
        "train_n": int(train.x.shape[0]),
        "color_vmin": vmin,
        "color_vmax": vmax,
        "mixture": sample.mixture,
        "panels": [
            {
                "title": r["title"],
                "fixed_v1": r["fixed_v1"].tolist(),
                "estimated_best_v2": r["estimated_best_v2"].tolist(),
                "oracle_best_v2": r["oracle_best_v2"].tolist(),
                "estimated_best_objective": r["estimated_best_objective"],
                "oracle_best_objective": r["oracle_best_objective"],
                "estimated_score_min": float(np.min(r["estimated_grid"])),
                "estimated_score_max": float(np.max(r["estimated_grid"])),
            }
            for r in results
        ],
        "outputs": {"png": str(png_path), "pdf": str(pdf_path)},
    }

    with open(outdir / args.output_json, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(json.dumps(meta, indent=2, default=str))
    return meta


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Estimated-nuisance DR-ME contour with oracle maximizer marker only.")

    p.add_argument("--outdir", type=Path, default=results_dir("contour"))
    p.add_argument("--output-png", type=str, default="contour_estimated_with_oracle_v2_overlay_tight.png")
    p.add_argument("--output-pdf", type=str, default="contour_estimated_with_oracle_v2_overlay_tight.pdf")
    p.add_argument("--output-json", type=str, default="contour_estimated_with_oracle_v2_overlay_metadata.json")

    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--train-fraction", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=2)

    p.add_argument("--grid-nx", type=int, default=49)
    p.add_argument("--grid-ny", type=int, default=34)
    p.add_argument("--xmin", type=float, default=-2.7)
    p.add_argument("--xmax", type=float, default=2.7)
    p.add_argument("--ymin", type=float, default=-1.3)
    p.add_argument("--ymax", type=float, default=1.3)

    p.add_argument("--side-location", type=float, default=1.80)
    p.add_argument("--center-fixed-x", type=float, default=0.00)
    p.add_argument("--center-fixed-y", type=float, default=-0.05)
    p.add_argument("--left-fixed-dx", type=float, default=0.00)
    p.add_argument("--left-fixed-y", type=float, default=-0.06)

    p.add_argument("--left-weight-y0", type=float, default=0.44)
    p.add_argument("--center-weight-y0", type=float, default=0.44)
    p.add_argument("--left-weight-y1", type=float, default=0.08)
    p.add_argument("--center-weight-y1", type=float, default=0.44)
    p.add_argument("--side-sd", type=float, default=0.34)
    p.add_argument("--center-sd", type=float, default=0.25)

    p.add_argument("--kernel-scale", type=float, default=0.8, help="Squared Gaussian bandwidth in exp(-||y-v||^2/(2*scale)).")
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--n-folds", type=int, default=2)
    p.add_argument("--propensity-l2", type=float, default=1e-2)
    p.add_argument("--regression-l2", type=float, default=1.0)
    p.add_argument("--n-basis-anchors", type=int, default=16)
    p.add_argument("--interaction-dim", type=int, default=3)

    p.add_argument("--oracle-points", type=int, default=340)
    p.add_argument("--point-size", type=float, default=14.0)
    p.add_argument("--point-alpha", type=float, default=0.25)

    p.add_argument("--shared-color-scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--color-quantile-low", type=float, default=0.02)
    p.add_argument("--color-quantile-high", type=float, default=0.985)
    p.add_argument("--n-levels", type=int, default=22)
    p.add_argument("--cmap", type=str, default="RdYlBu_r")
    p.add_argument("--show-contour-lines", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--fig-width", type=float, default=10.3)
    p.add_argument("--fig-height", type=float, default=4.45)
    p.add_argument("--dpi", type=int, default=320)
    p.add_argument("--title-fontsize", type=float, default=12.5)
    p.add_argument("--label-fontsize", type=float, default=11.0)
    p.add_argument("--tick-fontsize", type=float, default=10.0)
    p.add_argument("--legend-fontsize", type=float, default=10.2)
    p.add_argument("--colorbar-fontsize", type=float, default=11.5)
    p.add_argument("--fixed-marker-size", type=float, default=95.0)
    p.add_argument("--estimated-marker-size", type=float, default=170.0)
    p.add_argument("--oracle-marker-size", type=float, default=70.0)

    return p.parse_args()


def main():
    args = parse_args()
    make_figure(args)


if __name__ == "__main__":
    main()
