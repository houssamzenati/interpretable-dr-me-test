#!/usr/bin/env python3
"""OCTMNIST residual-location figure."""

import argparse
import math
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression

from octmnist_utils import OCTMNIST_CACHE, load_octmnist, save_image
from repo_paths import results_dir

LABEL_DME = 1
LABEL_NORMAL = 3


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def oct_features(images):
    images = np.asarray(images, dtype=np.float64)
    n, h, w = images.shape
    rows = np.arange(h, dtype=np.float64)[:, None]
    cols = np.arange(w, dtype=np.float64)[None, :]
    mass = images.sum(axis=(1, 2)) + 1e-8
    cy = (images * rows[None]).sum(axis=(1, 2)) / mass / (h - 1)
    cx = (images * cols[None]).sum(axis=(1, 2)) / mass / (w - 1)
    mean = images.mean(axis=(1, 2))
    std = images.std(axis=(1, 2))
    top = images[:, : h // 3].mean(axis=(1, 2))
    mid = images[:, h // 3 : 2 * h // 3].mean(axis=(1, 2))
    bot = images[:, 2 * h // 3 :].mean(axis=(1, 2))
    row_profile = images.mean(axis=2)
    # Coarse vertical profile features are useful for OCT layers.
    bands = []
    for a, b in [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28)]:
        bands.append(row_profile[:, a:b].mean(axis=1))
    feats = np.column_stack([mean, std, cx, cy, top, mid, bot] + bands)
    feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-8)
    return feats


def build_dme_mask(
    normal_images,
    dme_images,
    mask_source="dme_positive",
    percentile=85.0,
    smooth_sigma=1.0,
    center_prior=True,
):
    h, w = normal_images.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]

    if mask_source == "ellipse":
        mask = np.exp(-0.5 * ((yy - 12.0) / 3.0) ** 2 - 0.5 * ((xx - 14.0) / 6.0) ** 2)
        mask = gaussian_filter(mask, sigma=smooth_sigma)
    elif mask_source == "dme_positive":
        mean_normal = normal_images.mean(axis=0)
        mean_dme = dme_images.mean(axis=0)
        diff = gaussian_filter(mean_dme - mean_normal, sigma=1.0)
        pos = np.maximum(diff, 0.0)
        if center_prior:
            prior = np.exp(-0.5 * ((yy - 11.5) / 5.0) ** 2 - 0.5 * ((xx - 14.0) / 8.0) ** 2)
            pos = pos * prior
        positive_vals = pos[pos > 0]
        if positive_vals.size == 0:
            mask = np.exp(-0.5 * ((yy - 12.0) / 3.0) ** 2 - 0.5 * ((xx - 14.0) / 6.0) ** 2)
        else:
            thr = float(np.percentile(positive_vals, percentile))
            mask = np.where(pos >= thr, pos, 0.0)
            mask = gaussian_filter(mask, sigma=smooth_sigma)
    else:
        raise ValueError(f"Unknown mask_source={mask_source}")

    mask = np.maximum(mask, 0.0)
    if mask.max() <= 1e-12:
        mask = np.exp(-0.5 * ((yy - 12.0) / 3.0) ** 2 - 0.5 * ((xx - 14.0) / 6.0) ** 2)
    mask = mask / (mask.max() + 1e-12)
    info = {
        "mask_source": mask_source,
        "mask_percentile": float(percentile),
        "mask_smooth_sigma": float(smooth_sigma),
        "center_prior": bool(center_prior),
        "mask_mean": float(mask.mean()),
        "mask_max": float(mask.max()),
    }
    return mask, info


def ridge_projection(design, ridge_alpha):
    q = design.shape[1]
    gram = design.T @ design + ridge_alpha * np.eye(q)
    return np.linalg.solve(gram, design.T)


def make_design(X):
    return np.column_stack([np.ones(len(X)), X])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", type=str, default=None, help="Path to octmnist.npz. If omitted, download from Zenodo.")
    parser.add_argument("--data-home", type=str, default=str(OCTMNIST_CACHE))
    parser.add_argument("--outdir", type=str, default=str(results_dir("octmnist_results")))
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--theta", type=float, default=0.60, help="Severe residual strength.")
    parser.add_argument("--rare-prob", type=float, default=0.25, help="Probability of severe treated residual.")
    parser.add_argument("--noise-std", type=float, default=0.025)
    parser.add_argument("--effect-noise-std", type=float, default=0.20,
                        help="Std. dev. of lognormal mean-one amplitude noise on the structured treatment effect.")
    parser.add_argument("--ell", type=float, default=0.16, help="Gaussian kernel bandwidth in residual-image space.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--mask-source", choices=["dme_positive", "ellipse"], default="dme_positive")
    parser.add_argument("--mask-percentile", type=float, default=85.0)
    parser.add_argument("--mask-sigma", type=float, default=1.0)
    parser.add_argument("--no-center-prior", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--smooth-step", type=float, default=0.55)
    parser.add_argument("--anchor", type=float, default=0.03, help="Convex pullback to initialization after each step.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    images, labels = load_octmnist(args.source_path, Path(args.data_home))
    normal = images[labels == LABEL_NORMAL]
    dme = images[labels == LABEL_DME]
    if len(normal) == 0 or len(dme) == 0:
        raise ValueError("OCTMNIST normal or DME class is empty.")

    n = min(args.n, len(normal))
    baseline_idx = rng.choice(len(normal), size=n, replace=False)
    B = normal[baseline_idx].copy()

    M, mask_info = build_dme_mask(
        normal,
        dme,
        mask_source=args.mask_source,
        percentile=args.mask_percentile,
        smooth_sigma=args.mask_sigma,
        center_prior=not args.no_center_prior,
    )

    q = float(args.rare_prob)
    theta = float(args.theta)
    mild = q * theta
    effect_noise_std = float(args.effect_noise_std)

    eps0 = rng.normal(0.0, args.noise_std, size=B.shape)
    eps1 = rng.normal(0.0, args.noise_std, size=B.shape)
    severe_indicator = rng.binomial(1, q, size=n).astype(float)
    # Add stochasticity to the structured effect itself via mean-one amplitude
    # jitter, so the treatment pattern is not a deterministic copy of M.
    jitter0 = np.exp(effect_noise_std * rng.normal(size=n) - 0.5 * effect_noise_std**2)
    jitter1 = np.exp(effect_noise_std * rng.normal(size=n) - 0.5 * effect_noise_std**2)

    R0 = mild * jitter0[:, None, None] * M[None] + eps0
    R1 = severe_indicator[:, None, None] * theta * jitter1[:, None, None] * M[None] + eps1
    # Images for display only; learning happens on residuals R.
    Y0 = np.clip(B + R0, 0.0, 1.0)
    Y1 = np.clip(B + R1, 0.0, 1.0)

    X = oct_features(B)
    beta_true = np.array([0.9, -0.6, 0.7, -0.5, 0.4, -0.4, 0.55, 0.4, -0.3, 0.2, 0.3, -0.2, 0.25, -0.15], dtype=float)
    if beta_true.size < X.shape[1]:
        beta_true = np.concatenate([beta_true, np.zeros(X.shape[1] - beta_true.size)])
    beta_true = beta_true[: X.shape[1]]
    logit = X @ beta_true
    logit = 0.90 * logit / (np.std(logit) + 1e-8)
    pi_true = np.clip(sigmoid(logit), 0.07, 0.93)
    A = rng.binomial(1, pi_true, size=n).astype(float)

    R = np.where(A[:, None, None] == 1, R1, R0)
    Y = np.where(A[:, None, None] == 1, Y1, Y0)
    perm = rng.permutation(n)
    n_eta = int(0.35 * n)
    n_train = int(0.40 * n)
    eta_idx = perm[:n_eta]
    train_idx = perm[n_eta : n_eta + n_train]
    test_idx = perm[n_eta + n_train :]

    X_eta, X_train, X_test = X[eta_idx], X[train_idx], X[test_idx]
    A_eta, A_train, A_test = A[eta_idx], A[train_idx], A[test_idx]
    R_eta, R_train, R_test = R[eta_idx], R[train_idx], R[test_idx]

    # Propensity nuisance.
    prop_model = LogisticRegression(max_iter=2000, C=1.0)
    prop_model.fit(X_eta, A_eta.astype(int))
    pi_hat_train = np.clip(prop_model.predict_proba(X_train)[:, 1], 0.05, 0.95)
    pi_hat_test = np.clip(prop_model.predict_proba(X_test)[:, 1], 0.05, 0.95)

    # Outcome-kernel nuisance smoothers.
    D_eta = make_design(X_eta)
    D_train = make_design(X_train)
    D_test = make_design(X_test)
    arm0_eta = A_eta == 0
    arm1_eta = A_eta == 1
    if arm0_eta.sum() < 5 or arm1_eta.sum() < 5:
        raise RuntimeError("Too few nuisance samples in one treatment arm. Increase n or change seed.")
    P0 = ridge_projection(D_eta[arm0_eta], args.ridge_alpha)
    P1 = ridge_projection(D_eta[arm1_eta], args.ridge_alpha)
    H0_train = D_train @ P0
    H1_train = D_train @ P1
    H0_test = D_test @ P0
    H1_test = D_test @ P1

    R_eta_flat = R_eta.reshape(len(R_eta), -1)
    R_eta0 = R_eta_flat[arm0_eta]
    R_eta1 = R_eta_flat[arm1_eta]
    R_train_flat = R_train.reshape(len(R_train), -1)
    R_test_flat = R_test.reshape(len(R_test), -1)

    observed_resid_train_a0 = R_train[A_train == 0].mean(axis=0)
    observed_resid_train_a1 = R_train[A_train == 1].mean(axis=0)
    mean_resid_diff_train = observed_resid_train_a1 - observed_resid_train_a0

    p = R_train_flat.shape[1]
    beta2 = p * args.ell**2
    tau = 1e-6

    def score_block(v_img, R_hold_flat, A_hold, pi_hold, H0_hold, H1_hold, need_grad=False):
        v = np.asarray(v_img, dtype=float).reshape(-1)
        diff_hold = v[None] - R_hold_flat
        k_hold = np.exp(-np.sum(diff_hold * diff_hold, axis=1) / (2.0 * beta2))

        diff0 = v[None] - R_eta0
        t0 = np.exp(-np.sum(diff0 * diff0, axis=1) / (2.0 * beta2))
        m0 = H0_hold @ t0

        diff1 = v[None] - R_eta1
        t1 = np.exp(-np.sum(diff1 * diff1, axis=1) / (2.0 * beta2))
        m1 = H1_hold @ t1

        z = A_hold / pi_hold * (k_hold - m1) + m1 - (1.0 - A_hold) / (1.0 - pi_hold) * (k_hold - m0) - m0
        mean_z = float(z.mean())
        zc = z - mean_z
        var_z = float(np.mean(zc * zc))
        denom = math.sqrt(var_z + tau)
        score = mean_z / denom
        if not need_grad:
            return score, {"score": score, "mean_z": mean_z, "sd_z": math.sqrt(var_z)}

        grad_k = -(diff_hold / beta2) * k_hold[:, None]
        grad_t0 = -(diff0 / beta2) * t0[:, None]
        grad_t1 = -(diff1 / beta2) * t1[:, None]
        grad_m0 = H0_hold @ grad_t0
        grad_m1 = H1_hold @ grad_t1
        grad_z = (
            A_hold[:, None] / pi_hold[:, None] * (grad_k - grad_m1)
            + grad_m1
            - (1.0 - A_hold)[:, None] / (1.0 - pi_hold)[:, None] * (grad_k - grad_m0)
            - grad_m0
        )
        grad_mean = grad_z.mean(axis=0)
        grad_var = 2.0 * np.mean(zc[:, None] * grad_z, axis=0)
        grad_score = grad_mean / denom - mean_z * grad_var / (2.0 * denom**3)
        return score, grad_score.reshape(28, 28), {"score": score, "mean_z": mean_z, "sd_z": math.sqrt(var_z)}

    def train_score(v):
        return score_block(v, R_train_flat, A_train, pi_hat_train, H0_train, H1_train, need_grad=False)[0]

    def test_info(v):
        return score_block(v, R_test_flat, A_test, pi_hat_test, H0_test, H1_test, need_grad=False)[1]

    # Initialize from the observed residual mean difference on I_tr only.
    v_init = mean_resid_diff_train.copy()
    init_label = "training-split mean residual difference"
    init_train_score = float(train_score(v_init))
    init_test_score = float(test_info(v_init)["score"])

    # Smooth gradient ascent.
    v = v_init.copy()
    best_v = v.copy()
    best_train_score = init_train_score
    lr = float(args.learning_rate)
    history = []
    vmax_clip = max(theta * 1.25, float(np.max(np.abs(v_init))) * 1.25, 1e-6)
    for step in range(args.steps):
        current_score, grad, _ = score_block(v, R_train_flat, A_train, pi_hat_train, H0_train, H1_train, need_grad=True)
        direction = grad / (np.linalg.norm(grad) + 1e-12)
        proposal = v + lr * direction
        proposal = gaussian_filter(proposal, sigma=args.smooth_step)
        proposal = (1.0 - args.anchor) * proposal + args.anchor * v_init
        proposal = np.clip(proposal, -vmax_clip, vmax_clip)
        proposal_score = train_score(proposal)
        if proposal_score >= current_score:
            v = proposal
            lr = min(0.55, lr * 1.04 + 0.004)
            if proposal_score > best_train_score:
                best_train_score = float(proposal_score)
                best_v = v.copy()
        else:
            lr = max(0.03, lr * 0.70)
        history.append({"step": step + 1, "score": float(current_score), "proposal_score": float(proposal_score), "lr": float(lr)})

    v_star = best_v
    vstar_train = score_block(v_star, R_train_flat, A_train, pi_hat_train, H0_train, H1_train, need_grad=False)[1]
    vstar_test = test_info(v_star)

    # Diagnostics.
    oracle_R0_mean = R0.mean(axis=0)
    oracle_R1_mean = R1.mean(axis=0)
    oracle_resid_diff = oracle_R1_mean - oracle_R0_mean
    severe_indices = np.flatnonzero(severe_indicator == 1)
    mild_indices = np.flatnonzero(severe_indicator == 0)
    mild_idx = int(mild_indices[0]) if len(mild_indices) else 0
    severe_idx = int(severe_indices[0]) if len(severe_indices) else mild_idx
    sample_y0 = Y0[mild_idx]
    sample_y1 = Y1[severe_idx]
    example_control = R0[mild_idx]
    example_severe = R1[severe_idx]
    residual_vmax = max(
        float(np.abs(theta * M).max()),
        float(np.abs(v_star).max()),
        float(np.abs(mean_resid_diff_train).max()),
        float(np.abs(example_control).max()),
        float(np.abs(example_severe).max()),
        1e-6,
    )

    # Main paper panel: horizontal 1x4.
    panel_path = outdir / "octmnist_mean_matched_panel.png"
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.8))

    def show(ax, img, title, cmap="gray", vmin=None, vmax=None):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9.5)
        ax.axis("off")

    show(axes[0], sample_y0, "Sample $Y(0)$", vmin=0.0, vmax=1.0)
    show(axes[1], sample_y1, "Sample $Y(1)$", vmin=0.0, vmax=1.0)
    show(
        axes[2],
        mean_resid_diff_train,
        "Observed residual mean diff",
        cmap="coolwarm",
        vmin=-residual_vmax,
        vmax=residual_vmax,
    )
    show(
        axes[3],
        v_star,
        f"Learned $v^*$\ntrain={vstar_train['score']:.2f}, test={vstar_test['score']:.2f}",
        cmap="coolwarm",
        vmin=-residual_vmax,
        vmax=residual_vmax,
    )
    fig.tight_layout()
    fig.savefig(panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Appendix diagnostics panel with residual examples and mean-matched structure.
    appendix_panel_path = outdir / "octmnist_mean_matched_appendix_panel.png"
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.0))
    axs = axes.ravel()
    show(axs[0], sample_y0, "Sample $Y(0)$", vmin=0.0, vmax=1.0)
    show(axs[1], sample_y1, "Sample $Y(1)$", vmin=0.0, vmax=1.0)
    show(axs[2], example_control, "Sample $R(0)$: mild", cmap="coolwarm", vmin=-residual_vmax, vmax=residual_vmax)
    show(axs[3], example_severe, "Sample $R(1)$: severe", cmap="coolwarm", vmin=-residual_vmax, vmax=residual_vmax)
    show(axs[4], mean_resid_diff_train, "Observed residual mean diff", cmap="coolwarm", vmin=-residual_vmax, vmax=residual_vmax)
    show(axs[5], oracle_resid_diff, "Oracle residual mean diff", cmap="coolwarm", vmin=-residual_vmax, vmax=residual_vmax)
    show(axs[6], v_init, "Mean residual difference init", cmap="coolwarm", vmin=-residual_vmax, vmax=residual_vmax)
    show(
        axs[7],
        v_star,
        f"Learned $v^*$\ntrain={vstar_train['score']:.2f}, test={vstar_test['score']:.2f}",
        cmap="coolwarm",
        vmin=-residual_vmax,
        vmax=residual_vmax,
    )
    fig.tight_layout()
    fig.savefig(appendix_panel_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    vstar_path = outdir / "octmnist_mean_matched_vstar.png"
    init_path = outdir / "octmnist_mean_matched_init.png"
    save_image(str(vstar_path), v_star, "Learned v*", cmap="coolwarm", vmin=-residual_vmax, vmax=residual_vmax)
    save_image(str(init_path), v_init, "Mean residual difference init", cmap="coolwarm", vmin=-residual_vmax, vmax=residual_vmax)
    np.save(outdir / "octmnist_mean_matched_vstar.npy", v_star.astype(np.float32))

    metadata = {
        "n": int(n),
        "n_eta": int(n_eta),
        "n_train": int(n_train),
        "n_test": int(len(test_idx)),
        "theta": float(theta),
        "rare_prob": float(q),
        "mild_strength": float(mild),
        "noise_std": float(args.noise_std),
        "effect_noise_std": effect_noise_std,
        "effect_noise_model": "lognormal_mean_one_amplitude_jitter",
        "ell": float(args.ell),
        "ridge_alpha": float(args.ridge_alpha),
        "init_label": init_label,
        "init_train_score": init_train_score,
        "init_test_score": init_test_score,
        "vstar_train_score": float(vstar_train["score"]),
        "vstar_test_score": float(vstar_test["score"]),
        "vstar_train_mean_z": float(vstar_train["mean_z"]),
        "vstar_test_mean_z": float(vstar_test["mean_z"]),
        "oracle_mean_diff_l2": float(np.linalg.norm((oracle_R1_mean - oracle_R0_mean).reshape(-1))),
        "observed_resid_diff_l2": float(np.linalg.norm(mean_resid_diff_train.reshape(-1))),
        "observed_train_resid_diff_l2": float(np.linalg.norm(mean_resid_diff_train.reshape(-1))),
        **mask_info,
    }
    # Package outputs.
    package_path = outdir / "octmnist_mean_matched_package.zip"
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(__file__, arcname="octmnist_mean_matched_vstar.py")
        for filename in [
            "octmnist_mean_matched_panel.png",
            "octmnist_mean_matched_appendix_panel.png",
            "octmnist_mean_matched_vstar.png",
            "octmnist_mean_matched_init.png",
            "octmnist_mean_matched_vstar.npy",
        ]:
            zf.write(outdir / filename, arcname=f"results/{filename}")

    print("Saved:")
    print(panel_path)
    print(appendix_panel_path)
    print(vstar_path)
    print(init_path)
    print(package_path)
    print("Metadata:")
    print(metadata)


if __name__ == "__main__":
    main()
