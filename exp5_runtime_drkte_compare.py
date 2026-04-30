#!/usr/bin/env python3

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as sla

from repo_paths import results_dir


@dataclass
class Config:
    reps: int = 10
    seed_base: int = 270000
    outdir: str = str(results_dir("exp5"))
    n_grid: tuple = (300, 600, 1200, 3000)
    J_grid: tuple = (1, 2, 5, 10)
    M_grid: tuple = (40, 80, 160, 320)
    fixed_n: int = 1200
    fixed_J: int = 2
    fixed_M: int = 80
    grad_steps: int = 30
    grad_lr_scale: float = 0.08
    tau: float = 1e-3
    gamma: float = 1e-8
    p: int = 5
    ridge_alpha: float = 1e-2
    prop_l2: float = 1e-2
    prop_clip: float = 0.05
    setting: str = "localized_bump"
    bump_shift: float = 1.8
    bump_prob: float = 0.18
    variance_scale: float = 1.45
    mean_shift: float = 0.4


def sigmoid(x):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def prognostic_g(X):
    return (
        0.90 * X[:, 0]
        + 0.60 * np.sin(X[:, 1])
        + 0.35 * (X[:, 2] ** 2 - 1.0)
        + 0.25 * X[:, 0] * X[:, 3]
        - 0.20 * np.cos(X[:, 4])
    )


def true_propensity(X):
    logits = 0.90 * X[:, 0] - 0.75 * X[:, 1] + 0.55 * X[:, 2] - 0.40 * X[:, 3]
    return np.clip(sigmoid(logits), 0.06, 0.94)


def simulate_observational_sample(n, seed, cfg):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, cfg.p))
    pi = true_propensity(X)
    A = rng.binomial(1, pi).astype(int)
    base = prognostic_g(X)

    eps0 = rng.normal(size=n)
    eps1 = rng.normal(size=n)
    Y0 = base + eps0

    if cfg.setting == "null":
        Y1 = base + eps1
    elif cfg.setting == "mean_shift":
        Y1 = base + eps1 + cfg.mean_shift
    elif cfg.setting == "variance_shift":
        Y1 = base + cfg.variance_scale * eps1
    else:
        q = cfg.bump_prob
        B = rng.binomial(1, q, size=n).astype(float)
        bump = cfg.bump_shift * (B - q) / math.sqrt(max(q * (1.0 - q), 1e-8))
        Y1 = base + eps1 + bump

    Y = np.where(A == 1, Y1, Y0).reshape(-1, 1)
    return X, A, Y


def squared_distances(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    AA = np.sum(A * A, axis=1)[:, None]
    BB = np.sum(B * B, axis=1)[None, :]
    return np.maximum(AA + BB - 2.0 * (A @ B.T), 0.0)


def gaussian_kernel_y(Y, V, ell):
    return np.exp(-squared_distances(Y, np.asarray(V).reshape(-1, 1)) / (2.0 * ell * ell))


def gaussian_kernel_y_derivative(Y, V, ell):
    Y = np.asarray(Y).reshape(-1, 1)
    V = np.asarray(V).reshape(1, -1)
    K = np.exp(-0.5 * ((Y - V) / ell) ** 2)
    return ((Y - V) / (ell * ell)) * K


def median_bandwidth_y(Y, rng, max_points=700):
    Y = np.asarray(Y, float).reshape(len(Y), -1)
    if len(Y) > max_points:
        keep = rng.choice(len(Y), size=max_points, replace=False)
        Y = Y[keep]
    if len(Y) <= 1:
        return 1.0
    D = np.sqrt(squared_distances(Y, Y))
    vals = D[np.triu_indices(len(Y), k=1)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    return float(max(np.median(vals), 1e-3))


def safe_solve(A, B):
    try:
        factor = sla.cho_factor(A, check_finite=False)
        return sla.cho_solve(factor, B, check_finite=False)
    except Exception:
        try:
            return sla.solve(A, B, assume_a="sym", check_finite=False)
        except Exception:
            return np.linalg.pinv(A) @ B


def basis_features(X):
    X = np.asarray(X, float)
    n, p = X.shape
    parts = [X, X ** 2 - 1.0, np.sin(X), np.cos(X)]
    interactions = []
    for j in range(p):
        for k in range(j + 1, p):
            interactions.append((X[:, j] * X[:, k]).reshape(n, 1))
    if interactions:
        parts.append(np.hstack(interactions))
    return np.hstack(parts)


def standardize_with_intercept(Phi, mean=None, scale=None):
    if mean is None:
        mean = Phi.mean(axis=0, keepdims=True)
    if scale is None:
        scale = Phi.std(axis=0, keepdims=True)
        scale = np.where(scale < 1e-8, 1.0, scale)
    Phi_std = (Phi - mean) / scale
    Phi1 = np.hstack([np.ones((Phi.shape[0], 1)), Phi_std])
    return Phi1, mean, scale


def fit_logit(X, A, l2, max_iter=40):
    Phi, mean, scale = standardize_with_intercept(X.astype(float))
    y = A.astype(float)
    beta = np.zeros(Phi.shape[1])
    penalty = l2 * np.eye(Phi.shape[1])
    penalty[0, 0] = 0.0

    for _ in range(max_iter):
        p = np.clip(sigmoid(Phi @ beta), 1e-6, 1.0 - 1e-6)
        W = p * (1.0 - p)
        H = Phi.T @ (Phi * W[:, None]) + penalty
        grad = Phi.T @ (y - p) - penalty @ beta
        step = safe_solve(H, grad)
        beta = beta + step
        if np.linalg.norm(step) < 1e-8 * (1.0 + np.linalg.norm(beta)):
            break

    return {"beta": beta, "mean": mean, "scale": scale}


def predict_logit(model, X, clip):
    Phi, _, _ = standardize_with_intercept(X.astype(float), model["mean"], model["scale"])
    p = sigmoid(Phi @ model["beta"])
    return np.clip(p, clip, 1.0 - clip)


class Nuisance:
    def __init__(self, X_eta, A_eta, Y_eta, X_tr, A_tr, Y_tr, X_te, A_te, Y_te, cfg, rng):
        self.X_tr = X_tr
        self.A_tr = A_tr
        self.Y_tr = Y_tr
        self.X_te = X_te
        self.A_te = A_te
        self.Y_te = Y_te
        self.cfg = cfg

        self.ell = median_bandwidth_y(np.vstack([Y_eta, Y_tr]), rng)
        self.pi_model = fit_logit(X_eta, A_eta, cfg.prop_l2)
        self.pi_tr = predict_logit(self.pi_model, X_tr, cfg.prop_clip)
        self.pi_te = predict_logit(self.pi_model, X_te, cfg.prop_clip)

        Phi_eta, self.feature_mean, self.feature_scale = standardize_with_intercept(basis_features(X_eta))
        self.Phi_tr, _, _ = standardize_with_intercept(
            basis_features(X_tr), self.feature_mean, self.feature_scale
        )
        self.Phi_te, _, _ = standardize_with_intercept(
            basis_features(X_te), self.feature_mean, self.feature_scale
        )

        self.arm = {}
        for a in (0, 1):
            mask = A_eta == a
            Phi_a = Phi_eta[mask]
            Y_a = Y_eta[mask]
            if len(Y_a) < 5:
                Phi_a = Phi_eta
                Y_a = Y_eta
            G = Phi_a.T @ Phi_a
            penalty = cfg.ridge_alpha * np.eye(G.shape[0])
            penalty[0, 0] = 1e-10
            B = safe_solve(G + penalty, Phi_a.T)
            self.arm[a] = {"B": B, "Y": Y_a}

    def m(self, split, a, V):
        Phi = self.Phi_tr if split == "tr" else self.Phi_te
        U = gaussian_kernel_y(self.arm[a]["Y"], V, self.ell)
        return Phi @ self.arm[a]["B"] @ U

    def dm(self, split, a, V):
        Phi = self.Phi_tr if split == "tr" else self.Phi_te
        U = gaussian_kernel_y_derivative(self.arm[a]["Y"], V, self.ell)
        return Phi @ self.arm[a]["B"] @ U

    def z(self, split, V):
        V = np.asarray(V, float).reshape(-1)
        if split == "tr":
            A, Y, pi = self.A_tr, self.Y_tr, self.pi_tr
        else:
            A, Y, pi = self.A_te, self.Y_te, self.pi_te
        K = gaussian_kernel_y(Y, V, self.ell)
        m1 = self.m(split, 1, V)
        m0 = self.m(split, 0, V)
        a = A.reshape(-1, 1)
        p = pi.reshape(-1, 1)
        return (a / p) * (K - m1) + m1 - ((1.0 - a) / (1.0 - p)) * (K - m0) - m0

    def z_and_gradient(self, split, V):
        V = np.asarray(V, float).reshape(-1)
        if split == "tr":
            A, Y, pi = self.A_tr, self.Y_tr, self.pi_tr
        else:
            A, Y, pi = self.A_te, self.Y_te, self.pi_te

        K = gaussian_kernel_y(Y, V, self.ell)
        dK = gaussian_kernel_y_derivative(Y, V, self.ell)
        m1 = self.m(split, 1, V)
        m0 = self.m(split, 0, V)
        dm1 = self.dm(split, 1, V)
        dm0 = self.dm(split, 0, V)

        a = A.reshape(-1, 1)
        p = pi.reshape(-1, 1)
        Z = (a / p) * (K - m1) + m1 - ((1.0 - a) / (1.0 - p)) * (K - m0) - m0

        grad_terms = []
        for j in range(len(V)):
            dZ = np.zeros_like(Z)
            dZ[:, j] = (
                (A / pi) * (dK[:, j] - dm1[:, j])
                + dm1[:, j]
                - ((1.0 - A) / (1.0 - pi)) * (dK[:, j] - dm0[:, j])
                - dm0[:, j]
            )
            grad_terms.append(dZ)

        return Z, grad_terms


def mean_cov(Z):
    mu = Z.mean(axis=0)
    Zc = Z - mu
    S = Zc.T @ Zc / max(len(Z) - 1, 1)
    return mu, S, Zc


def location_objective(Z, tau):
    J = Z.shape[1]
    mu, S, _ = mean_cov(Z)
    return float(mu @ np.linalg.pinv(S + tau * np.eye(J)) @ mu)


def objective_gradient(nuisance, V, tau):
    Z, dZ_terms = nuisance.z_and_gradient("tr", V)
    n, J = Z.shape
    mu, S, Zc = mean_cov(Z)
    inv = np.linalg.pinv(S + tau * np.eye(J))
    inv_mu = inv @ mu
    value = float(mu @ inv_mu)
    grad = np.zeros(J)

    for j, dZ in enumerate(dZ_terms):
        dmu = dZ.mean(axis=0)
        dZc = dZ - dmu
        dS = (dZc.T @ Zc + Zc.T @ dZc) / max(n - 1, 1)
        grad[j] = float(2.0 * dmu @ inv_mu - inv_mu @ dS @ inv_mu)

    return value, grad


def make_dictionary(Y, M):
    y = Y.ravel()
    lo, hi = np.quantile(y, [0.02, 0.98])
    span = max(float(hi - lo), 1e-3)
    return np.linspace(lo - 0.05 * span, hi + 0.05 * span, M)


def greedy_select(ZC, J, tau):
    selected = []
    available = set(range(ZC.shape[1]))
    for _ in range(J):
        best = max(available, key=lambda c: location_objective(ZC[:, selected + [c]], tau))
        selected.append(int(best))
        available.remove(best)
    return selected


def learn_dictionary_locations(nuisance, J, M, tau):
    start = time.perf_counter()
    C = make_dictionary(nuisance.Y_tr, M)
    ZC = nuisance.z("tr", C)
    precompute = time.perf_counter() - start

    start = time.perf_counter()
    cols = greedy_select(ZC, J, tau)
    search = time.perf_counter() - start

    return C[cols], {"precompute_sec": precompute, "search_sec": search}


def learn_gradient_locations(nuisance, J, steps, lr_scale, tau, rng):
    qs = np.linspace(0.1, 0.9, max(J + 2, 5))[1:-1][:J]
    V = np.quantile(nuisance.Y_tr.ravel(), qs) + 0.05 * nuisance.ell * rng.normal(size=J)

    lo, hi = np.quantile(nuisance.Y_tr.ravel(), [0.005, 0.995])
    span = max(float(hi - lo), 1.0)
    lo = lo - 0.25 * span
    hi = hi + 0.25 * span

    lr = lr_scale * nuisance.ell
    m = np.zeros(J)
    v = np.zeros(J)
    best = V.copy()
    best_value = -np.inf
    start = time.perf_counter()

    for t in range(1, steps + 1):
        value, grad = objective_gradient(nuisance, V, tau)
        if np.isfinite(value) and value > best_value:
            best_value = value
            best = V.copy()
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 100:
            grad = grad * (100.0 / grad_norm)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad * grad)
        mhat = m / (1.0 - 0.9 ** t)
        vhat = v / (1.0 - 0.999 ** t)
        V = V + lr * mhat / (np.sqrt(vhat) + 1e-8)
        V = np.clip(V, lo, hi)

    search = time.perf_counter() - start
    return best, {"precompute_sec": 0.0, "search_sec": search}


def median_sqdist(X, max_points=700, seed=0):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(X) > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(X), size=max_points, replace=False)
        X = X[keep]
    if len(X) <= 1:
        return 1.0
    D = np.sqrt(squared_distances(X, X))
    vals = D[np.triu_indices(len(X), k=1)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    med = float(np.median(vals))
    return max(med * med, 1e-6)


def rbf_gamma(A, B, gamma):
    return np.exp(-float(gamma) * squared_distances(A, B))


def normal_survival(x):
    return 0.5 * math.erfc(float(x) / math.sqrt(2.0))


def dr_xkte_stat(Y, w, Xcov, T, ell_y, seed=0):
    Y = np.asarray(Y, float)
    Xcov = np.asarray(Xcov, float)
    T = np.asarray(T).astype(int).ravel()
    w = np.asarray(w, float).ravel()

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    N = len(Y)
    if N % 2 == 1:
        Y = Y[:-1]
        Xcov = Xcov[:-1]
        T = T[:-1]
        w = w[:-1]
        N -= 1
    if N < 20:
        return np.nan, np.nan

    N2 = N // 2
    Ta, Tb = T[:N2], T[N2:]
    Ya, Yb = Y[:N2], Y[N2:]
    Xa, Xb = Xcov[:N2], Xcov[N2:]

    Y0a = Ya[Ta == 0]
    Y1a = Ya[Ta == 1]
    Y0b = Yb[Tb == 0]
    Y1b = Yb[Tb == 1]
    X0a = Xa[Ta == 0]
    X1a = Xa[Ta == 1]
    X0b = Xb[Tb == 0]
    X1b = Xb[Tb == 1]

    if min(len(Y0a), len(Y1a), len(Y0b), len(Y1b)) < 3:
        return np.nan, np.nan

    Y_ord = np.vstack((Y0a, Y1a, Y0b, Y1b))
    X_ord = np.vstack((X0a, X1a, X0b, X1b))

    m1 = len(Y0a)
    n1 = len(Y1a)
    m0b = len(Y0b)
    n1b = len(Y1b)
    m = m1 + m0b
    nt = n1 + n1b

    wa = np.clip(w[:N2], 0.03, 0.97)
    wb = np.clip(w[N2:], 0.03, 0.97)
    ww = np.concatenate(
        (
            -1.0 / (1.0 - wa[Ta == 0]),
            1.0 / wa[Ta == 1],
            -1.0 / (1.0 - wb[Tb == 0]),
            1.0 / wb[Tb == 1],
        )
    )

    sigma_kx = median_sqdist(X_ord[N2:], seed=seed)
    KX = rbf_gamma(X_ord, X_ord, gamma=1.0 / sigma_kx)
    reg = sigma_kx

    mu0a = safe_solve(KX[:m1, :m1] + reg * np.eye(m1), KX[:m1, :m1 + n1])
    z0a = np.vstack((mu0a, np.zeros((n1, m1 + n1))))

    mu1a = safe_solve(
        KX[m1:m1 + n1, m1:m1 + n1] + reg * np.eye(n1),
        KX[m1:m1 + n1, :m1 + n1],
    )
    z1a = np.vstack((np.zeros((m1, m1 + n1)), mu1a))
    muAa = np.hstack((z0a[:, :m1], z1a[:, m1:m1 + n1]))

    mu0b = safe_solve(
        KX[m1 + n1:m + n1, m1 + n1:m + n1] + reg * np.eye(m0b),
        KX[m1 + n1:m + n1, m1 + n1:],
    )
    z0b = np.vstack((mu0b, np.zeros((n1b, m + nt - m1 - n1))))

    mu1b = safe_solve(KX[m + n1:, m + n1:] + reg * np.eye(n1b), KX[m + n1:, m1 + n1:])
    z1b = np.vstack((np.zeros((m0b, m + nt - m1 - n1)), mu1b))
    muAb = np.hstack((z0b[:, :m0b], z1b[:, m0b:]))

    left = z1a - z0a + ww[:N2, None] * (np.eye(m1 + n1) - muAa)
    right = z1b - z0b + ww[N2:, None] * (np.eye(m + nt - m1 - n1) - muAb)
    KY = gaussian_kernel_y(Y_ord[:N2], Y_ord[N2:].reshape(-1), ell_y)

    prod = left.T @ KY @ right
    U = prod.mean(axis=1)
    sd = float(U.std(ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        stat = 0.0
    else:
        stat = float(math.sqrt(len(U)) * U.mean() / sd)
    return stat, normal_survival(stat)


def final_test(nuisance, V, gamma):
    start = time.perf_counter()
    Z = nuisance.z("te", V)
    J = Z.shape[1]
    n = len(Z)
    mu, S, _ = mean_cov(Z)
    stat = float(n * mu @ np.linalg.pinv(S + gamma * np.eye(J)) @ mu)
    return time.perf_counter() - start, stat


def split_three_way(n, rng):
    perm = rng.permutation(n)
    n_eta = n // 3
    n_tr = n // 3
    return perm[:n_eta], perm[n_eta:n_eta + n_tr], perm[n_eta + n_tr:]


def run_trial(n, J, M, rep, axis, cfg):
    seed = cfg.seed_base + 1000000 * rep + 1000 * n + 37 * J + M
    rng = np.random.default_rng(seed)
    X, A, Y = simulate_observational_sample(n, seed, cfg)
    idx_eta, idx_tr, idx_te = split_three_way(n, rng)

    start = time.perf_counter()
    nuisance = Nuisance(
        X[idx_eta], A[idx_eta], Y[idx_eta],
        X[idx_tr], A[idx_tr], Y[idx_tr],
        X[idx_te], A[idx_te], Y[idx_te],
        cfg, rng,
    )
    nuisance_time = time.perf_counter() - start

    start = time.perf_counter()
    V_dict, dict_meta = learn_dictionary_locations(nuisance, J, M, cfg.tau)
    dict_learning_time = time.perf_counter() - start
    dict_test_time, dict_stat = final_test(nuisance, V_dict, cfg.gamma)

    base = {
        "axis": axis,
        "n": n,
        "J": J,
        "M": M,
        "rep": rep,
        "seed": seed,
        "n_eta": len(idx_eta),
        "n_tr": len(idx_tr),
        "n_te": len(idx_te),
        "grad_steps": cfg.grad_steps,
        "ell_y": nuisance.ell,
    }

    rows = [{
        **base,
        "method": "finite_dictionary",
        "nuisance_sec": nuisance_time,
        "learning_sec": dict_learning_time,
        "search_sec": dict_meta["search_sec"],
        "precompute_sec": dict_meta["precompute_sec"],
        "test_sec": dict_test_time,
        "core_sec": dict_learning_time + dict_test_time,
        "total_sec": nuisance_time + dict_learning_time + dict_test_time,
        "stat": dict_stat,
        "locations": ";".join(f"{x:.6g}" for x in V_dict),
    }]

    start = time.perf_counter()
    V_grad, grad_meta = learn_gradient_locations(
        nuisance, J, cfg.grad_steps, cfg.grad_lr_scale, cfg.tau, rng
    )
    grad_learning_time = time.perf_counter() - start
    grad_test_time, grad_stat = final_test(nuisance, V_grad, cfg.gamma)

    rows.append({
        **base,
        "method": "gradient_ascent",
        "nuisance_sec": nuisance_time,
        "learning_sec": grad_learning_time,
        "search_sec": grad_meta["search_sec"],
        "precompute_sec": 0.0,
        "test_sec": grad_test_time,
        "core_sec": grad_learning_time + grad_test_time,
        "total_sec": nuisance_time + grad_learning_time + grad_test_time,
        "stat": grad_stat,
        "locations": ";".join(f"{x:.6g}" for x in V_grad),
    })

    start = time.perf_counter()
    pi_model = fit_logit(X[idx_eta], A[idx_eta], cfg.prop_l2)
    w_te = predict_logit(pi_model, X[idx_te], cfg.prop_clip)
    dr_nuisance_time = time.perf_counter() - start

    order = np.random.default_rng(seed + 1234567).permutation(len(idx_te))
    Y_eval = Y[idx_te][order]
    X_eval = X[idx_te][order]
    A_eval = A[idx_te][order]
    w_eval = w_te[order]

    start = time.perf_counter()
    dr_stat, _ = dr_xkte_stat(Y_eval, w_eval, X_eval, A_eval, nuisance.ell, seed=seed + 99)
    dr_test_time = time.perf_counter() - start

    rows.append({
        **base,
        "method": "dr_xkte",
        "nuisance_sec": dr_nuisance_time,
        "learning_sec": 0.0,
        "search_sec": 0.0,
        "precompute_sec": 0.0,
        "test_sec": dr_test_time,
        "core_sec": dr_test_time,
        "total_sec": dr_nuisance_time + dr_test_time,
        "stat": dr_stat,
        "locations": "",
    })

    return rows


def build_tasks(cfg):
    all_tasks = []
    for n in cfg.n_grid:
        for rep in range(cfg.reps):
            all_tasks.append((n, cfg.fixed_J, cfg.fixed_M, rep, "n"))
    for J in cfg.J_grid:
        for rep in range(cfg.reps):
            all_tasks.append((cfg.fixed_n, J, cfg.fixed_M, rep, "J"))
    for M in cfg.M_grid:
        for rep in range(cfg.reps):
            all_tasks.append((cfg.fixed_n, cfg.fixed_J, M, rep, "M"))
    return all_tasks


def summarize(raw):
    group_cols = ["axis", "n", "J", "M", "grad_steps", "method"]
    summary = raw.groupby(group_cols).agg(
        reps=("rep", "count"),
        nuisance_sec_mean=("nuisance_sec", "mean"),
        learning_sec_mean=("learning_sec", "mean"),
        learning_sec_std=("learning_sec", "std"),
        learning_sec_median=("learning_sec", "median"),
        search_sec_mean=("search_sec", "mean"),
        precompute_sec_mean=("precompute_sec", "mean"),
        test_sec_mean=("test_sec", "mean"),
        core_sec_mean=("core_sec", "mean"),
        core_sec_std=("core_sec", "std"),
        core_sec_median=("core_sec", "median"),
        total_sec_mean=("total_sec", "mean"),
        total_sec_std=("total_sec", "std"),
        total_sec_median=("total_sec", "median"),
        stat_median=("stat", "median"),
    ).reset_index()

    summary["learning_sec_se"] = summary["learning_sec_std"].fillna(0.0) / np.sqrt(summary["reps"])
    summary["core_sec_se"] = summary["core_sec_std"].fillna(0.0) / np.sqrt(summary["reps"])
    summary["total_sec_se"] = summary["total_sec_std"].fillna(0.0) / np.sqrt(summary["reps"])
    return summary


def plot_axis(summary, axis, xcol, outpath, title, metric="total"):
    subset = summary[summary["axis"] == axis]
    ycol = f"{metric}_sec_mean"
    secol = f"{metric}_sec_se"

    plt.figure(figsize=(7.8, 4.7))
    for method, label in [
        ("finite_dictionary", "ODR-ME finite dictionary"),
        ("gradient_ascent", "ODR-ME gradient ascent"),
        ("dr_xkte", "DR-xKTE"),
    ]:
        dfm = subset[subset["method"] == method].sort_values(xcol)
        if dfm.empty:
            continue
        plt.errorbar(
            dfm[xcol],
            dfm[ycol],
            yerr=1.96 * dfm[secol],
            marker="o",
            capsize=3,
            linewidth=1.8,
            label=label,
        )

    plt.xlabel(xcol)
    plt.ylabel(f"{metric} runtime (seconds)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def plot_breakdown(raw, cfg, outpath):
    n0 = max(cfg.n_grid)
    subset = raw[(raw["axis"] == "n") & (raw["n"] == n0)]
    breakdown = subset.groupby("method").agg(
        nuisance=("nuisance_sec", "mean"),
        learning=("learning_sec", "mean"),
        test=("test_sec", "mean"),
    )

    methods = ["finite_dictionary", "gradient_ascent", "dr_xkte"]
    labels = ["ODR-ME finite", "ODR-ME gradient", "DR-xKTE"]
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))

    plt.figure(figsize=(8.0, 4.7))
    for component in ["nuisance", "learning", "test"]:
        values = np.array([breakdown.loc[m, component] if m in breakdown.index else 0.0 for m in methods])
        plt.bar(x, values, bottom=bottom, label=component)
        bottom = bottom + values

    plt.xticks(x, labels)
    plt.ylabel("seconds")
    plt.title(f"Runtime breakdown at n={n0}, J={cfg.fixed_J}, M={cfg.fixed_M}")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=Config.reps)
    parser.add_argument("--outdir", default=Config.outdir)
    parser.add_argument("--n-grid", type=int, nargs="+", default=list(Config.n_grid))
    parser.add_argument("--J-grid", type=int, nargs="+", default=list(Config.J_grid))
    parser.add_argument("--M-grid", type=int, nargs="+", default=list(Config.M_grid))
    parser.add_argument("--fixed-n", type=int, default=Config.fixed_n)
    parser.add_argument("--fixed-J", type=int, default=Config.fixed_J)
    parser.add_argument("--fixed-M", type=int, default=Config.fixed_M)
    parser.add_argument("--grad-steps", type=int, default=Config.grad_steps)
    parser.add_argument("--seed-base", type=int, default=Config.seed_base)
    args = parser.parse_args()

    return Config(
        reps=args.reps,
        outdir=args.outdir,
        n_grid=tuple(args.n_grid),
        J_grid=tuple(args.J_grid),
        M_grid=tuple(args.M_grid),
        fixed_n=args.fixed_n,
        fixed_J=args.fixed_J,
        fixed_M=args.fixed_M,
        grad_steps=args.grad_steps,
        seed_base=args.seed_base,
    )


def main():
    cfg = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)

    config_path = os.path.join(cfg.outdir, "exp5_config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    rows = []
    tasks = build_tasks(cfg)
    start = time.perf_counter()
    print(f"Experiment 5 runtime: {len(tasks)} trials, reps={cfg.reps}", flush=True)

    for i, (n, J, M, rep, axis) in enumerate(tasks, start=1):
        rows.extend(run_trial(n, J, M, rep, axis, cfg))
        if i % max(1, len(tasks) // 10) == 0 or i == len(tasks):
            elapsed = time.perf_counter() - start
            print(f"completed {i}/{len(tasks)} in {elapsed:.1f}s", flush=True)

    raw = pd.DataFrame(rows)
    raw_path = os.path.join(cfg.outdir, "exp5_runtime_raw.csv")
    raw.to_csv(raw_path, index=False)

    summary = summarize(raw)
    summary_path = os.path.join(cfg.outdir, "exp5_runtime_summary.csv")
    summary.to_csv(summary_path, index=False)

    plot_axis(
        summary,
        "n",
        "n",
        os.path.join(cfg.outdir, "exp5_runtime_total_vs_n_drkte.png"),
        f"Total runtime vs n (J={cfg.fixed_J}, M={cfg.fixed_M}, steps={cfg.grad_steps})",
        metric="total",
    )
    plot_axis(
        summary,
        "n",
        "n",
        os.path.join(cfg.outdir, "exp5_runtime_core_vs_n_drkte.png"),
        f"Core method runtime vs n (J={cfg.fixed_J}, M={cfg.fixed_M}, steps={cfg.grad_steps})",
        metric="core",
    )
    plot_axis(
        summary,
        "J",
        "J",
        os.path.join(cfg.outdir, "exp5_runtime_total_vs_J_drkte.png"),
        f"Total runtime vs J (n={cfg.fixed_n}, M={cfg.fixed_M}, steps={cfg.grad_steps})",
        metric="total",
    )
    plot_axis(
        summary,
        "M",
        "M",
        os.path.join(cfg.outdir, "exp5_runtime_total_vs_M_drkte.png"),
        f"Total runtime vs M (n={cfg.fixed_n}, J={cfg.fixed_J})",
        metric="total",
    )
    plot_breakdown(raw, cfg, os.path.join(cfg.outdir, "exp5_runtime_breakdown_nmax_drkte.png"))

    print(f"Saved {cfg.outdir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
