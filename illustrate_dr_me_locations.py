#!/usr/bin/env python3
"""Separate prototype meaning from DR-ME location learning.

Reuses the manuscript contour experiment without modifying its implementation.
Run: python illustrate_dr_me_locations.py --gif
See location_visuals_notes.md for interpretation and provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path

# Import this first: the original module sets BLAS limits and the Agg backend.
import contour_estimated_oracle as original
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
import matplotlib.patheffects as pe


Q_COLOR = "#70469A"
QPRIME_COLOR = "#AC527A"
INK = "#26313B"
GREY = "#697078"
FIXED = np.array([[0.0, -0.05], [-1.8, -0.06]])
LIMITS = (-2.7, 2.7, -1.3, 1.3)


def joint_training_criterion(fixed, candidates, ridge):
    """Unscaled joint criterion and Schur-complement gain for each candidate.

    Columns are DR features; covariance uses n-1. Positive ridge permits even
    duplicate locations. Neither simulated potential outcomes nor true nuisance
    functions enter this computation.
    """
    fixed = np.asarray(fixed, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    if (fixed.ndim != 1 or candidates.ndim != 2
            or len(fixed) < 2 or candidates.shape[0] != len(fixed)
            or candidates.shape[1] == 0):
        raise ValueError("Expected n fixed features and an n-by-M candidate matrix, n >= 2.")
    if (not np.isfinite(ridge) or ridge <= 0
            or not np.isfinite(fixed).all() or not np.isfinite(candidates).all()):
        raise ValueError("Features must be finite and ridge strictly positive.")
    m1, m2 = fixed.mean(), candidates.mean(axis=0)
    x1, x2 = fixed - m1, candidates - m2
    s11 = x1 @ x1 / (len(fixed) - 1)
    s22 = np.sum(x2 * x2, axis=0) / (len(fixed) - 1)
    s12 = x1 @ x2 / (len(fixed) - 1)
    a, b = s11 + ridge, s22 + ridge
    residual_variance = b - s12 * s12 / a
    if np.any(residual_variance <= 0):
        raise ValueError("Regularised covariance is numerically singular; increase ridge.")
    single = float(m1 * m1 / a)
    gain = (m2 - s12 * m1 / a) ** 2 / residual_variance
    covariance = np.empty((2, 2, candidates.shape[1]))
    covariance[0, 0] = s11
    covariance[1, 1] = s22
    covariance[0, 1] = covariance[1, 0] = s12
    return dict(scores=single + gain, single=single, gain=gain,
                mean=np.vstack([np.full_like(m2, m1), m2]), covariance=covariance)


def population_similarities(sample, locations, kernel_scale):
    """Exact two-dimensional mixture expectations, ordered q=1 then q'=0."""
    mix = sample.mixture
    return np.array([
        original.mixture_kernel_mean(
            locations, np.asarray(mix[key]), np.asarray(mix["means"]),
            np.asarray(mix["sds"]), kernel_scale)
        for key in ("w1", "w0")
    ])


def compute_experiment(args):
    sample = original.make_bimodal_shift(args.n, args.seed)
    train, _ = sample.split(args.train_fraction, args.seed + 11)
    folds = original.prepare_folds(train, 2, args.seed + 19, 16, 3, 1e-2, 1.0)
    xs = np.linspace(LIMITS[0], LIMITS[1], args.grid_nx)
    ys = np.linspace(LIMITS[2], LIMITS[3], args.grid_ny)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    # Fit once, then use the same folds and candidates in both scenarios.
    z = original.crossfit_dr_all_locations(
        train, np.vstack([FIXED, grid]), args.kernel_scale, folds)
    panels = []
    max_error = 0.0
    for j, fixed in enumerate(FIXED):
        result = joint_training_criterion(z[:, j], z[:, 2:], args.ridge)
        best = int(np.argmax(result["scores"]))
        # Independent numerical reference: original full-matrix pseudoinverse.
        checks = np.unique(np.r_[np.linspace(0, len(grid) - 1, 37, dtype=int), best])
        for k in checks:
            reference = original.hotelling_statistic(z[:, [j, k + 2]], args.ridge) / len(train.a)
            np.testing.assert_allclose(result["scores"][k], reference, rtol=2e-8, atol=2e-10)
            max_error = max(max_error, abs(result["scores"][k] - reference))
        result.update(fixed=fixed, selected=grid[best], best=best)
        panels.append(result)
    rng = np.random.default_rng(args.seed + 101)
    indices = rng.choice(args.n, min(args.n, 340), replace=False)
    # Only display layers use potential outcomes; learning used x,a,y above.
    clouds = (sample.y1[indices], sample.y0[indices])
    return dict(sample=sample, train_n=len(train.a), grid=grid, xx=xx, yy=yy,
                panels=panels, clouds=clouds, max_reference_error=max_error)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
        "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
        "axes.edgecolor": "#C8CCD0", "axes.spines.top": False,
        "axes.spines.right": False, "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })


def heading(fig, title, subtitle):
    fig.text(0.06, 0.93, title, fontsize=20, weight="semibold", va="top")
    fig.text(0.06, 0.855, subtitle, fontsize=12.5, color=GREY, va="top")


def legend(fig, y=0.775):
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=7, label=label)
               for c, label in [(Q_COLOR, r"$q$: set $A=1$"),
                                 (QPRIME_COLOR, r"$q'$: set $A=0$")]]
    leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.065, y),
                     ncol=2, frameon=False, columnspacing=2.2, handletextpad=0.5)
    for label, color in zip(leg.get_texts(), (Q_COLOR, QPRIME_COLOR)):
        label.set_color(color)


def outcome_axes(fig):
    ax = fig.add_axes([0.08, 0.30, 0.62, 0.43])
    ax.set(xlim=LIMITS[:2], ylim=LIMITS[2:], aspect="equal",
           xlabel="Outcome coordinate 1", ylabel="Outcome coordinate 2",
           xticks=[-2, -1, 0, 1, 2], yticks=[-1, 0, 1])
    ax.tick_params(labelsize=10)
    return ax


def scatter_clouds(ax, clouds, prototype=None, kernel_scale=None):
    for cloud, color in zip(clouds, (Q_COLOR, QPRIME_COLOR)):
        if prototype is None:
            alpha = 0.38
        else:
            similarity = original.kernel_features(cloud, prototype, kernel_scale).ravel()
            alpha = 0.10 + 0.63 * similarity
        ax.scatter(*cloud.T, s=15, c=color, alpha=alpha, linewidths=0, zorder=3)


def marker(ax, xy, label, shape="+", offset=(0, 35)):
    artist = ax.scatter(*xy, marker=shape, s=175 if shape == "+" else 230,
                        color=INK, linewidths=2.3, zorder=7)
    artist.set_path_effects([pe.withStroke(linewidth=5, foreground="white")])
    ann = ax.annotate(label, xy, xytext=offset, textcoords="offset points",
                      ha="center", va="center", fontsize=12, color=INK,
                      bbox=dict(boxstyle="round,pad=.24", fc="white", ec="none", alpha=.95),
                      arrowprops=dict(arrowstyle="-", color=INK, lw=.9), zorder=8)
    return ann


def prototype_figure(data, args):
    fig = plt.figure(figsize=(12, 6.4))
    heading(fig, "What does one outcome prototype measure?",
            "Compare how similar outcomes are to the same reference outcome.")
    legend(fig)
    ax = outcome_axes(fig)
    prototype = np.asarray(args.prototype)
    # These are similarity contours, not probability-region boundaries.
    gx, gy = np.meshgrid(np.linspace(-2.7, 2.7, 220), np.linspace(-1.3, 1.3, 130))
    similarity = original.kernel_features(
        np.c_[gx.ravel(), gy.ravel()], prototype, args.kernel_scale).reshape(gx.shape)
    ax.contourf(gx, gy, similarity, levels=[0, .1, .25, .5, .75, 1.001],
                colors=["#FFFFFF", "#FAFAFA", "#F1F1F1", "#E8E8E8", "#DEDEDE"])
    ax.contour(gx, gy, similarity, levels=[.25, .5, .75],
               colors="#ADB1B5", linewidths=.65, linestyles="dashed")
    scatter_clouds(ax, data["clouds"], prototype, args.kernel_scale)
    offset = (-70, 39) if prototype[0] > 1 else (65, 39)
    marker(ax, prototype, r"prototype $v$", offset=offset)
    means = population_similarities(data["sample"], prototype, args.kernel_scale)[:, 0]
    bars = fig.add_axes([.775, .36, .17, .31])
    bars.barh([1, 0], means, height=.37, color=[Q_COLOR, QPRIME_COLOR])
    bars.set(xlim=(0, max(.6, means.max() * 1.27)), ylim=(-.6, 1.6),
             yticks=[1, 0], yticklabels=[r"$q$", r"$q'$"], xticks=[0, .25, .5])
    bars.tick_params(axis="y", length=0, labelsize=16)
    for tick, color in zip(bars.get_yticklabels(), (Q_COLOR, QPRIME_COLOR)):
        tick.set_color(color)
    bars.spines["left"].set_visible(False)
    bars.set_title("Mean similarity to $v$", fontsize=13, pad=12, loc="left")
    for y, val in zip([1, 0], means):
        bars.text(val + .014, y, f"{val:.3f}", va="center", fontsize=12)
    bars.set_xlabel("Expected kernel similarity", fontsize=11)
    fig.text(.08, .205, "Nearby outcomes contribute more; distant outcomes contribute less.", fontsize=12)
    equation = HPacker(children=[TextArea(text, textprops={"fontsize": 19, "color": color})
        for text, color in [(r"$w(v)=$", INK), (r"$\mu_q(v)$", Q_COLOR),
                            (r"$-$", INK), (r"$\mu_{q'}(v)$", QPRIME_COLOR),
                            (f"$= {means[0] - means[1]:+.3f}$", INK)]],
        align="baseline", pad=0, sep=5)
    fig.add_artist(AnnotationBbox(equation, (.08, .13), xycoords=fig.transFigure,
                   box_alignment=(0, .5), frameon=True,
                   bboxprops=dict(boxstyle="round,pad=.6", fc="#F3F0F6", ec="none")))
    fig.text(.06, .032, "Synthetic population illustration. Dots are interventional samples; bars are exact mean similarities.",
             color=GREY, fontsize=10)
    return fig


def learning_figure(data, panel_index, phase, norm, cmap):
    """phase 0: fixed location; 1: genuine criterion surface; 2: grid selection."""
    panel = data["panels"][panel_index]
    fig = plt.figure(figsize=(12, 6.4))
    heading(fig, "Learn a second prototype given the first",
            "Keep the first prototype fixed; score each candidate pair using the DR estimates.")
    legend(fig)
    ax = outcome_axes(fig)
    if phase >= 1:
        ax.pcolormesh(data["xx"], data["yy"], panel["scores"].reshape(data["xx"].shape),
                       shading="nearest", cmap=cmap, norm=norm, rasterized=True, zorder=0)
    scatter_clouds(ax, data["clouds"])
    marker(ax, panel["fixed"], r"fixed first prototype $v_1$", offset=(0, -45))
    if phase == 2:
        offset = (70, 48) if panel["selected"][0] < -1.4 else (0, 48)
        marker(ax, panel["selected"], r"selected second prototype $v_2$", shape="*", offset=offset)
    elif phase == 1:
        candidate = data["grid"][np.argmin(np.sum((data["grid"] - [1.05, .65]) ** 2, axis=1))]
        marker(ax, candidate, r"candidate $v_2$", shape="o", offset=(0, 29))
    labels = [("1", "Fix the first\nprototype."),
              ("2", "Score each\ncandidate pair."),
              ("3", "Select the best\ngrid candidate.")]
    for step, (num, label) in enumerate(labels):
        color = INK if step <= phase else "#BCC0C4"
        fig.text(.76, .68 - .145 * step, num, fontsize=16, weight="bold", color=color)
        fig.text(.80, .68 - .145 * step, label, fontsize=13.5, va="top", color=color, linespacing=1.45)
    if phase >= 1:
        plot_box = ax.get_position()
        cax = fig.add_axes([plot_box.x0, .185, plot_box.width, .024])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          orientation="horizontal", ticks=np.linspace(norm.vmin, norm.vmax, 5), format="%.2f")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=10, length=3)
        cb.set_label("Joint training criterion  (larger is better)", fontsize=12)
    if phase == 2:
        message = ("The useful second prototype changes when the first prototype changes."
                   if panel_index == 1 else "Choose the pair with the strongest signal relative to its joint variability.")
        fig.text(.08, .07, message, fontsize=12.5)
    elif phase == 1:
        fig.text(.08, .07, "Each background value scores the pair: fixed first prototype + candidate at that position.", fontsize=12)
    else:
        fig.text(.08, .11, "One prototype is already chosen. Where should we place the second?", fontsize=15)
    footer = ("Colour: DR training criterion. Dots: synthetic interventional samples for context. Same colour scale throughout."
              if phase else "Dots: synthetic interventional samples for context. Learning will use only observed data and fitted nuisances.")
    fig.text(.06, .022, footer,
             color=GREY, fontsize=9.5)
    return fig


def save_outputs(data, args):
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    style()
    outputs = []
    fig = prototype_figure(data, args)
    for ext in ("png", "pdf"):
        path = out / f"prototype_similarity.{ext}"
        fig.savefig(path, dpi=args.dpi)
        outputs.append(path.name)
    plt.close(fig)
    scores = np.concatenate([p["scores"] for p in data["panels"]])
    norm = Normalize(vmin=float(scores.min()), vmax=float(scores.max()))
    # Neutral sequential background reserves violet and rose for interventions.
    cmap = LinearSegmentedColormap.from_list("criterion", ["#FFFFFF", "#DDE2E5", "#919EA5"])
    frames = []
    with PdfPages(out / "location_learning.pdf") as main_pdf, PdfPages(out / "location_learning_reveals.pdf") as reveal_pdf:
        for panel_index, name in enumerate(("center", "left")):
            for phase in range(3):
                fig = learning_figure(data, panel_index, phase, norm, cmap)
                path = out / f"learning_{panel_index * 3 + phase + 1:02d}_{name}.png"
                fig.savefig(path, dpi=args.dpi)
                frames.append(path)
                reveal_pdf.savefig(fig)
                if phase == 2:
                    main_pdf.savefig(fig)
                plt.close(fig)
    outputs.extend(["location_learning.pdf", "location_learning_reveals.pdf"])
    outputs.extend(path.name for path in frames)
    if args.gif:
        from PIL import Image
        with_images = [Image.open(path).convert("RGB") for path in frames]
        try:
            with_images[0].save(out / "location_learning.gif", save_all=True,
                                append_images=with_images[1:], duration=[1800, 2800, 3800] * 2,
                                loop=0, disposal=2)
        finally:
            for im in with_images:
                im.close()
        outputs.append("location_learning.gif")
    np.savez_compressed(
        out / "figure_data.npz", grid=data["grid"], fixed=FIXED,
        scores=np.stack([p["scores"] for p in data["panels"]]),
        means=np.stack([p["mean"] for p in data["panels"]]),
        covariances=np.stack([p["covariance"] for p in data["panels"]]),
        gains=np.stack([p["gain"] for p in data["panels"]]),
        selected=np.stack([p["selected"] for p in data["panels"]]),
        cloud_q=data["clouds"][0], cloud_qprime=data["clouds"][1],
        prototype=args.prototype, population_similarities=population_similarities(
            data["sample"], args.prototype, args.kernel_scale)[:, 0])
    metadata = {
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in (__file__, original.__file__)},
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "train_n": data["train_n"], "mixture": data["sample"].mixture,
        "interventions": {"q": {"arm": 1, "color": Q_COLOR}, "qprime": {"arm": 0, "color": QPRIME_COLOR}},
        "criterion": "mean.T @ solve(sample_covariance + ridge * I, mean); no sample-size factor",
        "covariance_ddof": 1, "kernel_scale_is_squared_bandwidth": True,
        "colour_limits": [norm.vmin, norm.vmax], "colour_clipping": False,
        "original_score_divisor": data["train_n"],
        "max_reference_error": data["max_reference_error"],
        "sampling_design": "Original two-fold cross-fitting inside learning split; no final test performed.",
        "nuisance_settings": {"folds": 2, "basis_anchors": 16, "interaction_dim": 3,
                              "propensity_l2": .01, "regression_l2": 1.0},
        "interpretation": [
            "Potential-outcome clouds and exact population bars are display-only.",
            "Learning uses estimated nuisance functions and observed X,A,Y only.",
            "This DGP has covariate-dependent assignment but no outcome confounding.",
            "Grid selection with the other location fixed is not unrestricted joint optimisation.",
            "Training criterion is not witness height, density, p-value, or exact power.",
            "All reveal surfaces are recomputed, not interpolated images."],
        "panels": [{"fixed": p["fixed"].tolist(), "selected": p["selected"].tolist(),
                    "grid_index": p["best"], "single_criterion": p["single"],
                    "joint_criterion": float(p["scores"][p["best"]]),
                    "gain": float(p["gain"][p["best"]])} for p in data["panels"]],
        "outputs": outputs + ["figure_data.npz", "metadata.json"],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"outdir": str(out.resolve()), "train_n": data["train_n"],
                      "max_reference_error": data["max_reference_error"],
                      "panels": metadata["panels"]}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("results/location_explainer"))
    parser.add_argument("--n", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=.55)
    parser.add_argument("--kernel-scale", type=float, default=.8, help="Squared Gaussian bandwidth.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Training ridge, not final-test ridge.")
    parser.add_argument("--grid-nx", type=int, default=49)
    parser.add_argument("--grid-ny", type=int, default=34)
    parser.add_argument("--prototype", type=float, nargs=2, default=[1.8, 0.0])
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--gif", action="store_true", help="Animate the six exact reveal frames (requires Pillow).")
    args = parser.parse_args()
    if args.n < 100 or not 0 < args.train_fraction < 1 or round(args.n * args.train_fraction) < 80:
        parser.error("Use n >= 100 and a train fraction in (0,1), with at least 80 training observations.")
    if (not np.isfinite([args.kernel_scale, args.ridge, *args.prototype]).all()
            or args.kernel_scale <= 0 or args.ridge <= 0):
        parser.error("Kernel scale and ridge must be finite and positive; prototype must be finite.")
    if args.grid_nx < 3 or args.grid_ny < 3 or args.dpi < 72 or args.seed < 0:
        parser.error("Use grid dimensions >= 3, dpi >= 72 and a nonnegative seed.")
    if not (LIMITS[0] <= args.prototype[0] <= LIMITS[1] and LIMITS[2] <= args.prototype[1] <= LIMITS[3]):
        parser.error("Prototype must lie in the plotted outcome domain.")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    save_outputs(compute_experiment(arguments), arguments)
