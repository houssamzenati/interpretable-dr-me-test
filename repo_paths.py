from pathlib import Path


RESULTS_ROOT = Path("results")

DEFAULT_RESULTS_DIRS = {
    "contour": RESULTS_ROOT / "contour",
    "exp1": RESULTS_ROOT / "exp1",
    "exp2": RESULTS_ROOT / "exp2",
    "exp3": RESULTS_ROOT / "exp3",
    "exp4": RESULTS_ROOT / "exp4",
    "exp5": RESULTS_ROOT / "exp5",
    "octmnist_causal": RESULTS_ROOT / "octmnist_causal",
    "octmnist_mean_matched": RESULTS_ROOT / "octmnist_mean_matched",
}


def results_dir(name):
    if name not in DEFAULT_RESULTS_DIRS:
        raise KeyError(f"Unknown results key: {name}")
    return DEFAULT_RESULTS_DIRS[name]
