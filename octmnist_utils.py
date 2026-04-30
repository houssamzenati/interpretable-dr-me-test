import hashlib
import math
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OCTMNIST_CACHE = Path("octmnist_cache")
OCTMNIST_FILENAME = "octmnist.npz"
OCTMNIST_URL = "https://zenodo.org/records/10519652/files/octmnist.npz?download=1"
OCTMNIST_MD5 = "c68d92d5b585d8d81f7112f81e2d0842"

OCT_LABELS = {
    0: "choroidal neovascularization",
    1: "diabetic macular edema",
    2: "drusen",
    3: "normal",
}


def check_md5(path, expected_md5):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_md5


def download_octmnist(data_home=OCTMNIST_CACHE, force=False):
    data_home = Path(data_home)
    data_home.mkdir(parents=True, exist_ok=True)
    output_path = data_home / OCTMNIST_FILENAME
    if output_path.exists() and not force and check_md5(output_path, OCTMNIST_MD5):
        return output_path
    print(f"Downloading OCTMNIST to {output_path} ...")
    urllib.request.urlretrieve(OCTMNIST_URL, output_path)
    if not check_md5(output_path, OCTMNIST_MD5):
        raise ValueError("Downloaded OCTMNIST file failed MD5 verification.")
    return output_path


def normalize_octmnist(images, labels):
    images = np.asarray(images)
    labels = np.asarray(labels).reshape(-1)

    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim == 2:
        side = int(round(math.sqrt(images.shape[1])))
        if side * side != images.shape[1]:
            raise ValueError(f"Could not infer square image side from shape {images.shape}")
        images = images.reshape(-1, side, side)
    if images.ndim != 3:
        raise ValueError(f"Unexpected OCT image shape {images.shape}")
    if images.shape[0] != labels.shape[0]:
        raise ValueError(f"Image/label count mismatch: {images.shape[0]} vs {labels.shape[0]}")

    images = images.astype(np.float64)
    if images.max(initial=0.0) > 1.5:
        images = images / 255.0
    images = np.clip(images, 0.0, 1.0)
    return images, labels.astype(int)


def load_octmnist(source_path=None, data_home=OCTMNIST_CACHE):
    source = download_octmnist(data_home=data_home) if source_path is None else Path(source_path)
    with np.load(source, allow_pickle=True) as data:
        images = np.concatenate([data["train_images"], data["val_images"], data["test_images"]], axis=0)
        labels = np.concatenate([data["train_labels"], data["val_labels"], data["test_labels"]], axis=0)
    return normalize_octmnist(images, labels)


def save_image(path, image, title="", cmap="gray", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
