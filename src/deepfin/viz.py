"""
UMAP scatter-plot
"""

from pathlib import Path
from typing import Sequence, Iterable

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np


def _distinct_palette(n: int) -> list[str]:
    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]

def plot_umap(
    xy: np.ndarray,
    labels: Sequence[int],
    outfile: str | Path,
) -> None:

    xy = np.asarray(xy)
    labels = np.asarray(labels)

    plt.figure(figsize=(6, 6), dpi=3000)
    unique = sorted(set(labels))
    palette = _distinct_palette(len(unique))

    for lab, col in zip(unique, palette, strict=True):
        mask = labels == lab
        if lab == -1:
            plt.scatter(xy[mask, 0], xy[mask, 1], c="grey", s=6, alpha=0.5, label="noise")
        else:
            plt.scatter(xy[mask, 0], xy[mask, 1], c=col, s=8, alpha=0.9, label=f"{lab}")

    plt.xticks([])
    plt.yticks([])
    plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.04, 1), loc="lower right")
    plt.tight_layout()
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()

"""
UMAP scatter plot with optional image thumbnails.
"""

def _to_rgb_array(img):
    """
    Return float array in [0,1]; keep alpha if present so
    OffsetImage can render transparency.
    """
    if hasattr(img, "copy"):                 # PIL.Image
        arr = np.asarray(img.convert("RGBA"), dtype=float) / 255.0
    else:                                    # already ndarray
        arr = np.asarray(img, dtype=float)
        if arr.max() > 1:
            arr = arr / 255.0
        if arr.shape[2] == 3:                # add fully-opaque alpha
            alpha = np.ones(arr.shape[:2] + (1,))
            arr = np.concatenate([arr, alpha], axis=2)
    return arr

def plot_umap_thumbnails(
    xy: np.ndarray,
    labels: Sequence[int],
    outfile: str | Path,
    *,
    images: Iterable | None = None,
    max_thumbnails: int = 60,
    thumb_zoom: float = 0.1,
    title: str = "Species clustering",
) -> None:
    """
    Save a 2-D UMAP scatter. If *images* is provided, overlay up to
    *max_thumbnails* thumbnails at their corresponding coordinates.

    Parameters
    ----------
    xy : np.ndarray, shape (n, 2)
    labels : 1-D cluster labels (−1 for noise)
    outfile : path-like
    images : iterable of images (optional)
    max_thumbnails : int
        Number of thumbnails to draw for speed/readability.
    thumb_zoom : float
        Scaling passed to OffsetImage(… , zoom=thumb_zoom)
    """
    xy = np.asarray(xy)
    labels = np.asarray(labels)

    plt.figure(figsize=(6, 6), dpi=150)
    unique = sorted(set(labels))
    palette = _distinct_palette(len(unique))

    for lab, col in zip(unique, palette, strict=True):
        mask = labels == lab
        plt.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c="grey" if lab == -1 else col,
            s=4 if lab == -1 else 6,
            alpha=0.6 if lab == -1 else 0.8,
            label="noise" if lab == -1 else str(lab),
        )

    # ---------- thumbnails ----------
    if images is not None:
        import random

        images = list(images)
        idx = list(range(len(images)))
        random.shuffle(idx)
        for i in idx[: max_thumbnails]:
            xi, yi = xy[i]
            # img_arr = _to_rgb_array(images[i])
            imagebox = OffsetImage(images[i], zoom=thumb_zoom)
            ab = AnnotationBbox(imagebox, (xi, yi), frameon=False)
            plt.gca().add_artist(ab)
    # ---------------------------------

    plt.xticks([])
    plt.yticks([])
    plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()