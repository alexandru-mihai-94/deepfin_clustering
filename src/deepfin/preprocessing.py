"""
Image preprocessing 

    -- Removes the background with rembg package (U-Net segmentation) 
    -- Resizes to 224 × 224 pixels
    -- Returns a list of NumPy RGB arrays scaled to [0, 1]
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import io

import numpy as np
from PIL import Image
from rembg import remove


def rembg_images(img_dir: str | Path) -> List[np.ndarray]:
    """
    Load all images in img_dir, remove the background 

    Parameters
    ----------
    img_dir : str | Path  Directory containing .jpg / .png files.

    Returns
    -------
    List[np.ndarray] List of images
    """
    img_dir = Path(img_dir)
    # images: list[np.ndarray] = []
    thumbs: list[Image.Image] = []

    for p in sorted(img_dir.glob("*")):
        if not p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            continue

        # with p.open("rb") as f:
        #     img = Image.open(f)
        #     # background removal
        #     no_bg = remove(img)
        # # img = Image.open(no_bg).convert("RGB").resize((224, 224))
        # # arr = np.asarray(no_bg, dtype=np.float32)
        # images.append(no_bg)

        data = p.read_bytes()                      # read file once
        rgba_bytes = remove(data)                  # rembg → bytes with alpha
        thumb = (
            Image.open(io.BytesIO(rgba_bytes))
            .convert("RGBA")                       # keep transparency
        )
        thumbs.append(thumb.copy())                # copy detaches from BytesIO

    if not thumbs:
        raise FileNotFoundError(f"No images in {img_dir}")

    return thumbs

def load_images(img_dir: str | Path) -> list[np.ndarray]:
    img_dir = Path(img_dir)
    images: list[np.ndarray] = []

    for p in sorted(img_dir.glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with p.open("rb") as f:
            img = Image.open(f).convert("RGB")
            img.load()            # read pixels now
            images.append(img)

    if not images:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return images