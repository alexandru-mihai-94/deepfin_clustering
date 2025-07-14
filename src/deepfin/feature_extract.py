"""
CNN feature extraction using EfficientNetV2L
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import tensorflow as tf
from keras.applications.efficientnet_v2 import EfficientNetV2L 
from keras.applications.vgg16 import preprocess_input 
from keras.models import Model

# from tensorflow.keras.applications.efficientnet import (
#     EfficientNetB0,
#     preprocess_input,
# )


# load once at import time
_base_model = EfficientNetV2L()
_base_model = Model(inputs=_base_model.inputs, outputs=_base_model.layers[-2].output)

def CNN_embed(
    images: list[np.ndarray] | np.ndarray,
    model: str = "EfficientNetV2L",
) -> np.ndarray:
    """
    Convert a list/array of RGB images to embedding vectors.

    Parameters
    ----------
    images : list[np.ndarray] | np.ndarray
        Each element shaped (224, 224, 3), scaled to [0, 1].
    model : str
        Placeholder for future model choices; only EfficientNetV2L is implemented.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_images, embedding_dim=1280).
    """
    if model != "EfficientNetV2L":
        raise ValueError("Only EfficientNetV2L is supported for now.")

    feature_vecs: list[np.ndarray] = []

    # x = np.asarray(images, dtype=np.float32)
    for img in images:
        img = img.resize((480, 480))
        img = np.array(img)

        if img.shape[-1] == 4:
            img = np.delete(img, -1, axis=2)

        img = img.reshape(1, 480, 480, 3)
        img = preprocess_input(img)

        features = _base_model.predict(img)
        feature_vecs.append(features)
        
    return feature_vecs