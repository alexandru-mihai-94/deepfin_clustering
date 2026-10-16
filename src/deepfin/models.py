"""
ImageNet feature extractors, each with its own preprocess_input (feature_extract.CNN_embed
applies vgg16 preprocessing to EfficientNetV2L and stretches images to a square).

Images are padded to a square (no stretching) with `bg`, resized to the model's input size,
and the features are the global-average-pooled last convolutional layer.
"""
from __future__ import annotations

import time

import numpy as np
from PIL import Image
import keras

A = keras.applications
MODELS = {  # name: (constructor, input size, preprocess_input)
    "EfficientNetV2L": (A.EfficientNetV2L, 480, A.efficientnet_v2.preprocess_input),
    "EfficientNetV2L_deepfin": (A.EfficientNetV2L, 480, A.vgg16.preprocess_input),  # the repo's CNN_embed choice
    "EfficientNetB0": (A.EfficientNetB0, 224, A.efficientnet.preprocess_input),
    "ResNet50": (A.ResNet50, 224, A.resnet50.preprocess_input),
    "MobileNetV2": (A.MobileNetV2, 224, A.mobilenet_v2.preprocess_input),
    "InceptionV3": (A.InceptionV3, 299, A.inception_v3.preprocess_input),
    "VGG16": (A.VGG16, 224, A.vgg16.preprocess_input),
    "DenseNet121": (A.DenseNet121, 224, A.densenet.preprocess_input),
}


def square(im: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    s = max(im.size)
    c = Image.new("RGB", (s, s), bg)
    c.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return c


def extract(images: list[Image.Image], name: str, bg=(255, 255, 255), tag: str = "") -> np.ndarray:
    """Features (n_images, n_features) from model `name`; prints progress every 40 images."""
    ctor, size, prep = MODELS[name]
    model = ctor(include_top=False, weights="imagenet", pooling="avg", input_shape=(size, size, 3))
    feats, t0 = [], time.time()
    for i in range(0, len(images), 8):
        batch = np.stack([np.asarray(square(im, bg).resize((size, size), Image.BICUBIC), np.float32)
                          for im in images[i:i + 8]])
        feats.append(model.predict(prep(batch), verbose=0))
        done = min(i + 8, len(images))
        if done % 40 == 0 or done == len(images):
            print(f"[features {tag}/{name}] {done}/{len(images)} ({(time.time() - t0) / done:.2f}s/img)", flush=True)
    keras.backend.clear_session()
    return np.vstack(feats)
