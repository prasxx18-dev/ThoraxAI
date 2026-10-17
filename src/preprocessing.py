"""Image loading, validation and input pipelines.

Design decision: intensity *normalisation* (ImageNet mean/std for DenseNet121,
[-1, 1] scaling for MobileNetV2) is part of the Keras model graph (see
``src/model.py``). Every pipeline here therefore produces resized RGB float
tensors in the range [0, 255], and training, evaluation, the Streamlit app and
Grad-CAM all share exactly one normalisation implementation.

Preprocessing steps
-------------------
1. Safe decode (Pillow, with a pixel limit against decompression bombs).
2. Convert grayscale / palette / RGBA / 16-bit images to 8-bit RGB by
   replicating the single X-ray channel (pretrained backbones expect 3 channels).
3. Resize to the model's square input (bilinear). NIH images are already square
   (1024x1024), so aspect distortion is negligible for this dataset.
4. Training-only augmentation: small rotation (±0.03 of a turn ≈ ±11°),
   ±8% zoom, ±5% translation and ±10% contrast. Horizontal flips are NOT used
   because left/right anatomy (heart position, side of effusion) is meaningful.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import config

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS


class ImageValidationError(ValueError):
    """Raised for unreadable, unsupported or implausible images.

    ``category`` is a short machine-readable reason shown in the UI.
    """

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def _open(source: bytes | str | Path) -> Image.Image:
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(Path(source))
        img.load()
        return img
    except FileNotFoundError as exc:
        raise ImageValidationError("not_found", f"Image file not found: {source}") from exc
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("too_large", "Image has too many pixels.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError("corrupted", f"Image could not be decoded: {exc}") from exc


def load_image(
    source: bytes | str | Path,
    filename: Optional[str] = None,
    max_bytes: int = config.MAX_UPLOAD_BYTES,
    min_side: int = config.MIN_IMAGE_SIDE,
) -> Image.Image:
    """Load and validate an image, returning an 8-bit RGB PIL image."""
    name = filename or (str(source) if not isinstance(source, (bytes, bytearray)) else None)
    if name and not name.lower().endswith(config.SUPPORTED_IMAGE_EXTENSIONS):
        raise ImageValidationError(
            "unsupported_format", f"Unsupported file type. Allowed: {', '.join(config.SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    if isinstance(source, (bytes, bytearray)):
        if len(source) == 0:
            raise ImageValidationError("empty", "The uploaded file is empty.")
        if len(source) > max_bytes:
            raise ImageValidationError("too_large", f"File exceeds {max_bytes // (1024 * 1024)} MB.")

    img = _open(source)
    if img.format and img.format.upper() not in {"PNG", "JPEG", "MPO"}:
        raise ImageValidationError("unsupported_format", f"Unsupported image format: {img.format}")
    width, height = img.size
    if min(width, height) < min_side:
        raise ImageValidationError("too_small", f"Image is {width}x{height}; minimum side is {min_side}px.")
    return to_rgb(img)


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any supported mode (L, I;16, I, P, LA, RGBA, RGB) to 8-bit RGB."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    if img.mode in ("LA", "RGBA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(background, img)
    if img.mode != "RGB":
        img = img.convert("L").convert("RGB") if img.mode != "RGBA" else img.convert("RGB")
    return img


def verify_image_file(path: Path) -> tuple[bool, str]:
    """Integrity check for dataset validation: returns (ok, reason)."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
            if min(img.size) < 1:
                return False, "zero-size image"
        return True, ""
    except Exception as exc:  # noqa: BLE001 - any decoder failure counts as corrupted
        return False, f"{type(exc).__name__}: {exc}"


def prepare_array(img: Image.Image, image_size: int) -> np.ndarray:
    """Resize an RGB PIL image to (image_size, image_size, 3) float32 in [0, 255]."""
    if img.mode != "RGB":
        img = to_rgb(img)
    resized = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def prepare_batch(img: Image.Image, image_size: int) -> np.ndarray:
    """Same as ``prepare_array`` with a leading batch dimension."""
    return prepare_array(img, image_size)[np.newaxis, ...]


# --------------------------------------------------------------------------- #
# tf.data pipelines (TensorFlow imported lazily)
# --------------------------------------------------------------------------- #
def _augmenter(seed: int):
    import keras

    return keras.Sequential(
        [
            keras.layers.RandomRotation(0.03, fill_mode="constant", seed=seed),
            keras.layers.RandomZoom(0.08, fill_mode="constant", seed=seed),
            keras.layers.RandomTranslation(0.05, 0.05, fill_mode="constant", seed=seed),
            keras.layers.RandomContrast(0.1, seed=seed),
        ],
        name="train_augmentation",
    )


def make_tf_dataset(
    paths: Sequence[str],
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    training: bool,
    augment: bool = True,
    seed: int = 42,
    cache_dir: Optional[str] = None,
):
    """Stream images from disk in batches (never loads the full dataset into RAM).

    Validation/test pipelines are deterministic: no shuffling, no augmentation.
    Undecodable files raise during iteration; run ``scripts/validate_dataset.py``
    first so corrupted files are excluded before training.
    """
    import tensorflow as tf

    paths = list(paths)
    labels = np.asarray(labels, dtype=np.float32)
    if len(paths) != len(labels):
        raise ValueError("paths and labels must have the same length")

    def _load(path, y):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False, dtype=tf.uint8)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (image_size, image_size), method="bilinear")
        return tf.cast(img, tf.float32), y

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 10_000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if cache_dir:
        ds = ds.cache(str(Path(cache_dir) / f"cache_{'train' if training else 'eval'}_{image_size}"))
    ds = ds.batch(batch_size)
    if training and augment:
        aug = _augmenter(seed)
        ds = ds.map(
            lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 255.0), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return ds.prefetch(tf.data.AUTOTUNE)
