"""Grad-CAM (Selvaraju et al., ICCV 2017) for ThoraxAI models.

For a chosen finding c, Grad-CAM takes the gradient of that finding's sigmoid
score with respect to the last convolutional feature maps A^k of the backbone,
global-average-pools the gradients to obtain channel weights alpha_k, and
computes ReLU(sum_k alpha_k * A^k). The map is upsampled and overlaid on the
X-ray.

Interpretation caveat: the map shows image regions that influenced the model's
score. It does NOT show where disease is, and a plausible-looking map does not
prove the prediction is correct (models can rely on artefacts such as text
markers, tubes or devices).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


class GradCAMError(RuntimeError):
    """Raised when Grad-CAM cannot be computed for the given model/input."""


def _model_parts(model: Any):
    try:
        import keras

        from src.model import get_backbone

        if not isinstance(model, keras.Model):
            raise GradCAMError("Grad-CAM requires a Keras model.")
        return (
            model.get_layer("input_norm"),
            get_backbone(model),
            model.get_layer("gap"),
            model.get_layer("predictions"),
        )
    except (ValueError, KeyError) as exc:
        raise GradCAMError(
            "Unsupported architecture for Grad-CAM: expected layers 'input_norm', a nested CNN backbone, "
            f"'gap' and 'predictions' ({exc})."
        ) from exc


def compute_gradcam(model: Any, batch: np.ndarray, class_index: int) -> tuple[np.ndarray, float]:
    """Return ``(heatmap, score)``; heatmap is (h, w) in [0, 1] at feature-map resolution."""
    import tensorflow as tf

    norm, backbone, gap, head = _model_parts(model)
    n_outputs = head.units
    if not 0 <= class_index < n_outputs:
        raise GradCAMError(f"class_index {class_index} out of range (0..{n_outputs - 1})")
    batch = np.asarray(batch, dtype=np.float32)
    if batch.ndim != 4 or batch.shape[0] != 1 or batch.shape[-1] != 3:
        raise GradCAMError(f"Expected input of shape (1, H, W, 3), got {batch.shape}")

    inputs = tf.convert_to_tensor(batch)
    with tf.GradientTape() as tape:
        features = backbone(norm(inputs), training=False)
        tape.watch(features)
        scores = head(gap(features))
        target = scores[:, class_index]
    grads = tape.gradient(target, features)
    if grads is None:
        raise GradCAMError("Gradient could not be computed for the selected output.")

    weights = tf.reduce_mean(grads, axis=(1, 2))  # (1, channels)
    cam = tf.nn.relu(tf.reduce_sum(features * weights[:, tf.newaxis, tf.newaxis, :], axis=-1))[0].numpy()
    peak = float(cam.max())
    heatmap = cam / peak if peak > 0 else np.zeros_like(cam)
    return heatmap.astype(np.float32), float(target.numpy()[0])


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4,
                    colormap: str = "jet") -> Image.Image:
    """Resize the heatmap to the image and alpha-blend it with a colour map."""
    import matplotlib

    if heatmap.ndim != 2:
        raise GradCAMError("Heatmap must be 2-D.")
    base = image.convert("RGB")
    hm = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(base.size, Image.BILINEAR)
    colored = matplotlib.colormaps[colormap](np.asarray(hm) / 255.0)[..., :3]
    blended = (1 - alpha) * np.asarray(base, dtype=np.float32) / 255.0 + alpha * colored
    return Image.fromarray(np.uint8(np.clip(blended, 0, 1) * 255))
