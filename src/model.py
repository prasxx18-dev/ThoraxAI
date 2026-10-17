"""Multi-label transfer-learning models (DenseNet121 primary, MobileNetV2 optional).

Model graph::

    image (H, W, 3) in [0, 255]
      -> input_norm      (backbone-specific normalisation, part of the model)
      -> backbone        (ImageNet-pretrained, include_top=False, frozen at first)
      -> gap             (GlobalAveragePooling2D)
      -> dropout
      -> predictions     (Dense(14, sigmoid))  -- 14 independent scores

A sigmoid per output (not softmax) lets several findings be positive for the
same image. The backbone is called with ``training=False`` so its BatchNorm
statistics stay frozen during fine-tuning (recommended for small datasets).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import numpy as np  # noqa: E402
from keras import layers, ops  # noqa: E402

import config  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger("model")


class ModelNotFoundError(FileNotFoundError):
    """Raised when a trained model checkpoint is not available."""


def _normalization_layer(architecture: str) -> layers.Layer:
    scheme = config.ARCHITECTURES[architecture]["normalization"]
    if scheme == "imagenet_torch":
        mean = [255.0 * m for m in config.IMAGENET_MEAN]
        variance = [(255.0 * s) ** 2 for s in config.IMAGENET_STD]
        return layers.Normalization(axis=-1, mean=mean, variance=variance, name="input_norm")
    if scheme == "tf_minus1_1":
        return layers.Rescaling(1.0 / 127.5, offset=-1.0, name="input_norm")
    raise ValueError(f"Unknown normalisation scheme: {scheme}")


def _create_backbone(architecture: str, image_size: int, weights: str | None) -> keras.Model:
    shape = (image_size, image_size, 3)
    if architecture == "densenet121":
        return keras.applications.DenseNet121(include_top=False, weights=weights, input_shape=shape)
    if architecture == "mobilenetv2":
        return keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=shape)
    raise ValueError(f"Unsupported architecture '{architecture}'. Choose from {list(config.ARCHITECTURES)}")


def build_model(
    architecture: str = "densenet121",
    image_size: int = 224,
    num_labels: int = config.NUM_LABELS,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> tuple[keras.Model, dict]:
    """Build the multi-label classifier.

    Returns ``(model, info)``; ``info["pretrained_weights_loaded"]`` records
    whether ImageNet weights were actually loaded (they need a one-time
    download; if that fails the model is built with random weights and this is
    reported rather than hidden).
    """
    if architecture not in config.ARCHITECTURES:
        raise ValueError(f"Unsupported architecture '{architecture}'. Choose from {list(config.ARCHITECTURES)}")
    loaded = False
    if pretrained:
        try:
            backbone = _create_backbone(architecture, image_size, "imagenet")
            loaded = True
        except Exception as exc:  # noqa: BLE001 - network/download errors vary
            log.warning("Could not load ImageNet weights (%s). Using random initialisation.", exc)
            backbone = _create_backbone(architecture, image_size, None)
    else:
        backbone = _create_backbone(architecture, image_size, None)
    backbone.trainable = False

    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = _normalization_layer(architecture)(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_labels, activation="sigmoid", name="predictions", dtype="float32")(x)
    model = keras.Model(inputs, outputs, name=f"thoraxai_{architecture}")
    info = {
        "architecture": architecture,
        "backbone_layer": backbone.name,
        "image_size": image_size,
        "num_labels": num_labels,
        "pretrained_weights_loaded": loaded,
        "normalization": config.ARCHITECTURES[architecture]["normalization"],
    }
    return model, info


def get_backbone(model: keras.Model) -> keras.Model:
    """Return the nested backbone sub-model."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("Model has no nested backbone; unsupported architecture for this operation.")


def unfreeze_backbone(model: keras.Model, num_layers: int) -> int:
    """Unfreeze the last ``num_layers`` backbone layers (BatchNorm stays frozen).

    Returns the number of layers made trainable.
    """
    backbone = get_backbone(model)
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - num_layers)
    count = 0
    for i, layer in enumerate(backbone.layers):
        trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = trainable
        count += int(trainable)
    return count


# --------------------------------------------------------------------------- #
# Loss and compilation
# --------------------------------------------------------------------------- #
def compute_label_weights(y: np.ndarray, clip: tuple[float, float] = (0.05, 50.0)) -> tuple[list, list]:
    """Per-label positive/negative weights for weighted BCE (class-balanced form).

    w_pos = N / (2 * P), w_neg = N / (2 * (N - P)). Rare findings therefore
    receive larger positive weights. Labels with no positives get weight 1.
    Computed from the TRAINING split only.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    pos = y.sum(axis=0)
    neg = n - pos
    pos_w = np.where(pos > 0, n / (2.0 * np.maximum(pos, 1)), 1.0)
    neg_w = np.where(neg > 0, n / (2.0 * np.maximum(neg, 1)), 1.0)
    return np.clip(pos_w, *clip).tolist(), np.clip(neg_w, *clip).tolist()


@keras.saving.register_keras_serializable(package="thoraxai")
class WeightedBinaryCrossentropy(keras.losses.Loss):
    """Binary cross-entropy with separate per-label positive/negative weights."""

    def __init__(self, pos_weights, neg_weights, name: str = "weighted_bce", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pos_weights = [float(w) for w in pos_weights]
        self.neg_weights = [float(w) for w in neg_weights]

    def call(self, y_true, y_pred):
        eps = keras.config.epsilon()
        y_pred = ops.clip(y_pred, eps, 1.0 - eps)
        y_true = ops.cast(y_true, y_pred.dtype)
        pw = ops.convert_to_tensor(self.pos_weights, dtype=y_pred.dtype)
        nw = ops.convert_to_tensor(self.neg_weights, dtype=y_pred.dtype)
        loss = -(pw * y_true * ops.log(y_pred) + nw * (1.0 - y_true) * ops.log(1.0 - y_pred))
        return ops.mean(loss, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"pos_weights": self.pos_weights, "neg_weights": self.neg_weights})
        return cfg


def compile_model(
    model: keras.Model,
    learning_rate: float,
    loss: str = "weighted_bce",
    label_weights: tuple[list, list] | None = None,
    weight_decay: float = 1e-5,
) -> None:
    if loss == "weighted_bce":
        if label_weights is None:
            raise ValueError("weighted_bce requires label_weights")
        loss_fn: keras.losses.Loss = WeightedBinaryCrossentropy(*label_weights)
    elif loss == "bce":
        loss_fn = keras.losses.BinaryCrossentropy(name="bce")
    else:
        raise ValueError(f"Unknown loss '{loss}'")
    n = model.output_shape[-1]
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss=loss_fn,
        metrics=[
            keras.metrics.AUC(multi_label=True, num_labels=n, name="auc"),
            keras.metrics.AUC(curve="PR", multi_label=True, num_labels=n, name="pr_auc"),
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        ],
    )


def load_trained_model(path: Path) -> keras.Model:
    """Load a saved ``.keras`` model for inference.

    ``safe_mode=True`` refuses arbitrary Python lambdas embedded in the file.
    Only models produced by this project's training script should be loaded;
    the app never accepts user-uploaded model files.
    """
    path = Path(path)
    if not path.is_file():
        raise ModelNotFoundError(f"Model checkpoint not found: {path}")
    if path.suffix != ".keras":
        raise ValueError(f"Expected a .keras file, got '{path.suffix}'")
    return keras.models.load_model(path, compile=False, safe_mode=True)
