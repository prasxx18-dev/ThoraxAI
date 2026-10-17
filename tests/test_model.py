"""Model construction, loss, checkpoint save/load and Grad-CAM error handling."""

import numpy as np
import pytest
from PIL import Image

import config
from src.model import (
    ModelNotFoundError,
    WeightedBinaryCrossentropy,
    build_model,
    compile_model,
    compute_label_weights,
    get_backbone,
    load_trained_model,
    unfreeze_backbone,
)


@pytest.fixture(scope="module")
def small_mobilenet():
    model, info = build_model("mobilenetv2", image_size=64, pretrained=False)
    return model, info


def test_model_has_14_sigmoid_outputs(small_mobilenet):
    model, info = small_mobilenet
    assert model.output_shape == (None, 14)
    assert model.get_layer("predictions").activation.__name__ == "sigmoid"
    assert info["pretrained_weights_loaded"] is False
    out = model(np.random.rand(2, 64, 64, 3).astype("float32") * 255, training=False).numpy()
    assert out.shape == (2, 14) and np.all((out >= 0) & (out <= 1))
    # Sigmoid (not softmax): rows are not forced to sum to 1.
    assert not np.allclose(out.sum(axis=1), 1.0)


def test_densenet121_builds():
    model, info = build_model("densenet121", image_size=64, pretrained=False)
    assert model.output_shape == (None, 14)
    assert get_backbone(model).name == "densenet121"
    assert all(not w.trainable for w in get_backbone(model).weights)


def test_unknown_architecture():
    with pytest.raises(ValueError):
        build_model("resnet9000", pretrained=False)


def test_unfreeze_keeps_batchnorm_frozen(small_mobilenet):
    model, _ = small_mobilenet
    n = unfreeze_backbone(model, 20)
    assert 0 < n <= 20
    import keras

    bns = [lyr for lyr in get_backbone(model).layers if isinstance(lyr, keras.layers.BatchNormalization)]
    assert all(not b.trainable for b in bns)
    get_backbone(model).trainable = False


def test_label_weights_favor_rare_labels():
    y = np.zeros((100, 14))
    y[:50, 0] = 1  # common
    y[:2, 1] = 1  # rare
    pos, neg = compute_label_weights(y)
    assert pos[1] > pos[0] and pos[13] == 1.0  # label without positives -> neutral weight
    assert len(pos) == len(neg) == 14


def test_weighted_bce_matches_plain_bce_with_unit_weights():
    import keras

    y_true = np.array([[1, 0, 1], [0, 0, 1]], dtype="float32")
    y_pred = np.array([[0.9, 0.2, 0.6], [0.3, 0.1, 0.8]], dtype="float32")
    w = WeightedBinaryCrossentropy([1, 1, 1], [1, 1, 1])
    ref = keras.losses.BinaryCrossentropy()
    assert float(w(y_true, y_pred)) == pytest.approx(float(ref(y_true, y_pred)), rel=1e-4)


def test_save_and_load_checkpoint(tmp_path, small_mobilenet):
    model, _ = small_mobilenet
    compile_model(model, 1e-3, "weighted_bce", compute_label_weights(np.eye(14)))
    path = tmp_path / "best_model.keras"
    model.save(path)
    loaded = load_trained_model(path)
    x = np.random.rand(1, 64, 64, 3).astype("float32") * 255
    np.testing.assert_allclose(model(x).numpy(), loaded(x).numpy(), atol=1e-5)


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        load_trained_model(tmp_path / "missing.keras")
    (tmp_path / "x.h5").write_text("x")
    with pytest.raises(ValueError):
        load_trained_model(tmp_path / "x.h5")


def test_gradcam_output_and_errors(small_mobilenet):
    from src.gradcam import GradCAMError, compute_gradcam, overlay_heatmap

    model, _ = small_mobilenet
    x = np.random.rand(1, 64, 64, 3).astype("float32") * 255
    heatmap, score = compute_gradcam(model, x, class_index=1)
    assert heatmap.ndim == 2 and heatmap.min() >= 0 and heatmap.max() <= 1
    assert 0 <= score <= 1
    overlay = overlay_heatmap(Image.new("RGB", (100, 100)), heatmap)
    assert overlay.size == (100, 100)
    with pytest.raises(GradCAMError):
        compute_gradcam(model, x, class_index=config.NUM_LABELS)
    with pytest.raises(GradCAMError):
        compute_gradcam(model, x[0], class_index=0)


def test_gradcam_rejects_unsupported_model():
    import keras

    from src.gradcam import GradCAMError, compute_gradcam

    inp = keras.Input((32, 32, 3))
    flat = keras.Model(inp, keras.layers.Dense(14, activation="sigmoid")(keras.layers.Flatten()(inp)))
    with pytest.raises(GradCAMError, match="Unsupported"):
        compute_gradcam(flat, np.zeros((1, 32, 32, 3)), 0)
    with pytest.raises(GradCAMError):
        compute_gradcam(object(), np.zeros((1, 32, 32, 3)), 0)
