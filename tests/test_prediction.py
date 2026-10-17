"""Inference wrapper, threshold mapping and missing-model handling (mocked models)."""

import io

import numpy as np
import pytest
from PIL import Image

import config
from src.model import ModelNotFoundError
from src.predict import PredictionError, Predictor, available_models, flag_findings


def _xray_bytes(size=(128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, 90).save(buf, format="PNG")
    return buf.getvalue()


def test_flag_findings_threshold_mapping():
    scores = [0.9, 0.2, 0.55, *([0.0] * 11)]
    thresholds = [0.5, 0.1, 0.6, *([0.5] * 11)]
    assert flag_findings(scores, thresholds) == ["Atelectasis", "Cardiomegaly"]
    with pytest.raises(ValueError):
        flag_findings([0.1], [0.5, 0.5])


def test_prediction_output_shape_and_multilabel(mock_model_factory):
    scores = np.linspace(0.05, 0.95, 14)
    model = mock_model_factory(scores, image_size=64)
    pred = Predictor(model, thresholds=[0.5] * 14)
    result = pred.predict(_xray_bytes(), filename="x.png")
    assert model.last_input.shape == (1, 64, 64, 3)
    assert len(result.scores) == 14 and set(result.scores) == set(config.LABELS)
    assert len(result.flagged) > 1  # several findings can be positive at once
    assert result.flagged[0] == "Hernia"  # highest score first
    assert not result.no_finding_predicted
    assert result.original_size == (128, 128)
    ranked = result.ranked()
    assert ranked[0][1] >= ranked[-1][1]


def test_no_finding_predicted_status(mock_model_factory):
    pred = Predictor(mock_model_factory(np.full(14, 0.1)), thresholds=[0.5] * 14)
    result = pred.predict(Image.new("RGB", (100, 100)))
    assert result.flagged == [] and result.no_finding_predicted


def test_custom_threshold_override(mock_model_factory):
    pred = Predictor(mock_model_factory(np.full(14, 0.3)), thresholds=[0.5] * 14)
    result = pred.predict(Image.new("RGB", (100, 100)), thresholds=[0.2] * 14)
    assert len(result.flagged) == 14 and not result.thresholds_validated


def test_missing_thresholds_are_disclosed(mock_model_factory):
    result = Predictor(mock_model_factory(np.full(14, 0.3))).predict(Image.new("RGB", (100, 100)))
    assert not result.thresholds_validated and result.notes


def test_bad_model_output_shape(mock_model_factory):
    with pytest.raises(PredictionError):
        Predictor(mock_model_factory(np.zeros(3))).predict(Image.new("RGB", (100, 100)))
    with pytest.raises(PredictionError):
        Predictor(mock_model_factory(np.full(14, np.nan))).predict(Image.new("RGB", (100, 100)))


def test_missing_checkpoint(tmp_path):
    (tmp_path / "densenet121").mkdir()
    with pytest.raises(ModelNotFoundError, match="train_model.py"):
        Predictor.from_model_dir(tmp_path / "densenet121")
    assert available_models(tmp_path) == []


def test_from_model_dir_with_real_small_model(tmp_path):
    from src.model import build_model
    from src.utils import save_json

    model, _ = build_model("mobilenetv2", image_size=64, pretrained=False)
    d = tmp_path / "mobilenetv2"
    d.mkdir()
    model.save(d / "best_model.keras")
    save_json({"thresholds": {lbl: 0.4 for lbl in config.LABELS}}, d / "threshold_config.json")
    save_json({"labels": config.LABELS}, d / "model_config.json")
    pred = Predictor.from_model_dir(d)
    assert pred.thresholds_validated and pred.image_size == 64
    assert available_models(tmp_path) == ["mobilenetv2"]
    res = pred.predict(_xray_bytes(), filename="a.png")
    assert len(res.scores) == 14


# ----------------------------------------------------------------- output activation / percentages
def _tiny_model(activation):
    import keras

    inp = keras.Input((32, 32, 3))
    x = keras.layers.GlobalAveragePooling2D()(inp)
    out = keras.layers.Dense(14, activation=activation, name="predictions")(x)
    return keras.Model(inp, out)


def test_sigmoid_model_scores_are_not_squashed_twice():
    model = _tiny_model("sigmoid")
    pred = Predictor(model, thresholds=[0.5] * 14)
    assert pred.activation == "sigmoid" and not pred.outputs_are_logits
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    from src.preprocessing import prepare_batch

    batch = prepare_batch(img, 32)  # exactly the pipeline the Predictor uses
    raw = np.asarray(model(batch, training=False), dtype=np.float64)[0]
    res = pred.predict(img)
    np.testing.assert_allclose([res.scores[l] for l in config.LABELS], raw, atol=1e-6)
    assert res.percentages == {l: round(float(v) * 100, 1) for l, v in zip(config.LABELS, raw)}


def test_logit_model_gets_sigmoid_applied_once():
    model = _tiny_model(None)  # linear output = raw logits
    pred = Predictor(model, thresholds=[0.5] * 14)
    assert pred.activation == "linear" and pred.outputs_are_logits
    batch = np.full((1, 32, 32, 3), 200.0, dtype=np.float32)
    logits = np.asarray(model(batch, training=False), dtype=np.float64)
    np.testing.assert_allclose(pred.predict_scores(batch), 1 / (1 + np.exp(-logits)), atol=1e-9)


def test_softmax_model_is_rejected_for_multilabel():
    with pytest.raises(PredictionError, match="Unsupported output activation"):
        Predictor(_tiny_model("softmax"))


def test_out_of_range_scores_are_refused(mock_model_factory):
    pred = Predictor(mock_model_factory(np.full(14, 2.5)))  # mock has no layers -> assumed sigmoid outputs
    with pytest.raises(PredictionError, match=r"outside \[0, 1\]"):
        pred.predict(Image.new("RGB", (100, 100)))


def test_percentages_map_to_the_correct_labels(mock_model_factory):
    scores = np.round(np.linspace(0.01, 0.14, 14), 3)  # distinct value per label index
    res = Predictor(mock_model_factory(scores)).predict(Image.new("RGB", (100, 100)))
    for i, label in enumerate(config.LABELS):
        assert res.scores[label] == pytest.approx(scores[i], abs=1e-6)
        assert res.percentages[label] == round(scores[i] * 100, 1)
