# ThoraxAI — System Architecture

## Layers

```
┌───────────────────────────── Offline pipeline (scripts/) ─────────────────────────────┐
│ validate_dataset.py ──► src/dataset.py ──► results/dataset_validation.{json,csv}        │
│ train_model.py ──► src/train.py ─┬─► src/model.py (DenseNet121 / MobileNetV2)          │
│                                  ├─► src/preprocessing.py (tf.data streaming)          │
│                                  └─► src/metrics.py (validation threshold selection)   │
│                                        └─► models/<arch>/*.keras, *.json               │
│ evaluate_model.py ──► src/evaluate.py ─► src/metrics.py + src/plots.py ─► results/<arch>│
└─────────────────────────────────────────────────────────────────────────────────────────┘
┌───────────────────────────── Interactive app (Streamlit) ──────────────────────────────┐
│ app.py (st.navigation, top bar) ─► app_pages/ home · analyze · insights · about        │
│      insights ─► src/views/ performance · dataset_view · system_view                    │
│      all pages use src/ui.py (design system, presentation only)                         │
│      ├─► src/predict.py   Predictor: load model + thresholds, score, flag findings     │
│      ├─► src/gradcam.py   Grad-CAM heatmap + overlay                                   │
│      └─► reads results/*.json|csv|png produced offline (never fabricates numbers)      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Module responsibilities

| Module | Responsibility | Depends on Streamlit? |
|---|---|---|
| `config.py` | Paths (env-overridable), label order, architectures, `TrainConfig` | No |
| `src/dataset.py` | Discovery, CSV parsing, label encoding, image index, patient splits, dev subset, validation report | No |
| `src/preprocessing.py` | Safe image decoding/validation, RGB conversion, resizing, `tf.data` pipelines, augmentation | No |
| `src/model.py` | Model construction, weighted BCE loss, compile, fine-tuning, safe loading | No |
| `src/train.py` | Orchestrates a training run and writes all artifacts | No |
| `src/metrics.py` | Per-label/macro/micro metrics, threshold selection, curve data | No |
| `src/evaluate.py` | Test-split evaluation, CSV/JSON/figures, model comparison | No |
| `src/predict.py` | Inference wrapper (`Predictor`), threshold mapping | No |
| `src/gradcam.py` | Grad-CAM computation and overlay | No |
| `src/plots.py` | Matplotlib/Seaborn figures | No |
| `src/ui.py` | Design system: theme CSS, icons, notices, stat tiles, score table, empty states, footer, status checks | Yes |
| `src/views/*.py` | Insights views (model performance, dataset, system status) | Yes |
| `app_pages/*.py` | Home, Analyze X-ray, Insights, About (not `pages/`, which triggers Streamlit's legacy navigation) | Yes |

## Model graph

```
Input (224×224×3, values 0–255)
  └─ input_norm      Normalization (ImageNet mean/std)  [MobileNetV2: Rescaling to [-1, 1]]
  └─ densenet121     ImageNet weights, include_top=False, called with training=False → 7×7×1024
  └─ gap             GlobalAveragePooling2D → 1024
  └─ dropout         p = 0.3
  └─ predictions     Dense(14, sigmoid)  → 14 independent scores
```

Normalisation is inside the saved model, so training, evaluation, the app and Grad-CAM cannot use
different preprocessing. Grad-CAM reuses the named sub-layers (`input_norm`, backbone, `gap`,
`predictions`) to differentiate a chosen output with respect to the 7×7 backbone feature maps.

## Artifacts

```
models/<arch>/best_model.keras        best validation macro-AUC checkpoint (used by the app)
models/<arch>/final_model.keras       weights at the end of training
models/<arch>/last_model.keras        most recent epoch (for --resume)
models/<arch>/model_config.json       architecture, input, train config, splits, timings, versions
models/<arch>/label_mapping.json      output index → label
models/<arch>/threshold_config.json   per-label thresholds selected on validation data
results/dataset_validation.json|*.csv dataset report (+ dataset_figures/)
results/<arch>/splits.csv             exact image → split assignment used in training
results/<arch>/training_history.json  per-epoch loss/AUC/PR-AUC (train & val)
results/<arch>/validation_metrics.json
results/<arch>/test_metrics.json      final held-out metrics
results/<arch>/per_label_metrics.csv
results/<arch>/confusion_matrices/, roc_curves/, precision_recall_curves/, *.png
results/model_comparison.csv          when more than one model has been evaluated
```

## Security and privacy decisions

- Models are loaded only from `models/` with `keras.models.load_model(..., safe_mode=True, compile=False)`;
  the app never accepts uploaded model files.
- Uploaded images are decoded with Pillow, size- and pixel-limited (decompression-bomb guard) and kept in memory.
- No network calls at inference time; no third-party APIs. (ImageNet weights are downloaded once at first training.)
- The UI hides filenames and patient IDs in the dataset explorer.
