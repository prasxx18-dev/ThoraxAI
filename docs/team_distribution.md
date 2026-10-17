# ThoraxAI — Team Work Distribution (4 members)

| Member | Primary responsibility | Files owned | Deliverables | Review partner |
|---|---|---|---|---|
| **Member 1** — Data | Dataset download & setup, metadata parsing, label encoding, patient-level splits, preprocessing, dataset analysis | `src/dataset.py`, `src/preprocessing.py`, `scripts/validate_dataset.py`, `tests/test_dataset.py`, `tests/test_preprocessing.py` | `results/dataset_validation.*`, dataset figures, report ch. 7 & 14 | Member 3 |
| **Member 2** — Model | DenseNet121/MobileNetV2 implementation, weighted loss, two-stage training, checkpoints, resume, run configuration | `src/model.py`, `src/train.py`, `scripts/train_model.py`, `tests/test_model.py` | `models/<arch>/*`, training curves, report ch. 13, 15, 19 | Member 4 |
| **Member 3** — Evaluation & XAI | Metrics, threshold selection, evaluation plots, Grad-CAM, model comparison | `src/metrics.py`, `src/evaluate.py`, `src/plots.py`, `src/gradcam.py`, `tests/test_metrics.py` | `results/<arch>/*`, Grad-CAM examples, report ch. 16, 17, 20 | Member 1 |
| **Member 4** — Application & integration | Streamlit UI, inference wrapper, integration, testing, documentation, report & presentation coordination | `app.py`, `pages/`, `src/ui.py`, `src/predict.py`, `tests/test_app.py`, `tests/test_prediction.py`, `docs/` | Working demo, test report, final report & slides | Member 2 |

## Suggested timeline (6 weeks)

| Week | Milestone |
|---|---|
| 1 | Environment, dataset download, validation report (M1); literature survey (all) |
| 2 | Preprocessing + splits finalised (M1); model built and smoke-trained in development mode (M2) |
| 3 | Metrics & thresholds (M3); app pages with empty states (M4) |
| 4 | Development-mode training + evaluation; Grad-CAM in app (M2, M3, M4) |
| 5 | Full or larger training run if hardware allows; optional MobileNetV2 comparison; tests (all) |
| 6 | Report results chapter, slides, viva practice — each member explains the full pipeline |

All members must be able to explain the complete workflow: data → splits → model → loss → thresholds → metrics → Grad-CAM → app.
