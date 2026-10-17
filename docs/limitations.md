# ThoraxAI — Limitations and Status

## What is implemented vs. what has been done

| Item | Status |
|---|---|
| Source code for validation, training, evaluation, inference, Grad-CAM, app | Implemented |
| Unit / integration tests | Implemented and run (see `docs/testing.md`) |
| End-to-end pipeline execution | Verified on synthetic images (software tests) and on the real dataset |
| Training on NIH ChestX-ray14 | **Development mode only**: 2,087 training images / 705 patients (224 px copy) |
| Test-set evaluation on real data | Done on a 687-image held-out subset (67 patients): macro ROC-AUC 0.584, macro PR-AUC 0.099, macro F1 0.108 |
| Full-dataset training | **Not performed** |
| Clinical validation | **Not performed and out of scope** |

The current results come from a small development subset. They are only modestly better than chance and do not
represent what the architecture achieves with full-dataset training. Quote them only with that context.

## Technical limitations

- **Labels:** NLP-mined from reports, not image-level radiologist annotations; documented label noise. No explicit uncertainty labels.
- **Single source:** one hospital system; performance may drop on other scanners, populations or protocols.
- **Resolution:** 224 × 224 input discards detail relevant to small nodules or subtle pneumothorax.
- **Imbalance:** e.g. Hernia is very rare; per-label metrics for rare findings have wide uncertainty (no confidence intervals are computed).
- **Thresholds:** chosen to maximise validation F1, not a clinically chosen operating point.
- **Calibration:** scores are not calibrated probabilities.
- **Shortcut learning:** models can exploit chest drains, lines, text markers or view position (AP vs PA).
- **Grad-CAM:** coarse (7 × 7), can be visually plausible yet unfaithful; not evidence of correctness.
- **Input checking:** the app cannot detect that an upload is not a chest X-ray.
- **Reproducibility:** seeds are fixed, but GPU/oneDNN kernels may introduce small run-to-run differences.
- **Hardware:** native-Windows TensorFlow ≥ 2.11 is CPU-only; GPU training requires Linux or WSL2.

## Ethical considerations

- Not for diagnosis, triage or treatment; disclaimers are shown on every analysis-related page.
- Bias across age, sex and view position has not been audited.
- Data use must follow the NIH/Kaggle dataset terms; images must not be redistributed.
- No patient identifiers are displayed; uploaded images are processed locally and not stored unless the user saves a Grad-CAM image.
