# ThoraxAI — Test Report

**Run:** 2026-10-17 · Windows 11 · Python 3.12.10 · TensorFlow 2.18.0 · Keras 3.15.1 · Streamlit 1.66.0 · CPU only
**Command:** `.venv\Scripts\python -m pytest -q tests`
**Result:** **108 passed, 0 failed, 0 skipped** (≈50 s). One warning: Keras notes that `shuffle=True` is ignored for
`tf.data` input. This is expected, because the training pipeline shuffles inside `tf.data`.

> All tests use **synthetic random-noise images** and mocked or randomly initialised models. They verify the
> software. They are **not** model evaluation results and must not be quoted as performance.

## Coverage by requirement

| # | Requirement | Tests | Result |
|---|---|---|---|
| 1 | Metadata CSV loading | `test_dataset.py::test_load_metadata`, `::test_load_metadata_kaggle_header_format` | Pass |
| 2 | Missing / malformed metadata | `::test_load_metadata_missing_file`, `::test_load_metadata_missing_columns`, `::test_load_metadata_empty`, `::test_duplicate_metadata_rows_are_reported` | Pass |
| 3 | Label mapping | `::test_label_mapping_has_14_unique_labels`, `::test_parse_finding_labels[6 cases]` | Pass |
| 4 | Multi-label encoding | `::test_multilabel_encoding_roundtrip` | Pass |
| 5 | Image preprocessing | `test_preprocessing.py::test_resize_and_value_range`, `::test_jpeg_supported`, `::test_tf_pipeline_is_deterministic_for_eval` | Pass |
| 6 | Grayscale / 16-bit / RGBA handling | `::test_grayscale_converted_to_rgb`, `::test_16bit_and_rgba_images` | Pass |
| 7 | Corrupted / invalid image rejection | `::test_invalid_uploads_rejected[3 cases]`, `::test_truncated_png_rejected`, `::test_too_small_and_too_large`, `::test_verify_image_file` | Pass |
| 8 | Dataset validation | `test_dataset.py::test_validate_dataset_report`, `::test_image_index_detects_duplicates_and_missing`, `::test_discovery_and_missing_dataset` | Pass |
| — | Patient-level splits, no leakage | `::test_official_split_has_no_patient_overlap`, `::test_grouped_split_is_reproducible_and_leak_free`, `::test_development_subset_keeps_patients_together` | Pass |
| 9 | Missing model checkpoint | `test_model.py::test_missing_checkpoint_raises`, `test_prediction.py::test_missing_checkpoint` | Pass |
| 10 | Prediction output shape | `test_prediction.py::test_prediction_output_shape_and_multilabel`, `::test_bad_model_output_shape`, `::test_from_model_dir_with_real_small_model`; `test_model.py::test_model_has_14_sigmoid_outputs`, `::test_densenet121_builds` | Pass |
| 11 | Threshold-based label mapping | `test_prediction.py::test_flag_findings_threshold_mapping`, `::test_no_finding_predicted_status`, `::test_custom_threshold_override`, `::test_missing_thresholds_are_disclosed` | Pass |
| 12 | Metric calculation | `test_metrics.py` (8 tests incl. undefined AUC, macro vs micro, threshold selection f1/youden) | Pass |
| 13 | Missing evaluation files | `test_metrics.py::test_missing_evaluation_files`, `test_app.py::test_performance_page_empty_state` | Pass |
| 14 | Grad-CAM error handling | `test_model.py::test_gradcam_output_and_errors`, `::test_gradcam_rejects_unsupported_model` | Pass |
| — | Local dataset configuration, image-size survey, training dry run | `test_config.py` (4 tests) | Pass |
| 15 | Streamlit startup | `test_app.py` (19 tests: 4 pages and 3 Insights views without dataset/model, empty states, home call-to-action navigation, usage notices, no academic wording, system-status dataset checks, stale-config regression, no legacy `pages/` folder) | Pass |
| — | Model internals | `test_model.py::test_unfreeze_keeps_batchnorm_frozen`, `::test_label_weights_favor_rare_labels`, `::test_weighted_bce_matches_plain_bce_with_unit_weights`, `::test_save_and_load_checkpoint`, `::test_unknown_architecture` | Pass |
| — | End-to-end pipeline | `test_pipeline.py::test_train_and_evaluate_end_to_end`: trains (head + fine-tune), selects thresholds, evaluates, writes all artifacts, then renders the Analysis, Performance and Dataset pages from those artifacts | Pass |

## Manual verification

| Check | How | Result |
|---|---|---|
| App starts with no dataset and no model | `streamlit run app.py`, every page opened in the browser | Home shows "Dataset: Missing, Trained models: 0". Analysis is disabled with training commands. Performance shows "No evaluation results yet" with no numbers. Dataset Explorer shows "Dataset not configured". Setup Guide shows an unchecked checklist |
| Upload → analysis → Grad-CAM in the browser | App pointed (via `THORAXAI_*` env vars) at a throwaway model trained on synthetic noise in a temp folder. A generated grayscale PNG was uploaded | Upload validated, 14 scores shown, Grad-CAM overlay rendered, download/save buttons present |
| Bug found and fixed in the browser | The Performance page's training-history chart crashed (`melt` `value_name` collided with a column name) | Fixed. Regression covered by `test_pipeline.py` |
| CLI scripts with missing data | `validate_dataset.py` and `evaluate_model.py` with no dataset/model | Clear error message, exit code 2 |
| CLI scripts end to end | validate → train → evaluate on a synthetic dataset in a temp folder | All artifacts written. Metrics at chance level (macro AUC 0.5), as expected for noise, which suggests no leakage |
| ImageNet weights | Built DenseNet121 and MobileNetV2 with `pretrained=True` | Weights downloaded and loaded |
| CPU timing | Frozen-backbone training step at 224 px, batch 16 | DenseNet121 ≈ 72 ms/image, MobileNetV2 ≈ 23 ms/image |

## Percentage results and simplified cards verification (2026-10-17)

**Output type.** The saved `best_model.keras` ends in `predictions` with activation `sigmoid` (14 units). Outputs on
test inputs fall in [0, 1] and do not sum to 1, so they are multi-label sigmoid outputs and not softmax.
Percentage = output × 100, and no extra sigmoid is applied. No calibration has been performed, so the UI says
"Model-estimated likelihood".

**Automated tests:**
- `test_prediction.py` covers:
  - a sigmoid model's percentages equal its raw output × 100 (no double sigmoid)
  - a linear (logit) model gets sigmoid exactly once
  - a softmax model is rejected
  - out-of-range outputs are refused
  - each percentage maps to the correct label
- `test_findings.py` covers:
  - `to_percent` rounding, clamping and NaN/inf rejection
  - one- or two-sentence explanations for all 14 labels from trusted sources
  - cards contain only the name, percentage and "What it means" (no decimals, thresholds or old sections)
  - the neutral "No findings crossed the current display threshold." message
- `test_app.py` checks that the analysis page shows exactly one notice and no repeated warnings.

**Live app** (headless Edge via the DevTools protocol, real DenseNet121, two held-out NIH test images):

| Check | Result |
|---|---|
| Image A results | 7 cards: Consolidation 79.4%, Cardiomegaly 74.4%, Infiltration 74.1%, Edema 68.9%, Pneumonia 63.2%, Atelectasis 59.2%, Pleural Thickening 55.8% |
| Same image via `Predictor` outside the UI | Identical percentages |
| Card content | Name, percentage, "What it means" only; no raw decimals, thresholds or old sections |
| All 14 outputs list | 14 rows, percentages only |
| Notices | Exactly one notice at the bottom; no repeated warnings |
| Replace image (B) | A's results removed before re-analysis (0 cards); B gives 6 cards (Fibrosis 80.2% …), identical to the direct run |
| Remove image | Results and preview cleared |
| Corrupted file | Friendly message, Analyze X-ray disabled, no traceback, path or decoder internals shown |
| Grad-CAM | "Original X-ray" and "Grad-CAM heatmap · Consolidation" rendered side by side |
| About | Evaluation tiles read from `results/densenet121/test_metrics.json`: ROC-AUC 0.584, PR-AUC 0.099, F1 0.108, 687 test images |
| Responsive | 390 px: 1 card column, no horizontal scroll. 820 px: 2 columns. 1366 px: 3 columns. No overflowing elements |

Not verified: clicking "Download heatmap" (it saves a file), and the missing-model page in a live browser (it is
covered by `test_app.py` instead).

## Plain-language findings verification (2026-10-17, superseded by the simplified cards above)

`tests/test_findings.py` (9 tests) checks the following:
- the knowledge base covers all 14 labels in model order
- every source is an HTTPS link on an allow-listed reputable domain
- no advice or certainty wording appears ("you should", "prescrib…", "definitely", …)
- symptom text states when sources say symptoms can be absent
- summary wording for the flagged, none-flagged (never "normal") and near-threshold cases, with real scores unchanged
- the finding cards render

In the live app with the trained DenseNet121 and a held-out NIH test image:

| Check | Result |
|---|---|
| AI summary | "7 of 14 findings reached their review threshold", listing the real scores (Consolidation 0.794 …) |
| Flagged cards | 7 cards with score and threshold at 3 decimals (e.g. Edema 0.689 vs 0.688 — previously shown as 0.69, now fixed) |
| No-finding path | Single threshold at 0.95 → "No finding reached its review threshold … not the same as a normal X-ray" |
| Grad-CAM | Uploaded X-ray and heatmap side by side, plus the "how to read it" panel |
| Reference guide | 14 general cards, clearly separated from image-specific results, with source links |

## Website redesign verification (2026-10-17)

Browser checks against a fresh server (port 8502) used the trained DenseNet121 and a held-out test-split image
(`00000013_004.png`, dataset labels Effusion|Emphysema|Infiltration|Pneumothorax), uploaded through the real file input.

| Check | Result |
|---|---|
| Upload → preview | 224 × 224 image previewed with file details |
| Analyze → results | 14 scores; 7 above threshold; highest Consolidation 0.794; ~520–650 ms |
| Scores are genuine | Identical to `Predictor.predict` run outside the UI (0.794 / 0.770 / 0.744 …) |
| Grad-CAM | Generated for the top finding; switching to Cardiomegaly regenerated it (score 0.744) |
| Reset | "Analyze another image" clears upload, results and heatmaps |
| Corrupt file | Friendly error, Analyze disabled (decoder internals are no longer shown) |
| Insights | Real test metrics (macro ROC-AUC 0.584), curves, confusion matrices, training history, dataset counts, system status (dataset found at the configured local folder) |
| About anchors | `/about#privacy` scrolls to the Privacy section |
| Mobile (375 px) | Layout stacks, navigation collapses to a menu, no horizontal scroll |
| Navigation bug found and fixed | Page files in a `pages/` folder collided with Streamlit's legacy auto-navigation, which replaced the top bar with a hidden sidebar on direct URLs. Fixed by moving them to `app_pages/`; a guard test prevents regressions |

Not verified in the browser: clicking *Download heatmap (PNG)*, because it saves a file to the machine. The button
renders, and its PNG bytes come from the same `overlay_heatmap` path that is unit-tested.

## Real dataset checks (2026-10-17)

Dataset: a local Kaggle copy with `images-224/`, kept outside the project folder and configured through `local_config.json`.

| Check | Command | Result |
|---|---|---|
| Validation, every image decoded | `scripts\validate_dataset.py --verify all` (68 s) | 112,120 metadata rows, 112,120 images found, 0 missing, 0 duplicates, **0 corrupted**, 30,805 patients, all sampled images 224×224 (grayscale, rarely grayscale+alpha) |
| Official split + leakage | same | train 77,738 / val 8,786 / test 25,596 images; patient overlap 0 / 0 / 0 |
| Grayscale+alpha decoding | TF `decode_image(channels=3)` vs Pillow | Identical 3-channel grayscale |
| Training compatibility | `scripts\train_model.py --mode development --dry-run` | Subset 3,042 images (train 2,087 / val 268 / test 687); batches `[16,224,224,3]`, labels `[16,14]`; ImageNet weights loaded; output `[16,14]`; nothing written |

These are data checks only. **No model has been trained, and no performance metrics exist.**

## Not executed

- Training and evaluation on the real dataset: deliberately not started yet.
- GPU training (native-Windows TensorFlow is CPU-only).
- Upload interaction under automated tests: Streamlit's `AppTest` cannot drive `st.file_uploader`. The upload path
  is covered by unit tests of `load_image` / `Predictor` and by the manual browser check above.

## Remaining issues

- `scripts/make_presentation.py` produces a file that passes OOXML validation but does not open in PowerPoint.
  It is parked for later, and no `.pptx` is shipped.
- Grad-CAM is coarse (7×7 feature map), and scores are uncalibrated. These are documented limitations, not defects.
