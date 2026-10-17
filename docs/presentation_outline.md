# ThoraxAI — Presentation Outline (15 slides)

A PowerPoint generator (`scripts/make_presentation.py`) exists but is **not finished**: its current output does
not open in PowerPoint, so no `.pptx` is shipped yet. Slide 13 and parts of slide 7 are **placeholders** — replace
them only with real outputs from your own training/evaluation run.

| # | Title | Content | Visual |
|---|---|---|---|
| 1 | ThoraxAI | Title, subtitle, team names, guide, department, year | Logo |
| 2 | Introduction | Chest X-ray volume; DL for image analysis; aim of the project | — |
| 3 | Problem Statement | 14-finding multi-label scoring + explanation, leakage-free evaluation, non-diagnostic | — |
| 4 | Motivation | Binary demos ignore co-occurrence; imbalance; leakage; need for XAI | — |
| 5 | Objectives | 5 objectives (pipeline, model, imbalance/thresholds, evaluation, XAI + app) | — |
| 6 | Literature Survey | ChestX-ray14, CheXNet/CheXNeXt, DenseNet, Grad-CAM, label-noise & bias papers | Table |
| 7 | Dataset & 14 Findings | Source, size (from validation report), labels, imbalance | Prevalence chart *(from results/dataset_figures)* |
| 8 | Proposed Methodology | Validate → split → preprocess → train → thresholds → evaluate → app | Flow diagram |
| 9 | System Architecture | Offline pipeline vs Streamlit app; artifacts | Block diagram |
| 10 | Model Architecture | Input → norm → DenseNet121 → GAP → Dropout → Dense(14, sigmoid); weighted BCE | Layer diagram |
| 11 | Multi-Label Workflow | Sigmoid vs softmax, per-finding thresholds, "No Finding predicted" | Example score vector (illustrative, not a result) |
| 12 | Grad-CAM | Formula, steps, interpretation caveat | Grad-CAM screenshot *(from your app)* |
| 13 | Experimental Results | **PLACEHOLDER** — macro/micro AUC, PR-AUC, F1, per-finding table, ROC/PR curves | From `results/densenet121/` |
| 14 | Limitations & Future Scope | Labels, single source, resolution, calibration, XAI limits; future work | — |
| 15 | Conclusion & References | Key takeaways; 5–6 core references | — |

**Demo plan (after slide 12 or at the end):** Home → Setup status → X-ray Analysis (upload a test-split image,
run, adjust thresholds, Grad-CAM) → Model Performance → Dataset Explorer.
