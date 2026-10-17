# ThoraxAI: Multi-Label Chest X-ray Disease Classification Using Deep Learning and Explainable AI

> **Report draft status.** Chapters 1–18 and 21–25 are complete drafts. Chapters 19–20 (experimental setup and
> results) are **templates**: fill them only with numbers produced by `scripts/evaluate_model.py` on the real
> dataset. No experimental results exist at the time of writing.

---

## Title Page (template)

```
                        ThoraxAI: Multi-Label Chest X-ray Disease Classification
                            Using Deep Learning and Explainable AI

                     A Mini-Project Report submitted in partial fulfilment of the
                    requirements for the degree of Bachelor of Engineering in
               Computer Science and Engineering (Artificial Intelligence and Machine Learning)

                                         Submitted by
                     <Student Name 1>  (<USN/Roll No.>)
                     <Student Name 2>  (<USN/Roll No.>)
                     <Student Name 3>  (<USN/Roll No.>)
                     <Student Name 4>  (<USN/Roll No.>)

                                   Under the guidance of
                              <Guide Name>, <Designation>

                     Department of Computer Science and Engineering (AI & ML)
                                   <College Name>, <City>
                                   <Academic Year 20XX–20XX>
```

(Add certificate, declaration and acknowledgement pages according to your institution's format.)

---

## Abstract

Chest radiography is among the most frequently performed imaging examinations, and a single radiograph can show
several abnormalities at once. This project presents ThoraxAI, an educational system that assigns independent
scores to 14 thoracic findings in a frontal chest X-ray using the NIH ChestX-ray14 dataset. A DenseNet121
convolutional neural network pretrained on ImageNet is adapted through transfer learning with a new head of 14
sigmoid units, trained with a class-balanced weighted binary cross-entropy loss to counter severe label imbalance.
The data pipeline validates the dataset, detects missing and corrupted images, and enforces patient-level
separation between training, validation and test sets to prevent leakage. Per-finding decision thresholds are
selected on the validation split only, and the model is evaluated on the held-out official test split using
per-label ROC-AUC, average precision, precision, recall, specificity and F1-score with macro and micro averages.
Gradient-weighted Class Activation Mapping (Grad-CAM) produces heatmaps indicating which image regions influenced
each score. A multi-page Streamlit application integrates image upload, multi-label prediction with adjustable
thresholds, Grad-CAM visualisation, performance dashboards and dataset exploration, and is designed never to
display simulated predictions or metrics. The system is a research prototype and is not intended for clinical use.
*[Add one sentence summarising the actual test results once evaluation has been run.]*

**Keywords:** chest X-ray, multi-label classification, DenseNet121, transfer learning, Grad-CAM, explainable AI, ChestX-ray14.

---

## 1. Introduction

Medical imaging is central to modern diagnosis, and the chest radiograph is one of the most common examinations
because it is fast, inexpensive and widely available. Interpreting radiographs requires trained radiologists,
whose workload is high. Deep learning, and in particular convolutional neural networks (CNNs), has shown strong
results in image classification, motivating research on automated assistance for radiograph interpretation.

The release of large public datasets such as ChestX-ray14 [1], CheXpert [6] and MIMIC-CXR [7] enabled the training
of deep models on chest X-rays. Rajpurkar et al. [2] trained a 121-layer DenseNet (CheXNet) on ChestX-ray14 and
reported performance comparable to radiologists for pneumonia detection on a small test set, and later extended
the approach to multiple findings [15]. At the same time, studies have highlighted label noise [10], poor
generalisation across hospitals [11] and bias across patient groups [12], which are essential considerations for
any honest project in this area.

ThoraxAI implements a complete, reproducible pipeline — from dataset validation to an explainable web
application — at a scale appropriate for an undergraduate mini-project.

## 2. Problem Statement

Given a frontal chest radiograph, produce an independent likelihood score for each of 14 thoracic findings
(Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema,
Emphysema, Fibrosis, Pleural Thickening, Hernia), allow any combination of findings to be flagged, explain which
image regions influenced each score, and evaluate the system without patient-level data leakage — while clearly
communicating that the output is not a diagnosis.

## 3. Motivation

- Many student projects address binary tasks (e.g. pneumonia vs normal) that ignore co-occurring findings.
- Multi-label classification, class imbalance, threshold selection and leakage-free evaluation are core
  machine-learning skills that a realistic medical dataset exercises well.
- Explainability is increasingly expected for AI in healthcare; implementing and critically assessing Grad-CAM
  builds understanding of both its value and its limits [16, 17].

## 4. Objectives

1. Build a reproducible pipeline to validate and split ChestX-ray14 at patient level.
2. Implement a DenseNet121 transfer-learning model with 14 sigmoid outputs (MobileNetV2 as an optional comparison).
3. Address label imbalance with a weighted loss and select per-finding thresholds on validation data only.
4. Evaluate on the held-out test split with appropriate multi-label metrics.
5. Provide Grad-CAM explanations.
6. Deliver a Streamlit application that degrades gracefully when the dataset or model is missing.
7. Document limitations and ethical considerations.

## 5. Scope

**In scope:** frontal chest X-rays (PNG/JPEG), the 14 ChestX-ray14 labels, image-level classification, offline
evaluation, Grad-CAM, a local web interface.
**Out of scope:** clinical deployment, regulatory approval, DICOM ingestion, lateral views, lesion segmentation,
report generation, prospective or external-site validation.

## 6. Literature Survey

| # | Work | Contribution | Relevance |
|---|---|---|---|
| [1] | Wang et al., CVPR 2017 — ChestX-ray8/14 | Hospital-scale dataset with NLP-mined labels; weakly-supervised localisation baselines | Source dataset and label definitions |
| [2] | Rajpurkar et al., 2017 — CheXNet | DenseNet121 on ChestX-ray14, 14-label output | Primary architecture choice |
| [15] | Rajpurkar et al., PLOS Med 2018 — CheXNeXt | Multi-finding comparison with radiologists | Multi-label clinical framing |
| [3] | Huang et al., CVPR 2017 — DenseNet | Dense connectivity, feature reuse | Backbone |
| [4] | Sandler et al., CVPR 2018 — MobileNetV2 | Inverted residuals, efficient CNN | Lightweight comparison model |
| [6] | Irvin et al., AAAI 2019 — CheXpert | Uncertainty labels, expert-labelled test set | Label-uncertainty handling (future work) |
| [7] | Johnson et al., Sci. Data 2019 — MIMIC-CXR | Large public CXR dataset with reports | External validation (future work) |
| [5] | Selvaraju et al., ICCV 2017 — Grad-CAM | Gradient-weighted class activation maps | Explainability method |
| [16] | Adebayo et al., NeurIPS 2018 | Sanity checks show some saliency maps are insensitive to the model | Caution in interpreting heatmaps |
| [17] | Ghassemi et al., Lancet Digit. Health 2021 | Critique of post-hoc explanations in health care | Ethical framing of XAI |
| [10] | Oakden-Rayner, Acad. Radiol. 2020 | Label quality problems in public CXR datasets | Limitations |
| [11] | Zech et al., PLOS Med 2018 | Models learn hospital-specific confounders; poor cross-site generalisation | Limitations |
| [12] | Seyyed-Kalantari et al., Nat. Med. 2021 | Under-diagnosis bias in under-served groups | Fairness |
| [13] | Yosinski et al., NeurIPS 2014 | Transferability of CNN features | Transfer learning rationale |
| [14] | Saito & Rehmsmeier, PLOS ONE 2015 | PR curves more informative than ROC under imbalance | Metric choice |

## 7. Dataset Description

ChestX-ray14 was released by the NIH Clinical Center [1] and is mirrored on Kaggle
(`https://www.kaggle.com/datasets/nih-chest-xrays/data`). It is commonly described as 112,120 frontal-view
radiographs of 30,805 unique patients with 14 finding labels text-mined from radiology reports, distributed as
1024 × 1024 PNGs with a metadata CSV (`Data_Entry_2017.csv`) and official patient-disjoint split lists.

*Table 7.1 — Verified dataset statistics (fill from `results/dataset_validation.json` / `dataset_label_counts.csv`):*

| Quantity | Value |
|---|---|
| Metadata rows | _from report_ |
| Image files found | _from report_ |
| Patients | _from report_ |
| Missing / corrupted images | _from report_ |
| Train / val / test images (patients) | _from report_ |

*Table 7.2 — Positives per finding (fill from `dataset_label_counts.csv`)*, and include
`results/dataset_figures/label_prevalence.png` and `label_cooccurrence.png` as figures.

## 8. Limitations of Existing Systems

- Binary classifiers ignore co-occurring findings.
- Random image-level splits leak patients between training and test sets and inflate reported performance.
- Accuracy-only reporting hides poor sensitivity on rare findings.
- Fixed 0.5 thresholds are rarely appropriate for imbalanced labels.
- Many demos show predictions without explanation or show heatmaps as if they were proof.
- Some student applications display hard-coded or simulated outputs.

## 9. Proposed System

ThoraxAI addresses these points with (i) a multi-label sigmoid model, (ii) patient-level splits that preserve the
official test set, (iii) a class-balanced weighted loss, (iv) per-finding thresholds chosen on validation data,
(v) comprehensive per-label and averaged metrics, (vi) Grad-CAM with explicit caveats, and (vii) an application
that reads only real artifacts and shows setup guidance otherwise.

## 10. System Architecture

See `docs/architecture.md` for the full diagram. The system has an offline pipeline (validation → training →
evaluation, implemented as scripts over framework-independent modules) and an interactive Streamlit application
that loads the trained model for inference and Grad-CAM and reads the evaluation files for its dashboards.
*[Insert the architecture figure — e.g. a screenshot of the diagram on the app's About page.]*

## 11. Methodology

1. Acquire and extract the dataset locally.
2. Validate: counts, missing/duplicate/corrupted files, label statistics, patient overlap.
3. Split at patient level (official test list preserved; 10 % of train/val patients for validation).
4. Optionally select a development subset of whole patients.
5. Preprocess and stream batches with `tf.data`; augment training data only.
6. Train the head, then optionally fine-tune upper backbone layers.
7. Select thresholds on validation predictions.
8. Evaluate once on the test split.
9. Serve the model in Streamlit with Grad-CAM.

## 12. Multi-Label Classification

In multi-class classification each image has exactly one label, and softmax outputs sum to one. In multi-label
classification each image has a *set* of labels; the task is decomposed into C binary problems that share a
feature extractor. Each output uses a sigmoid, `p_c = σ(z_c)`, so any number of findings may exceed their
thresholds. Targets are multi-hot vectors (e.g. `[0,1,1,0,…]` for cardiomegaly + effusion); "No Finding" is the
all-zero vector. Evaluation is per label and then averaged (macro: equal weight per finding; micro: pooled over all
image–finding pairs).

## 13. CNNs and Transfer Learning

A CNN learns hierarchical features via convolution, non-linearity and pooling. DenseNet121 [3] connects each layer
in a dense block to all preceding layers through concatenation, which improves gradient flow and feature reuse;
it has about 7 million parameters without its classifier. Features learned on ImageNet transfer to other visual
domains [13]; ThoraxAI therefore initialises the backbone with ImageNet weights, freezes it to train a new head,
and optionally fine-tunes the final layers with a low learning rate while keeping BatchNorm statistics frozen.

## 14. Data Preprocessing

See `docs/methodology.md` §5. Summary: safe decoding, grayscale-to-RGB replication, resizing to 224 × 224,
backbone-specific normalisation embedded in the model, and mild training-only augmentation (no horizontal flips,
which would swap anatomical laterality).

## 15. Training Procedure

- Optimiser AdamW (lr 1e-3, weight decay 1e-5); batch 16 (CPU default).
- Loss: weighted BCE, `L = −(1/C) Σ_c [w⁺_c y_c log p_c + w⁻_c (1−y_c) log(1−p_c)]` with
  `w⁺_c = N/(2P_c)`, `w⁻_c = N/(2(N−P_c))` computed on the training split.
- Early stopping on validation macro ROC-AUC (patience 3) with best-weight restoration; learning-rate reduction on plateau.
- Artifacts: best/final/last checkpoints, configuration, label mapping, thresholds, history, software versions, training time.

## 16. Evaluation Metrics

| Metric | Definition | Why it matters |
|---|---|---|
| Precision | TP / (TP + FP) | Fraction of flagged findings that are labelled positive |
| Recall / sensitivity | TP / (TP + FN) | Fraction of positives found; missed findings are costly |
| Specificity | TN / (TN + FP) | Fraction of negatives correctly not flagged |
| F1 | Harmonic mean of precision and recall | Balances FP and FN at a threshold |
| ROC-AUC | Area under TPR-vs-FPR curve | Threshold-independent ranking quality |
| Average precision (PR-AUC) | Area under precision–recall curve | More informative than ROC under imbalance [14] |
| Macro / micro averages | Mean over labels / pooled over pairs | Rare vs common finding emphasis |

Confusion matrices are reported per finding at the validation-selected thresholds.

## 17. Grad-CAM

For finding c and last-layer feature maps `A^k`, Grad-CAM [5] computes `α_k = mean_ij ∂y^c/∂A^k_ij` and
`L^c = ReLU(Σ_k α_k A^k)`, which is upsampled and overlaid on the input. In ThoraxAI the map is 7 × 7 for a 224 px
input, so it is coarse. Saliency methods can fail sanity checks [16] and post-hoc explanations should not be
treated as justification of individual decisions [17]; the application states this next to every heatmap.

## 18. Implementation

Python 3.12, TensorFlow 2.18 / Keras 3, NumPy, Pandas, Pillow, scikit-learn, Matplotlib, Seaborn, Streamlit.
Code is modular (`src/`), with CLI scripts (`scripts/`), a multi-page app (`app.py`, `pages/`) and a pytest suite
(`tests/`). See `README.md` for commands and `docs/testing.md` for test outcomes.

## 19. Experimental Setup — TEMPLATE

| Item | Value |
|---|---|
| Hardware | _e.g. CPU model / GPU model, RAM_ |
| Software | _from `models/densenet121/model_config.json` → environment_ |
| Mode | _development (N images) / full_ |
| Image size, batch size | _ |
| Epochs (head / fine-tune) | _ |
| Learning rates | _ |
| Loss | weighted BCE |
| Threshold method | max-F1 on validation |
| Split method and sizes | _from model_config.json_ |
| Training time | _from model_config.json_ |

Measured on the development laptop (Windows 11, 12-thread CPU, TensorFlow CPU): one training step with a frozen
DenseNet121 backbone took ≈72 ms per image at 224 px (MobileNetV2 ≈23 ms). This is a timing measurement only, not
a performance result.

## 20. Results and Discussion — TEMPLATE (do not fill with estimates)

*Table 20.1 — Test-set metrics per finding* (copy from `results/densenet121/per_label_metrics.csv`).

| Finding | Support | Threshold | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| … | | | | | | | | |
| **Macro** | | | | | | | | |
| **Micro** | | | | | | | | |

Figures to include: training curves (`training_curves.png`), ROC (`roc_curves/roc_all.png`), PR
(`precision_recall_curves/pr_all.png`), confusion matrices (`confusion_matrices/confusion_matrices_all.png`),
Grad-CAM examples (saved from the app to `results/gradcam/`), and `model_comparison.csv` if MobileNetV2 was trained.

Discussion prompts: Which findings are ranked well (high AUC) and which are not? How do PR-AUC values compare with
prevalence (the baseline AP equals prevalence)? Where do false negatives concentrate? Did fine-tuning help on
validation? If development mode was used, state the subset size and that results do not represent full-dataset
performance. Compare cautiously with published numbers [2, 15], noting differences in splits, labels and resolution.

## 21. Limitations

See `docs/limitations.md`. Key points: noisy NLP-derived labels, single-institution data, 224 px resolution,
uncalibrated scores, F1-optimised (non-clinical) thresholds, possible shortcut learning, coarse and potentially
unfaithful heatmaps, no subgroup analysis, and no clinical validation.

## 22. Ethical Considerations

Patient safety (explicit non-diagnostic disclaimers), privacy (local processing, no identifiers shown, dataset
terms respected), fairness (bias not yet audited; known under-diagnosis risks [12]), transparency (all numbers
read from generated files) and accountability (decisions remain with clinicians).

## 23. Conclusion

ThoraxAI delivers an end-to-end, tested and documented pipeline for multi-label chest X-ray classification with
explainability. Its design emphasises correctness — patient-level splits, validation-only threshold selection,
imbalance-aware training and metrics, and refusal to show fabricated outputs. *[Add a summary of the measured test
results and what they show once evaluation is complete.]*

## 24. Future Enhancements

Higher resolution and full fine-tuning on GPU; ensembles; probability calibration; uncertainty-aware training with
CheXpert; external validation on MIMIC-CXR; bootstrap confidence intervals; subgroup fairness analysis; localisation
with the NIH bounding-box subset; alternative explainers (Grad-CAM++, Score-CAM) with sanity checks; DICOM support.

## 25. References

1. X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, R. M. Summers, "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases," *Proc. IEEE CVPR*, 2017.
2. P. Rajpurkar, J. Irvin, K. Zhu, B. Yang, H. Mehta, T. Duan, D. Ding, A. Bagul, C. Langlotz, K. Shpanskaya, M. P. Lungren, A. Y. Ng, "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning," arXiv:1711.05225, 2017.
3. G. Huang, Z. Liu, L. van der Maaten, K. Q. Weinberger, "Densely Connected Convolutional Networks," *Proc. IEEE CVPR*, 2017.
4. M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," *Proc. IEEE CVPR*, 2018.
5. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," *Proc. IEEE ICCV*, 2017.
6. J. Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison," *Proc. AAAI*, 2019.
7. A. E. W. Johnson et al., "MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports," *Scientific Data*, 6, 317, 2019.
8. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," *Proc. IEEE CVPR*, 2009.
9. D. P. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization," *ICLR*, 2015; I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization," *ICLR*, 2019.
10. L. Oakden-Rayner, "Exploring Large-scale Public Medical Image Datasets," *Academic Radiology*, 27(1), 106–112, 2020.
11. J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, E. K. Oermann, "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study," *PLOS Medicine*, 15(11), e1002683, 2018.
12. L. Seyyed-Kalantari, H. Zhang, M. B. A. McDermott, I. Y. Chen, M. Ghassemi, "Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations," *Nature Medicine*, 27, 2176–2182, 2021.
13. J. Yosinski, J. Clune, Y. Bengio, H. Lipson, "How transferable are features in deep neural networks?," *NeurIPS*, 2014.
14. T. Saito, M. Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets," *PLOS ONE*, 10(3), e0118432, 2015.
15. P. Rajpurkar et al., "Deep learning for chest radiograph diagnosis: A retrospective comparison of the CheXNeXt algorithm to practicing radiologists," *PLOS Medicine*, 15(11), e1002686, 2018.
16. J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, B. Kim, "Sanity Checks for Saliency Maps," *NeurIPS*, 2018.
17. M. Ghassemi, L. Oakden-Rayner, A. L. Beam, "The false hope of current approaches to explainable artificial intelligence in health care," *The Lancet Digital Health*, 3(11), e745–e750, 2021.

*Before submission, verify every reference (title, venue, year, pages) against the original publication or
Google Scholar, and format according to your department's citation style.*
