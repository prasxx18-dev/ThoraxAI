# ThoraxAI — Methodology

## 1. Dataset

NIH Clinical Center ChestX-ray14 (Wang et al., 2017), Kaggle mirror `nih-chest-xrays/data`.
Commonly reported size: 112,120 frontal images from 30,805 patients, 14 report-mined labels plus
"No Finding". **Always use the counts produced by `scripts/validate_dataset.py` for your copy.**

Files used: `Data_Entry_2017*.csv` (columns `Image Index`, `Finding Labels`, `Patient ID`, …),
`train_val_list.txt`, `test_list.txt`, and the PNGs (Kaggle layout `images_001/images/` … `images_012/images/`).

## 2. Label policy

- 14 independent binary targets in a fixed order (`config.LABELS`).
- `Finding Labels` is split on `|`; tokens are matched case-insensitively; unknown tokens are reported, not silently mapped.
- "No Finding" → all-zero target. It is **not** a 15th output.
- Labels are NLP-extracted from reports; there is no explicit "uncertain" label in this dataset. Any uncertainty
  is therefore unrepresented, an absent label is treated as negative for training, and this limitation is disclosed.
- The app's "No Finding predicted" status = no score ≥ its threshold (a model-generated status, not a normal diagnosis).

## 3. Splitting and leakage prevention

1. If `train_val_list.txt` and `test_list.txt` exist, the official (patient-disjoint) test list is the test set.
2. Validation = 10 % of the *patients* in the train/val list (seeded, default 42).
3. Without the lists, a reproducible patient-grouped random split (70/10/20) is created.
4. Splits are computed on the full metadata before missing files are removed, so a partial download cannot change them.
5. Training aborts if any patient appears in more than one split. The exact split is saved to `results/<arch>/splits.csv`
   and reused by evaluation.

## 4. Development vs full mode

| | Development | Full |
|---|---|---|
| Images | `--dev-max-images` (default 3,000) | All available |
| Selection | Whole patients sampled per split (proportional budget), then one extra patient added per split for any label without a positive | — |
| Purpose | Prove the pipeline on a CPU in reasonable time | Final experiment (GPU recommended) |
| Reporting | Every artifact is labelled "DEVELOPMENT SUBSET" | — |

No oversampling by copying images is performed.

## 5. Preprocessing

| Step | Choice | Reason |
|---|---|---|
| Decode | Pillow (app) / `tf.io.decode_image` (training), 3 channels | Robust decoding of PNG/JPEG; grayscale replicated to RGB for ImageNet backbones |
| 16-bit / alpha | Min–max to 8-bit; alpha composited on black | Consistent input range |
| Resize | 224 × 224 bilinear | ImageNet default; NIH images are square (1024²) |
| Normalise | Inside model: ImageNet mean/std (DenseNet121) or [-1, 1] (MobileNetV2) | Single source of truth |
| Augment (train only) | Rotation ±0.03 turn (≈±11°), zoom ±8 %, shift ±5 %, contrast ±10 % | Mild positional/intensity variability |
| Not used | Horizontal flip, strong crops, colour jitter | Flips swap anatomical laterality (heart side, effusion side); crops could remove findings |
| Validation | Corrupted files excluded after integrity check; size limits for uploads | Prevents silent training crashes |

## 6. Model and training

- Backbone: DenseNet121 (ImageNet), frozen; head: GAP → Dropout(0.3) → Dense(14, sigmoid).
- Loss: class-balanced weighted BCE, `w+ = N/(2P)`, `w− = N/(2(N−P))`, clipped to [0.05, 50], computed on the training split only. Plain BCE is available with `--loss bce`.
- Optimiser: AdamW (lr 1e-3, weight decay 1e-5).
- Stage 1: head only, `--epochs` (default 5). Stage 2 (optional): unfreeze last `--fine-tune-layers` (default 40) backbone layers except BatchNorm, lr 1e-5.
- Callbacks: ModelCheckpoint (best val macro AUC), last-epoch checkpoint, EarlyStopping (patience 3, restore best), ReduceLROnPlateau (val loss), CSVLogger.
- Seeds fixed for Python/NumPy/TF/Keras (GPU kernels may still be non-deterministic).

## 7. Thresholds

Per-label threshold that maximises F1 on the **validation** split (`--threshold-method youden` maximises
sensitivity + specificity − 1 instead). Labels with only one class in validation keep 0.5. Thresholds are saved to
`threshold_config.json` and used unchanged for the test set and the app. The test set is never used for tuning.

## 8. Evaluation

Per label: precision, recall (sensitivity), specificity, F1, ROC-AUC, average precision (PR-AUC), confusion counts,
support. Macro = unweighted mean over labels where defined (labels without positives are excluded from macro
AUC/AP and listed). Micro = pooled over all image-label pairs. Also: exact-match ratio, Hamming loss, model size,
batch throughput and single-image latency. Accuracy is not used as a headline metric because of class imbalance.

Scores are uncalibrated sigmoid outputs; no calibration analysis has been performed.

## 9. Explainability

Grad-CAM on the backbone's last feature map (7 × 7 for 224 input). Gradients of the chosen sigmoid score are
global-average-pooled into channel weights; ReLU of the weighted sum gives the map, normalised to [0, 1] and
upsampled bilinearly. Interpretation is limited: the map shows influence on the score, not disease location.
