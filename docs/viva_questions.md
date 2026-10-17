# ThoraxAI — Viva Preparation (35 questions)

**1. Why did you choose this project?**
It combines core ML topics — CNNs, transfer learning, multi-label learning, class imbalance, leakage-free evaluation and explainability — on a realistic public medical dataset, and goes beyond the common binary pneumonia classifier.

**2. Why are chest X-rays suitable for deep learning?**
They are common, standardised 2-D images available in large public datasets (e.g. ChestX-ray14 with ~112k images), and many findings have visual patterns a CNN can learn.

**3. What dataset did you use?**
NIH ChestX-ray14 from Kaggle: frontal radiographs with 14 report-mined labels, patient IDs and official patient-disjoint train/test lists. We verify the actual counts with our validation script.

**4. How were the labels created, and why does that matter?**
By NLP on radiology reports, not by radiologists annotating each image. Labels are noisy, so model performance is limited by label quality, and an absent label doesn't prove a finding is absent.

**5. Multi-label vs multi-class?**
Multi-class: exactly one label per image. Multi-label: any subset of labels. A radiograph can show cardiomegaly and effusion together, so this is multi-label.

**6. Why sigmoid instead of softmax?**
Softmax makes outputs compete and sum to 1, which implies one class. Sigmoid scores each finding independently in [0, 1], so several can be high at once.

**7. What loss did you use and why?**
Binary cross-entropy per output, averaged over 14 labels, with class-balanced weights so rare positive labels contribute more (w⁺ = N/2P, w⁻ = N/2(N−P)).

**8. How is "No Finding" handled?**
As an all-zero target, not a 15th class. In the app, "No Finding predicted" means no score crossed its threshold — a model status, not a normal diagnosis.

**9. What is a CNN?**
A network using learned convolution filters, non-linear activations and pooling to build features hierarchically, from edges to complex shapes, with shared weights giving translation equivariance.

**10. Why DenseNet121?**
Dense connections give strong gradient flow and feature reuse with few parameters (~7M without top); it is the architecture used by CheXNet on this dataset, so it is a well-established baseline.

**11. What is transfer learning?**
Reusing a model trained on a large dataset (ImageNet) as initialisation. Low-level features transfer, so we need less data and time than training from scratch.

**12. Explain your two-stage training.**
Stage 1 freezes the backbone and trains only the new head. Stage 2 (optional) unfreezes the last ~40 layers with a 100× lower learning rate, keeping BatchNorm frozen to avoid destroying pretrained statistics.

**13. Why keep BatchNorm frozen during fine-tuning?**
Small batches give noisy batch statistics; updating them can wreck pretrained features. Calling the backbone with `training=False` keeps the moving mean/variance fixed.

**14. What preprocessing do you apply?**
Decode, convert grayscale to 3-channel RGB, resize to 224×224, and normalise with ImageNet mean/std — normalisation is a layer inside the model so training and inference can't differ.

**15. What augmentations, and why no horizontal flip?**
Small rotation, zoom, shift and contrast. Flipping swaps left and right — the heart is on the left and effusions are side-specific — creating anatomically unrealistic images.

**16. What is data leakage here and how did you prevent it?**
The same patient's images in both training and test sets let the model recognise the patient, inflating scores. We split by patient (using the official test list), verify zero overlap, and abort training if any is found.

**17. Train, validation, test — roles?**
Train: fit weights. Validation: early stopping, checkpoint choice, threshold selection. Test: used once at the end for the final, unbiased estimate.

**18. Why not tune thresholds on the test set?**
That leaks test information into model decisions and makes the reported performance optimistic.

**19. How did you handle class imbalance?**
Weighted BCE, per-label thresholds, and metrics that are robust to imbalance (PR-AUC, per-label recall/F1, macro averages) instead of accuracy. We do not oversample by duplicating images.

**20. Why not report accuracy?**
With ~5% prevalence, predicting "absent" everywhere gives ~95% accuracy while finding nothing.

**21. Precision vs recall?**
Precision = TP/(TP+FP): how many flags are correct. Recall = TP/(TP+FN): how many real positives were found. Missing disease (FN) and false alarms (FP) have different costs.

**22. What are ROC-AUC and PR-AUC?**
ROC-AUC: probability a random positive scores above a random negative — threshold-independent ranking. PR-AUC (average precision) summarises precision vs recall; its baseline equals prevalence, so it is more informative for rare findings.

**23. Macro vs micro average?**
Macro: mean of per-label metrics, every finding equal. Micro: pool all image–label decisions — dominated by common findings.

**24. How did you choose thresholds?**
Per finding, the threshold maximising F1 on validation predictions (Youden's J is an option). Saved with the model and reused unchanged.

**25. Are your scores probabilities?**
They are uncalibrated sigmoid outputs, especially after weighted loss. We call them model scores; calibration (e.g. temperature scaling) is future work.

**26. What is overfitting and how did you control it?**
Fitting training data but not generalising. Controls: frozen pretrained backbone, dropout, weight decay, augmentation, early stopping on validation AUC, best-checkpoint restore.

**27. Explain Grad-CAM.**
Take the gradient of a finding's score w.r.t. the last conv feature maps, average gradients per channel to get weights, take ReLU of the weighted sum of maps, upsample and overlay on the image.

**28. Does a Grad-CAM heatmap prove the model is right?**
No. It shows what influenced the score; it is coarse (7×7) and may highlight shortcuts like tubes or text markers. Research shows saliency maps can be misleading.

**29. Main limitations?**
Noisy labels, single hospital, low resolution, imbalance, uncalibrated scores, possible shortcut learning, no external or clinical validation.

**30. What dataset biases exist?**
One US institution, specific scanners, many follow-up images from sick patients, demographic imbalance, AP vs PA view differences — the model may not generalise.

**31. How would you check generalisation?**
External validation on another dataset (e.g. CheXpert, MIMIC-CXR), subgroup analysis by age/sex/view, and confidence intervals via bootstrapping.

**32. Ethical issues?**
Patient safety (not a diagnostic tool), privacy (local processing, no identifiers), fairness (possible under-diagnosis of some groups), transparency (no fabricated results), accountability (clinician remains responsible).

**33. How is Streamlit integrated?**
`app.py` registers pages via `st.navigation`. Pages call framework-independent modules (`predict.py`, `gradcam.py`) and read generated result files; the model is cached with `st.cache_resource`. Without a model, the app shows setup steps instead of predictions.

**34. How did you test the software?**
Pytest: unit tests with synthetic images and mock models (label encoding, preprocessing, metrics, thresholds, error handling), an end-to-end pipeline test on synthetic noise, and Streamlit `AppTest` page-startup tests. Synthetic results are never reported as model performance.

**35. Future improvements?**
Higher resolution and GPU fine-tuning, ensembles, calibration, uncertainty labels (CheXpert), external validation, localisation with bounding boxes, better explainers with sanity checks, DICOM support.
