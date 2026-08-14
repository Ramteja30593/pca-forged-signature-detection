# PCA-Based Forged Signature Detection Using Machine Learning

## Project Overview

This project is a machine learning extension of a linear algebra mini-project on forged signature detection.  
It uses grayscale signature images, applies PCA for dimensionality reduction, and then classifies signatures as **Genuine** or **Forged** with supervised machine learning models.

The project is intentionally small and reproducible for an M.Tech Data Science mini-project.

## Problem Statement

Given handwritten signature images, the goal is to determine whether a signature is genuine or forged.  
Unlike the earlier linear algebra version that used reconstruction error as an anomaly score, this version uses PCA as a preprocessing step for supervised classification.

## Dataset Link

Kaggle Signature Verification Dataset:  
<https://www.kaggle.com/datasets/akashgundu/signature-verification-dataset>

Local dataset path used in this project:

`/Users/medepatiramtejareddy/Downloads/maths_sign/extract`

## Dataset Structure

- Numeric folders such as `001`, `002`, `003` contain genuine signatures.
- Folders ending in `_forg` contain forged signatures.
- The ML experiment uses a balanced subset of:
  - 150 genuine signatures
  - 150 forged signatures
  - 300 total images

Images are:

- converted to grayscale,
- cropped to the foreground signature region,
- autocontrasted,
- resized to `32 x 32`,
- normalized to `[0, 1]`,
- flattened into `1024`-dimensional vectors.

## Methodology

The project follows a leakage-safe machine learning workflow:

1. Load image paths from the local dataset.
2. Build a balanced subset of genuine and forged signatures.
3. Preprocess images into 1024-dimensional vectors.
4. Perform a stratified train/test split with `test_size=0.20` and `random_state=42`.
5. Fit `StandardScaler` on training data only.
6. Fit PCA on training data only.
7. Train classifiers on PCA-transformed data.
8. Evaluate on the held-out test set.

## PCA Explanation

PCA is used here for dimensionality reduction.

- In the linear algebra project, PCA was built manually from the covariance matrix and eigenvalue decomposition.
- In the ML project, `sklearn.decomposition.PCA` is used inside a leakage-safe workflow.

Two PCA settings are compared:

- `n_components = 0.95` to retain 95% explained variance
- `n_components = 50` for fixed dimensionality

For this subset, the 95% variance setting selected **142 components**.

## Models Used

- Baseline Logistic Regression without PCA
- Logistic Regression + PCA
- SVM with RBF kernel + PCA
- Random Forest + PCA

An additional small `GridSearchCV` is performed only for SVM:

- `C = [0.1, 1, 10]`
- `gamma = ['scale', 'auto']`
- `cv = 5`
- `scoring = 'f1'`

## Results Table

| Model | PCA Components | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Logistic Regression (No PCA) | No PCA | 0.7667 | 0.7667 | 0.7667 | 0.7667 | 0.7944 |
| Logistic Regression + PCA 95% | 138 | 0.7667 | 0.7500 | 0.8000 | 0.7742 | 0.8122 |
| SVM RBF + PCA 95% | 138 | 0.7167 | 0.7407 | 0.6667 | 0.7018 | 0.8311 |
| Random Forest + PCA 95% | 138 | 0.7333 | 0.7500 | 0.7000 | 0.7241 | 0.8028 |
| Logistic Regression + PCA 50 | 50 | 0.7500 | 0.7273 | 0.8000 | 0.7619 | 0.7533 |
| SVM RBF + PCA 50 | 50 | 0.7667 | 0.7667 | 0.7667 | 0.7667 | 0.8211 |
| Random Forest + PCA 50 | 50 | 0.8167 | 0.8065 | 0.8333 | 0.8197 | 0.8822 |
| Tuned SVM RBF + PCA 95% | 138 | 0.6167 | 0.5686 | 0.9667 | 0.7160 | 0.7844 |

## Key Findings

- PCA reduced the original `1024`-dimensional feature space to **138 components** for 95% explained variance.
- Foreground cropping before resizing improved signal quality by reducing the amount of empty background seen by PCA and the classifiers.
- PCA slightly improved Logistic Regression over the raw-pixel baseline in this subset.
- The **best overall model** was **Random Forest + PCA 50**, with:
  - Accuracy = `0.8167`
  - Precision = `0.8065`
  - Recall = `0.8333`
  - F1-score = `0.8197`
  - ROC-AUC = `0.8822`
- In this improved preprocessing setup, the fixed 50-component PCA representation performed better than the 95%-variance PCA setting for Random Forest.
- The tuned SVM did not outperform the untuned SVM models on the test set in this run, even though its cross-validation F1 was reasonable.
- The manual PCA and `sklearn` PCA showed the same principal variance structure on the training data.

## Limitations

- Only 300 images were used for the ML experiment.
- The subset is balanced and reproducible, but still small.
- Results depend on image size, PCA setting, classifier choice, and the selected subset.
- This is a proof-of-concept academic project, not a forensic-grade authentication system.

## How to Run

1. Make sure the local dataset is present at:

   `/Users/medepatiramtejareddy/Downloads/maths_sign/extract`

2. Install dependencies:

   ```bash
   pip install -r requirements_ml.txt
   ```

3. Open the notebook:

   `PCA_Forged_Signature_ML.ipynb`

4. Run the notebook from top to bottom.

If you need to regenerate the notebook with executed outputs:

```bash
python build_ml_notebook.py
```

## Project Structure

```text
maths_sign/
├── extract/                              # local dataset, not for GitHub
├── PCA_Forged_Signature_ML.ipynb         # executed ML notebook
├── PCA_Forged_Signature_Detection.ipynb  # earlier linear algebra notebook
├── build_ml_notebook.py                  # notebook generator for the ML version
├── README_ML.md
└── requirements_ml.txt
```

## Note on GitHub

Do **not** include the dataset in GitHub.  
Only the code, notebooks, documentation, and dependency files should be committed.
