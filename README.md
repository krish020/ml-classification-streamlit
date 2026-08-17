# Multi-Model Classification Explorer

An end-to-end machine learning classification project: six classifiers trained on the
UCI **Breast Cancer Wisconsin (Diagnostic)** dataset, exposed through an interactive
Streamlit web application deployed on Streamlit Community Cloud.

---

## a. Problem Statement

Breast cancer diagnosis from a fine needle aspirate (FNA) biopsy traditionally depends on a
pathologist's visual assessment of cell nuclei — a process that is slow, requires scarce
expertise, and carries inter-observer variability. The clinical cost of the two possible
errors is highly asymmetric: a false negative (a malignant tumour classified as benign)
delays treatment and can be fatal, whereas a false positive triggers additional testing.

**Objective.** Build and compare supervised classification models that predict whether a
breast mass is **Malignant** or **Benign** from 30 numeric features computed from digitised
images of FNA biopsies, and identify which algorithm gives the best diagnostic performance
on this dataset.

- **Task type:** Binary classification
- **Target variable:** `diagnosis` — `1 = Malignant` (positive class), `0 = Benign`
- **Positive class choice:** Malignant is treated as the positive class so that **Recall**
  directly measures the proportion of cancers correctly caught — the metric that matters
  most clinically.
- **Primary selection metric:** **Matthews Correlation Coefficient (MCC)**, because the
  dataset is moderately imbalanced (37% malignant) and MCC only scores highly when all four
  confusion-matrix quadrants are good.

---

## b. Dataset Description

| Property | Value |
|---|---|
| **Dataset name** | Breast Cancer Wisconsin (Diagnostic) — WDBC |
| **Source** | UCI Machine Learning Repository, Dataset ID 17 |
| **URL** | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| **Instances** | **569** (requirement: ≥ 500 ✔) |
| **Features** | **30** numeric predictors (requirement: ≥ 12 ✔) |
| **Target** | `diagnosis` — binary (Malignant / Benign) |
| **Class balance** | 212 Malignant (37.3%) · 357 Benign (62.7%) |
| **Missing values** | 0 |
| **Train / Test split** | 398 / 171 rows — 70/30 stratified, `random_state=42` |

### Feature structure

Ten real-valued characteristics are computed for each cell nucleus in the image, and each is
reported in **three statistical variants** — giving 10 × 3 = 30 features:

| Base measurement | Description |
|---|---|
| `radius` | Mean distance from the nucleus centre to points on the perimeter |
| `texture` | Standard deviation of grey-scale values |
| `perimeter` | Nucleus boundary length |
| `area` | Nucleus area |
| `smoothness` | Local variation in radius lengths |
| `compactness` | perimeter² / area − 1.0 |
| `concavity` | Severity of concave portions of the contour |
| `concave_points` | Number of concave portions of the contour |
| `symmetry` | Symmetry of the nucleus |
| `fractal_dimension` | "Coastline approximation" − 1 |

The three variants are `mean_*` (average over nuclei), `*_error` (standard error), and
`worst_*` (mean of the three largest values). Example column names: `mean_radius`,
`radius_error`, `worst_radius`.

### Preprocessing applied

1. **Column normalisation** — names lower-cased, spaces replaced with underscores.
2. **Target re-encoding** — scikit-learn ships this dataset with `0 = malignant`; the label
   is inverted so `1 = Malignant`, making the positive class the clinically important one.
3. **Feature scaling** — every model is wrapped in a `Pipeline([StandardScaler, classifier])`.
   Scaling is essential for Logistic Regression and kNN (distance-based), and harmless for the
   tree models. The pipeline means the app can feed **raw, unscaled CSV values** to any model
   and get correct predictions, with no risk of train/serve skew.
4. **Stratified split** — preserves the 37/63 class ratio in both train and test sets.
5. **No leakage** — the scaler is fitted inside the pipeline on training folds only.

### Files in this repository

| File | Description |
|---|---|
| `test_data.csv` | **171 held-out test rows** (30 features + `diagnosis`) — upload this to the app |
| `data/train_data.csv` | 398 training rows |
| `data/full_dataset.csv` | All 569 rows |

---

## c. GitHub Repository Link

> **Repository:** `https://github.com/krish020/ml-classification-streamlit`

Public repository containing the complete source code, `requirements.txt`, this README,
`test_data.csv`, and a `model/` folder with all six saved model files.

### Repository structure

```
ml-classification-streamlit/
│-- app.py                  # Streamlit web application
│-- requirements.txt        # Python dependencies
│-- README.md               # This file
│-- test_data.csv           # Test data (171 rows) for upload
│-- .gitignore
│-- data/
│   │-- train_data.csv      # Training split (398 rows)
│   │-- full_dataset.csv    # Complete dataset (569 rows)
│-- model/
    │-- train_models.py                   # Training + evaluation script
    │-- metrics.json                      # Saved metrics & metadata
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- random_forest_ensemble.pkl
    │-- gradient_boosting_ensemble.pkl
```

---

## d. Models Used

Six classification models were implemented on the same dataset, each inside an identical
`StandardScaler → classifier` pipeline.

| # | Model | Key hyperparameters |
|---|---|---|
| 1 | Logistic Regression | `C=1.0`, `max_iter=5000` |
| 2 | Decision Tree Classifier | `max_depth=5`, `min_samples_leaf=5`, `criterion='gini'` |
| 3 | K-Nearest Neighbours | `n_neighbors=7`, `weights='distance'`, `metric='minkowski'` |
| 4 | Naive Bayes (Gaussian) | `GaussianNB()` defaults |
| 5 | Random Forest (Ensemble) | `n_estimators=300`, `max_depth=None` |
| 6 | Gradient Boosting (Ensemble) | `n_estimators=200`, `learning_rate=0.1`, `max_depth=3` |

> **Note on model count:** the assignment lists five named models but asks for a comparison
> table of six. **Gradient Boosting** was added as a second ensemble method so that all six
> table rows are filled with a genuinely distinct algorithm family.

### Comparison Table — Evaluation Metrics

All metrics are computed on the **held-out test set (171 rows)** that was never seen during
training. Positive class = Malignant. **Best score in each column is in bold.**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9708 | **0.9975** | 0.9836 | **0.9375** | 0.9600 | 0.9376 |
| Decision Tree | 0.9064 | 0.9262 | **1.0000** | 0.7500 | 0.8571 | 0.8077 |
| kNN | 0.9649 | 0.9893 | **1.0000** | 0.9062 | 0.9508 | 0.9264 |
| Naive Bayes | 0.9357 | 0.9933 | 0.9492 | 0.8750 | 0.9106 | 0.8622 |
| **Random Forest (Ensemble)** | **0.9766** | 0.9963 | **1.0000** | **0.9375** | **0.9677** | **0.9506** |
| Gradient Boosting (Ensemble) | 0.9649 | 0.9972 | 0.9833 | 0.9219 | 0.9516 | 0.9253 |

### Supporting diagnostics

| ML Model Name | 5-Fold CV Accuracy (train) | Train Accuracy | Confusion Matrix `[[TN, FP], [FN, TP]]` |
|---|---|---|---|
| Logistic Regression | 0.9723 ± 0.0218 | 0.9874 | `[[106, 1], [4, 60]]` |
| Decision Tree | 0.9320 ± 0.0237 | 0.9698 | `[[107, 0], [16, 48]]` |
| kNN | 0.9648 ± 0.0304 | 1.0000 | `[[107, 0], [6, 58]]` |
| Naive Bayes | 0.9296 ± 0.0207 | 0.9472 | `[[104, 3], [8, 56]]` |
| Random Forest (Ensemble) | 0.9496 ± 0.0358 | 1.0000 | `[[107, 0], [4, 60]]` |
| Gradient Boosting (Ensemble) | 0.9572 ± 0.0389 | 1.0000 | `[[106, 1], [5, 59]]` |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Outstanding for such a simple model — 97.08% accuracy and the **highest AUC of all six (0.9975)**, meaning its predicted probabilities rank malignant cases better than any other model. This is strong evidence that the 30 features are close to **linearly separable** after standardisation, which is unsurprising given that malignant nuclei are systematically larger and more irregular. It ties Random Forest on Recall (0.9375, only 4 missed cancers) and gives up almost nothing to the ensembles while remaining fully interpretable via its coefficients. Train accuracy (0.9874) sits just above test accuracy (0.9708), indicating **no meaningful overfitting**. The best accuracy-per-unit-complexity in the study. |
| **Decision Tree** | Clearly the **weakest model** — lowest score on Accuracy (0.9064), AUC (0.9262), Recall (0.7500), F1 (0.8571) and MCC (0.8077). Its Precision of 1.0000 is misleading: the tree is extremely conservative and simply refuses to predict "Malignant" unless certain, so it produces **16 false negatives — four times more missed cancers than Random Forest**, which is the most dangerous error type here. Its AUC is depressed because a depth-5 tree yields only a handful of distinct leaf probabilities, giving a coarse, step-like ROC curve with poor ranking resolution. The single-tree structure imposes axis-parallel splits on features that are strongly correlated, so it captures the decision boundary far less smoothly than the other methods. Not competitive, but valuable as the baseline that demonstrates exactly how much the ensemble adds. |
| **kNN** | Very strong at 96.49% accuracy, with perfect Precision (1.0000) — every mass it flagged as malignant genuinely was. Recall of 0.9062 (6 false negatives) is its weak spot: malignant cases lying near the class boundary get out-voted by their benign neighbours, since benign points outnumber malignant roughly 5:3. Performance depends **entirely on the `StandardScaler` step** — the raw features span wildly different ranges (`area` in the hundreds vs `smoothness` ≈ 0.1), and without scaling the Euclidean distance would be dominated by area alone. Train accuracy of 1.0000 is an artefact of `weights='distance'` (each training point is its own nearest neighbour at distance zero), not true overfitting; the honest estimate is the 0.9648 CV score, which matches test accuracy almost exactly. |
| **Naive Bayes** | The most interesting split personality in the set: a **near-top AUC of 0.9933 alongside the second-lowest accuracy (0.9357) and MCC (0.8622)**. The high AUC shows it *ranks* cases very well; the low accuracy shows its probabilities are **poorly calibrated**, so the default 0.5 threshold sits in the wrong place. The cause is its core conditional-independence assumption, which is badly violated here — `mean_radius`, `mean_perimeter` and `mean_area` are near-perfectly correlated by construction, so the same geometric evidence is triple-counted and the posteriors are pushed to overconfident extremes. It is also the only model with a materially imperfect Precision (0.9492, 3 false positives). Notably, it is the model that would gain the most from simple **threshold tuning** — its excellent AUC means the ranking is already there. Compensating virtues: it trains almost instantly and was the most stable model in cross-validation (lowest std, ±0.0207). |
| **Random Forest (Ensemble)** | **The overall winner.** Best on four of the six metrics: Accuracy (0.9766), Precision (1.0000), F1 (0.9677) and MCC (0.9506), while tying Logistic Regression for the best Recall (0.9375). Its confusion matrix `[[107, 0], [4, 60]]` is the cleanest of all six — **zero false positives and only 4 false negatives out of 171 cases**. Bagging 300 decorrelated trees cancels out the variance that crippled the single Decision Tree, lifting MCC from 0.8077 to 0.9506 — a **direct, controlled demonstration of the value of ensembling**, since both models share the same base learner. Train accuracy of 1.0000 against test accuracy of 0.9766 looks like overfitting but is normal and benign for unpruned forests: the out-of-sample gap stays small precisely because averaging controls variance. Its one soft spot is a marginally lower AUC (0.9963) than Logistic Regression, as forest probabilities come from discrete vote proportions and are slightly coarser. |
| **Gradient Boosting (Ensemble)** | A close and consistent runner-up: 96.49% accuracy and the **second-highest AUC (0.9972)**, essentially matching Logistic Regression on probability ranking. Its metrics are tightly clustered (F1 0.9516, MCC 0.9253) with no obvious weakness, and it beats kNN and Gradient-Boosting-free baselines on Recall (0.9219, 5 false negatives). It nonetheless trails Random Forest on every headline metric. Boosting fits residuals **sequentially**, which is most valuable on noisy, complex boundaries; this dataset is clean, well-separated and small (398 training rows), so there is little residual structure left for boosting to exploit and it cannot out-earn the simpler variance-reduction strategy of bagging. It was also the least stable model in cross-validation (±0.0389), consistent with sequential fitting being more sensitive to the particular training fold. |

### Overall Winner

> ### 🏆 **Random Forest (Ensemble)**
> **MCC 0.9506 · Accuracy 0.9766 · Precision 1.0000 · Recall 0.9375 · F1 0.9677 · AUC 0.9963**

**Why Random Forest wins:**

1. **Best on the primary metric.** It has the highest MCC (0.9506), the most trustworthy
   single number for a moderately imbalanced binary problem because it accounts for all four
   confusion-matrix quadrants rather than just the ones a model happens to be good at.
2. **Best confusion matrix, and the best clinical error profile.** `[[107, 0], [4, 60]]` —
   the only model that combines **zero false positives with the fewest false negatives**.
   Given that a missed cancer is the costliest error here, minimising false negatives without
   trading away precision is exactly the right behaviour.
3. **Wins or ties on 5 of 6 metrics.** It is top on Accuracy, Precision, F1 and MCC, and tied
   top on Recall. Only AUC goes elsewhere, by 0.0012 — a difference of no practical
   significance.
4. **Robust by construction.** Averaging 300 decorrelated trees makes it insensitive to
   hyperparameter choices, feature scaling and outliers, so the result is unlikely to be a
   lucky split.

**Honest caveat — the runner-up deserves a mention.** Logistic Regression finishes within
0.6 percentage points of accuracy, matches Random Forest exactly on Recall, and posts the
**best AUC of all six models (0.9975)**. It is also ~50× faster to train, produces calibrated
probabilities, and is directly interpretable through its coefficients — a real advantage in a
clinical setting where a model's reasoning must be explained and audited. Its 5-fold CV
accuracy (0.9723) is in fact *higher* than Random Forest's (0.9496), which is a genuine signal
that the two models are closer than the single test split suggests.

**Recommendation:** *Random Forest* on raw predictive performance and error profile; but if
interpretability, calibration or inference latency matter, *Logistic Regression* is the better
engineering choice at negligible cost in accuracy. The broader finding is that this dataset is
**close to linearly separable** — which is why the simple linear model competes with the
ensembles, and why the single Decision Tree, the only model that cannot express a smooth
oblique boundary, is the one that fails.

---

## Streamlit Application Features

Live app: **`https://ml-classification-app-jquhinl4vwhccwty8nurss.streamlit.app`**
*Deployed on Streamlit Community Cloud — opens directly into an interactive frontend.*

| # | Required feature | Implementation |
|---|---|---|
| **a** | **Dataset upload option (CSV)** | Sidebar file uploader accepting test CSVs. Schema is validated on upload — missing feature columns or a missing `diagnosis` column produce a clear error instead of a crash. A bundled `test_data.csv` can be used as a fallback, and is downloadable from within the app. |
| **b** | **Model selection dropdown** | `st.selectbox` listing all six trained models; the entire results panel updates reactively on change. |
| **c** | **Display of evaluation metrics** | All six required metrics — Accuracy, AUC, Precision, Recall, F1, MCC — rendered as metric cards for the selected model, plus a full six-model comparison table with best/worst cells colour-highlighted. |
| **d** | **Confusion matrix / classification report** | **Both.** A seaborn heatmap confusion matrix with a TN/FP/FN/TP breakdown, *and* a full scikit-learn classification report (per-class precision, recall, F1, support). |

**Additional features beyond the requirement:**

- ROC curve for the selected model, and an overlaid ROC plot for all six models
- Grouped bar chart comparing all six metrics across all six models
- Automatic "overall winner" callout computed live on whatever data is uploaded
- Row-level predictions with malignancy probabilities and a correct/incorrect flag
- CSV download buttons for predictions and for the comparison table
- Dataset summary cards (row count, feature count, class balance)
- **Self-healing model loading** — if the pickled models cannot be deserialised (e.g. a
  scikit-learn version mismatch on the cloud runtime), the app automatically retrains from
  `data/train_data.csv` at startup and reports this, rather than failing to launch

---

## Running Locally

```bash
git clone https://github.com/krish020/ml-classification-streamlit.git
cd ml-classification-streamlit

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Optional — regenerate models, metrics and the data splits from scratch
python model/train_models.py

streamlit run app.py
```

The app opens at `http://localhost:8501`.

---


## Tech Stack

Python · scikit-learn · pandas · NumPy · Streamlit · Matplotlib · Seaborn · joblib

## License

Released for academic assignment purposes. Dataset © UCI Machine Learning Repository
(Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian, University of Wisconsin).
