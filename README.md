# ✈️ Flight Price Prediction: ML Course Project

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-regression-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](Flight_Price_Regression_Analysis.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prakhoo3012/ml-project-iiitb/blob/main/Flight_Price_Regression_Analysis.ipynb)

A supervised **regression** project for a Machine Learning course Kaggle competition. The goal is to predict the ticket price of a domestic flight from its characteristics (airline, route, class, stops, duration, days before departure, etc.).

> **Course:** Machine Learning · **Project:** Course Project I · **Deadline:** 15 October 2026
> **Team:** [@Prakhoo3012](https://github.com/Prakhoo3012), `<Name 2>`, `<Name 3>`, `<Name 4>`

## Quick Links

| What | Link |
|---|---|
| 📓 Main notebook (EDA → preprocessing → models → submission) | [`Flight_Price_Regression_Analysis.ipynb`](Flight_Price_Regression_Analysis.ipynb) |
| 🧪 Every model tried and its scores | [`experiment_log.csv`](experiment_log.csv) |
| 📂 Data folder | [`data/`](data/) → [`train.csv`](data/train.csv) · [`test.csv`](data/test.csv) · [`submission.csv`](data/submission.csv) (Kaggle-format predictions) |
| 📦 Dependencies | [`requirements.txt`](requirements.txt) |
| 🏁 Kaggle competition | `<add competition link here>` |
| 🐙 Repository | <https://github.com/Prakhoo3012/ml-project-iiitb> |

## Contents
- [✈️ Flight Price Prediction: ML Course Project](#️-flight-price-prediction-ml-course-project)
  - [Quick Links](#quick-links)
  - [Contents](#contents)
  - [Problem Statement](#problem-statement)
  - [Dataset](#dataset)
  - [Repository Structure](#repository-structure)
  - [Approach](#approach)
    - [Preprocessing](#preprocessing)
    - [Validation](#validation)
  - [Models and Results](#models-and-results)
  - [Key Findings](#key-findings)
  - [Best Model](#best-model)
  - [How to Run](#how-to-run)
    - [Option A: Locally](#option-a-locally)
    - [Option B: Google Colab (no installation)](#option-b-google-colab-no-installation)
    - [Output files](#output-files)

---

## Problem Statement

Given details of a flight, predict its **price** (a continuous value), so this is a regression problem. Models are scored on a hidden private test set on Kaggle, so we rely on a **hold-out split and K-fold cross-validation** for model selection rather than only the public leaderboard.

## Dataset

All files are in [`data/`](data/).

| File | Rows | Contents |
|---|---|---|
| [`train.csv`](data/train.csv) | 18,076 | 11 features + the target `price` (no missing values) |
| [`test.csv`](data/test.csv) | 4,520 | the same 11 features, **no** `price`; used to make Kaggle predictions |
| [`submission.csv`](data/submission.csv) | 4,520 | predictions in the Kaggle format (columns `id`, `price`, same row order as `test.csv`) |

| Column | Description | Type |
|---|---|---|
| `id` | Row identifier (not used as a feature) | ID |
| `airline` | Airline name (7 values) | Categorical |
| `flight` | Flight code, e.g. `WN 3596` (1,123 unique values) | Categorical |
| `source_city`, `destination_city` | Origin and destination (5 cities each) | Categorical |
| `departure_time`, `arrival_time` | Time-of-day bucket | Categorical |
| `stops` | `zero`, `one`, `two_or_more` | Ordinal |
| `flight_class` | Economy / Business | Categorical |
| `duration` | Flight duration in hours | Numeric |
| `days_left` | Days between booking and departure | Numeric |
| **`price`** | **Target variable** | Numeric |

**Key EDA observations**
- `price` is heavily right-skewed (skewness ≈ 7.07 → ≈ −0.74 after `log(1 + price)`).
- Flight class and number of stops are the strongest drivers of price (Business averages roughly twice the Economy price).
- Prices fall as `days_left` increases, and the pattern differs between Economy and Business (an interaction effect).
- Data-quality issues: 25 rows with `price = 0`, 6 duplicate rows, and ultra-rare categories (one airline with a single row, `Late_Night` departures with 6 rows).

## Repository Structure

```
ml-project-iiitb/
├── data/
│   ├── train.csv                              # labelled training data
│   ├── test.csv                               # unlabelled data to predict
│   └── submission.csv                         # predictions in Kaggle format (id, price)
├── Flight_Price_Regression_Analysis.ipynb     # full analysis and all models
├── experiment_log.csv                         # results table written by the notebook
├── requirements.txt                           # Python dependencies
└── README.md
```

Running the notebook also creates `my_submission.csv` (Kaggle predictions) and `flight_price_model.pickle` (the saved best model). `my_submission.csv` is deliberately **not** named `submission.csv`, so re-running the notebook can never overwrite a file in `data/`.

## Approach

The notebook follows a standard flow: **EDA** (univariate → bivariate → multivariate) → **data diagnosis** → **preprocessing** → **modelling and tuning** → **evaluation** → **Kaggle submission**.

### Preprocessing
| Step | Reason |
|---|---|
| Drop duplicate rows | Prevents the same row from appearing in both train and validation parts |
| Remove `price = 0` rows | Free tickets are data errors that distort the error metric |
| Remove rows with ultra-rare categories (< 30 rows) | Too few examples to learn from; they cause wild extrapolation in flexible models |
| Drop `id` and `flight` | `id` carries no signal; `flight` has 1,000+ values (airline is already captured) |
| Map `flight_class` to 0/1 and `stops` to 0/1/2 | Both are binary/ordered, so a numeric code is meaningful |
| Dummy-encode the other categoricals (`drop_first=True`) | Avoids inventing a false order between categories |
| Standardise features inside the polynomial / Ridge / Lasso pipelines | Needed for stable squared terms and a fair regularization penalty |
| Train on `log(1 + price)`, predict back in dollars | Reduces the effect of the skewed target |

### Validation
- **Hold-out split:** 75 % train / 25 % test (`random_state=101`), used for the main comparison table.
- **5-fold cross-validation** (`random_state=101`) for the main contenders, as a more reliable estimate of private-leaderboard performance.
- Hyper-parameters such as Ridge/Lasso `alpha` are tuned by 5-fold CV on the training part only.
- Metrics: **RMSE**, **MAE**, **R²**, plus train RMSE to detect over-fitting.

## Models and Results

Hold-out test scores (25 % split). Lower RMSE / MAE is better; higher R² is better. The full table is saved in [`experiment_log.csv`](experiment_log.csv).

| # | Model | Test RMSE | Test MAE | Test R² | Key observation |
|---|---|---|---|---|---|
| 0 | Baseline (predict mean) | 211.0 | 133.5 | 0.000 | Reference point |
| 1 | Simple linear regression (best single feature: `flight_class`) | 190.4 | 116.4 | 0.185 | One feature explains little |
| 2 | Multiple linear regression | 156.1 | 85.6 | 0.453 | Combining features helps a lot |
| 3 | Multiple linear regression + log target | 157.8 | 79.6 | 0.440 | Log target improves MAE, not RMSE |
| 4 | Quadratic regression (numeric features) + log | 158.3 | 80.0 | 0.437 | Curvature alone adds nothing |
| 5 | Quadratic regression (all features + interactions) + log | 133.0 | 58.5 | 0.603 | Interactions matter a lot |
| 7a | Ridge on plain features + log (α = 100) | 157.7 | 79.5 | 0.441 | Same as unregularized |
| 7b | Lasso on plain features + log (α = 1e-5) | 157.8 | 79.6 | 0.440 | Same as unregularized |
| 7c | Ridge on quadratic features + log (α = 100) | 133.5 | 58.3 | 0.600 | No gain over model 5 |
| 7d | Lasso on quadratic features + log (α = 3e-4) | 133.3 | 58.2 | 0.601 | Same accuracy, 111 of 377 terms zeroed |
| 6a | Regression tree (depth 11) | 132.1 | 50.1 | 0.608 | Learns interactions automatically |
| 6b | Regression tree (depth 11) + log | 131.8 | 48.7 | 0.610 | |
| 6c | **Regression tree (depth 11, `min_samples_leaf=10`) + log** | **130.3** | **49.4** | **0.619** | **Best on the hold-out split** |

**5-fold cross-validation of the main contenders**

| Model | CV RMSE (mean ± std) |
|---|---|
| Multiple linear regression | 152.0 ± 8.5 |
| Multiple linear regression + log | 152.2 ± 8.9 |
| Quadratic (all features) + log | 118.7 ± 10.0 |
| Ridge on quadratic + log | 119.0 ± 10.0 |
| Lasso on quadratic + log | 118.9 ± 10.0 |
| Regression tree (depth 11) + log | **117.5 ± 10.3** |

> The CV standard deviations are about 10, so the tree, the quadratic model and its Ridge/Lasso variants are **statistically tied**. Your own numbers may differ slightly if you change the split or settings.

| Model | Best public Kaggle score |
|---|---|
| `<fill in after submitting>` | `<score>` |

## Key Findings

- **One feature is not enough.** The best single feature (flight class) explains about 18 % of price variance; combining all features raises this to about 45 %.
- **Interactions drive performance.** Squared terms on the numeric columns alone add nothing, but quadratic terms across *all* features cut RMSE from about 156 to about 133 (CV: 152 → 119). Class, stops, airline and booking time affect price *together*.
- **Trees learn this automatically** and give the best hold-out score. The `max_depth` curve shows the classic bias–variance trade-off, and an unrestricted tree (`min_samples_leaf=1`) over-fits (train RMSE ≈ 86 vs test ≈ 144).
- **Ridge and Lasso did not improve accuracy.** With about 18,000 rows and a few dozen features, over-fitting from large weights is not the main problem. Lasso's benefit is a *simpler model* at the same accuracy.
- **Rare categories are a pitfall.** Grouping ultra-rare categories into an `Other` label made the quadratic model produce extreme predictions (negative R²) even though its MAE looked fine. This is why we report more than one metric.
- **MLE connection.** Under Gaussian noise, least-squares regression is the maximum-likelihood estimator of the weights; the notebook demonstrates this by computing the MLE noise variance.

## Best Model

- **Best hold-out model:** Regression tree, `max_depth = 11`, `min_samples_leaf = 10`, trained on `log(1 + price)`.
- **Best public-leaderboard model:** `<fill in>`
- **Public leaderboard score:** `<fill in>`

## How to Run

### Option A: Locally
```bash
git clone https://github.com/Prakhoo3012/ml-project-iiitb.git
cd ml-project-iiitb
pip install -r requirements.txt
jupyter notebook Flight_Price_Regression_Analysis.ipynb
```
Run all cells from top to bottom. The notebook finds the data in `data/` automatically.

### Option B: Google Colab (no installation)
Click the **Open in Colab** badge at the top. If `data/` is not present, the notebook reads the files directly from this repository.

### Output files
| File | Created by |
|---|---|
| `my_submission.csv` | final section: predictions on `test.csv`, columns `id`, `price` (the file to upload to Kaggle) |
| `experiment_log.csv` | results table of all models |
| `flight_price_model.pickle` | the saved best model and its feature columns |

---