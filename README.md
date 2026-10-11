# ✈️ Flight Price Prediction: ML Course Project

A supervised **regression** project built for a Machine Learning course Kaggle competition. The goal is to predict the ticket price of a flight from its characteristics (airline, route, class, stops, duration, days before departure, etc.).

> **Course:** Machine Learning · **Project:** Course Project I · **Deadline:** 15 October 2026
> **Kaggle competition:** `<add link here>`
> **Team:** `<Name 1>`, `<Name 2>`, `<Name 3>`, `<Name 4>`

---

## 📌 Table of Contents
- [✈️ Flight Price Prediction: ML Course Project](#️-flight-price-prediction-ml-course-project)
  - [📌 Table of Contents](#-table-of-contents)
  - [🎯 Problem Statement](#-problem-statement)
  - [📊 Dataset](#-dataset)
  - [📁 Repository Structure](#-repository-structure)
  - [🛠️ Approach](#️-approach)
    - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
    - [2. Preprocessing](#2-preprocessing)
    - [3. Validation](#3-validation)
  - [🧪 Models and Results](#-models-and-results)
  - [💡 Key Findings](#-key-findings)
  - [🏆 Best Model](#-best-model)
  - [🚀 How to Run](#-how-to-run)
  - [👥 Team Contributions](#-team-contributions)
  - [📚 Tools and Libraries](#-tools-and-libraries)

---

## 🎯 Problem Statement

Given details of a flight, predict its **price** (a continuous value). Because the target is numeric, this is a regression problem. Models are evaluated on a hidden private test set on Kaggle, so we use **K-fold cross-validation** as our main guide for model selection rather than relying only on the public leaderboard.

## 📊 Dataset

`train.csv` contains **18,076 rows** and **12 columns**, with **no missing values**.

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
- Prices drop as `days_left` increases, and the relationship differs between Economy and Business (an interaction effect).
- Data-quality issues: 25 rows with `price = 0`, 6 duplicate rows, and ultra-rare categories (e.g. one airline with a single row, `Late_Night` departures with 6 rows).

## 📁 Repository Structure

```
.
├── Flight_Price_Regression_Starter.ipynb   # Main notebook: EDA → preprocessing → models → submission
├── train.csv                               # Training data (from Kaggle)
├── test.csv                                # Test data (from Kaggle)
├── submission.csv                          # Generated predictions for Kaggle
├── experiment_log.csv                      # Table of every model tried and its scores
├── README.md
└── requirements.txt
```

> Adjust this tree to match what is actually in your repo.

## 🛠️ Approach

### 1. Exploratory Data Analysis
Distribution of the target (raw and log), boxplots of price against every categorical feature, scatter plots of price against `days_left` and `duration` split by class, a correlation heatmap, and category-frequency checks.

### 2. Preprocessing
| Step | Reason |
|---|---|
| Drop duplicates | Avoids the same row appearing in both train and validation folds |
| Remove `price = 0` rows | Free tickets are data errors that distort the error metric |
| Remove rows with ultra-rare categories (< 30 rows) | Too few examples to learn from, and they cause wild extrapolation in flexible models |
| Drop `id` and `flight` | `id` carries no signal; `flight` has 1,000+ values (airline already captured) |
| Ordinal-encode `stops` | The categories have a natural order |
| One-hot encode other categoricals | Avoids inventing a false order between categories |
| Standardise numeric features | Needed for stable polynomial features and comparable coefficients |
| Train on `log(1 + price)` | Reduces the effect of the skewed target; predictions are converted back to dollars |
| Use scikit-learn `Pipeline`s | Prevents data leakage during cross-validation |

### 3. Validation
5-fold cross-validation (`KFold`, `shuffle=True`, `random_state=42`) reporting **RMSE**, **MAE** and **R²**, along with training RMSE to detect over-fitting.

## 🧪 Models and Results

All scores are 5-fold cross-validation results. Lower RMSE and MAE are better; higher R² is better.

| # | Model | CV RMSE | CV MAE | CV R² | Key observation |
|---|---|---|---|---|---|
| 0 | Baseline (predict mean) | 211.1 | 135.2 | 0.000 | Reference point |
| 1a | Simple linear regression (flight class only) | 189.5 | 118.0 | 0.195 | Best single feature |
| 1b | Simple linear regression (days left only) | 205.1 | 129.2 | 0.057 | Weak on its own |
| 1c | Simple linear regression (duration only) | 202.2 | 129.0 | 0.083 | Weak on its own |
| 2 | Multiple linear regression | 152.0 | 88.6 | 0.482 | Combining features helps a lot |
| 3 | Multiple linear regression + log target | 152.2 | 82.0 | 0.481 | Log target improves MAE |
| 4 | Quadratic regression (numeric features) + log | 152.9 | 82.3 | 0.476 | Curvature alone adds little |
| 5 | Quadratic regression (all features + interactions) + log | 119.1 | 58.9 | 0.680 | Interactions matter a lot |
| 6a | Regression tree (depth = 19) | 117.5 | 48.0 | 0.688 | Captures interactions automatically |
| 6b | **Regression tree (depth = 19) + log target** | **113.9** | **45.2** | **0.707** | **Best CV score** |

The notebook also records experiments on the tree's `min_samples_leaf` parameter (1, 5, 10, 20, 50), and the full table is saved to `experiment_log.csv`.

| Model | Best public Kaggle score |
|---|---|
| `<fill in after submitting>` | `<score>` |

> ⚠️ **Replace the numbers above with your own final results** after re-running the notebook, as small differences can arise from your settings and any experiments you add.

## 💡 Key Findings

- **A single feature is not enough.** The best single-feature model (flight class) explains about 19.5% of price variance. Combining all features raises this to about 48%.
- **Interactions drive performance.** Adding squared and interaction terms for numeric features only barely helps, but adding them across *all* features cuts RMSE from about 152 to about 119. Class, stops and booking time affect price *together*.
- **Trees do well with little tuning.** The regression tree automatically learns these interactions and gave the best CV score. The validation curve over `max_depth` shows the classic bias-variance trade-off.
- **Rare categories are a pitfall.** Grouping ultra-rare categories into an `Other` label caused the quadratic model to produce extreme predictions (negative R²) even though its MAE looked fine. This showed why we should always check more than one metric.
- **MLE connection.** Under Gaussian noise, least-squares regression is the maximum-likelihood estimator of the weights. The notebook demonstrates this by computing the MLE noise variance.

## 🏆 Best Model

- **Best CV model:** Regression tree (`max_depth = 19`, `min_samples_leaf = 5`) trained on `log(1 + price)`
- **Best public leaderboard model:** `<fill in>`
- **Public leaderboard score:** `<fill in>`

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

**2. Install dependencies** (Python 3.9+)
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

**3. Add the data**
Place `train.csv` and `test.csv` from the Kaggle competition in the repository root.

**4. Launch the notebook**
```bash
jupyter notebook Flight_Price_Regression_Starter.ipynb
```
Run all cells from top to bottom. The last section writes `submission.csv`, which you can upload to Kaggle.

## 👥 Team Contributions

| Member | Contribution |
|---|---|
| `<Name 1>` | `<e.g. EDA and visualisation>` |
| `<Name 2>` | `<e.g. preprocessing and feature engineering>` |
| `<Name 3>` | `<e.g. linear and polynomial models>` |
| `<Name 4>` | `<e.g. regression tree, report>` |

## 📚 Tools and Libraries

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter

---

*Built as part of the Machine Learning course project. Dataset provided via the course Kaggle competition.*