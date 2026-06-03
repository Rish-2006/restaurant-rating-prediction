# 🍽️ Restaurant Rating Predictor
### Cognifyz Technologies Internship — Task 1

> **Predict the aggregate rating of a restaurant** using machine learning, based on features like votes, cost, location, price range, and service availability.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Project Structure](#-project-structure)
4. [Tech Stack](#-tech-stack)
5. [How It Works](#-how-it-works)
6. [Feature Engineering](#-feature-engineering)
7. [Models Trained](#-models-trained)
8. [Results & Performance](#-results--performance)
9. [Feature Importance](#-feature-importance)
10. [Output Plots](#-output-plots)
11. [Installation & Setup](#-installation--setup)
12. [Usage](#-usage)
13. [Predict a New Restaurant](#-predict-a-new-restaurant)
14. [Demo Predictions](#-demo-predictions)
15. [Key Insights](#-key-insights)
16. [Author](#-author)

---

## 🔍 Project Overview

This project builds a **supervised machine learning pipeline** to predict the **aggregate rating** (on a scale of 0.0 to 4.9) of a restaurant based on its characteristics — without knowing what customers actually rated it.

This is a **regression problem**. Four different ML models are trained, compared, and the best one (Gradient Boosting) is saved to disk and used for real-time predictions.

**Why does this matter?**
A restaurant platform like Zomato or Swiggy could use such a model to:
- Estimate the expected rating of a newly onboarded restaurant
- Identify which features most drive higher ratings
- Flag restaurants that underperform compared to their expected rating

---

## 📊 Dataset

| Property | Value |
|---|---|
| **File** | `Dataset_.csv` |
| **Source** | Zomato restaurant dataset (public) |
| **Total Rows** | 9,551 restaurants |
| **Total Columns** | 21 |
| **Countries Covered** | 15 |
| **Cities Covered** | 141 |
| **Unique Cuisine Combinations** | 1,716 |
| **Target Variable** | `Aggregate rating` (float, 0.0 – 4.9) |
| **Encoding** | UTF-8 with BOM (`utf-8-sig`) |

### Column Reference

| Column | Type | Description |
|---|---|---|
| `Restaurant ID` | int | Unique identifier |
| `Restaurant Name` | str | Name of the restaurant |
| `Country Code` | int | Numeric country identifier |
| `City` | str | City where the restaurant is located |
| `Address` | str | Full address |
| `Locality` | str | Area/neighbourhood |
| `Locality Verbose` | str | City + Locality string |
| `Longitude` | float | GPS longitude |
| `Latitude` | float | GPS latitude |
| `Cuisines` | str | Comma-separated list of cuisines |
| `Average Cost for two` | int | Average dining cost for 2 people |
| `Currency` | str | Local currency symbol |
| `Has Table booking` | str (Yes/No) | Accepts advance table bookings |
| `Has Online delivery` | str (Yes/No) | Offers online food delivery |
| `Is delivering now` | str (Yes/No) | Currently delivering |
| `Switch to order menu` | str (Yes/No) | Has order-menu switching |
| `Price range` | int (1–4) | 1 = Budget, 4 = Luxury |
| `Aggregate rating` | float | **Target variable** — overall rating |
| `Rating color` | str | Color label for the rating band |
| `Rating text` | str | Text label (Excellent/Good/etc.) |
| `Votes` | int | Total number of votes received |

> ⚠️ Restaurants with `Aggregate rating == 0` (unrated) are excluded from training — they represent restaurants with no review history and would mislead the model.

---

## 📁 Project Structure

```
aggreagate_restaurant_task_1/
│
├── restaurant_rating_predictor.py   # Main script — full ML pipeline
├── Dataset_.csv                     # Raw dataset (9,551 rows × 21 columns)
├── best_model.pkl                   # Saved best model (Gradient Boosting)
│
└── plots/
    ├── 1_model_comparison.png       # R², RMSE, MAE across all 4 models
    ├── 2_predicted_vs_actual.png    # Scatter: predicted vs real ratings
    ├── 3_feature_importance.png     # Feature importance bar chart (GB)
    ├── 4_residuals.png              # Residual distribution & residual vs predicted
    └── 5_insights.png               # Avg rating by price range & delivery status
```

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.3 | Data loading, cleaning, feature engineering |
| `numpy` | ≥ 1.21 | Numerical operations, log transforms |
| `scikit-learn` | ≥ 1.0 | ML models, train/test split, metrics, cross-validation |
| `matplotlib` | ≥ 3.4 | Plotting |
| `seaborn` | ≥ 0.11 | Statistical visualisation (imported, available for extension) |
| `joblib` | ≥ 1.0 | Saving and loading the trained model |
| `Python` | ≥ 3.8 | Language |

---

## ⚙️ How It Works

The script runs as a single end-to-end pipeline across **4 steps**:

```
[1/4] Load Data
       ↓
[2/4] Preprocess & Feature Engineer
       ↓
[3/4] Train 4 Models → Compare → Pick Best → Save
       ↓
[4/4] Generate 5 Analysis Plots
       ↓
      DONE — best_model.pkl + plots/ ready
```

---

## 🔧 Feature Engineering

Before training, raw columns are transformed into model-ready features:

| Feature | Source | Transformation |
|---|---|---|
| `Country Code` | Raw column | Used as-is (numeric) |
| `Average Cost for two` | Raw column | Used as-is |
| `Has Table booking` | Yes/No → 1/0 | Binary encoding |
| `Has Online delivery` | Yes/No → 1/0 | Binary encoding |
| `Is delivering now` | Yes/No → 1/0 | Binary encoding |
| `Switch to order menu` | Yes/No → 1/0 | Binary encoding |
| `Price range` | Raw column (1–4) | Used as-is |
| `Votes` | Raw column | Used as-is |
| `City_encoded` | `City` column | Top-20 cities kept; rest → "Other"; LabelEncoder applied |
| `Cuisine_count` | `Cuisines` column | Count of commas + 1 = number of cuisines offered |
| `Log_cost` | `Average Cost for two` | `np.log1p(cost)` — reduces skew |
| `Log_votes` | `Votes` | `np.log1p(votes)` — reduces skew |

**Why log transforms?**
Both `Votes` and `Average Cost for two` are heavily right-skewed — a few restaurants have extremely high votes or costs. Log-transforming pulls these outliers in and helps linear and tree models learn better.

**Total features used: 12**

---

## 🤖 Models Trained

Four regression models are trained and evaluated:

| Model | Key Hyperparameters |
|---|---|
| **Linear Regression** | Default (ordinary least squares) |
| **Decision Tree** | `max_depth=8`, `random_state=42` |
| **Random Forest** | `n_estimators=100`, `n_jobs=-1`, `random_state=42` |
| **Gradient Boosting** | `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`, `random_state=42` |

Each model is evaluated on:
- **R²** — proportion of variance explained (higher is better)
- **RMSE** — root mean squared error (lower is better)  
- **MAE** — mean absolute error (lower is better)
- **CV R²** — 5-fold cross-validation score on training set (checks overfitting)

---

## 📈 Results & Performance

Train/Test Split: **80% / 20%** → 5,922 training rows | 1,481 test rows

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.5254 | 0.3831 | 0.2832 |
| Decision Tree | 0.5490 | 0.3735 | 0.2758 |
| Random Forest | 0.5565 | 0.3704 | 0.2743 |
| **Gradient Boosting** ✅ | **0.5986** | **0.3524** | **0.2620** |

**🏆 Winner: Gradient Boosting**

- Best R² of **0.5986** — explains ~60% of rating variance
- Lowest RMSE of **0.3524** — on average, predictions are within 0.35 stars
- Lowest MAE of **0.2620** — median error under 0.26 stars
- Saved to `best_model.pkl` for reuse

> An R² of ~0.60 is reasonable here because restaurant ratings also depend on food taste, service quality, and ambiance — factors that simply don't exist as structured columns in this dataset.

---

## 🎯 Feature Importance

Ranked by importance in the Gradient Boosting model:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Votes | **34.7%** |
| 2 | Log_votes | **30.4%** |
| 3 | City_encoded | **18.2%** |
| 4 | Log_cost | 4.9% |
| 5 | Average Cost for two | 3.8% |
| 6 | Country Code | 2.7% |
| 7 | Cuisine_count | 2.2% |
| 8 | Has Online delivery | 1.1% |
| 9 | Price range | 1.0% |
| 10 | Has Table booking | 0.7% |
| 11 | Is delivering now | 0.2% |
| 12 | Switch to order menu | 0.0% |

**Key takeaway:** Votes (raw + log-transformed) together drive **65% of prediction power**. This makes intuitive sense — restaurants with many votes have had enough time and exposure to stabilise at an accurate rating. Location (city) adds another 18%.

---

## 🖼️ Output Plots

All plots are saved to the `plots/` folder at 150 DPI:

| File | What it shows |
|---|---|
| `1_model_comparison.png` | Side-by-side horizontal bar charts comparing R², RMSE, and MAE across all 4 models. Best model highlighted in teal. |
| `2_predicted_vs_actual.png` | Scatter plot of predicted vs actual ratings for Gradient Boosting. A perfect model would lie on the diagonal dashed line. |
| `3_feature_importance.png` | Horizontal bar chart of all 12 feature importances. Top-3 in teal, next 2 in blue, rest in grey. |
| `4_residuals.png` | Left: histogram of residuals (errors). Right: residuals vs predicted values — checks for bias at different rating levels. |
| `5_insights.png` | Left: average rating by price range (1–4). Right: average rating for restaurants with vs without online delivery. |

---

## 💻 Installation & Setup

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd aggreagate_restaurant_task_1
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

Or with a versions-pinned file:

```bash
pip install pandas>=1.3 numpy>=1.21 scikit-learn>=1.0 matplotlib>=3.4 seaborn>=0.11 joblib>=1.0
```

### 4. Confirm the dataset is present

Make sure `Dataset_.csv` is in the same folder as `restaurant_rating_predictor.py`.

---

## 🚀 Usage

### Run the full pipeline

```bash
python restaurant_rating_predictor.py
```

This will:
1. Load and preprocess `Dataset_.csv`
2. Train all 4 models and print a comparison table to the console
3. Save the best model as `best_model.pkl`
4. Generate all 5 plots inside `plots/`
5. Run 4 demo predictions and print the results

**Expected console output (abbreviated):**
```
============================================================
  COGNIFYZ TASK 1: RESTAURANT RATING PREDICTION
============================================================

[1/4] Loading data...
      Raw shape      : 9551 rows × 21 columns
      ...

[3/4] Training & evaluating models...
      Model                  R²     RMSE      MAE       CV R²
      ---------------------------------------------------------
      Linear Regression   0.5254   0.3831   0.2832   0.5172±0.0097
      Decision Tree       0.5490   0.3735   0.2758   0.4872±0.0214
      Random Forest       0.5565   0.3704   0.2743   0.5345±0.0112
      Gradient Boosting   0.5986   0.3524   0.2620   0.5789±0.0089 ◄ BEST
```

---

## 🔮 Predict a New Restaurant

You can call `predict_rating()` directly in a Python session after running the pipeline:

```python
rating, label = predict_rating(
    votes=1500,
    avg_cost=1200,
    price_range=3,
    has_table_booking=1,
    has_online_delivery=0
)
print(f"Predicted Rating: {rating} ({label})")
# Example output: Predicted Rating: 3.87 (Very Good)
```

### Function signature

```python
def predict_rating(
    votes,                    # int   — number of customer votes
    avg_cost,                 # int   — average cost for two (local currency)
    price_range,              # int   — 1 (Budget) to 4 (Luxury)
    has_table_booking=0,      # 0/1   — accepts table bookings?
    has_online_delivery=0,    # 0/1   — offers delivery?
    country_code=1,           # int   — country code (default India = 1)
    city_encoded=0,           # int   — encoded city integer
    cuisine_count=1,          # int   — number of cuisines offered
    is_delivering=0,          # 0/1   — currently delivering?
    switch_menu=0             # 0/1   — switch to order menu?
) -> tuple[float, str]
```

Returns a `(rating, label)` tuple where label is one of:
`"Excellent"` | `"Very Good"` | `"Good"` | `"Average"` | `"Below Average"`

### Load the saved model independently

```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load("best_model.pkl")

# Build a feature row manually
row = pd.DataFrame([{
    'Country Code': 1,
    'Average Cost for two': 800,
    'Has Table booking': 1,
    'Has Online delivery': 1,
    'Is delivering now': 0,
    'Switch to order menu': 0,
    'Price range': 2,
    'Votes': 600,
    'Cuisine_count': 2,
    'City_encoded': 5,
    'Log_cost': np.log1p(800),
    'Log_votes': np.log1p(600),
}])

predicted_rating = model.predict(row)[0]
print(f"Predicted: {predicted_rating:.2f}")
```

---

## 🧪 Demo Predictions

The script automatically runs these 4 examples at the end:

| Scenario | Votes | Cost | Price Range | Table Booking | Delivery | Predicted |
|---|---|---|---|---|---|---|
| Upscale dine-in, many votes | 3,000 | ₹2,000 | 4 (Luxury) | ✅ | ❌ | ~4.0+ (Very Good) |
| Budget delivery, few votes | 50 | ₹300 | 1 (Budget) | ❌ | ✅ | ~2.8 (Average) |
| Mid-range with delivery | 500 | ₹800 | 2 | ❌ | ✅ | ~3.5 (Good) |
| Fine dining, table bookings | 1,200 | ₹1,500 | 3 | ✅ | ❌ | ~3.8 (Very Good) |

---

## 💡 Key Insights

From the data analysis (see `plots/5_insights.png`):

1. **Votes drive ratings the most** — restaurants with more votes consistently have higher and more stable ratings. A restaurant with 3,000 votes is almost always rated higher than one with 50 votes, even at the same price point.

2. **Location matters significantly** — the city a restaurant is in contributes 18% to its predicted rating, reflecting regional differences in food culture and reviewer behaviour.

3. **Higher price range = higher rating** — luxury restaurants (price range 4) have a noticeably higher average rating (~4.1) compared to budget restaurants (price range 1, ~3.1).

4. **Online delivery correlates with slightly higher ratings** — restaurants offering delivery tend to have marginally better ratings, possibly because they attract a wider and more engaged reviewer base.

5. **Cuisine diversity has minor impact** — the number of cuisines offered contributes only ~2.2%, suggesting that specialisation vs variety is not a strong signal for quality perception.

---

## 👤 Author

**Risjit Dev O**
Internship Task 1 — Cognifyz Technologies
Date: May 23, 2026

---

## 📄 License

This project was created as part of an internship assignment. The dataset is publicly available via the Zomato restaurant dataset. The code is free to use for educational purposes.
