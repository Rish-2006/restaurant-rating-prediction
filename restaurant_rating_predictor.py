# ============================================================
#  COGNIFYZ INTERNSHIP — TASK 1
#  Restaurant Rating Prediction — Full ML Model
#  Usage: python restaurant_rating_predictor.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# ── CONFIG ─────────────────────────────────────────────────
DATA_PATH   = "Dataset_.csv"          # ← change if needed
MODEL_PATH  = "best_model.pkl"
PLOTS_DIR   = "plots"
TEST_SIZE   = 0.2
RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
#  STEP 1 — LOAD DATA
# ============================================================
print("=" * 60)
print("  COGNIFYZ TASK 1: RESTAURANT RATING PREDICTION")
print("=" * 60)
print("\n[1/4] Loading data...")

df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
print(f"      Raw shape      : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"      Columns        : {df.columns.tolist()}")
print(f"      Missing values :\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ============================================================
#  STEP 2 — PREPROCESS
# ============================================================
print("\n[2/4] Preprocessing...")

# Remove unrated restaurants
df = df[df['Aggregate rating'] > 0].copy()
print(f"      After removing unrated : {len(df)} rows")

# Fill missing cuisines
df['Cuisines'] = df['Cuisines'].fillna('Unknown')

# Encode binary Yes/No columns
binary_cols = ['Has Table booking', 'Has Online delivery',
               'Is delivering now', 'Switch to order menu']
for col in binary_cols:
    df[col] = (df[col] == 'Yes').astype(int)

# Encode City (top 20 + Other)
top_cities = df['City'].value_counts().nlargest(20).index
df['City_cat'] = df['City'].apply(lambda x: x if x in top_cities else 'Other')
le_city = LabelEncoder()
df['City_encoded'] = le_city.fit_transform(df['City_cat'])

# Feature engineering
df['Cuisine_count'] = df['Cuisines'].apply(lambda x: len(x.split(',')))
df['Log_cost']      = np.log1p(df['Average Cost for two'])
df['Log_votes']     = np.log1p(df['Votes'])

FEATURES = [
    'Country Code', 'Average Cost for two', 'Has Table booking',
    'Has Online delivery', 'Is delivering now', 'Switch to order menu',
    'Price range', 'Votes', 'Cuisine_count', 'City_encoded',
    'Log_cost', 'Log_votes'
]

X = df[FEATURES]
y = df['Aggregate rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"      Features used  : {len(FEATURES)}")
print(f"      Train size     : {len(X_train)}")
print(f"      Test  size     : {len(X_test)}")
print(f"      Target range   : {y.min():.1f} – {y.max():.1f}  (mean={y.mean():.3f})")

# ============================================================
#  STEP 3 — TRAIN & EVALUATE ALL MODELS
# ============================================================
print("\n[3/4] Training & evaluating models...")

models = {
    'Linear Regression' : LinearRegression(),
    'Decision Tree'     : DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
    'Random Forest'     : RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting' : GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                     max_depth=5, random_state=RANDOM_STATE),
}

results = {}
print(f"\n      {'Model':<22} {'R²':>7} {'RMSE':>8} {'MAE':>8} {'CV R²':>10}")
print("      " + "-"*57)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse  = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    results[name] = {
        'model' : model,
        'y_pred': y_pred,
        'MSE'   : round(mse,  4),
        'RMSE'  : round(rmse, 4),
        'MAE'   : round(mae,  4),
        'R2'    : round(r2,   4),
        'CV_mean': round(cv.mean(), 4),
        'CV_std' : round(cv.std(),  4),
    }
    star = " ◄ BEST" if name == 'Gradient Boosting' else ""
    print(f"      {name:<22} {r2:>7.4f} {rmse:>8.4f} {mae:>8.4f} "
          f"  {cv.mean():.4f}±{cv.std():.4f}{star}")

# ── Pick best model ─────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]['R2'])
best       = results[best_name]
best_model = best['model']

print(f"\n      ✓ Best model: {best_name}")
print(f"        R²={best['R2']}  RMSE={best['RMSE']}  MAE={best['MAE']}")

# ── Save model ──────────────────────────────────────────────
joblib.dump(best_model, MODEL_PATH)
print(f"      ✓ Model saved → {MODEL_PATH}")

# ============================================================
#  STEP 4 — FEATURE IMPORTANCE & PLOTS
# ============================================================
print("\n[4/4] Generating plots & feature analysis...")

gb_model  = results['Gradient Boosting']['model']
feat_imp  = sorted(zip(FEATURES, gb_model.feature_importances_), key=lambda x: -x[1])
feat_names = [f[0].replace('_', ' ') for f in feat_imp]
feat_vals  = [f[1] for f in feat_imp]

print("\n      Feature Importances (Gradient Boosting):")
for fn, fv in zip(feat_names, feat_vals):
    bar = '█' * int(fv * 80)
    print(f"        {fn:<25} {fv:.1%}  {bar}")

# ── COLOUR PALETTE ──────────────────────────────────────────
TEAL  = '#1D9E75'
PINK  = '#D4537E'
BLUE  = '#378ADD'
AMBER = '#BA7517'
GRAY  = '#888780'
LGRAY = '#F1EFE8'

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : 'white',
    'axes.grid'         : True,
    'grid.alpha'        : 0.25,
    'grid.linestyle'    : '--',
})

# ── Plot 1: Model Comparison ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Task 1 — Model Comparison', fontsize=14, fontweight='bold', y=1.01)

names  = list(results.keys())
colors = [TEAL if n == best_name else LGRAY for n in names]
ec     = ['#2C2C2A' if n == best_name else GRAY for n in names]

for ax, metric, title in zip(axes,
    ['R2', 'RMSE', 'MAE'],
    ['R² (higher = better)', 'RMSE (lower = better)', 'MAE (lower = better)']):
    vals = [results[n][metric] for n in names]
    bars = ax.barh(names, vals, color=colors, edgecolor=ec, linewidth=0.8, height=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + max(vals)*0.01,
                bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(0, max(vals) * 1.2)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/1_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: Predicted vs Actual ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
y_pred_best = best['y_pred']
ax.scatter(y_test, y_pred_best, alpha=0.3, s=14, color=TEAL, label='Predictions')
lims = [min(y_test.min(), y_pred_best.min()) - 0.1,
        max(y_test.max(), y_pred_best.max()) + 0.1]
ax.plot(lims, lims, '--', color=PINK, lw=1.8, label='Perfect fit')
ax.set_xlabel('Actual Rating', fontsize=11)
ax.set_ylabel('Predicted Rating', fontsize=11)
ax.set_title(f'{best_name}\nR²={best["R2"]}  RMSE={best["RMSE"]}', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/2_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 3: Feature Importance ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = [TEAL if i < 3 else BLUE if i < 5 else GRAY for i in range(len(feat_names))]
bars = ax.barh(feat_names[::-1], feat_vals[::-1],
               color=bar_colors[::-1], edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, feat_vals[::-1]):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.1%}', va='center', fontsize=9)
ax.set_title('Feature Importance — Gradient Boosting', fontsize=11, fontweight='bold')
ax.set_xlabel('Importance')
ax.set_xlim(0, max(feat_vals) * 1.2)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/3_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 4: Residuals ───────────────────────────────────────
residuals = y_test.values - y_pred_best
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(residuals, bins=40, color=BLUE, edgecolor='white', alpha=0.85)
axes[0].axvline(0, color=PINK, lw=1.5, linestyle='--')
axes[0].set_title('Residuals Distribution', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Residual (Actual − Predicted)')

axes[1].scatter(y_pred_best, residuals, alpha=0.3, s=12, color=BLUE)
axes[1].axhline(0, color=PINK, lw=1.5, linestyle='--')
axes[1].set_xlabel('Predicted Rating'); axes[1].set_ylabel('Residual')
axes[1].set_title('Residuals vs Predicted', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/4_residuals.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 5: Rating Distribution by Price Range ──────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
price_rating = df.groupby('Price range')['Aggregate rating'].mean()
axes[0].bar(price_rating.index.astype(str), price_rating.values,
            color=[LGRAY, BLUE, TEAL, PINK], edgecolor='#2C2C2A', linewidth=0.8, width=0.55)
for i, val in enumerate(price_rating.values):
    axes[0].text(i, val + 0.02, f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
axes[0].set_title('Avg Rating by Price Range', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Price Range (1=Budget → 4=Luxury)')
axes[0].set_ylabel('Avg Rating'); axes[0].set_ylim(2.8, 4.3)

delivery = df.groupby('Has Online delivery')['Aggregate rating'].mean()
bars = axes[1].bar(['No Delivery', 'Has Delivery'], delivery.values,
                   color=[TEAL, BLUE], edgecolor='#2C2C2A', linewidth=0.8, width=0.4)
for bar, val in zip(bars, delivery.values):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
axes[1].set_title('Avg Rating: Delivery vs No Delivery', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Avg Rating'); axes[1].set_ylim(3.0, 3.8)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/5_insights.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n      ✓ All 5 plots saved to ./{PLOTS_DIR}/")

# ============================================================
#  PREDICT FUNCTION — use this to score new restaurants
# ============================================================
def predict_rating(votes, avg_cost, price_range,
                   has_table_booking=0, has_online_delivery=0,
                   country_code=1, city_encoded=0,
                   cuisine_count=1, is_delivering=0, switch_menu=0):
    """
    Predict the rating of a new restaurant.

    Parameters
    ----------
    votes               : int   – Number of votes (0 – 10000)
    avg_cost            : int   – Average cost for two people (in local currency)
    price_range         : int   – 1 (Budget) to 4 (Luxury)
    has_table_booking   : 0/1   – Does the restaurant accept table bookings?
    has_online_delivery : 0/1   – Does it offer online delivery?
    country_code        : int   – Country code (default 1)
    city_encoded        : int   – Encoded city integer (default 0)
    cuisine_count       : int   – Number of cuisines offered
    is_delivering       : 0/1   – Is it currently delivering?
    switch_menu         : 0/1   – Switch to order menu?

    Returns
    -------
    float : Predicted aggregate rating (1.0 – 5.0)
    """
    log_cost  = np.log1p(avg_cost)
    log_votes = np.log1p(votes)

    row = pd.DataFrame([{
        'Country Code'        : country_code,
        'Average Cost for two': avg_cost,
        'Has Table booking'   : has_table_booking,
        'Has Online delivery' : has_online_delivery,
        'Is delivering now'   : is_delivering,
        'Switch to order menu': switch_menu,
        'Price range'         : price_range,
        'Votes'               : votes,
        'Cuisine_count'       : cuisine_count,
        'City_encoded'        : city_encoded,
        'Log_cost'            : log_cost,
        'Log_votes'           : log_votes,
    }])

    pred = best_model.predict(row)[0]
    pred = float(np.clip(pred, 1.0, 5.0))

    label = ('Excellent' if pred >= 4.5 else
             'Very Good' if pred >= 4.0 else
             'Good'      if pred >= 3.5 else
             'Average'   if pred >= 3.0 else 'Below Average')

    return round(pred, 2), label


# ── DEMO PREDICTIONS ────────────────────────────────────────
print("\n" + "=" * 60)
print("  DEMO PREDICTIONS")
print("=" * 60)

demos = [
    dict(votes=3000, avg_cost=2000, price_range=4,
         has_table_booking=1, has_online_delivery=0,
         desc="Upscale dine-in, many votes"),
    dict(votes=50,   avg_cost=300,  price_range=1,
         has_table_booking=0, has_online_delivery=1,
         desc="Budget delivery place, few votes"),
    dict(votes=500,  avg_cost=800,  price_range=2,
         has_table_booking=0, has_online_delivery=1,
         desc="Mid-range with delivery"),
    dict(votes=1200, avg_cost=1500, price_range=3,
         has_table_booking=1, has_online_delivery=0,
         desc="Fine dining, table bookings"),
]

for d in demos:
    desc = d.pop('desc')
    rating, label = predict_rating(**d)
    print(f"  {desc:<40} → {rating} ({label})")

print("\n" + "=" * 60)
print("  DONE! Files created:")
print(f"    • {MODEL_PATH}        (trained model — load with joblib.load)")
print(f"    • {PLOTS_DIR}/        (5 analysis plots)")
print("=" * 60)
