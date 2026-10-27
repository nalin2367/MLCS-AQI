"""
Train and export the MQ sensor air-quality proxy prediction model.
Run: python train_model.py "mq data.csv"
"""
import sys
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "mq data.csv"
OUTPUT_PATH = "aqi_prediction_model.joblib"
FEATURES = ["humidity", "mq2", "mq4", "mq6"]
TARGET = "mq135"

df = pd.read_csv(DATA_PATH)
model_df = df[FEATURES + [TARGET]].copy()
for col in FEATURES + [TARGET]:
    model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
model_df = model_df.loc[model_df[TARGET].notna()].reset_index(drop=True)

if len(model_df) < 10:
    raise ValueError("Not enough usable rows to train and evaluate the model.")

n = len(model_df)
train_end, val_end = int(n * .60), int(n * .80)
train_df, val_df, test_df = model_df.iloc[:train_end], model_df.iloc[train_end:val_end], model_df.iloc[val_end:]

def preprocessor():
    return ColumnTransformer([("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), FEATURES)], remainder="drop")

models = {
    "Linear Regression": Pipeline([("preprocessor", preprocessor()), ("model", LinearRegression())]),
    "Random Forest": Pipeline([("preprocessor", preprocessor()), ("model", RandomForestRegressor(
        n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1
    ))]),
}

comparison = []
for name, model in models.items():
    model.fit(train_df[FEATURES], train_df[TARGET])
    pred = model.predict(val_df[FEATURES])
    comparison.append({
        "model": name,
        "validation_mae": float(mean_absolute_error(val_df[TARGET], pred)),
        "validation_rmse": float(mean_squared_error(val_df[TARGET], pred) ** 0.5),
        "validation_r2": float(r2_score(val_df[TARGET], pred)),
    })

comparison.sort(key=lambda item: item["validation_rmse"])
best_name = comparison[0]["model"]
best_model = models[best_name]

# Fit on train + validation; keep the final chronological segment untouched for testing.
fit_df = model_df.iloc[:val_end]
best_model.fit(fit_df[FEATURES], fit_df[TARGET])
test_pred = best_model.predict(test_df[FEATURES])
test_metrics = {
    "test_mae": float(mean_absolute_error(test_df[TARGET], test_pred)),
    "test_rmse": float(mean_squared_error(test_df[TARGET], test_pred) ** 0.5),
    "test_r2": float(r2_score(test_df[TARGET], test_pred)),
}
artifact = {
    "model": best_model,
    "model_name": best_name,
    "feature_columns": FEATURES,
    "target_column": TARGET,
    "description": "Predicts mq135 as a continuous air-quality proxy, not a calibrated regulatory AQI.",
    "dataset_rows": int(len(model_df)),
    "train_rows_used_for_final_fit": int(len(fit_df)),
    "test_rows": int(len(test_df)),
    "validation_comparison": comparison,
    "test_metrics": test_metrics,
}
joblib.dump(artifact, OUTPUT_PATH)
print("Selected model:", best_name)
print("Validation comparison:")
print(json.dumps(comparison, indent=2))
print("Held-out chronological test metrics:")
print(json.dumps(test_metrics, indent=2))
print("Saved model to:", OUTPUT_PATH)
