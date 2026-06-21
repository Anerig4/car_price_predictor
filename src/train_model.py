"""
train_model.py
---------------
Trains a car price prediction model on the cleaned dataset and saves it
to models/car_price_model.joblib

Usage:
    python src/train_model.py
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_data(path: str = "data/cleaned_car_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["car_age"] = 2026 - df["year"]
    return df


def build_pipeline() -> Pipeline:
    categorical_features = ["name", "company", "fuel_type"]
    numeric_features = ["car_age", "kms_driven"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",  # keeps numeric_features as-is
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def main():
    df = load_data()

    feature_cols = ["name", "company", "fuel_type", "car_age", "kms_driven"]
    X = df[feature_cols]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("Model trained.")
    print(f"R^2 Score : {r2:.3f}")
    print(f"MAE       : Rs {mae:,.0f}")
    print(f"RMSE      : Rs {rmse:,.0f}")

    joblib.dump(pipeline, "models/car_price_model.joblib")
    print("\nSaved -> models/car_price_model.joblib")


if __name__ == "__main__":
    main()
