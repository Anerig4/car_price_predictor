"""
main.py
-------
Runs the full pipeline end-to-end:
1. Clean the raw data
2. Train the model
3. Run one example prediction

Usage:
    python main.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from clean_data import load_raw, clean
from train_model import load_data, build_pipeline
from predict import predict_price, fair_value_tip

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


def main():
    print("=== Step 1: Cleaning data ===")
    raw = load_raw("data/quikr_car.csv")
    cleaned = clean(raw)
    cleaned.to_csv("data/cleaned_car_data.csv", index=False)
    print(f"Cleaned dataset: {cleaned.shape[0]} rows\n")

    print("=== Step 2: Training model ===")
    df = load_data("data/cleaned_car_data.csv")
    feature_cols = ["name", "company", "fuel_type", "car_age", "kms_driven"]
    X = df[feature_cols]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"R^2 Score: {r2_score(y_test, y_pred):.3f}")
    print(f"MAE      : Rs {mean_absolute_error(y_test, y_pred):,.0f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/car_price_model.joblib")
    print("Saved -> models/car_price_model.joblib\n")

    print("=== Step 3: Example prediction ===")
    example = dict(
        name="Maruti Suzuki Swift",
        company="Maruti",
        fuel_type="Petrol",
        year=2016,
        kms_driven=42000,
    )
    predicted = predict_price(pipeline, **example)
    tip = fair_value_tip(predicted, example["kms_driven"], 2026 - example["year"])

    print(f"Car            : {example['name']} ({example['company']}, "
          f"{example['fuel_type']}, {example['year']})")
    print(f"Kms Driven     : {example['kms_driven']:,}")
    print(f"Predicted Price: Rs {predicted:,.0f}")
    print(f"Tip            : {tip}")

    print("\nDone! Try your own prediction with:")
    print('  python src/predict.py --name "Hyundai Grand i10" --company Hyundai '
          '--fuel Petrol --year 2015 --kms 30000')


if __name__ == "__main__":
    main()
