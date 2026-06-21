"""
predict.py
----------
Loads the trained model and predicts a car's resale price from
command-line arguments or interactive input.

Usage (CLI args):
    python src/predict.py --name "Maruti Suzuki Alto" --company Maruti \\
        --fuel Petrol --year 2017 --kms 35000

Usage (interactive, no args):
    python src/predict.py
"""

import argparse
import joblib
import pandas as pd

MODEL_PATH = "models/car_price_model.joblib"


def predict_price(model, name, company, fuel_type, year, kms_driven):
    car_age = 2026 - year
    input_df = pd.DataFrame([{
        "name": name,
        "company": company,
        "fuel_type": fuel_type,
        "car_age": car_age,
        "kms_driven": kms_driven,
    }])

    predicted = model.predict(input_df)[0]
    predicted = max(0, round(predicted, -2))  # round to nearest 100

    return predicted


def fair_value_tip(predicted_price, kms_driven, car_age):
    if car_age <= 3 and kms_driven < 40000:
        return "Relatively low age/mileage — should hold resale value well."
    elif car_age >= 10 or kms_driven > 120000:
        return "High age/mileage — expect a noticeably lower resale value."
    else:
        return "Typical depreciation range for a car of this age and mileage."


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a used car's price.")
    parser.add_argument("--name", type=str, help="Car name, e.g. 'Maruti Suzuki Alto'")
    parser.add_argument("--company", type=str, help="Manufacturer, e.g. 'Maruti'")
    parser.add_argument("--fuel", type=str, choices=["Petrol", "Diesel", "LPG"],
                         help="Fuel type")
    parser.add_argument("--year", type=int, help="Year of purchase, e.g. 2017")
    parser.add_argument("--kms", type=int, help="Kilometers driven, e.g. 35000")
    return parser.parse_args()


def get_interactive_input():
    print("Enter car details:")
    name = input("  Car name (e.g. Maruti Suzuki Alto): ").strip()
    company = input("  Company/Brand (e.g. Maruti): ").strip()
    fuel_type = input("  Fuel type (Petrol/Diesel/LPG): ").strip()
    year = int(input("  Year of purchase (e.g. 2017): ").strip())
    kms_driven = int(input("  Kilometers driven (e.g. 35000): ").strip())
    return name, company, fuel_type, year, kms_driven


def main():
    args = parse_args()

    if all([args.name, args.company, args.fuel, args.year, args.kms]):
        name, company, fuel_type, year, kms_driven = (
            args.name, args.company, args.fuel, args.year, args.kms
        )
    else:
        name, company, fuel_type, year, kms_driven = get_interactive_input()

    model = joblib.load(MODEL_PATH)
    predicted = predict_price(model, name, company, fuel_type, year, kms_driven)
    car_age = 2026 - year
    tip = fair_value_tip(predicted, kms_driven, car_age)

    print("\n--- Prediction ---")
    print(f"Car            : {name} ({company}, {fuel_type}, {year})")
    print(f"Kms Driven     : {kms_driven:,}")
    print(f"Predicted Price: Rs {predicted:,.0f}")
    print(f"Tip            : {tip}")


if __name__ == "__main__":
    main()
