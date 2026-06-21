"""
clean_data.py
--------------
Cleans the raw Quikr used-car dataset (data/quikr_car.csv) and saves a
tidy version to data/cleaned_car_data.csv.

The raw data has several real-world messiness issues:
- Price contains "Ask For Price" and comma-formatted numbers ("3,25,000")
- kms_driven contains "kms" text and commas ("45,000 kms")
- year sometimes contains junk values (non-4-digit strings)
- fuel_type has missing values
- Some rows are exact duplicates
"""

import pandas as pd


def load_raw(path: str = "data/quikr_car.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- year: keep only rows where year is a clean 4-digit number ---
    df = df[df["year"].str.isnumeric()]
    df["year"] = df["year"].astype(int)

    # --- Price: drop "Ask For Price" rows, remove commas, convert to int ---
    df = df[df["Price"] != "Ask For Price"]
    df["Price"] = df["Price"].str.replace(",", "", regex=False).astype(int)

    # --- kms_driven: remove "kms" and commas, convert to int ---
    df = df[df["kms_driven"].notna()]
    df["kms_driven"] = (
        df["kms_driven"]
        .str.replace(" kms", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df = df[df["kms_driven"].str.isnumeric()]
    df["kms_driven"] = df["kms_driven"].astype(int)

    # --- fuel_type: drop missing ---
    df = df[df["fuel_type"].notna()]

    # --- name: keep only first 3 words (e.g. "Maruti Suzuki Alto") ---
    df["name"] = df["name"].str.split().str.slice(0, 3).str.join(" ")

    # --- sensible value ranges (remove extreme outliers / data errors) ---
    df = df[df["Price"] < 6_000_000]      # under 60 lakh
    df = df[df["kms_driven"] < 400_000]
    df = df[df["year"] >= 1995]

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    return df


def main():
    raw = load_raw()
    print(f"Raw data: {raw.shape[0]} rows")

    cleaned = clean(raw)
    print(f"Cleaned data: {cleaned.shape[0]} rows")

    cleaned.to_csv("data/cleaned_car_data.csv", index=False)
    print("Saved -> data/cleaned_car_data.csv")
    print(cleaned.head())


if __name__ == "__main__":
    main()
