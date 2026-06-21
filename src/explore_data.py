"""
explore_data.py
----------------
Generates a few exploratory plots from the cleaned dataset and saves
them as PNG files in the plots/ folder (handy when working in VS Code,
since there's no notebook to display plots inline).

Usage:
    python src/explore_data.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
os.makedirs("plots", exist_ok=True)


def main():
    df = pd.read_csv("data/cleaned_car_data.csv")
    df["car_age"] = 2026 - df["year"]

    print(df.describe())

    # Price distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(df["Price"], bins=40, kde=True, color="steelblue")
    plt.title("Distribution of Selling Price")
    plt.xlabel("Price (Rs)")
    plt.tight_layout()
    plt.savefig("plots/price_distribution.png")
    plt.close()

    # Price vs company
    plt.figure(figsize=(10, 5))
    top_companies = df["company"].value_counts().head(10).index
    sns.boxplot(
        data=df[df["company"].isin(top_companies)],
        x="company", y="Price"
    )
    plt.xticks(rotation=45)
    plt.title("Price by Company (Top 10 by listing count)")
    plt.tight_layout()
    plt.savefig("plots/price_by_company.png")
    plt.close()

    # Price vs car age
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x="car_age", y="Price", hue="fuel_type", alpha=0.7)
    plt.title("Car Age vs Price")
    plt.tight_layout()
    plt.savefig("plots/price_vs_age.png")
    plt.close()

    # Price vs kms driven
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x="kms_driven", y="Price", hue="fuel_type", alpha=0.7)
    plt.title("Kms Driven vs Price")
    plt.tight_layout()
    plt.savefig("plots/price_vs_kms.png")
    plt.close()

    print("\nSaved plots to plots/ folder:")
    for f in os.listdir("plots"):
        print(" -", f)


if __name__ == "__main__":
    main()
