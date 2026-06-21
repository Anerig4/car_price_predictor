# Car Price Predictor

A machine learning project that predicts the resale price of a used car based on its name, manufacturer, fuel type, year of purchase, and kilometers driven. Built using a real-world dataset of scraped Quikr car listings.

## Overview

This project demonstrates a complete, end-to-end regression pipeline:

1. **Data Cleaning** — handling missing values, inconsistent formatting, and invalid entries in raw scraped data
2. **Exploratory Data Analysis** — visualizing price trends across brand, age, and mileage
3. **Model Training** — a Random Forest Regressor trained via a scikit-learn pipeline with one-hot encoding
4. **Inference** — a command-line tool for predicting prices on new, unseen cars

## Dataset

The raw dataset (`data/quikr_car.csv`) contains 892 used car listings with the following fields: `name`, `company`, `year`, `Price`, `kms_driven`, `fuel_type`.

As with most scraped data, it required cleaning before use:

| Issue | Resolution |
|---|---|
| `Price` contained non-numeric values (`"Ask For Price"`) and comma-separated strings (`"3,25,000"`) | Removed unparseable rows; stripped commas and cast to integer |
| `kms_driven` contained embedded text (`"45,000 kms"`) | Stripped non-numeric characters and cast to integer |
| `year` contained invalid entries (e.g. `"TOUR"`, `"150k"`) | Filtered to valid four-digit years |
| `fuel_type` had missing values | Dropped incomplete rows |
| Extreme outliers and duplicate rows | Filtered and removed |

After cleaning, the dataset contains approximately 718 usable rows, saved to `data/cleaned_car_data.csv`.

## Project Structure

```
car_price_predictor/
├── data/
│   ├── quikr_car.csv             # Raw dataset
│   └── cleaned_car_data.csv      # Cleaned dataset (generated)
├── models/
│   └── car_price_model.joblib    # Trained model (generated)
├── plots/                        # EDA visualizations (generated)
├── src/
│   ├── clean_data.py             # Data cleaning pipeline
│   ├── explore_data.py           # Exploratory data analysis
│   ├── train_model.py            # Model training pipeline
│   └── predict.py                # Command-line inference tool
├── main.py                       # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run the full pipeline

```bash
python main.py
```

This cleans the raw data, trains the model, and runs a sample prediction.

### Run individual steps

```bash
python src/clean_data.py      # Generates data/cleaned_car_data.csv
python src/explore_data.py    # Generates plots in plots/
python src/train_model.py     # Generates models/car_price_model.joblib
```

### Predict a price

With command-line arguments:

```bash
python src/predict.py --name "Maruti Suzuki Alto" --company Maruti --fuel Petrol --year 2017 --kms 35000
```

Interactively:

```bash
python src/predict.py
```

## Model Details

| | |
|---|---|
| Algorithm | Random Forest Regressor (`n_estimators=300`, `max_depth=12`) |
| Features | car name, company, fuel type, car age, kilometers driven |
| Encoding | One-hot encoding for categorical features via `ColumnTransformer` |
| Train/test split | 80/20 |

### Performance

| Metric | Value |
|---|---|
| R² Score | ~0.54 |
| MAE | ~₹1.4 lakh |

These results reflect the inherent noise and limited size of the underlying dataset, which is typical of real-world scraped data rather than a curated benchmark dataset.

## Potential Improvements

- Evaluate gradient boosting methods (e.g. XGBoost, LightGBM) for comparison
- Consolidate low-frequency car names/brands into broader categories to reduce overfitting
- Expand the dataset with additional listings or features (transmission type, ownership history)
- Apply hyperparameter tuning (grid search / randomized search)
