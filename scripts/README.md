# Scripts

This directory contains executable scripts for the data pipeline, modeling, and database management.

## Pipeline Scripts

- **`build_features.py`** (Task 3):
  - Reads raw data from `data/raw/`.
  - Applies feature engineering (date extraction, categorical encoding, etc.).
  - Saves processed data to `data/processed/xente_processed.csv`.

- **`create_credit_risk_target.py`** (Task 4):
  - Reads `data/processed/xente_processed.csv`.
  - Performs RFM analysis and K-Means clustering.
  - Assigns a `is_high_risk` label to the least engaged cluster.
  - Saves the labeled dataset to `data/processed/xente_processed_with_risk.csv`.

- **`generate_insights.py`**:
  - Generates visualizations and summary reports from the processed data.
  - Outputs to `outputs/figures/` and `outputs/reports/`.

- **`train_sentiment_analysis_model.py`**:
  - Trains a sentiment analysis model (if applicable to the dataset).

- **`check_processed_df.py`**:
  - Utility to inspect the processed dataframe.

## Database Scripts

- **`create_database.py`**: Creates the PostgreSQL database and tables.
- **`run_migrations.py`**: Applies SQL schema migrations.
- **`dump_db.py`**: Dumps the database content.
- **`test_db_connection.py`**: Verifies connectivity to the database.

## Setup

- **`setup_venv.ps1`**: PowerShell script to set up the Python virtual environment.
