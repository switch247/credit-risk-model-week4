"""Reusable helpers for the insurance risk analytics dataset.

The functions here are intentionally project-specific (insurance schema) but
standalone so they can be copied into other projects with similar tabular data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

# Columns that should be parsed as dates. TransactionMonth is a period-like date,
# VehicleIntroDate is a month/year field in the raw data.
DATE_COLS: List[str] = ["TransactionMonth", "VehicleIntroDate"]

# Columns that are numeric; everything else defaults to object and is stripped.
NUMERIC_COLS: List[str] = [
    "RegistrationYear",
    "Cylinders",
    "cubiccapacity",
    "kilowatts",
    "NumberOfDoors",
    "NumberOfVehiclesInFleet",
    "CustomValueEstimate",
    "CapitalOutstanding",
    "SumInsured",
    "CalculatedPremiumPerTerm",
    "TotalPremium",
    "TotalClaims",
]

# Boolean-like columns with Yes/No/True/False values.
BOOL_COLS: List[str] = [
    "IsVATRegistered",
    "AlarmImmobiliser",
    "TrackingDevice",
    "NewVehicle",
    "WrittenOff",
    "Rebuilt",
    "Converted",
    "CrossBorder",
]


def load_insurance_raw(path: Path | str, nrows: int | None = None) -> pd.DataFrame:
    """Load the pipe-delimited insurance dataset with light dtype hints."""
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="|",
        nrows=nrows,
        low_memory=False,
        parse_dates=[c for c in DATE_COLS if c not in {"VehicleIntroDate"}],
    )
    return df


def clean_strings(df: pd.DataFrame, exclude: Iterable[str] = ()) -> pd.DataFrame:
    """Strip whitespace and normalize empty strings to NaN for object columns."""
    df = df.copy()
    object_cols = [c for c in df.columns if df[c].dtype == "object" and c not in exclude]
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan})
    return df


def coerce_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize boolean-ish columns to True/False/NaN."""
    df = df.copy()
    mapping = {"Yes": True, "No": False, "True": True, "False": False, True: True, False: False}
    for col in [c for c in BOOL_COLS if c in df.columns]:
        df[col] = df[col].map(mapping).astype("boolean")
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert known numeric columns to float for downstream math."""
    df = df.copy()
    for col in [c for c in NUMERIC_COLS if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def standardize_postal_code(df: pd.DataFrame) -> pd.DataFrame:
    """Keep postal codes as strings to avoid loss of leading zeros."""
    df = df.copy()
    if "PostalCode" in df.columns:
        df["PostalCode"] = (
            df["PostalCode"].astype(str).str.replace(".0$", "", regex=True).str.strip().replace({"": np.nan})
        )
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute loss_ratio, margin, and claim_flag with safe division."""
    df = df.copy()
    premium = df.get("TotalPremium")
    claims = df.get("TotalClaims")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["loss_ratio"] = np.where(premium > 0, claims / premium, np.nan)
    df["margin"] = premium - claims
    df["claim_flag"] = claims > 0
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate policy-month rows if present."""
    df = df.copy()
    key_cols = [c for c in ["UnderwrittenCoverID", "PolicyID", "TransactionMonth"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols)
    return df


def prep_insurance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing chain for EDA-ready data."""
    df = clean_strings(df, exclude=DATE_COLS)
    df = standardize_postal_code(df)
    df = coerce_booleans(df)
    df = coerce_numeric(df)
    df = add_derived_metrics(df)
    df = deduplicate(df)
    return df


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy missingness table with counts and percentages."""
    total = len(df)
    summary = pd.DataFrame({"missing": df.isna().sum()})
    summary["pct"] = (summary["missing"] / total) * 100
    summary = summary.sort_values("pct", ascending=False)
    return summary


def iqr_bounds(series: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    """Compute IQR-based lower and upper bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def flag_outliers(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Return a boolean mask for IQR outliers."""
    lower, upper = iqr_bounds(series, k=k)
    return (series < lower) | (series > upper)


def aggregate_loss(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Aggregate premium/claims and loss ratio by grouping columns."""
    agg = (
        df.groupby(group_cols)
        .agg(
            policies=("PolicyID", "nunique"),
            exposure_months=("TransactionMonth", "count"),
            premium_sum=("TotalPremium", "sum"),
            claims_sum=("TotalClaims", "sum"),
            claim_rate=("claim_flag", "mean"),
        )
        .reset_index()
    )
    agg["loss_ratio"] = np.where(agg["premium_sum"] > 0, agg["claims_sum"] / agg["premium_sum"], np.nan)
    return agg


def summarize_numerics(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Quick descriptive statistics for selected numeric columns."""
    cols = [c for c in cols if c in df.columns]
    return df[cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T


def monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly aggregated premium, claims, loss ratio, and claim frequency."""
    if "TransactionMonth" not in df.columns:
        return pd.DataFrame()
    monthly = (
        df.assign(month=pd.to_datetime(df["TransactionMonth"]).dt.to_period("M"))
        .groupby("month")
        .agg(
            premium_sum=("TotalPremium", "sum"),
            claims_sum=("TotalClaims", "sum"),
            claim_rate=("claim_flag", "mean"),
            policies=("PolicyID", "nunique"),
        )
        .reset_index()
    )
    monthly["loss_ratio"] = np.where(monthly["premium_sum"] > 0, monthly["claims_sum"] / monthly["premium_sum"], np.nan)
    monthly["month"] = monthly["month"].astype(str)
    return monthly
