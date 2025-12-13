"""Reusable EDA utilities for missingness, cardinality, and outlier checks."""

from __future__ import annotations

import pandas as pd
from typing import List


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Percent missing, dtype, and non-null counts."""
    missing_pct = df.isna().mean().mul(100).round(2)
    return (
        pd.DataFrame({"missing_pct": missing_pct, "dtype": df.dtypes.astype(str)})
        .assign(non_null=lambda d: df.shape[0] - (d["missing_pct"] * df.shape[0] / 100))
        .sort_values("missing_pct", ascending=False)
    )


def top_frequencies(df: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    """Return top-n value counts with share."""
    counts = df[column].value_counts(dropna=False).head(n)
    return (
        counts.to_frame(name="count")
        .assign(share=lambda d: d["count"] / len(df))
        .reset_index()
        .rename(columns={"index": column})
    )


def flag_outliers_iqr(series: pd.Series, iqr_multiplier: float = 1.5) -> pd.Series:
    """Boolean mask for IQR-based outliers."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
    return (series < lower) | (series > upper)


def constant_columns(df: pd.DataFrame) -> List[str]:
    """Columns with a single unique value (including NaN)."""
    nunique = df.nunique(dropna=False)
    return nunique[nunique <= 1].index.tolist()


def cardinality_report(df: pd.DataFrame, cat_columns: List[str], max_rows: int = 12) -> pd.DataFrame:
    """Unique counts and share for categorical columns."""
    rows = []
    total = len(df)
    for col in cat_columns:
        if col in df.columns:
            uniq = df[col].nunique(dropna=False)
            rows.append({"column": col, "unique": uniq, "unique_pct": uniq / total if total else 0})
    return pd.DataFrame(rows).sort_values("unique_pct", ascending=False).head(max_rows)


def class_balance(df: pd.DataFrame, candidates: List[str]) -> pd.DataFrame:
    """Return class balance for the first available target column."""
    present = [c for c in candidates if c in df.columns]
    if not present:
        return pd.DataFrame()
    col = present[0]
    counts = df[col].value_counts(dropna=False)
    return (
        counts.to_frame(name="count")
        .assign(share=lambda d: d["count"] / len(df))
        .reset_index()
        .rename(columns={"index": col})
    )


__all__ = [
    "missingness_summary",
    "top_frequencies",
    "flag_outliers_iqr",
    "constant_columns",
    "cardinality_report",
    "class_balance",
]
