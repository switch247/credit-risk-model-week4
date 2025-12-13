import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class TemporalExtractor(BaseEstimator, TransformerMixin):
    """Extracts hour/day/month/year from a datetime column."""

    def __init__(self, time_col: str):
        self.time_col = time_col

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = X.copy()
        if self.time_col in Xt.columns:
            dt = pd.to_datetime(Xt[self.time_col], errors='coerce')
            Xt['transaction_hour'] = dt.dt.hour
            Xt['transaction_day'] = dt.dt.day
            Xt['transaction_month'] = dt.dt.month
            Xt['transaction_year'] = dt.dt.year
        return Xt


class CustomerAggregates(BaseEstimator, TransformerMixin):
    """Computes per-customer aggregates and merges back to row-level for modeling.

    Aggregates:
    - total_amount, avg_amount, txn_count, std_amount
    """

    def __init__(self, customer_id_col: str, amount_col: str):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self._agg_df: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y=None):
        grp = (
            X.groupby(self.customer_id_col)[self.amount_col]
            .agg(['sum', 'mean', 'count', 'std'])
            .rename(columns={
                'sum': 'total_amount',
                'mean': 'avg_amount',
                'count': 'txn_count',
                'std': 'std_amount',
            })
        )
        # Handle NaN std for single transactions
        grp['std_amount'] = grp['std_amount'].fillna(0.0)
        grp = grp.reset_index()
        self._agg_df = grp
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._agg_df is None:
            raise RuntimeError('CustomerAggregates must be fitted before transform.')
        Xt = X.merge(self._agg_df, on=self.customer_id_col, how='left')
        return Xt


def build_feature_pipeline(
    *,
    customer_id_col: str = 'CustomerId',
    amount_col: str = 'Amount',
    time_col: str = 'TransactionStartTime',
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
) -> Tuple[Pipeline, List[str]]:
    """Constructs a sklearn Pipeline that:
    - Extracts temporal features
    - Computes per-customer aggregates
    - Encodes categoricals (OHE)
    - Imputes missing values
    - Scales numerical features

    Returns (pipeline, output_feature_names_estimate)
    """

    # Default column sets based on Xente schema
    if categorical_cols is None:
        categorical_cols = [
            'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId',
            'PricingStrategy', 'FraudResult'
        ]
    if numerical_cols is None:
        numerical_cols = [
            amount_col, 'Value', 'transaction_hour', 'transaction_day', 'transaction_month', 'transaction_year',
            'total_amount', 'avg_amount', 'txn_count', 'std_amount'
        ]

    pre_steps = [
        ('temporal', TemporalExtractor(time_col=time_col)),
        ('cust_aggs', CustomerAggregates(customer_id_col=customer_id_col, amount_col=amount_col)),
    ]

    transformers = [
        (
            'cat',
            Pipeline([
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]),
            categorical_cols,
        ),
        (
            'num',
            Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler()),
            ]),
            numerical_cols,
        ),
    ]

    column_tf = ColumnTransformer(transformers=transformers, remainder='drop')
    pipe = Pipeline(steps=pre_steps + [('columns', column_tf)])

    # We cannot know exact expanded OHE feature names without fitting; provide an estimate list
    output_features_estimate = [
        # numerical (post-scale)
        *numerical_cols,
        # categorical will expand; we return base cols for reference
        *categorical_cols,
    ]

    return pipe, output_features_estimate


def build_woe_iv(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Optional[pd.DataFrame]:
    """Optionally compute WoE/IV using xverse if available; returns dataframe of IV per feature.
    This does not modify the pipeline; it is for analysis/selection only.
    """
    try:
        from xverse.transformer import WOE
    except Exception:
        return None

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # xverse requires categorical/binned features; it will bin numericals internally
    w = WOE()
    w.fit(X, y)
    iv_df = w.iv_df
    return iv_df


def load_raw_xente(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def process_xente(
    raw_path: str,
    *,
    output_npy_path: Optional[str] = None,
    return_array: bool = True,
):
    """Load raw Xente data, fit-transform the feature pipeline, and optionally persist numpy array.
    Returns the transformed array if return_array is True.
    """
    df = load_raw_xente(raw_path)
    pipe, feat_names = build_feature_pipeline()
    X = pipe.fit_transform(df)
    if output_npy_path:
        os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
        np.save(output_npy_path, X)
    return X if return_array else None
