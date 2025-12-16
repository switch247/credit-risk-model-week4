import pytest
import pandas as pd
import numpy as np
from src.features.xente_features import TemporalExtractor, CustomerAggregates, build_feature_pipeline

@pytest.fixture
def sample_df():
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4'],
        'CustomerId': ['C1', 'C1', 'C2', 'C3'],
        'Amount': [100.0, 200.0, 500.0, 50.0],
        'TransactionStartTime': [
            '2023-01-01 10:00:00',
            '2023-01-02 11:00:00',
            '2023-02-01 12:00:00',
            '2023-03-01 13:00:00'
        ],
        'ProviderId': ['P1', 'P1', 'P2', 'P1'],
        'ProductCategory': ['Cat1', 'Cat1', 'Cat2', 'Cat1']
    }
    return pd.DataFrame(data)

def test_temporal_extractor(sample_df):
    extractor = TemporalExtractor(time_col='TransactionStartTime')
    df_transformed = extractor.fit_transform(sample_df)
    
    assert 'transaction_hour' in df_transformed.columns
    assert 'transaction_day' in df_transformed.columns
    assert 'transaction_month' in df_transformed.columns
    assert 'transaction_year' in df_transformed.columns
    
    # Check specific values
    assert df_transformed.loc[0, 'transaction_hour'] == 10
    assert df_transformed.loc[0, 'transaction_month'] == 1
    assert df_transformed.loc[2, 'transaction_month'] == 2

def test_customer_aggregates(sample_df):
    aggregator = CustomerAggregates(customer_id_col='CustomerId', amount_col='Amount')
    aggregator.fit(sample_df)
    df_transformed = aggregator.transform(sample_df)
    
    assert 'total_amount' in df_transformed.columns
    assert 'avg_amount' in df_transformed.columns
    assert 'txn_count' in df_transformed.columns
    assert 'std_amount' in df_transformed.columns
    
    # Check C1 values (2 transactions: 100, 200)
    c1_rows = df_transformed[df_transformed['CustomerId'] == 'C1']
    assert c1_rows['total_amount'].iloc[0] == 300.0
    assert c1_rows['avg_amount'].iloc[0] == 150.0
    assert c1_rows['txn_count'].iloc[0] == 2
    assert c1_rows['std_amount'].iloc[0] > 0 # Should have std dev
    
    # Check C2 values (1 transaction: 500)
    c2_rows = df_transformed[df_transformed['CustomerId'] == 'C2']
    assert c2_rows['total_amount'].iloc[0] == 500.0
    assert c2_rows['std_amount'].iloc[0] == 0.0 # Single value std is 0 (filled)

def test_build_feature_pipeline():
    pipeline, feature_names = build_feature_pipeline(
        customer_id_col='CustomerId',
        amount_col='Amount',
        time_col='TransactionStartTime',
        categorical_cols=['ProviderId'],
        numerical_cols=['Amount']
    )
    
    assert pipeline is not None
    assert isinstance(feature_names, list)
