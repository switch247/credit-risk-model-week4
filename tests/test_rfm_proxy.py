import pytest
import pandas as pd
import numpy as np
from src.features.rfm_proxy import RFMProxyLabeler

@pytest.fixture
def rfm_sample_df():
    # Create a dataset with clear clusters
    # C1: High engagement (Recent, Frequent, High Value)
    # C2: Medium engagement
    # C3: Low engagement (Old, Infrequent, Low Value) -> Should be High Risk
    
    data = {
        'CustomerId': ['C1', 'C1', 'C1', 'C2', 'C2', 'C3'],
        'TransactionStartTime': [
            '2023-12-31', '2023-12-30', '2023-12-29', # C1: Very recent
            '2023-06-01', '2023-06-02',               # C2: Mid year
            '2023-01-01'                              # C3: Very old
        ],
        'Amount': [
            1000, 1000, 1000, # C1: High value
            500, 500,         # C2: Mid value
            10                # C3: Low value
        ]
    }
    df = pd.DataFrame(data)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def test_rfm_calculation(rfm_sample_df):
    labeler = RFMProxyLabeler(
        customer_id_col='CustomerId',
        transaction_time_col='TransactionStartTime',
        amount_col='Amount',
        n_clusters=3,
        random_state=42
    )
    
    rfm = labeler._calculate_rfm(rfm_sample_df)
    
    # Check C1
    c1 = rfm[rfm['CustomerId'] == 'C1'].iloc[0]
    assert c1['Frequency'] == 3
    assert c1['Monetary'] == 3000
    # Recency should be small (snapshot is max date + 1 day = 2024-01-01)
    # C1 max date is 2023-12-31. Recency = 1 day.
    assert c1['Recency'] == 1
    
    # Check C3
    c3 = rfm[rfm['CustomerId'] == 'C3'].iloc[0]
    assert c3['Frequency'] == 1
    assert c3['Monetary'] == 10
    # C3 max date is 2023-01-01. Recency ~ 365 days.
    assert c3['Recency'] >= 360

def test_fit_transform_assigns_risk(rfm_sample_df):
    labeler = RFMProxyLabeler(
        customer_id_col='CustomerId',
        transaction_time_col='TransactionStartTime',
        amount_col='Amount',
        n_clusters=3,
        random_state=42
    )
    
    df_labeled = labeler.fit_transform(rfm_sample_df)
    
    assert 'is_high_risk' in df_labeled.columns
    
    # Check that we have both 0 and 1 labels (assuming 3 clusters on this data produces separation)
    # With 3 distinct profiles, we expect C3 to be high risk (1) and C1 to be low risk (0)
    
    c1_risk = df_labeled[df_labeled['CustomerId'] == 'C1']['is_high_risk'].iloc[0]
    c3_risk = df_labeled[df_labeled['CustomerId'] == 'C3']['is_high_risk'].iloc[0]
    
    # C3 is the "worst" customer, so it should be high risk (1)
    # C1 is the "best", so it should be low risk (0)
    
    # Note: K-Means is unsupervised, but our heuristic (max Recency - Freq - Mon) 
    # should consistently pick the "worst" cluster.
    
    assert c3_risk == 1
    assert c1_risk == 0

def test_transform_new_data(rfm_sample_df):
    labeler = RFMProxyLabeler(
        customer_id_col='CustomerId',
        transaction_time_col='TransactionStartTime',
        amount_col='Amount',
        n_clusters=3,
        random_state=42
    )
    
    labeler.fit(rfm_sample_df)
    
    # Create new data similar to C3 (High Risk)
    new_data = pd.DataFrame({
        'CustomerId': ['C4'],
        'TransactionStartTime': ['2023-01-02'], # Old
        'Amount': [5] # Low value
    })
    new_data['TransactionStartTime'] = pd.to_datetime(new_data['TransactionStartTime'])
    
    df_new_labeled = labeler.transform(new_data)
    
    assert df_new_labeled['is_high_risk'].iloc[0] == 1
