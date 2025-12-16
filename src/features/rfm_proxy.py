import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class RFMProxyLabeler(BaseEstimator, TransformerMixin):
    """
    Calculates RFM metrics, performs K-Means clustering, and assigns a 'high_risk' label
    to the cluster with the lowest engagement (Recency, Frequency, Monetary).

    Basel II Context:
    In the absence of observed default events, this proxy serves as the target variable for 
    Predictive Modelling (Probability of Default - PD). We define 'High Risk' (Proxy Default) 
    as customers with significant disengagement (High Recency, Low Frequency/Monetary), 
    aligning with the behavioral assumption that inactivity precedes default or churn.
    """
    def __init__(self, customer_id_col: str, transaction_time_col: str, amount_col: str, n_clusters: int = 3, random_state: int = 42):
        self.customer_id_col = customer_id_col
        self.transaction_time_col = transaction_time_col
        self.amount_col = amount_col
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_ = None
        self.scaler_ = None
        self.high_risk_cluster_ = None
        self.rfm_df_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Calculates RFM, scales it, and fits K-Means.
        Identifies the high-risk cluster.
        """
        # 1. Calculate RFM
        rfm = self._calculate_rfm(X)
        self.rfm_df_ = rfm # Store for inspection if needed

        # 2. Scale RFM
        self.scaler_ = StandardScaler()
        rfm_scaled = self.scaler_.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

        # 3. Cluster
        # cluster by K-Means on scaled RFM (random_state for reproducibility)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(rfm_scaled)
        
        # Assign clusters to the RFM dataframe
        rfm['Cluster'] = self.kmeans_.labels_

        # 4. Identify High-Risk Cluster
        # High risk = "least engaged" = Low Frequency, Low Monetary, High Recency (long time since last txn)
        # We can look at the centroids.
        
        cluster_means = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        
        # We want the cluster with:
        # - Max Recency (Least recent)
        # - Min Frequency
        # - Min Monetary
        
        # A simple heuristic: Rank the clusters. 
        # We want the "worst" behavior.
        # High Recency is "bad" (rank descending)
        # Low Frequency is "bad" (rank ascending)
        # Low Monetary is "bad" (rank ascending)
        
        # Let's create a "Risk Score" for the cluster means to pick the winner.
        # Normalize means for comparison? Or just use the scaled centers.
        
        centers = self.kmeans_.cluster_centers_
        # centers is shape (n_clusters, 3) -> (Recency, Frequency, Monetary) (if columns were in that order)
        # The scaler was fitted on [['Recency', 'Frequency', 'Monetary']]
        
        # Higher Recency (index 0) -> More Risk
        # Lower Frequency (index 1) -> More Risk
        # Lower Monetary (index 2) -> More Risk
        
        # We can define a score: Center_Recency - Center_Frequency - Center_Monetary
        # The cluster with the highest score is the highest risk.
        
        risk_scores = centers[:, 0] - centers[:, 1] - centers[:, 2]
        self.high_risk_cluster_ = np.argmax(risk_scores)
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the original dataframe with an added 'is_high_risk' column.
        """
        if self.kmeans_ is None:
            raise RuntimeError("Fit the model first.")

        # We need to map the calculated risk back to the original transactions.
        # Since RFM is per customer, we merge on CustomerId.
        
        # Re-calculate RFM or use the one from fit? 
        # If X is different from fit X, we must re-calculate RFM for the new customers 
        # and predict their cluster.
        
        # For the purpose of this task (creating a target variable for the training set),
        # we usually fit_transform on the whole dataset.
        
        # If we are transforming new data, we calculate its RFM, scale, predict cluster, check if high risk.
        
        rfm = self._calculate_rfm(X)
        rfm_scaled = self.scaler_.transform(rfm[['Recency', 'Frequency', 'Monetary']])
        clusters = self.kmeans_.predict(rfm_scaled)
        
        rfm['is_high_risk'] = (clusters == self.high_risk_cluster_).astype(int)
        
        # Merge back to X
        # We only want the 'is_high_risk' column
        X_out = X.copy()
        X_out = X_out.merge(rfm[[self.customer_id_col, 'is_high_risk']], on=self.customer_id_col, how='left')
        
        # Fill NaNs if any (e.g. if a customer in X was not in the RFM calculation - shouldn't happen if X is same)
        X_out['is_high_risk'] = X_out['is_high_risk'].fillna(0).astype(int)
        
        return X_out

    def _calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_clean[self.transaction_time_col]):
            df_clean[self.transaction_time_col] = pd.to_datetime(df_clean[self.transaction_time_col], errors='coerce')
            
        # Snapshot date: max date + 1 day
        snapshot_date = df_clean[self.transaction_time_col].max() + pd.Timedelta(days=1)
        
        # Calculate RFM
        rfm = df_clean.groupby(self.customer_id_col).agg({
            self.transaction_time_col: lambda x: (snapshot_date - x.max()).days,
            self.customer_id_col: 'count',
            self.amount_col: 'sum'
        }).rename(columns={
            self.transaction_time_col: 'Recency',
            self.customer_id_col: 'Frequency',
            self.amount_col: 'Monetary'
        })
        
        # Reset index to make CustomerId a column
        rfm = rfm.reset_index()
        
        return rfm
