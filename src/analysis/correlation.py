"""
Correlation Analysis Module

Statistical correlation tests between sentiment and stock returns.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple


def calculate_pearson_correlation(sentiment: pd.Series, returns: pd.Series) -> Dict[str, float]:
    """
    Calculate Pearson correlation coefficient with p-value.
    
    Args:
        sentiment: Series of sentiment scores
        returns: Series of stock returns
        
    Returns:
        Dictionary with correlation, p_value, and significance flag
    """
    # Remove NaN values
    valid_idx = ~(sentiment.isna() | returns.isna())
    sent_clean = sentiment[valid_idx]
    ret_clean = returns[valid_idx]
    
    if len(sent_clean) < 3:
        return {'correlation': 0.0, 'p_value': 1.0, 'significant': False, 'n_samples': 0}
    
    corr, p_value = pearsonr(sent_clean, ret_clean)
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_samples': len(sent_clean)
    }


def calculate_spearman_correlation(sentiment: pd.Series, returns: pd.Series) -> Dict[str, float]:
    """
    Calculate Spearman rank correlation (non-parametric).
    
    Args:
        sentiment: Series of sentiment scores
        returns: Series of stock returns
        
    Returns:
        Dictionary with correlation, p_value, and significance
    """
    valid_idx = ~(sentiment.isna() | returns.isna())
    sent_clean = sentiment[valid_idx]
    ret_clean = returns[valid_idx]
    
    if len(sent_clean) < 3:
        return {'correlation': 0.0, 'p_value': 1.0, 'significant': False}
    
    corr, p_value = spearmanr(sent_clean, ret_clean)
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def calculate_lagged_correlation(sentiment: pd.Series, returns: pd.Series,
                                 max_lag: int = 5) -> pd.DataFrame:
    """
    Test correlation with time lags (news impact delay).
    
    Args:
        sentiment: Series of sentiment scores
        returns: Series of stock returns
        max_lag: Maximum number of days to lag (default 5)
        
    Returns:
        DataFrame with lag, correlation, and p_value
    """
    results = []
    
    for lag in range(max_lag + 1):
        if lag == 0:
            lagged_returns = returns
        else:
            lagged_returns = returns.shift(-lag)
        
        corr_result = calculate_pearson_correlation(sentiment, lagged_returns)
        corr_result['lag'] = lag
        results.append(corr_result)
    
    df = pd.DataFrame(results)
    return df[['lag', 'correlation', 'p_value', 'significant', 'n_samples']]


def test_correlation_significance(correlation: float, n: int, alpha: float = 0.05) -> bool:
    """
    Test if correlation is statistically significant.
    
    Args:
        correlation: Correlation coefficient
        n: Sample size
        alpha: Significance level (default 0.05)
        
    Returns:
        True if significant, False otherwise
    """
    if n < 3:
        return False
    
    # Calculate t-statistic
    t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
    
    # Critical value for two-tailed test (approximate)
    from scipy.stats import t
    critical_value = t.ppf(1 - alpha/2, n - 2)
    
    return abs(t_stat) > critical_value


def calculate_rolling_correlation(sentiment: pd.Series, returns: pd.Series,
                                  window: int = 30) -> pd.Series:
    """
    Calculate rolling correlation over time.
    
    Args:
        sentiment: Series of sentiment scores
        returns: Series of stock returns
        window: Rolling window size (default 30 days)
        
    Returns:
        Series of rolling correlations
    """
    rolling_corr = sentiment.rolling(window).corr(returns)
    return rolling_corr
