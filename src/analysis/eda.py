"""
Exploratory Data Analysis Module

Functions for analyzing customer review data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class EDA:
    """Class for Exploratory Data Analysis"""

    def __init__(self, df: pd.DataFrame = None):
        """
        Initialize EDA with a DataFrame
        
        Args:
            df: DataFrame to analyze
        """
        self.df = df

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file"""
        self.df = pd.read_csv(filepath)
        return self.df

    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset"""
        if self.df is None:
            raise ValueError("No data loaded")

        stats = {
            'total_reviews': len(self.df),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }

        if 'rating' in self.df.columns:
            stats['rating_distribution'] = self.df['rating'].value_counts().to_dict()
            stats['avg_rating'] = self.df['rating'].mean()

        if 'bank_name' in self.df.columns:
            stats['reviews_per_bank'] = self.df['bank_name'].value_counts().to_dict()

        if 'text_length' in self.df.columns:
            stats['avg_text_length'] = self.df['text_length'].mean()
            stats['median_text_length'] = self.df['text_length'].median()

        return stats

    def get_rating_distribution(self) -> pd.Series:
        """Get rating distribution"""
        if 'rating' not in self.df.columns:
            raise ValueError("No 'rating' column found")
        return self.df['rating'].value_counts().sort_index()

    def get_reviews_by_bank(self) -> pd.Series:
        """Get review counts by bank"""
        if 'bank_name' not in self.df.columns:
            raise ValueError("No 'bank_name' column found")
        return self.df['bank_name'].value_counts()

    def get_date_range(self) -> Dict[str, Any]:
        """Get date range of reviews"""
        if 'review_date' not in self.df.columns:
            raise ValueError("No 'review_date' column found")
        
        self.df['review_date'] = pd.to_datetime(self.df['review_date'])
        return {
            'min_date': self.df['review_date'].min(),
            'max_date': self.df['review_date'].max(),
            'date_range_days': (self.df['review_date'].max() - self.df['review_date'].min()).days
        }

    def get_top_words(self, column: str = 'review_text', n: int = 20) -> Dict[str, int]:
        """
        Get top N most common words from text column
        
        Args:
            column: Column name containing text
            n: Number of top words to return
            
        Returns:
            Dictionary of word: count
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found")

        from collections import Counter
        import re

        # Combine all text
        all_text = ' '.join(self.df[column].dropna().astype(str))
        
        # Simple tokenization (lowercase, remove punctuation)
        words = re.findall(r'\b[a-z]+\b', all_text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
                      'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her',
                      'its', 'our', 'their', 'me', 'him', 'us', 'them'}
        
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        word_counts = Counter(words)
        return dict(word_counts.most_common(n))

    def summary_report(self) -> str:
        """Generate a summary report"""
        if self.df is None:
            raise ValueError("No data loaded")

        stats = self.get_basic_stats()
        
        report = []
        report.append("=" * 60)
        report.append("EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Reviews: {stats['total_reviews']}")
        
        if 'reviews_per_bank' in stats:
            report.append("\nReviews per Bank:")
            for bank, count in stats['reviews_per_bank'].items():
                report.append(f"  {bank}: {count}")
        
        if 'rating_distribution' in stats:
            report.append("\nRating Distribution:")
            for rating in sorted(stats['rating_distribution'].keys(), reverse=True):
                count = stats['rating_distribution'][rating]
                pct = (count / stats['total_reviews']) * 100
                report.append(f"  {'⭐' * int(rating)}: {count} ({pct:.1f}%)")
            report.append(f"\nAverage Rating: {stats['avg_rating']:.2f}")
        
        if 'avg_text_length' in stats:
            report.append(f"\nText Statistics:")
            report.append(f"  Average Length: {stats['avg_text_length']:.0f} characters")
            report.append(f"  Median Length: {stats['median_text_length']:.0f} characters")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
