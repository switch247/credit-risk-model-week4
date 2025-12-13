"""CustomerFeedbackPipeline class moved into the pipeline package.

This module exposes the `CustomerFeedbackPipeline` class for programmatic use
within the package. It mirrors the previous implementation but uses package
relative imports so it can be imported as
`from src.pipeline.customer_feedback_pipeline import CustomerFeedbackPipeline`.
"""
from pathlib import Path
import json
import pandas as pd
from typing import Optional, Dict, Any

from . import sentiment as sent
from . import text_analysis as ta_module
from . import preprocessing as pp
from ..utils import metrics as metrics_lib
from ..utils.generate_sample import generate_sample_reviews
from ..config import settings

try:
    from joblib import dump as joblib_dump
except Exception:
    joblib_dump = None


class CustomerFeedbackPipeline:
    """Class-based pipeline for sentiment and theme extraction.

    Usage:
        pipeline = CustomerFeedbackPipeline(config)
        pipeline.run()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Load config from settings if not provided
        self.config = config or {}
        self.data_path = self.config.get('data_path', settings.DATA_PATHS.get('processed_reviews'))
        self.out_dir = Path(self.config.get('out_dir', 'outputs/models'))
        self.method = self.config.get('method', 'vader')
        self.n_themes = int(self.config.get('n_themes', 5))
        self.analyzer = ta_module.TextAnalyzer()
        self.df = None
        self.df_processed = None
        self.themes = {}

    def load_data(self):
        p = Path(self.data_path)
        if p.exists():
            self.df = pd.read_csv(p)
        else:
            print(f"Processed data not found at {p}, generating sample data (n=500)...")
            self.df = generate_sample_reviews(n=500)

        # Ensure expected columns exist
        for c in ['review_text', 'rating', 'bank_name']:
            if c not in self.df.columns:
                self.df[c] = None

        print(f"Loaded {len(self.df)} reviews from {self.data_path}")
        return self.df

    def compute_sentiment(self, method: Optional[str] = None):
        method = method or self.method
        print(f"Computing sentiment with method={method}...")
        # Preprocess text into a column used for both sentiment and theme extraction
        try:
            self.df = pp.preprocess_dataframe(self.df, text_col='review_text', out_col='review_text_preprocessed')
        except Exception:
            # If preprocessing fails, fall back to original text column
            self.df['review_text_preprocessed'] = self.df.get('review_text', '').fillna('').astype(str)

        self.df_processed = sent.batch_sentiment(
            self.df,
            text_col='review_text_preprocessed',
            out_score_col='sentiment_score',
            out_label_col='sentiment_label',
            method=method,
        )

        # Coverage check: assert at least 90% of reviews got a non-null sentiment score
        coverage = metrics_lib.evaluate_sentiment_coverage(self.df_processed, 'sentiment_score')
        print(f"Sentiment coverage: {coverage:.2%}")
        try:
            assert coverage >= 0.9, f"Sentiment coverage below required threshold: {coverage:.2%}"
        except AssertionError:
            # Provide a clear message and re-raise so pipelines/tests can catch it
            print(f"WARNING: sentiment coverage {coverage:.2%} < 90%")
            raise
        return self.df_processed

    def extract_themes(self, n_themes: Optional[int] = None):
        n_themes = n_themes or self.n_themes
        print(f"Extracting up to {n_themes} themes per bank...")
        # Prefer preprocessed text when available
        text_col = 'review_text_preprocessed' if 'review_text_preprocessed' in self.df_processed.columns else 'review_text'
        self.themes = self.analyzer.get_themes_by_bank(self.df_processed, text_col=text_col, bank_col='bank_name', n_themes=n_themes)
        return self.themes

    def attach_primary_theme(self):
        def pick_theme_for_row(row):
            bank = row.get('bank_name')
            bank_themes = self.themes.get(bank, [])
            if not bank_themes:
                return None
            return ', '.join(bank_themes[0][:5])

        self.df_processed['identified_theme'] = self.df_processed.apply(pick_theme_for_row, axis=1)
        return self.df_processed

    def aggregate_sentiment_by_bank_rating(self):
        return self.df_processed.groupby(['bank_name', 'rating']).agg({'sentiment_score': 'mean', 'review_text': 'count'}).reset_index().rename(columns={'review_text': 'count'})

    def save_models(self):
        # Save per-bank fitted models (if joblib available)
        if joblib_dump is None:
            return
        for bank in self.themes.keys():
            # Prefer the preprocessed text for fitting models
            text_col = 'review_text_preprocessed' if 'review_text_preprocessed' in self.df_processed.columns else 'review_text'
            texts = self.df_processed[self.df_processed['bank_name'] == bank][text_col].dropna().astype(str).tolist()
            if not texts:
                continue
            try:
                self.analyzer.fit_topic_model(texts, n_topics=self.n_themes, n_top_words=8)
                model_base = Path(self.out_dir) / f"{bank.replace(' ', '_')}_topic_model"
                if self.analyzer.nmf_model is not None:
                    joblib_dump(self.analyzer.nmf_model, str(model_base) + '_nmf.joblib')
                if self.analyzer.tfidf_vectorizer is not None:
                    joblib_dump(self.analyzer.tfidf_vectorizer, str(model_base) + '_tfidf.joblib')
            except Exception:
                continue

    def save_outputs(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = self.out_dir / 'reviews_with_sentiment_and_themes.csv'
        self.df_processed.to_csv(out_csv, index=False)

        themes_path = self.out_dir / 'themes_by_bank.json'
        with open(themes_path, 'w', encoding='utf-8') as f:
            json.dump(self.themes, f, indent=2)

        summary = self.aggregate_sentiment_by_bank_rating()
        summary_path = self.out_dir / 'sentiment_summary_by_bank_rating.csv'
        summary.to_csv(summary_path, index=False)

        metrics = {
            'total_reviews': len(self.df_processed),
            'sentiment_coverage': metrics_lib.evaluate_sentiment_coverage(self.df_processed, 'sentiment_score'),
            'banks': list(self.themes.keys())
        }
        metrics_path = self.out_dir / 'task2_metrics.json'
        metrics_lib.save_metrics(metrics, str(metrics_path))

        theme_counts = metrics_lib.summarize_theme_counts(self.df_processed, bank_col='bank_name', theme_col='identified_theme')
        theme_counts.to_csv(self.out_dir / 'theme_counts.csv', index=False)

        # Attempt to save models
        self.save_models()

        print(f"Saved outputs to {self.out_dir}")

    def run(self, method: Optional[str] = None, n_themes: Optional[int] = None):
        self.load_data()
        self.compute_sentiment(method=method)
        self.extract_themes(n_themes=n_themes)
        self.attach_primary_theme()
        self.save_outputs()
