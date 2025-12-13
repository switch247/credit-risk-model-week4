import pytest
from src.pipeline.text_analysis import TextAnalyzer
from src.utils.generate_sample import generate_sample_reviews
import pandas as pd


def test_text_analyzer_extract_keywords():
    texts = ["login error, cannot login", "fast transfer but UI slow", "crash when sending money"]
    ta = TextAnalyzer()
    kws = ta.extract_keywords(texts, top_n=5)
    assert isinstance(kws, list)
    assert len(kws) <= 5


def test_get_themes_by_bank_on_sample():
    df = generate_sample_reviews(n=30)
    ta = TextAnalyzer()
    themes = ta.get_themes_by_bank(df, text_col='review_text', bank_col='bank_name', n_themes=2)
    assert isinstance(themes, dict)
    # banks should be present
    assert all(isinstance(k, str) for k in themes.keys())
