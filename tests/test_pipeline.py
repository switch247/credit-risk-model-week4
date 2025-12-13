"""
Test pipeline modules
"""
import pytest


def test_pipeline_module_exists():
    """Test pipeline module can be imported"""
    from src import pipeline
    assert pipeline is not None


def test_sentiment_functions():
    """Test sentiment analysis functions exist"""
    from src.pipeline import (
        setup_nltk_resources,
        analyze_headline_sentiment
    )
    assert setup_nltk_resources is not None
    assert analyze_headline_sentiment is not None
