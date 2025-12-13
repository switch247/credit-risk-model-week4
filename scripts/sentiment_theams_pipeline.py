"""Thin CLI wrapper for the CustomerFeedbackPipeline.

This script should be a small entrypoint that imports the pipeline
implemented in the package and runs it. The heavy implementation lives in
`src.pipeline.customer_feedback_pipeline`.
"""

from pathlib import Path
import argparse
import sys

# Ensure repository root is on sys.path so `src` package is importable when running the script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.sentiment_theams_pipeline import CustomerFeedbackPipeline
from src.config import settings


def main():
    parser = argparse.ArgumentParser(description='Run CustomerFeedbackPipeline')
    parser.add_argument('--data', default=settings.DATA_PATHS.get('processed_reviews'))
    parser.add_argument('--out', default='outputs/models')
    parser.add_argument('--method', default='vader', choices=['vader', 'textblob', 'transformer'])
    parser.add_argument('--n_themes', default=5, type=int)
    args = parser.parse_args()

    cfg = {'data_path': args.data, 'out_dir': args.out, 'method': args.method, 'n_themes': args.n_themes}
    pipeline = CustomerFeedbackPipeline(cfg)
    pipeline.run()


if __name__ == '__main__':
    main()

