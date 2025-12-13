"""Check processed CSV contents for expected sentiment columns."""
from pathlib import Path
import pandas as pd
from src.config import settings


def main():
    p = Path('outputs') / 'models' / 'reviews_with_sentiment_and_themes.csv'
    if not p.exists():
        p = Path(settings.DATA_PATHS.get('processed_reviews'))
    print('Using', p)
    df = pd.read_csv(p)
    print('Columns:', df.columns.tolist())
    print('\nDtypes:')
    print(df.dtypes)
    print('\nSample sentiment columns:')
    print(df[['review_id','sentiment_label','sentiment_score']].head(10))


if __name__ == '__main__':
    main()
