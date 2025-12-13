#!/usr/bin/env python3
"""Generate insights, visualizations and a markdown summary from processed reviews.

Saves figures to `outputs/figures/` and writes `outputs/reports/final_insights.md`.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import sys

try:
    from wordcloud import WordCloud
    HAVE_WORDCLOUD = True
except Exception:
    HAVE_WORDCLOUD = False


DATA_P = Path('outputs') / 'models' / 'reviews_with_sentiment_and_themes.csv'
FIG_DIR = Path('outputs') / 'figures'
REPORT_MD = Path('outputs') / 'reports' / 'final_insights.md'


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed reviews file not found: {path}")
    df = pd.read_csv(path, parse_dates=['review_date'], dayfirst=False, low_memory=False)
    return df


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)


def sentiment_trends(df: pd.DataFrame):
    df['year_month'] = df['review_date'].dt.to_period('M').astype(str)
    monthly = df.groupby(['year_month', 'bank_name']).agg(
        mean_sentiment=('sentiment_score', 'mean'),
        n_reviews=('review_id', 'count')
    ).reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly, x='year_month', y='mean_sentiment', hue='bank_name', marker='o')
    plt.xticks(rotation=45)
    plt.title('Monthly Mean Sentiment by Bank')
    plt.tight_layout()
    p = FIG_DIR / 'sentiment_trends.png'
    plt.savefig(p)
    plt.close()
    return p


def rating_distribution(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='rating', hue='bank_name', palette='Set2')
    plt.title('Rating Distribution by Bank')
    plt.tight_layout()
    p = FIG_DIR / 'rating_distribution.png'
    plt.savefig(p)
    plt.close()
    return p


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def top_words_by_bank(df: pd.DataFrame, bank: str, sentiment_label=None, top_n=25):
    sel = df[df['bank_name'] == bank]
    if sentiment_label:
        sel = sel[sel['sentiment_label'] == sentiment_label]
    texts = sel['review_text'].fillna('').map(clean_text)
    words = Counter()
    for t in texts:
        words.update([w for w in t.split() if len(w) > 2])
    return words.most_common(top_n)


def plot_word_freq(freq, out_path: Path, title: str = None):
    if HAVE_WORDCLOUD:
        wc = WordCloud(width=800, height=400, background_color='white')
        wc.generate_from_frequencies(dict(freq))
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    else:
        # fallback to horizontal bar chart
        words, counts = zip(*freq[:20]) if freq else ([], [])
        plt.figure(figsize=(8, 6))
        # use a single color to avoid seaborn palette/hue deprecation warning
        sns.barplot(x=list(counts), y=list(words), color='C0')
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    return out_path


def derive_insights(df: pd.DataFrame):
    banks = df['bank_name'].unique().tolist()
    insights = {}
    for bank in banks:
        pos_top = top_words_by_bank(df, bank, sentiment_label='positive', top_n=30)
        neg_top = top_words_by_bank(df, bank, sentiment_label='negative', top_n=30)
        # heuristics for drivers and pain points
        drivers = [w for w, _ in pos_top[:10]]
        pain = [w for w, _ in neg_top[:12]]
        insights[bank] = {
            'drivers': drivers,
            'pain_points': pain,
            'pos_top': pos_top,
            'neg_top': neg_top
        }
    return insights


def check_kpis(df: pd.DataFrame):
    kpis = {}
    kpis['n_reviews'] = int(df.shape[0])
    kpis['banks_covered'] = sorted(df['bank_name'].unique().tolist())
    kpis['sentiment_scores_present'] = 'sentiment_score' in df.columns and df['sentiment_score'].notna().any()
    kpis['themes_present'] = 'identified_theme' in df.columns and df['identified_theme'].notna().any()
    # AHT (Average Handling Time) is not present in the dataset
    kpis['AHT_present'] = 'AHT' in df.columns or 'aht' in df.columns
    return kpis


def write_report(md_path: Path, kpis, insights, figure_paths):
    lines = []
    lines.append('# Final Insights & Recommendations')
    lines.append('')
    lines.append('**Key Performance Indicators (KPI) check**:')
    lines.append('')
    lines.append(f'- **Total reviews**: {kpis["n_reviews"]}')
    lines.append(f'- **Banks covered**: {", ".join(kpis["banks_covered"]) }')
    lines.append(f'- **Sentiment scores available**: {kpis["sentiment_scores_present"]}')
    lines.append(f'- **Thematic labels available**: {kpis["themes_present"]}')
    if not kpis['AHT_present']:
        lines.append('- **AHT (Average Handling Time)**: MISSING — KPI unmet (no `AHT` column).')
    else:
        lines.append('- **AHT (Average Handling Time)**: Present.')
    lines.append('')

    lines.append('## Visualizations')
    lines.append('')
    for k, p in figure_paths.items():
        lines.append(f'- **{k}**: `{p}`')
    lines.append('')

    lines.append('## Bank-level Insights and Recommendations')
    lines.append('')
    for bank, info in insights.items():
        lines.append(f'### {bank}')
        lines.append('')
        drivers = ', '.join(info['drivers'][:8])
        pain = ', '.join(info['pain_points'][:8])
        lines.append(f'- **Top drivers**: {drivers or "(not enough positive reviews)"}.')
        lines.append(f'- **Top pain points**: {pain or "(not enough negative reviews)"}.')
        lines.append('')
        # Recommendations: generic mapping from pain points
        recs = []
        if any(w in ' '.join(info['pain_points']).lower() for w in ['slow', 'lag', 'loading', 'load']):
            recs.append('Optimize app performance and reduce launch/load times; add performance monitoring.')
        if any(w in ' '.join(info['pain_points']).lower() for w in ['crash', 'crushed', 'freeze', 'not working']):
            recs.append('Improve crash handling, automated error reporting and QA for releases.')
        if any(w in ' '.join(info['pain_points']).lower() for w in ['otp', 'security', 'auth', 'disable']):
            recs.append('Review authentication flows (OTP, security questions) and add clearer UX/error messages.')
        if not recs:
            recs = [
                'Perform targeted UX testing to validate flows users report as confusing.',
                'Improve monitoring and release rollback procedures.'
            ]
        for r in recs[:5]:
            lines.append(f'- **Recommendation**: {r}')
        lines.append('')

    lines.append('## Cross-bank comparison')
    lines.append('')
    lines.append('- Compare banks by mean sentiment and rating distributions (see visualizations).')
    lines.append('- Look for differences in common pain points (e.g., one bank may show more connection/OTP problems while another shows crashes).')
    lines.append('')

    lines.append('## Ethics & Bias')
    lines.append('')
    lines.append('- Reviews are self-selected and may over-represent frustrated or highly satisfied users (selection bias).')
    lines.append('- Language differences and automated translation can distort sentiment scores.')
    lines.append('- Consider demographic and channel biases (Google Play users differ from in-branch customers).')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append('Generated with `scripts/generate_insights.py`')

    md_path.write_text('\n'.join(lines), encoding='utf8')


def main():
    ensure_dirs()
    df = load_data(DATA_P)
    fig_paths = {}
    fig_paths['sentiment_trends'] = sentiment_trends(df)
    fig_paths['rating_distribution'] = rating_distribution(df)

    # word/keyword visualizations per bank
    insights = derive_insights(df)
    for bank in insights.keys():
        pos_freq = insights[bank]['pos_top']
        neg_freq = insights[bank]['neg_top']
        ppos = plot_word_freq(pos_freq, FIG_DIR / f'wordcloud_{bank}_positive.png', title=f'{bank} - positive keywords')
        pneg = plot_word_freq(neg_freq, FIG_DIR / f'wordcloud_{bank}_negative.png', title=f'{bank} - negative keywords')
        fig_paths[f'{bank}_positive_keywords'] = ppos
        fig_paths[f'{bank}_negative_keywords'] = pneg

    kpis = check_kpis(df)
    write_report(REPORT_MD, kpis, insights, fig_paths)
    print(f'Report written: {REPORT_MD}')


if __name__ == '__main__':
    main()
