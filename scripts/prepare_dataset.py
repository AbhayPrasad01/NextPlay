"""
Prepare the FronkonGames Steam dataset (games.csv) for the recommendation engine.
Outputs cleaned_steam.csv compatible with src/models.py HybridRecommender.
"""
import pandas as pd
import numpy as np
import os
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return ' '.join(text.split())

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    input_csv = os.path.join(data_dir, 'games.csv')

    print("=" * 60)
    print("  PREPARING FRONKONGAMES DATASET")
    print("=" * 60)

    # ── The CSV header has "DiscountDLC count" merged as one column ──
    # ── but data rows have 40 fields. Provide correct 40-column names. ──
    correct_columns = [
        'AppID', 'Name', 'Release date', 'Estimated owners', 'Peak CCU',
        'Required age', 'Price', 'Discount', 'DLC count', 'About the game',
        'Supported languages', 'Full audio languages', 'Reviews',
        'Header image', 'Website', 'Support url', 'Support email',
        'Windows', 'Mac', 'Linux', 'Metacritic score', 'Metacritic url',
        'User score', 'Positive', 'Negative', 'Score rank', 'Achievements',
        'Recommendations', 'Notes', 'Average playtime forever',
        'Average playtime two weeks', 'Median playtime forever',
        'Median playtime two weeks', 'Developers', 'Publishers',
        'Categories', 'Genres', 'Tags', 'Screenshots', 'Movies'
    ]

    print(f"\nLoading {input_csv} ...")
    df = pd.read_csv(input_csv, header=0, names=correct_columns, skiprows=1,
                     low_memory=False, on_bad_lines='skip')
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ── Quick validation ──
    sample = df['Name'].dropna().head(3).tolist()
    print(f"  Sample names: {[str(n)[:50] for n in sample]}")

    # ── Rename columns to expected format ──
    rename = {
        'AppID': 'appid',
        'Name': 'name',
        'Release date': 'release_date',
        'Estimated owners': 'owners',
        'Price': 'price',
        'About the game': 'about_the_game',
        'Header image': 'header_image',
        'Website': 'website',
        'Windows': 'windows',
        'Mac': 'mac',
        'Linux': 'linux',
        'Metacritic score': 'metacritic_score',
        'User score': 'user_score',
        'Positive': 'positive_ratings',
        'Negative': 'negative_ratings',
        'Achievements': 'achievements',
        'Average playtime forever': 'average_playtime',
        'Median playtime forever': 'median_playtime',
        'Developers': 'developer',
        'Publishers': 'publisher',
        'Categories': 'categories',
        'Genres': 'genres',
        'Tags': 'steamspy_tags',
        'Screenshots': 'screenshots',
        'Movies': 'movies',
    }
    rename_existing = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=rename_existing)

    # ── Drop duplicates by name ──
    df = df.drop_duplicates(subset=['name'])
    print(f"  After dedup: {len(df):,} games")

    # ── Filter: must have name and genres ──
    df = df[df['name'].notna() & (df['name'].astype(str).str.strip() != '')]
    df = df[df['genres'].notna() & (df['genres'].astype(str).str.strip() != '')]
    print(f"  After filtering (name+genres required): {len(df):,} games")

    # ── Fill NaN for text fields ──
    text_cols = ['categories', 'genres', 'steamspy_tags', 'developer', 'publisher',
                 'header_image', 'screenshots', 'movies', 'about_the_game', 'website',
                 'release_date', 'owners']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # ── Numeric fields ──
    num_cols = ['positive_ratings', 'negative_ratings', 'average_playtime',
                'median_playtime', 'metacritic_score', 'price', 'achievements', 'appid']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ── Build combined_features for TF-IDF ──
    print("  Building combined features ...")
    cats = df['categories'].astype(str).str.replace(',', ' ', regex=False)
    gens = df['genres'].astype(str).str.replace(',', ' ', regex=False)
    tags = df['steamspy_tags'].astype(str).str.replace(',', ' ', regex=False) if 'steamspy_tags' in df.columns else ''
    df['combined_features'] = (cats + ' ' + gens + ' ' + tags).apply(clean_text)

    # ── Build clean_name ──
    print("  Building clean names ...")
    df['clean_name'] = df['name'].apply(clean_text)

    # ── Steam store link ──
    df['steam_link'] = 'https://store.steampowered.com/app/' + df['appid'].astype(int).astype(str)

    # ── Platforms string ──
    def build_platforms(row):
        parts = []
        if str(row.get('windows', '')).strip().lower() in ['true', '1']: parts.append('windows')
        if str(row.get('mac', '')).strip().lower() in ['true', '1']: parts.append('mac')
        if str(row.get('linux', '')).strip().lower() in ['true', '1']: parts.append('linux')
        return ';'.join(parts) if parts else 'windows'
    df['platforms'] = df.apply(build_platforms, axis=1)

    # ── Select and save ──
    output_cols = [
        'appid', 'name', 'release_date', 'developer', 'publisher', 'platforms',
        'categories', 'genres', 'steamspy_tags', 'achievements',
        'positive_ratings', 'negative_ratings', 'average_playtime', 'median_playtime',
        'owners', 'price', 'combined_features', 'clean_name',
        'header_image', 'screenshots', 'movies', 'about_the_game', 'website',
        'steam_link', 'metacritic_score',
    ]
    output_cols = [c for c in output_cols if c in df.columns]

    out_path = os.path.join(data_dir, 'cleaned_steam.csv')
    df[output_cols].to_csv(out_path, index=False)
    print(f"\n  Saved {len(df):,} games to cleaned_steam.csv")

    # ── Generate synthetic user interactions ──
    print("\n  Generating synthetic user interactions ...")
    played = df[df['positive_ratings'] > 0].copy()
    print(f"    {len(played):,} games have ratings")

    np.random.seed(42)
    n_users = 8000
    interactions = []

    weights = np.log1p(played['positive_ratings'].values).astype(float)
    weights = weights / weights.sum()

    for uid in range(n_users):
        n_games = np.random.randint(3, 25)
        idxs = np.random.choice(len(played), size=min(n_games, len(played)), replace=False, p=weights)
        for idx in idxs:
            row = played.iloc[idx]
            avg_pt = max(row['average_playtime'], 10)
            playtime = int(np.random.lognormal(np.log(avg_pt), 0.8))
            playtime = max(1, min(playtime, 50000))
            interactions.append({
                'user_id': uid,
                'game_name': row['name'],
                'clean_name': row['clean_name'],
                'playtime': playtime,
            })

    ui_df = pd.DataFrame(interactions)
    ui_path = os.path.join(data_dir, 'cleaned_user_interactions.csv')
    ui_df.to_csv(ui_path, index=False)
    print(f"    Saved {len(ui_df):,} interactions ({n_users:,} users)")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)

if __name__ == '__main__':
    main()
