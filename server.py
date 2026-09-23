"""
Flask backend API for the Game Recommendation System.
Serves the HTML/CSS/JS frontend and provides REST endpoints for ML recommendations.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from models import HybridRecommender

app = Flask(__name__, static_folder='static')

# ── Global state ──
rec = None
steam_df = None
user_df = None

def init_recommender():
    global rec, steam_df, user_df
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    print("Loading recommendation engine ...")
    rec = HybridRecommender(data_dir)
    steam_df = rec.steam_df
    user_df = rec.user_df
    print(f"Engine ready — {len(steam_df):,} games, {len(user_df):,} interactions")

# ── Helpers ──
def game_to_dict(row):
    """Convert a DataFrame row to a JSON-serializable dict with game details."""
    screenshots = str(row.get('screenshots', '')).split(',') if pd.notna(row.get('screenshots')) and str(row.get('screenshots')).strip() else []
    screenshots = [s.strip() for s in screenshots if s.strip().startswith('http')][:8]

    movies = str(row.get('movies', '')).split(',') if pd.notna(row.get('movies')) and str(row.get('movies')).strip() else []
    movies = [m.strip() for m in movies if m.strip().startswith('http')][:3]

    total = float(row.get('positive_ratings', 0)) + float(row.get('negative_ratings', 0))
    quality = round(float(row.get('positive_ratings', 0)) / total * 100, 1) if total > 0 else 0

    about = str(row.get('about_the_game', '')) if pd.notna(row.get('about_the_game')) else ''
    # Clean up "nan" strings
    if about.lower() in ('nan', 'none', ''):
        about = ''

    tags_raw = str(row.get('steamspy_tags', '')) if pd.notna(row.get('steamspy_tags')) else ''
    cats_raw = str(row.get('categories', '')) if pd.notna(row.get('categories')) else ''
    genres_raw = str(row.get('genres', '')) if pd.notna(row.get('genres')) else ''

    # Platforms
    platforms = str(row.get('platforms', '')) if pd.notna(row.get('platforms')) else ''

    return {
        'name': str(row.get('name', '')),
        'appid': int(row.get('appid', 0)),
        'developer': str(row.get('developer', '')),
        'publisher': str(row.get('publisher', '')),
        'genres': genres_raw,
        'categories': cats_raw,
        'tags': tags_raw,
        'steamspy_tags': tags_raw,
        'platforms': platforms,
        'release_date': str(row.get('release_date', '')),
        'price': float(row.get('price', 0)),
        'header_image': str(row.get('header_image', '')),
        'screenshots': screenshots,
        'movies': movies,
        'description': about,
        'about_the_game': about,
        'website': str(row.get('website', '')),
        'steam_link': str(row.get('steam_link', f"https://store.steampowered.com/app/{int(row.get('appid', 0))}")),
        'positive_ratings': int(row.get('positive_ratings', 0)),
        'negative_ratings': int(row.get('negative_ratings', 0)),
        'quality_score': quality,
        'metacritic_score': int(row.get('metacritic_score', 0)),
        'average_playtime': int(row.get('average_playtime', 0)),
        'median_playtime': int(row.get('median_playtime', 0)),
        'owners': str(row.get('owners', '')),
        'achievements': int(row.get('achievements', 0)),
    }

def playtime_label(mins):
    h = mins / 60
    if h < 10: return "Short"
    if h < 40: return "Medium"
    return "Long"

# ── Static file serving ──
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/css/<path:filename>')
def css(filename):
    return send_from_directory('static/css', filename)

@app.route('/js/<path:filename>')
def js(filename):
    return send_from_directory('static/js', filename)

# ── API Endpoints ──

@app.route('/api/stats')
def api_stats():
    n_games = len(steam_df)
    n_users = int(user_df['user_id'].nunique())
    n_interactions = len(user_df)
    avg_playtime = round(user_df['playtime'].mean() / 60, 1)

    # Genre distribution
    genres = steam_df['genres'].dropna().str.split(',').explode().str.strip()
    genre_counts = genres.value_counts().head(10).to_dict()

    return jsonify({
        'n_games': n_games,
        'n_users': n_users,
        'n_interactions': n_interactions,
        'avg_playtime_hrs': avg_playtime,
        'genre_distribution': genre_counts,
    })

@app.route('/api/games')
def api_games():
    import re
    q = request.args.get('q', '').lower().strip()
    limit = min(int(request.args.get('limit', 20)), 50)

    if not q:
        return jsonify([])
        
    q_clean = re.sub(r'[^a-z0-9]', '', q)
    mask = steam_df['clean_name'].str.replace(' ', '', regex=False).str.contains(q_clean, na=False)
    matches = steam_df[mask]

    results = []
    for _, row in matches.head(limit).iterrows():
        results.append({
            'name': row['name'],
            'header_image': str(row.get('header_image', '')),
            'developer': str(row.get('developer', '')),
            'genres': str(row.get('genres', '')),
        })
    return jsonify(results)

@app.route('/api/game/<path:name>')
def api_game_detail(name):
    matches = steam_df[steam_df['name'] == name]
    if matches.empty:
        return jsonify({'error': 'Game not found'}), 404
    return jsonify(game_to_dict(matches.iloc[0]))

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    data = request.get_json()
    game = data.get('game', '')
    algo = data.get('algorithm', 'Hybrid')
    count = min(int(data.get('count', 10)), 30)

    if not game:
        return jsonify({'error': 'No game specified'}), 400

    raws = rec.get_recommendations(game, rec_type=algo, top_n=count + 5)
    if raws is None or raws.empty:
        return jsonify({'results': [], 'message': 'No recommendations found for this title.'})

    # Merge with full details
    rated = steam_df.copy()
    total = rated['positive_ratings'] + rated['negative_ratings']
    rated['quality_score'] = np.where(total > 0, rated['positive_ratings'] / total, 0.0)

    detail_cols = ['name', 'genres', 'average_playtime', 'positive_ratings', 'negative_ratings',
                   'developer', 'header_image', 'screenshots', 'movies', 'website', 'steam_link',
                   'appid', 'release_date', 'price', 'metacritic_score', 'quality_score',
                   'about_the_game', 'steamspy_tags', 'categories', 'owners', 'publisher', 'mood']
    detail_cols = [c for c in detail_cols if c in rated.columns]
    merged = pd.merge(raws, rated[detail_cols], on='name', how='left')
    merged = merged[merged['name'] != game].head(count)

    results = []
    for _, row in merged.iterrows():
        d = game_to_dict(row)
        d['match_score'] = round(float(row.get('final_score', 0)) * 100, 1)
        d['explanation'] = str(row.get('explanation', ''))
        results.append(d)

    return jsonify({'results': results, 'algorithm': algo, 'source_game': game})

@app.route('/api/recommend/multi', methods=['POST'])
def api_recommend_multi():
    data = request.get_json()
    games = data.get('games', [])
    count = min(int(data.get('count', 10)), 30)

    if not games:
        return jsonify({'error': 'No games specified'}), 400

    raws = rec.get_recommendations_from_multiple(games, top_n=count + 5)
    if raws is None or raws.empty:
        return jsonify({'results': [], 'message': 'Could not build profile from those titles.'})

    rated = steam_df.copy()
    total = rated['positive_ratings'] + rated['negative_ratings']
    rated['quality_score'] = np.where(total > 0, rated['positive_ratings'] / total, 0.0)

    detail_cols = ['name', 'genres', 'average_playtime', 'positive_ratings', 'negative_ratings',
                   'developer', 'header_image', 'screenshots', 'movies', 'website', 'steam_link',
                   'appid', 'release_date', 'price', 'metacritic_score', 'quality_score',
                   'about_the_game', 'steamspy_tags', 'categories', 'owners', 'publisher', 'mood']
    detail_cols = [c for c in detail_cols if c in rated.columns]
    merged = pd.merge(raws, rated[detail_cols], on='name', how='left')
    merged = merged[~merged['name'].isin(games)].head(count)

    results = []
    for _, row in merged.iterrows():
        d = game_to_dict(row)
        d['match_score'] = round(float(row.get('final_score', 0)) * 100, 1)
        d['explanation'] = str(row.get('explanation', ''))
        results.append(d)

    return jsonify({'results': results, 'source_games': games})

@app.route('/api/mood/<mood>')
def api_mood(mood):
    mood_map = {
        'action': { 'title': 'Action', 'genres': ['Action', 'Shooter'], 'tags': ['FPS', 'Hack and Slash', 'Third-Person Shooter', 'Action-Adventure'] },
        'relaxing': { 'title': 'Relaxing', 'genres': ['Casual', 'Simulation'], 'tags': ['Relaxing', 'Puzzle', 'Cozy', 'Atmospheric', 'Walking Simulator'] },
        'competitive': { 'title': 'Competitive', 'genres': ['Sports', 'Racing'], 'tags': ['Competitive', 'Multiplayer', 'MOBA', 'PvP', 'eSports'] },
        'story': { 'title': 'Story-Driven', 'genres': ['RPG', 'Adventure'], 'tags': ['Story Rich', 'Narrative', 'Choices Matter', 'Visual Novel', 'JRPG'] },
        'horror': { 'title': 'Horror', 'genres': [], 'tags': ['Horror', 'Survival Horror', 'Psychological Horror', 'Zombies', 'Dark'] },
        'strategy': { 'title': 'Strategy', 'genres': ['Strategy'], 'tags': ['Grand Strategy', 'City Builder', 'Tactical', 'Tower Defense', 'Turn-Based Strategy', 'RTS'] },
        'retro': { 'title': 'Retro', 'genres': [], 'tags': ['Retro', 'Pixel Graphics', 'Arcade', '2D', 'Classic', '8-Bit', '16-bit'] },
    }

    mood_config = mood_map.get(mood.lower())
    if not mood_config:
        return jsonify({'results': []})

    genre_set = {g.lower() for g in mood_config['genres']}
    tag_set = {t.lower() for t in mood_config['tags']}

    def matches_mood(row):
        # Exact token matching — split by comma, strip, lowercase
        row_genres = {g.strip().lower() for g in str(row.get('genres', '')).split(',') if g.strip()}
        row_tags = {t.strip().lower() for t in str(row.get('steamspy_tags', '')).split(',') if t.strip()}

        # Exclude non-game software (Wallpaper Engine, etc.)
        exclude_genres = {'utilities', 'design & illustration', 'animation & modeling',
                          'audio production', 'software training', 'accounting',
                          'photo editing', 'video production', 'web publishing'}
        if row_genres & exclude_genres:
            return False

        return bool(row_genres & genre_set) or bool(row_tags & tag_set)

    df = steam_df.copy()
    mdf = df[df.apply(matches_mood, axis=1)].copy()

    if mdf.empty:
        return jsonify({'results': []})

    total = mdf['positive_ratings'] + mdf['negative_ratings']
    mdf['quality_score'] = np.where(total > 0, mdf['positive_ratings'] / total, 0.0)
    # Balanced score: quality matters more, with a popularity floor
    mdf = mdf[total >= 50]  # Minimum 50 reviews
    mdf['mood_score'] = mdf['quality_score'] * np.log1p(mdf['positive_ratings'])
    top = mdf.sort_values('mood_score', ascending=False).head(20)

    results = []
    for _, row in top.iterrows():
        d = game_to_dict(row)
        d['match_score'] = round(float(row['quality_score']) * 100, 1)
        d['explanation'] = f"Top-rated {mood_config['title'].lower()} title with {int(row['positive_ratings']):,} positive reviews."
        results.append(d)

    return jsonify({'results': results, 'mood': mood_config['title']})

@app.route('/api/playtime/<bracket>')
def api_playtime(bracket):
    df = steam_df.copy()
    df['hrs'] = df['average_playtime'] / 60
    total = df['positive_ratings'] + df['negative_ratings']
    df['quality_score'] = np.where(total > 0, df['positive_ratings'] / total, 0.0)

    bracket_map = {
        'short': (0, 10),
        'medium': (10, 30),
        'long': (30, 100),
        'endless': (100, 999999),
    }
    lo, hi = bracket_map.get(bracket.lower(), (0, 999999))
    target = df[(df['hrs'] >= lo) & (df['hrs'] < hi)]
    top = target.sort_values('positive_ratings', ascending=False).head(15)

    results = []
    for _, row in top.iterrows():
        d = game_to_dict(row)
        d['match_score'] = round(float(row['quality_score']) * 100, 1)
        hrs = int(row['hrs'])
        d['explanation'] = f"Average playtime: {hrs} hours. Quality score: {d['quality_score']}%."
        results.append(d)

    return jsonify({'results': results, 'bracket': bracket})

@app.route('/api/trending')
def api_trending():
    limit = min(int(request.args.get('limit', 15)), 30)
    g = user_df.groupby('game_name').agg(
        total_hours=('playtime', 'sum'),
        n_players=('user_id', 'nunique')
    ).reset_index()
    g['total_hours'] = (g['total_hours'] / 60).round(0).astype(int)
    g['score'] = g['total_hours'] + g['n_players'] * 100
    top = g.sort_values('score', ascending=False).head(limit)

    results = []
    for _, row in top.iterrows():
        game_match = steam_df[steam_df['name'] == row['game_name']]
        header = ''
        if not game_match.empty:
            header = str(game_match.iloc[0].get('header_image', ''))
        results.append({
            'name': row['game_name'],
            'total_hours': int(row['total_hours']),
            'n_players': int(row['n_players']),
            'header_image': header,
        })

    return jsonify({'results': results})

@app.route('/api/popular')
def api_popular():
    limit = min(int(request.args.get('limit', 12)), 60)
    page = max(int(request.args.get('page', 1)), 1)
    offset = (page - 1) * limit
    
    df = steam_df.copy()
    df['min_owners'] = df['owners'].apply(
        lambda x: int(str(x).replace(',', '').split('-')[0].strip()) if isinstance(x, str) and '-' in str(x) else 0)
    
    top = df.sort_values('min_owners', ascending=False).iloc[offset:offset+limit]

    results = []
    for _, row in top.iterrows():
        d = game_to_dict(row)
        d['min_owners'] = int(row['min_owners'])
        results.append(d)

    return jsonify({
        'results': results,
        'page': page,
        'has_more': offset + limit < len(df)
    })

@app.route('/api/explore/<path:name>')
def api_explore(name):
    matches = steam_df[steam_df['name'] == name]
    if matches.empty:
        return jsonify({'error': 'Game not found'}), 404

    game_detail = game_to_dict(matches.iloc[0])

    # Get similar games
    crecs = rec.get_content_recommendations(name, top_n=8)
    similar = []
    if not crecs.empty:
        detail_cols = [c for c in ['name', 'genres', 'developer', 'header_image', 'appid',
                                    'positive_ratings', 'negative_ratings', 'steam_link',
                                    'release_date', 'mood'] if c in steam_df.columns]
        crecs = pd.merge(crecs, steam_df[detail_cols], on='name', how='left')
        for _, row in crecs[crecs['name'] != name].head(6).iterrows():
            d = game_to_dict(row)
            d['similarity'] = round(float(row.get('content_score', 0)) * 100, 1)
            similar.append(d)

    return jsonify({'game': game_detail, 'similar': similar})


if __name__ == '__main__':
    init_recommender()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
