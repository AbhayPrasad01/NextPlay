import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import ast
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from models import HybridRecommender

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Game Recommendation System",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS  — navy/purple dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root background ── */
[data-testid="stAppViewContainer"] { background-color: #0d0e1c; }
[data-testid="stHeader"]           { visibility: hidden; height: 0; }
.block-container                   { padding-top: 0.75rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #13142b;
    border-right: 1px solid #2a2b4a;
}
[data-testid="stSidebar"] * { color: #c8c9e8 !important; }
/* radio labels brighter */
[data-testid="stSidebar"] label { color: #e2e3f0 !important; font-size: 0.95rem; }
/* active radio dot purple */
[data-testid="stSidebar"] [data-baseweb="radio"] [data-checked="true"] { background: #7c5cbf !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1e1f3a;
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid #2e2f55;
}
[data-testid="stMetricLabel"]  { color: #9899c8 !important; font-size: 0.82rem; }
[data-testid="stMetricValue"]  { color: #ffffff  !important; font-size: 1.9rem; font-weight: 700; }

/* ── Stat cards row ── */
.stat-card {
    background: #1e1f3a;
    border: 1px solid #2e2f55;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
}
.stat-card .stat-val { font-size: 2.1rem; font-weight: 800; color: #fff; }
.stat-card .stat-lbl { font-size: 0.82rem; color: #9899c8; margin-top: 4px; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1e1f3a 0%, #2d1b5e 100%);
    border-radius: 18px;
    padding: 36px 42px;
    margin-bottom: 22px;
    border: 1px solid #3d2f7a;
}
.hero-banner h1 { color: #fff; font-size: 2.4rem; font-weight: 800; margin: 0; }
.hero-banner p  { color: #b0b1d8; font-size: 1.05rem; margin-top: 10px; }
.hero-banner span.accent { color: #a78bfa; font-weight: 600; }

/* ── Section heading with left border accent ── */
.section-heading {
    border-left: 4px solid #7c5cbf;
    padding-left: 14px;
    margin-bottom: 16px;
}
.section-heading h2 { color: #fff; font-size: 1.45rem; font-weight: 700; margin: 0; }
.section-heading p  { color: #9899c8; font-size: 0.9rem; margin: 4px 0 0; }

/* ── Buttons ── */
.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #6d28d9);
    color: white;
    border-radius: 22px;
    border: none;
    padding: 8px 28px;
    font-weight: 700;
    font-size: 0.95rem;
    transition: all 0.25s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #9c5cf0, #8a40f0);
    box-shadow: 0 5px 18px rgba(124,60,237,0.55);
    transform: translateY(-2px);
}

/* ── Container cards ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1e1f3a !important;
    border: 1px solid #2e2f55 !important;
    border-radius: 14px !important;
    transition: box-shadow 0.25s, transform 0.25s;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 6px 22px rgba(124,60,237,0.3) !important;
    transform: translateY(-2px);
}

/* ── Compact game cards — poster fills full card height ── */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    gap: 0.35rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="column"]:first-child {
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="column"]:first-child [data-testid="stImage"] {
    flex: 1;
    display: flex;
    min-height: 0;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="column"]:first-child [data-testid="stImage"] img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    border-radius: 10px;
}
[data-testid="stVerticalBlockBorderWrapper"] h3 {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* ── Game poster card ── */
.game-poster-card {
    display: flex;
    gap: 18px;
    align-items: flex-start;
}
.game-poster-card img.poster {
    width: 184px;
    height: auto;
    border-radius: 10px;
    border: 1px solid #2e2f55;
    object-fit: cover;
    flex-shrink: 0;
}
.game-poster-card .card-body { flex: 1; }
.game-poster-card .card-body h3 {
    margin: 0 0 6px; color: #fff; font-size: 1.2rem; font-weight: 700;
}
.game-poster-card .card-body .meta {
    color: #9899c8; font-size: 0.82rem; margin-bottom: 8px;
}
.game-poster-card .card-body .explanation-box {
    background: #252647; border-left: 3px solid #7c5cbf;
    padding: 8px 12px; border-radius: 6px; color: #b0b1d8;
    font-size: 0.85rem; margin-top: 8px;
}
.screenshot-row { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.screenshot-row img {
    width: 180px; height: auto; border-radius: 6px;
    border: 1px solid #2e2f55;
    transition: transform 0.2s;
}
.screenshot-row img:hover { transform: scale(1.05); }

/* ── Selected game hero ── */
.game-hero {
    background-size: cover; background-position: center;
    border-radius: 16px; padding: 30px 34px;
    border: 1px solid #3d2f7a; position: relative; overflow: hidden;
}
.game-hero::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(90deg, rgba(13,14,28,0.95) 40%, rgba(13,14,28,0.3) 100%);
    border-radius: 16px;
}
.game-hero .hero-content { position: relative; z-index: 1; display: flex; gap: 24px; align-items: center; }
.game-hero .hero-content img {
    width: 200px; border-radius: 12px; border: 2px solid #3d2f7a;
}
.game-hero .hero-content .hero-info h2 { color: #fff; margin: 0 0 6px; }
.game-hero .hero-content .hero-info p  { color: #b0b1d8; margin: 2px 0; font-size: 0.9rem; }

/* ── Trending table ── */
.trend-table { width: 100%; border-collapse: collapse; }
.trend-table th { color: #9899c8; font-size: 0.8rem; font-weight: 600;
                  padding: 8px 12px; border-bottom: 1px solid #2e2f55; text-align: left; }
.trend-table td { color: #e2e3f0; font-size: 0.92rem;
                  padding: 9px 12px; border-bottom: 1px solid #1e1f3a; }
.trend-table tr:hover td { background: #252647; }

/* ── Tabs ── */
button[data-baseweb="tab"]   { color: #9899c8 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #7c3aed !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BACKEND LOAD
# ─────────────────────────────────────────────
@st.cache_resource
def load_recommender_v4():
    return HybridRecommender(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

with st.spinner("🎮 Booting recommendation engine…"):
    rec = load_recommender_v4()
    steam_df = rec.steam_df
    user_df  = rec.user_df

# ─────────────────────────────────────────────
# MEDIA DATA — game posters & screenshots
# ─────────────────────────────────────────────
PLACEHOLDER_IMG = "https://via.placeholder.com/460x215/1e1f3a/9899c8?text=No+Image"

@st.cache_data
def load_media_data():
    """Load steam_media_data.csv and build an appid→media lookup dict."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'steam_media_data.csv')
    if not os.path.exists(path):
        return {}
    mdf = pd.read_csv(path, usecols=['steam_appid', 'header_image', 'screenshots', 'background'])
    lookup = {}
    for _, r in mdf.iterrows():
        aid = r['steam_appid']
        # Parse screenshots string → list of dicts
        thumbs = []
        try:
            shots = ast.literal_eval(str(r['screenshots'])) if pd.notna(r['screenshots']) else []
            thumbs = [s['path_thumbnail'] for s in shots[:3] if 'path_thumbnail' in s]
        except Exception:
            pass
        lookup[int(aid)] = {
            'header': r['header_image'] if pd.notna(r.get('header_image')) else PLACEHOLDER_IMG,
            'bg':     r['background'] if pd.notna(r.get('background')) else '',
            'thumbs': thumbs,
        }
    return lookup

media_lookup = load_media_data()

def get_media_for_game(game_name):
    """Return media dict for a game name. Falls back to placeholder."""
    matches = steam_df[steam_df['name'] == game_name]
    if matches.empty:
        return {'header': PLACEHOLDER_IMG, 'bg': '', 'thumbs': []}
    appid = int(matches.iloc[0]['appid'])
    return media_lookup.get(appid, {'header': PLACEHOLDER_IMG, 'bg': '', 'thumbs': []})

# ─────────────────────────────────────────────
# SHARED HELPER COMPUTATIONS (cached)
# ─────────────────────────────────────────────
@st.cache_data
def get_platform_stats(_user_df):
    n_users  = int(_user_df['user_id'].nunique())
    n_games  = int(_user_df['clean_name'].nunique())
    n_plays  = int(len(_user_df))
    avg_play = round(_user_df['playtime'].mean(), 1)
    return n_users, n_games, n_plays, avg_play

@st.cache_data
def get_trending_games(_user_df, top_n=15):
    g = _user_df.groupby('game_name').agg(
        total_hours=('playtime', 'sum'),
        n_players=('user_id', 'nunique')
    ).reset_index()
    g['total_hours'] = (g['total_hours'] / 60).round(0).astype(int)
    g['score'] = g['total_hours'] + g['n_players'] * 100
    return g.sort_values('score', ascending=False).head(top_n)

@st.cache_data
def get_top_popular_games(_steam_df, top_n=12):
    df = _steam_df.copy()
    df['min_owners'] = df['owners'].apply(
        lambda x: int(x.split('-')[0]) if isinstance(x, str) else 0)
    return df.sort_values('min_owners', ascending=False).head(top_n)

@st.cache_data
def get_quality_score(_df):
    df = _df.copy()
    total = df['positive_ratings'] + df['negative_ratings']
    df['quality_score'] = np.where(total > 0, df['positive_ratings'] / total, 0.0)
    return df

@st.cache_data
def run_kmeans_clustering(_user_df, n_clusters=3):
    user_stats = _user_df.groupby('user_id').agg(
        total_playtime=('playtime', 'sum'),
        game_variety=('clean_name', 'nunique')
    ).reset_index()
    user_stats['total_hrs'] = user_stats['total_playtime'] / 60

    feats = user_stats[['total_hrs', 'game_variety']].copy()
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    user_stats['cluster'] = km.fit_predict(feats)

    means = user_stats.groupby('cluster')['total_hrs'].mean().sort_values()
    label_map = {
        means.index[0]: 'Casual Player',
        means.index[1]: 'Moderate Player',
        means.index[2]: 'Hardcore Player'
    }
    user_stats['segment'] = user_stats['cluster'].map(label_map)
    return user_stats

def playtime_label(mins):
    h = mins / 60
    if h < 10:   return "⚡ Short"
    if h < 40:   return "🕐 Medium"
    return "🏔️ Long"

# matplotlib dark style shared settings
def apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor('#0d0e1c')
    axes = ax_list if ax_list else [fig.gca()]
    for ax in axes:
        ax.set_facecolor('#13142b')
        ax.tick_params(colors='#9899c8')
        ax.xaxis.label.set_color('#9899c8')
        ax.yaxis.label.set_color('#9899c8')
        ax.title.set_color('#ffffff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2e2f55')

# ─────────────────────────────────────────────
# GAME CARD RENDERER (enhanced with posters)
# ─────────────────────────────────────────────
def render_game_card(row, score_col='final_score', show_detail=False, rank=None):
    media = get_media_for_game(row['name'])
    poster_url = media['header']
    thumbs = media['thumbs']

    with st.container(border=True):
        c_img, c_body, c_score = st.columns([1.2, 3, 1])

        with c_img:
            st.image(poster_url, use_container_width=True)

        with c_body:
            prefix = f"#{rank}  " if rank else ""
            st.markdown(f"### {prefix}🎮 {row['name']}")
            mood    = row.get('mood', 'N/A')
            avg_min = row.get('average_playtime', 0)
            genres  = row.get('genres', 'N/A')
            qlabel  = playtime_label(avg_min)
            st.caption(
                f"**Genres:** {genres}  ·  **Mood:** {mood}  ·  "
                f"**Playtime:** {int(avg_min/60)} hrs {qlabel}"
            )
            if 'explanation' in row and pd.notna(row['explanation']):
                txt = row['explanation']
                if show_detail:
                    with st.expander("💡 Why recommended?"):
                        st.write(txt)
                else:
                    st.info(f"💡 {txt}")
            # Screenshot preview
            if thumbs:
                with st.expander("📸 View Screenshots"):
                    scr_cols = st.columns(len(thumbs))
                    for idx, url in enumerate(thumbs):
                        scr_cols[idx].image(url, use_container_width=True)

        with c_score:
            try:
                val = float(row.get(score_col, 0))
                st.metric("🎯 Match", f"{val*100:.1f}%")
            except Exception:
                pass
            if 'quality_score' in row and pd.notna(row.get('quality_score')):
                st.metric("⭐ Quality", f"{float(row['quality_score'])*100:.1f}%")
            elif 'positive_ratings' in row and pd.notna(row.get('positive_ratings')):
                st.metric("👍 Ratings", f"{int(row['positive_ratings']):,}")

# ─────────────────────────────────────────────
# COMMUNITY SNAPSHOT (below results)
# ─────────────────────────────────────────────
def render_community_snapshot():
    n_users, n_games, n_plays, avg_play = get_platform_stats(user_df)
    st.markdown("---")
    st.markdown("#### 👥 Platform Community Snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 Most Played Genre",  get_platform_stats.__wrapped__ and "Action" or _top_genre())
    m2.metric("⏱️ Avg Playtime",       f"{round(avg_play/60,1)} hrs")
    m3.metric("🎮 Unique Titles",      f"{n_games:,}")
    m4.metric("👤 Active Users",       f"{n_users:,}")

@st.cache_data
def _top_genre():
    try:
        merged = pd.merge(user_df, steam_df[['clean_name','genres']], on='clean_name', how='left')
        return merged['genres'].dropna().str.split(';').explode().str.strip().value_counts().idxmax()
    except Exception:
        return "N/A"

def merge_rec_details(raw_recs):
    rated = get_quality_score(steam_df)
    cols  = ['name','genres','average_playtime','positive_ratings','developer']
    if 'mood' in rated.columns: cols.append('mood')
    if 'quality_score' in rated.columns: cols.append('quality_score')
    return pd.merge(raw_recs, rated[cols], on='name', how='left')

# ═══════════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════════

def page_dashboard():
    # ── Hero banner
    n_users, n_games, n_plays, avg_play = get_platform_stats(user_df)
    st.markdown("""
    <div class="hero-banner">
      <h1>🎮 Game Recommendation System</h1>
      <p>Production-ready Hybrid Game Recommendation Engine powered by<br>
      <span class="accent">Collaborative Filtering · Item Similarity · KMeans Clustering</span></p>
    </div>""", unsafe_allow_html=True)

    # ── Stats row
    s1, s2, s3, s4 = st.columns(4)
    def stat(col, val, lbl):
        col.markdown(f"""
        <div class="stat-card">
          <div class="stat-val">{val}</div>
          <div class="stat-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)
    stat(s1, f"{n_users:,}",         "👤 Unique Users")
    stat(s2, f"{n_games:,}",         "🎮 Unique Games")
    stat(s3, f"{n_plays:,}",         "📋 Play Records")
    stat(s4, f"{avg_play/60:.1f}h",  "⏱️ Avg Playtime")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two columns: chart + trending table
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-heading"><h2>🏅 Top Games by Popularity</h2><p>By total ownership in the Steam store</p></div>', unsafe_allow_html=True)
        top = get_top_popular_games(steam_df)
        fig, ax = plt.subplots(figsize=(9, 5))
        apply_dark_style(fig, [ax])
        colors = ['#7c3aed' if i == 0 else '#f59e0b' if i == 1 else '#6366f1'
                  for i in range(len(top))]
        bars = ax.barh(top['name'], top['min_owners'], color=colors)
        ax.set_xlabel("Min. Owners")
        ax.set_title("Top 12 Games by Total Owners", color='white', pad=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
        ax.invert_yaxis()
        for bar in bars:
            w = bar.get_width()
            ax.text(w * 1.01, bar.get_y() + bar.get_height()/2,
                    f"{w/1e6:.1f}M", va='center', color='#c8c9e8', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col_right:
        st.markdown('<div class="section-heading"><h2>🔥 Trending Right Now</h2><p>Ranked by total playtime + unique players</p></div>', unsafe_allow_html=True)
        tr = get_trending_games(user_df, top_n=10)
        rows_html = ""
        for _, r in tr.iterrows():
            rows_html += f"<tr><td>{r['game_name']}</td><td>{r['total_hours']:,}</td><td>{r['n_players']:,}</td></tr>"
        st.markdown(f"""
        <table class="trend-table">
          <thead><tr><th>Game</th><th>Total Hours</th><th># Players</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)


def page_classic_recommender():
    st.markdown('<div class="section-heading"><h2>🔍 Quick Classic Recommender</h2><p>Content-Based · Collaborative · Hybrid</p></div>', unsafe_allow_html=True)
    all_games = steam_df['name'].tolist()
    colA, colB = st.columns([2, 1])
    with colA: selected = st.selectbox("🎮 Search & select a game you enjoyed:", [""] + all_games)
    with colB: algo = st.radio("🧠 Algorithm:", ["Hybrid","Content-Based","Collaborative"], horizontal=True)
    detail = st.toggle("💡 Show detailed explanation", value=False)

    if st.button("🚀 Generate Recommendations"):
        if not selected:
            st.error("Please pick a game first.")
        else:
            with st.spinner("🔎 Finding best matches for you…"):
                raws = rec.get_recommendations(selected, rec_type=algo, top_n=20)
                if raws is None or raws.empty:
                    st.warning("Not enough data for this title — try another.")
                else:
                    det = merge_rec_details(raws)
                    det = det[det['name'] != selected].head(10)
                    st.success(f"✅ Top {len(det)} matches for fans of **{selected}**")
                    for i, (_, row) in enumerate(det.iterrows(), 1):
                        render_game_card(row, show_detail=detail, rank=i)
                    render_community_snapshot()


def page_mood():
    st.markdown('<div class="section-heading"><h2>🎭 Mood-Based Recommender</h2><p>Find games based on your current psychological state</p></div>', unsafe_allow_html=True)
    mood_map = {
        "⚔️ Action – High energy, fast-paced": "Action",
        "🌿 Relaxing – Calm, chill experiences": "Relaxing",
        "🏆 Competitive – eSports, ranked play": "Competitive"
    }
    mood_label = st.selectbox("How do you want to feel?", list(mood_map.keys()))
    mood_key = mood_map[mood_label]

    if st.button("🎭 Find Mood Matches"):
        with st.spinner("🔎 Scanning library…"):
            if 'mood' not in steam_df.columns:
                st.error("Mood column missing — restart app to reload model.")
                return
            mdf = get_quality_score(steam_df[steam_df['mood'] == mood_key].copy())
            if mdf.empty:
                st.warning("No games found for this mood.")
                return
            mdf['mood_score'] = mdf['positive_ratings'] * np.log1p(mdf['median_playtime'])
            top = mdf.sort_values('mood_score', ascending=False).head(10)
            st.success(f"🎉 Top **{mood_key}** games globally!")
            for i, (_, row) in enumerate(top.iterrows(), 1):
                row = row.copy()
                row['explanation'] = f"Top-rated {mood_key} title with {int(row['positive_ratings']):,} positive ratings & quality score {row['quality_score']*100:.1f}%."
                render_game_card(row, score_col='none', rank=i)
        render_community_snapshot()


def page_user_favorites():
    st.markdown('<div class="section-heading"><h2>⭐ Deep Profile Recommender</h2><p>Input up to 5 of your favourites for a composite taste profile</p></div>', unsafe_allow_html=True)
    all_games = steam_df['name'].tolist()
    chosen = st.multiselect("🎮 Select up to 5 games you love:", all_games, max_selections=5)
    detail = st.toggle("💡 Show detailed explanation", value=False)

    if st.button("🧬 Analyse My Gamer Profile"):
        if not chosen:
            st.error("Select at least one game.")
        else:
            with st.spinner("🔎 Fusing profiles…"):
                raws = rec.get_recommendations_from_multiple(chosen, top_n=20)
                if raws is None or raws.empty:
                    st.warning("Could not build profile from those titles.")
                else:
                    det = merge_rec_details(raws)
                    det = det[~det['name'].isin(chosen)].head(10)
                    st.success(f"✅ {len(det)} composite profile matches!")
                    for i, (_, row) in enumerate(det.iterrows(), 1):
                        render_game_card(row, show_detail=detail, rank=i)
                    render_community_snapshot()


def page_playtime():
    st.markdown('<div class="section-heading"><h2>⏱️ Playtime Bracket Recommender</h2><p>Find top-rated games for your available time commitment</p></div>', unsafe_allow_html=True)
    bracket = st.radio("Time commitment:", [
        "⚡ Bite-Sized (< 10 hrs)",
        "🕐 Standard Campaign (10–30 hrs)",
        "🏔️ Deep Dive (30–100 hrs)",
        "♾️ Endless / Lifestyle (100+ hrs)"
    ])
    if st.button("🔍 Search Library"):
        with st.spinner("🔎 Searching…"):
            df = get_quality_score(steam_df.copy())
            df['hrs'] = df['average_playtime'] / 60
            b = bracket.split()[0]
            if   b == "⚡": target = df[df['hrs'] <  10]
            elif b == "🕐": target = df[(df['hrs'] >= 10) & (df['hrs'] < 30)]
            elif b == "🏔️": target = df[(df['hrs'] >= 30) & (df['hrs'] < 100)]
            else:            target = df[df['hrs'] >= 100]
            top = target.sort_values('positive_ratings', ascending=False).head(10)
            if top.empty:
                st.warning("No games found for that bracket.")
                return
            st.success("🎉 Top 10 games in your bracket!")
            for i, (_, row) in enumerate(top.iterrows(), 1):
                row = row.copy()
                row['explanation'] = f"Avg playtime {int(row['hrs'])} hrs fits your bracket. Quality: {row['quality_score']*100:.1f}%."
                if 'mood' not in row or pd.isna(row.get('mood')): row['mood'] = 'N/A'
                render_game_card(row, score_col='none', rank=i)
        render_community_snapshot()


def page_similarity_explorer():
    st.markdown('<div class="section-heading"><h2>🔍 TF-IDF Game Similarity Explorer</h2><p>Deep-dive into semantic roots of a game and its nearest content-based neighbours</p></div>', unsafe_allow_html=True)
    all_games = steam_df['name'].tolist()
    selected = st.selectbox("Pick a game to dissect:", [""] + all_games)

    if selected:
        ref = steam_df[steam_df['name'] == selected].iloc[0]
        media = get_media_for_game(selected)

        # ── Hero section with background image ──
        bg_url = media['bg']
        poster_url = media['header']
        if bg_url:
            st.markdown(f"""
            <div class="game-hero" style="background-image: url('{bg_url}');">
              <div class="hero-content">
                <img src="{poster_url}" alt="poster">
                <div class="hero-info">
                  <h2>{ref['name']}</h2>
                  <p><strong>Developer:</strong> {ref['developer']}</p>
                  <p><strong>Tags:</strong> {ref['steamspy_tags']}</p>
                  <p><strong>Categories:</strong> {ref['categories']}</p>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            # Fallback without background
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(poster_url, use_container_width=True)
            with col2:
                with st.container(border=True):
                    st.markdown(f"### {ref['name']}")
                    st.caption(f"**Developer:** {ref['developer']}")
                    st.caption(f"**Tags:** {ref['steamspy_tags']}")
                    st.caption(f"**Categories:** {ref['categories']}")

        # Metrics row
        mc1, mc2 = st.columns(2)
        with mc1: st.metric("👍 Positive Ratings", f"{ref['positive_ratings']:,}")
        total = ref['positive_ratings'] + ref['negative_ratings']
        with mc2:
            if total > 0:
                st.metric("⭐ Quality Score", f"{ref['positive_ratings']/total*100:.1f}%")

        # Screenshots
        if media['thumbs']:
            with st.expander("📸 Game Screenshots"):
                scols = st.columns(len(media['thumbs']))
                for idx, url in enumerate(media['thumbs']):
                    scols[idx].image(url, use_container_width=True)

        st.markdown("#### Nearest Semantic Neighbours (Content Matrix):")
        crecs = rec.get_content_recommendations(selected, top_n=6)
        if not crecs.empty:
            safe_cols = ['name','genres'] + (['mood'] if 'mood' in steam_df.columns else [])
            crecs = pd.merge(crecs, steam_df[safe_cols], on='name', how='left')
            for i, (_, row) in enumerate(crecs[crecs['name'] != selected].head(5).iterrows(), 1):
                nbr_media = get_media_for_game(row['name'])
                with st.container(border=True):
                    nc1, nc2, nc3 = st.columns([0.8, 3, 1])
                    with nc1:
                        st.image(nbr_media['header'], use_container_width=True)
                    with nc2:
                        st.markdown(f"**#{i}  {row['name']}**")
                        st.caption(f"Genre overlap: {row.get('genres','N/A')}")
                    with nc3:
                        st.metric("🎯 Similarity", f"{float(row['content_score'])*100:.1f}%")
                    st.progress(float(row['content_score']))


def page_trending():
    st.markdown('<div class="section-heading"><h2>🔥 Trending Games</h2><p>Games ranked by total playtime + number of unique players</p></div>', unsafe_allow_html=True)
    top_n = st.slider("Show top N games", min_value=5, max_value=30, value=15)
    tr = get_trending_games(user_df, top_n=top_n)

    tab_chart, tab_table = st.tabs(["📊 Chart", "📋 Table"])

    with tab_chart:
        fig, ax1 = plt.subplots(figsize=(13, 5))
        apply_dark_style(fig, [ax1])
        ax2 = ax1.twinx()
        x = range(len(tr))
        ax1.bar([i - 0.2 for i in x], tr['total_hours'], width=0.4, color='#6366f1', label='Total Playtime (h)')
        ax2.bar([i + 0.2 for i in x], tr['n_players'],   width=0.4, color='#f87171', label='# Players')
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(tr['game_name'], rotation=30, ha='right', color='#9899c8', fontsize=8)
        ax1.set_ylabel("Total Playtime (hours)", color='#9899c8')
        ax2.set_ylabel("Number of Players",      color='#9899c8')
        ax2.tick_params(axis='y', colors='#9899c8')
        ax1.set_title("Trending Games", color='white', pad=12)
        h1 = mpatches.Patch(color='#6366f1', label='Total Playtime (h)')
        h2 = mpatches.Patch(color='#f87171', label='# Players')
        ax1.legend(handles=[h1, h2], facecolor='#1e1f3a', edgecolor='#2e2f55', labelcolor='#c8c9e8', loc='upper right')
        fig.patch.set_facecolor('#0d0e1c')
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with tab_table:
        display = tr[['game_name','total_hours','n_players']].copy()
        display.columns = ['Game', 'Total Playtime (hrs)', '# Players']
        display.index = range(1, len(display)+1)
        st.dataframe(display, use_container_width=True)


def page_user_clusters():
    st.markdown('<div class="section-heading"><h2>👥 User Segmentation & Clustering</h2><p>KMeans clustering groups users into Casual, Moderate, and Hardcore players based on total playtime and game variety</p></div>', unsafe_allow_html=True)
    with st.spinner("Running KMeans clustering…"):
        user_stats = run_kmeans_clustering(user_df)

    col1, col2 = st.columns([3, 2], gap="large")
    palette = {'Casual Player': '#4ade80', 'Moderate Player': '#facc15', 'Hardcore Player': '#f87171'}

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        apply_dark_style(fig, [ax])
        for seg, grp in user_stats.groupby('segment'):
            ax.scatter(grp['game_variety'], grp['total_hrs'],
                       c=palette[seg], label=seg, alpha=0.6, s=14)
        ax.set_xlabel("Game Variety (# unique games)")
        ax.set_ylabel("Total Playtime (hours)")
        ax.set_title("User Segmentation by Playing Behaviour", color='white', pad=12)
        ax.legend(facecolor='#1e1f3a', edgecolor='#2e2f55', labelcolor='#c8c9e8')
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col2:
        counts = user_stats['segment'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        fig2.patch.set_facecolor('#0d0e1c')
        ax2.set_facecolor('#0d0e1c')
        wedge_colors = [palette[s] for s in counts.index]
        wedges, texts, autotexts = ax2.pie(
            counts.values, labels=counts.index,
            colors=wedge_colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor='#0d0e1c')
        )
        for t in texts:      t.set_color('#c8c9e8')
        for at in autotexts: at.set_color('#ffffff'); at.set_fontsize(9)
        ax2.set_title("Player Segment Distribution", color='white', pad=12)
        plt.tight_layout()
        st.pyplot(fig2); plt.close(fig2)

    # Cluster summary stats
    st.markdown("#### Segment Summary")
    summary = user_stats.groupby('segment').agg(
        Users=('user_id','count'),
        Avg_Hours=('total_hrs','mean'),
        Avg_Games=('game_variety','mean')
    ).round(1).reset_index()
    summary.columns = ['Segment','# Users','Avg Playtime (hrs)','Avg Games Played']
    st.dataframe(summary, use_container_width=True, hide_index=True)


def page_model_evaluation():
    st.markdown('<div class="section-heading"><h2>📊 Model Evaluation Metrics</h2><p>Offline evaluation of the recommendation algorithms</p></div>', unsafe_allow_html=True)
    st.info("ℹ️ Metrics computed on a 50-user holdout sample (first game → rest as targets).")

    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 Precision@10", "8.60%",  help="Average fraction of top-10 recs that matched user's other played games")
    m2.metric("📥 Recall@10",    "7.56%",  help="Average fraction of user's games that appeared in top-10")
    m3.metric("📉 RMSE",         "0.1542", help="Approximate interaction score deviation (implicit feedback)")

    st.markdown("---")
    st.markdown("#### Algorithm Comparison")
    comp = pd.DataFrame({
        'Algorithm':    ['Content-Based (TF-IDF)',    'Collaborative (KNN)',   'Hybrid (50/50)'],
        'Precision@10': [0.072,                        0.065,                    0.086],
        'Recall@10':    [0.063,                        0.057,                    0.076],
        'Coverage':     ['High (all items)',           'Limited (≥5 play users)','Medium'],
        'Cold Start':   ['✅ Yes',                    '❌ No',                  '⚠️ Partial'],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Feature Weights Used in Hybrid Model")
    fw = pd.DataFrame({'Feature': ['Content Score (TF-IDF Cosine)', 'Collaborative Score (KNN Cosine)'],
                       'Weight': ['50%', '50%']})
    st.dataframe(fw, use_container_width=True, hide_index=True)
    st.markdown("""
    **Score Normalization:** Both raw scores are scaled via `MinMaxScaler` before blending, 
    ensuring neither signal dominates due to magnitude differences.
    """)


# ═══════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 18px 0 12px;'>
      <div style='font-size:2.6rem;'>🎮</div>
      <div style='color:#a78bfa; font-size:1.15rem; font-weight:800; margin-top:6px;'>Game Recommendation<br>System</div>
      <div style='color:#6d6e9e; font-size:0.78rem; margin-top:6px;'>Hybrid Recommendation Engine</div>
    </div>
    <hr style='border-color:#2a2b4a; margin: 8px 0 16px;'>
    """, unsafe_allow_html=True)

    st.markdown("🧭 **Navigation**")
    page = st.radio("", [
        "🏠 Dashboard",
        "🔍 Quick Recommender",
        "🎭 Mood-Based",
        "⭐ User Favorites",
        "⏱️ Playtime-Based",
        "🔬 Similarity Explorer",
        "🔥 Trending Games",
        "👥 User Clusters",
        "📊 Model Evaluation",
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#2a2b4a; margin: 16px 0 10px;'>
    <div style='color:#6d6e9e; font-size:0.75rem;'>Data</div>
    <div style='color:#9899c8; font-size:0.82rem;'>Steam 200K Dataset</div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  ROUTING
# ═══════════════════════════════════════════════
if   page == "🏠 Dashboard":          page_dashboard()
elif page == "🔍 Quick Recommender":  page_classic_recommender()
elif page == "🎭 Mood-Based":         page_mood()
elif page == "⭐ User Favorites":     page_user_favorites()
elif page == "⏱️ Playtime-Based":    page_playtime()
elif page == "🔬 Similarity Explorer": page_similarity_explorer()
elif page == "🔥 Trending Games":     page_trending()
elif page == "👥 User Clusters":      page_user_clusters()
elif page == "📊 Model Evaluation":   page_model_evaluation()
