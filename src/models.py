import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import os
import re

def clean_text_for_search(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

class HybridRecommender:
    def __init__(self, data_dir):
        self.steam_df = pd.read_csv(os.path.join(data_dir, 'cleaned_steam.csv'))
        self.user_df = pd.read_csv(os.path.join(data_dir, 'cleaned_user_interactions.csv'))
        
        # Add search key
        self.steam_df['search_key'] = self.steam_df['name'].apply(clean_text_for_search)
        
        # Mood Mapping
        self._map_moods()
        
        self._build_content_model()
        self._build_collaborative_model()
        
    def _map_moods(self):
        """Map games to predefined moods based on genres and tags"""
        def determine_mood(row):
            text = str(row['genres']).lower() + " " + str(row['categories']).lower() + " " + str(row['steamspy_tags']).lower()
            
            # Simple keyword matching for moods
            if any(word in text for word in ['competitive', 'multi-player', 'esports', 'sports', 'racing']):
                return 'Competitive'
            elif any(word in text for word in ['casual', 'puzzle', 'relaxing', 'simulation', 'story rich', 'atmospheric']):
                return 'Relaxing'
            elif any(word in text for word in ['action', 'shooter', 'fps', 'fighting', 'hack and slash']):
                return 'Action'
            return 'Other'
            
        self.steam_df['mood'] = self.steam_df.apply(determine_mood, axis=1)
        
    def _build_content_model(self):
        print("Building Content-Based Model...")
        self.steam_df['combined_features'] = self.steam_df['combined_features'].fillna('')
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.steam_df['combined_features'])
        
        self.content_nn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.content_nn.fit(self.tfidf_matrix)
        print("Content Model Built.")

    def _build_collaborative_model(self):
        print("Building Collaborative Model (Item-Item via User Interactions)...")
        # Log scaling the playtime
        self.user_df['rating'] = np.log1p(self.user_df['playtime'])
        
        self.games_in_cf = self.user_df['clean_name'].unique()
        self.game_to_idx = {name: idx for idx, name in enumerate(self.games_in_cf)}
        self.idx_to_game = {idx: name for name, idx in self.game_to_idx.items()}
        
        users = self.user_df['user_id'].unique()
        self.user_to_idx = {u: idx for idx, u in enumerate(users)}
        
        row = self.user_df['user_id'].map(self.user_to_idx).values
        col = self.user_df['clean_name'].map(self.game_to_idx).values
        data = self.user_df['rating'].values
        
        # User-Item Matrix
        self.user_item_matrix = csr_matrix((data, (row, col)), shape=(len(users), len(self.games_in_cf)))
        
        # Item-User Matrix (transpose)
        self.item_user_matrix = self.user_item_matrix.T
        
        # Fit KNN on Item-User Matrix
        self.collab_nn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.collab_nn.fit(self.item_user_matrix)
        print("Collaborative Model Built.")

    def _get_game_index(self, input_game):
        search_key = clean_text_for_search(input_game)
        matches = self.steam_df[self.steam_df['search_key'] == search_key]
        if len(matches) == 0:
            matches = self.steam_df[self.steam_df['search_key'].str.contains(search_key)]
            if len(matches) == 0:
                return None
        return matches.index[0]

    def get_content_recommendations(self, game_name, top_n=50):
        idx = self._get_game_index(game_name)
        if idx is None:
            return pd.DataFrame()
            
        distances, indices = self.content_nn.kneighbors(self.tfidf_matrix[idx], n_neighbors=top_n+1)
        
        recs = []
        for i in range(1, len(distances[0])):
            match_idx = indices[0][i]
            sim_score = 1 - distances[0][i]  # Cosine similarity
            recs.append({
                'name': self.steam_df.iloc[match_idx]['name'],
                'clean_name': self.steam_df.iloc[match_idx]['clean_name'],
                'content_score': sim_score
            })
            
        return pd.DataFrame(recs)

    def get_collaborative_recommendations(self, game_name, top_n=50):
        idx = self._get_game_index(game_name)
        if idx is None:
            return pd.DataFrame()
            
        clean_name = self.steam_df.iloc[idx]['clean_name']
        
        if clean_name not in self.game_to_idx:
            return pd.DataFrame()
            
        cf_idx = self.game_to_idx[clean_name]
        distances, indices = self.collab_nn.kneighbors(self.item_user_matrix[cf_idx], n_neighbors=top_n+1)
        
        recs = []
        for i in range(1, len(distances[0])):
            match_idx = indices[0][i]
            game_clean = self.idx_to_game[match_idx]
            sim_score = 1 - distances[0][i]
            
            real_name_matches = self.steam_df[self.steam_df['clean_name'] == game_clean]['name'].values
            real_name = real_name_matches[0] if len(real_name_matches) > 0 else game_clean
            
            recs.append({
                'name': real_name,
                'clean_name': game_clean,
                'collab_score': sim_score
            })
            
        return pd.DataFrame(recs)

    def get_recommendations(self, game_name, rec_type="Hybrid", top_n=50):
        """Unified method to get recommendations based on selected type"""
        
        # Retrieve broad subset to allow filtering downstream
        c_recs = self.get_content_recommendations(game_name, top_n=100)
        cf_recs = self.get_collaborative_recommendations(game_name, top_n=100)
        
        if c_recs.empty and cf_recs.empty:
            return pd.DataFrame()
            
        # Normalize scores before returning or combining
        scaler = MinMaxScaler()
        if not c_recs.empty:
            c_recs['content_score'] = scaler.fit_transform(c_recs[['content_score']])
        if not cf_recs.empty:
            cf_recs['collab_score'] = scaler.fit_transform(cf_recs[['collab_score']])
            
        if rec_type == "Content-Based":
            if c_recs.empty: return pd.DataFrame()
            # Generate explanation
            c_recs['explanation'] = "Recommended because it shares highly similar genres, themes, and tags with your selection."
            c_recs['final_score'] = c_recs['content_score']
            return c_recs.sort_values('final_score', ascending=False).head(top_n)
            
        elif rec_type == "Collaborative":
            if cf_recs.empty: return pd.DataFrame()
            # Generate explanation
            cf_recs['explanation'] = "Recommended because players who heavily enjoyed your selection also played this."
            cf_recs['final_score'] = cf_recs['collab_score']
            return cf_recs.sort_values('final_score', ascending=False).head(top_n)
            
        else: # Hybrid
            if cf_recs.empty:
                c_recs['final_score'] = c_recs['content_score']
                c_recs['explanation'] = "Recommended mostly due to shared themes and genres (lack of player overlap data)."
                return c_recs.sort_values('final_score', ascending=False).head(top_n)
            if c_recs.empty:
                cf_recs['final_score'] = cf_recs['collab_score']
                cf_recs['explanation'] = "Recommended due to player overlap (lack of text data matching)."
                return cf_recs.sort_values('final_score', ascending=False).head(top_n)
                
            merged = pd.merge(c_recs, cf_recs, on=['name', 'clean_name'], how='outer').fillna(0)
            merged['final_score'] = (merged['content_score'] * 0.5) + (merged['collab_score'] * 0.5)
            
            # Explanation based on dominant score
            def get_explanation(row):
                if row['content_score'] > 0.7 and row['collab_score'] > 0.7:
                    return "Perfect match! Shares strong thematic similarities and is highly popular among similar players."
                elif row['content_score'] > row['collab_score']:
                    return "Recommended primarily because it shares similar genres, themes, and tags."
                else:
                    return "Recommended primarily because players who enjoyed your selection also heavily played this."
                    
            merged['explanation'] = merged.apply(get_explanation, axis=1)
            
            return merged.sort_values('final_score', ascending=False).head(top_n)

    def get_recommendations_from_multiple(self, game_names, top_n=50):
        """Aggregate recommendations for multiple input games to create a profile match."""
        all_c_recs = []
        all_cf_recs = []
        
        valid_games = []
        for name in game_names:
            c = self.get_content_recommendations(name, top_n=50)
            cf = self.get_collaborative_recommendations(name, top_n=50)
            if not c.empty or not cf.empty:
                valid_games.append(name)
            if not c.empty: all_c_recs.append(c)
            if not cf.empty: all_cf_recs.append(cf)
            
        if not all_c_recs and not all_cf_recs:
            return pd.DataFrame()
            
        final_recs = None
        
        # Combine Content
        if all_c_recs:
            merged_c = pd.concat(all_c_recs)
            # Filter out inputted games from results
            merged_c = merged_c[~merged_c['name'].isin(valid_games)]
            if not merged_c.empty:
                merged_c = merged_c.groupby(['name', 'clean_name'])['content_score'].sum().reset_index()
                scaler = MinMaxScaler()
                merged_c['content_score'] = scaler.fit_transform(merged_c[['content_score']])
                final_recs = merged_c
                
        # Combine Collab
        if all_cf_recs:
            merged_cf = pd.concat(all_cf_recs)
            merged_cf = merged_cf[~merged_cf['name'].isin(valid_games)]
            if not merged_cf.empty:
                merged_cf = merged_cf.groupby(['name', 'clean_name'])['collab_score'].sum().reset_index()
                scaler = MinMaxScaler()
                merged_cf['collab_score'] = scaler.fit_transform(merged_cf[['collab_score']])
                
                if final_recs is not None:
                    final_recs = pd.merge(final_recs, merged_cf, on=['name', 'clean_name'], how='outer').fillna(0)
                    final_recs['final_score'] = (final_recs['content_score'] * 0.5) + (final_recs['collab_score'] * 0.5)
                else:
                    final_recs = merged_cf
                    final_recs['final_score'] = final_recs['collab_score']
                    final_recs['content_score'] = 0
                    
        if final_recs is None or final_recs.empty:
            return pd.DataFrame()
            
        if 'final_score' not in final_recs.columns:
            final_recs['final_score'] = final_recs['content_score']
            final_recs['collab_score'] = 0
            
        final_recs['explanation'] = f"A strong composite match based on similar genres, player overlap, and aesthetic to your {len(valid_games)} selected favorites!"
        return final_recs.sort_values('final_score', ascending=False).head(top_n)

