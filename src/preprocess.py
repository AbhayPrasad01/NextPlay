import pandas as pd
import numpy as np
import os
import re

def clean_text(text):
    """Lowercase text and remove special characters"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return ' '.join(text.split())

def preprocess_steam_data(file_path):
    """Preprocess the Steam Store Games Dataset."""
    print("Loading Steam Dataset...")
    df = pd.read_csv(file_path)
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['name'])
    
    # Fill missing values
    df.fillna('', inplace=True)
    
    # Combine features for Content-Based Filtering
    print("Combining text features...")
    # Categories, genres, steamspy_tags
    df['combined_features'] = df['categories'] + " " + df['genres'] + " " + df['steamspy_tags']
    df['combined_features'] = df['combined_features'].apply(lambda x: x.replace(';', ' '))
    df['combined_features'] = df['combined_features'].apply(clean_text)
    
    # Clean game names for easier matching
    df['clean_name'] = df['name'].apply(clean_text)
    
    return df

def preprocess_user_data(file_path, min_interactions=3):
    """Preprocess the Steam User Behavior Dataset."""
    print("Loading User Behavior Dataset...")
    # The dataset has no header
    df = pd.read_csv(file_path, header=None, names=['user_id', 'game_name', 'behavior', 'playtime', 'zero'])
    
    # We only care about "play" behaviour or if they "purchase" we can consider it implicit (playtime = 0.1)
    # Let's keep both, but score them. Play > Purchase.
    
    # Create an implicit rating: 
    # For simplicity, if play, rating = playtime. If purchase but not play, rating = 0 (we'll filter out just purchase or consider as low rating).
    # Since we want a robust simple model, let's just keep 'play' interactions, as we know the user experienced the game.
    df_play = df[df['behavior'] == 'play'].copy()
    
    # Clean game names to match the steam store dataset
    df_play['clean_name'] = df_play['game_name'].apply(clean_text)
    
    # Filter users and games with very few interactions to reduce sparsity and memory size
    user_counts = df_play['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    
    df_filtered = df_play[df_play['user_id'].isin(valid_users)]
    
    return df_filtered

def run_preprocessing(data_dir):
    steam_csv_path = os.path.join(data_dir, 'steam.csv')
    user_csv_path = os.path.join(data_dir, 'steam-200k.csv')
    
    if not os.path.exists(steam_csv_path) or not os.path.exists(user_csv_path):
        print(f"Error: Datasets not found in {data_dir}. Ensure 'steam.csv' and 'steam-200k.csv' exist.")
        return
    
    steam_df = preprocess_steam_data(steam_csv_path)
    user_df = preprocess_user_data(user_csv_path, min_interactions=5)
    
    # Only keep users' games that actually exist in our steam store dataset
    valid_games = set(steam_df['clean_name'])
    user_df = user_df[user_df['clean_name'].isin(valid_games)]
    
    # Save the cleaned datasets
    print("Saving preprocessed datasets...")
    steam_df.to_csv(os.path.join(data_dir, 'cleaned_steam.csv'), index=False)
    user_df.to_csv(os.path.join(data_dir, 'cleaned_user_interactions.csv'), index=False)
    print("Preprocessing complete!")

if __name__ == "__main__":
    data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    run_preprocessing(data_folder)
