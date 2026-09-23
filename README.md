# NextPlay 🎮

A premium, production-ready Game Recommendation System built with Python, Flask, and Vanilla JS/CSS. NextPlay utilizes a sophisticated hybrid machine-learning algorithm combining Content-Based Filtering (TF-IDF) and Collaborative Filtering (K-Nearest Neighbors) to generate highly accurate, robust game recommendations. 

The frontend has been meticulously crafted to resemble a professional, art-directed editorial storefront, completely avoiding generic "dashboard" layouts in favor of a sleek, content-first experience.

## ✨ Features

- **Hybrid Recommendation Engine:** Suggests similar games based on combined tags, genres, and semantic descriptions (TF-IDF), alongside mathematical user playtime overlap (Item-Item CF).
- **Deep Dive Pages:** Fully responsive game detail pages featuring high-resolution library hero banners, dynamic HTML5 YouTube trailer embeds, and nearest-neighbor game matches.
- **My Taste Profile Builder:** Select up to 5 favorite games to amalgamate their combined content and collaborative vectors into a unified master profile match.
- **Mood-Based Discovery:** Recommends games based on emotional states and pacing (e.g., Action, Horror & Suspense, Relaxing). Uses strict set-intersection filtering to ensure absolute accuracy (e.g., removing software/utilities from game searches).
- **Playtime Bracket Filtering:** Recommends games specifically fitting a user’s available free time (Quick Sessions vs Lifestyle games).
- **Custom UI / UX:** 100% custom-built design system using vanilla CSS variables, `Inter` typography, native dark mode toggling, and fully responsive layouts.

## 📊 Dataset Details

This system is powered by a heavily processed version of the **FronkonGames Steam Dataset**, containing metadata for over **116,000+ PC games**.

- **Source:** Steam Store API / FronkonGames dataset.
- **Coverage:** 116,252 unique titles.
- **Injected Titles:** Manual overrides were added for missing platform exclusives (e.g., *Marvel's Spider-Man Remastered*, *Marvel's Spider-Man 2*).
- **Synthetic Interactions:** To fuel the collaborative filtering engine, synthetic user interaction logs (107,000+ interactions across 8,000 mock users) were generated using a log-normal distribution based on real Steam average playtime and rating weights.

## 🛠️ Technology Stack

- **Backend:** Python 3, Flask, Pandas, Scikit-Learn
- **Machine Learning:** `TfidfVectorizer` (Content), `cosine_similarity` (Content), `NearestNeighbors` (Collaborative)
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (ES6+), Chart.js
- **Media Delivery:** Steam CDN Assets, YouTube iframe API

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AbhayPrasad01/NextPlay.git
   cd NextPlay
   ```

2. **Setup Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Dataset Setup:**
   Ensure you have the required Steam dataset CSV files inside a `data/` directory at the project root. (Note: Raw `.csv` files are ignored by git to prevent pushing massive datasets).
   Run the dataset preparation script:
   ```bash
   python scripts/prepare_dataset.py
   python scripts/inject_spiderman.py
   ```

4. **Launch the Flask Server:**
   ```bash
   python server.py
   ```
   Open `http://127.0.0.1:5000` in your web browser.

## 🎨 Design System

NextPlay utilizes a custom CSS framework built strictly for this project:
- **Typography:** `Inter` sans-serif scale.
- **Palette:** Monochromatic charcoal/off-white core with 1px border structures. No heavy gradients or glassmorphism.
- **Modularity:** Fully responsive using CSS Grid and Flexbox, collapsing gracefully into mobile hamburger navigation.

## 📄 License
This project is for educational and portfolio purposes. Data belongs to Valve Corporation and respective publishers.
