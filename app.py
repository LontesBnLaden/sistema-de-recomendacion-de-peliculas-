import requests
from flask import Flask, request, render_template, jsonify
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Carga datasets (archivos .csv deben estar en la misma carpeta que este app.py)
movies = pd.read_csv("movies_metadata_cleaned.csv")
ratings = pd.read_csv("ratings_small.csv")

# Asegurar que 'id' sea numérico
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies.dropna(subset=['id'], inplace=True)
movies['id'] = movies['id'].astype(int)

# Rellenar overviews vacíos
movies['overview'] = movies['overview'].fillna("No description available")

# Entrenar modelo SVD (Filtrado colaborativo)
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)
svd = SVD()
svd.fit(trainset)

# Calcular TF-IDF y similitud coseno (Filtrado basado en contenido)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_streaming_platforms(movie_title):
    if not movie_title:
        return ["Not Available"]
    movie_title_cleaned = movie_title.replace(" ", "+")
    return [f"https://www.justwatch.com/us/search?q={movie_title_cleaned}"]

@app.route('/recommend_user', methods=['POST'])
def recommend_user():
    data = request.get_json(force=True)
    user_id = int(data.get('user_id', -1))

    if user_id not in ratings['userId'].values:
        return jsonify({"error": "User ID not found in dataset"}), 404

    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].unique()

    if len(rated_movies) == 0:
        return jsonify({"error": "No ratings found for this user"}), 404

    all_movie_ids = movies['id'].unique()
    unrated_movie_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movies]

    if not unrated_movie_ids:
        return jsonify({"error": "No unrated movies found"}), 404

    predictions = [svd.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)

    top_movie_ids = [pred.iid for pred in predictions[:5]]
    recommended_movies = movies[movies['id'].isin(top_movie_ids)][['title', 'id']].to_dict(orient='records')

    return jsonify(recommended_movies)

@app.route('/recommend_movie', methods=['POST'])
def recommend_movie():
    data = request.get_json(force=True)
    movie_title = data.get('movie_title', '').strip().lower()

    movies['title_lower'] = movies['title'].str.lower()

    if movie_title not in movies['title_lower'].values:
        return jsonify({"error": "Movie not found"}), 404

    idx_list = movies[movies['title_lower'] == movie_title].index.tolist()
    if not idx_list:
        return jsonify({"error": "Movie not found in database"}), 404

    idx = idx_list[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    movie_indices = [i[0] for i in sim_scores]
    similar_movies = movies.iloc[movie_indices][['title', 'id']].to_dict(orient='records')

    for movie in similar_movies:
        movie["platforms"] = get_streaming_platforms(movie["title"])

    return jsonify(similar_movies)

@app.route('/search_suggestions')
def search_suggestions():
    query = request.args.get('query', '').strip().lower()
    if not query:
        return jsonify([])

    suggestions = movies[movies['title'].str.lower().str.contains(query, na=False)]['title'].head(5).tolist()
    return jsonify(suggestions)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
