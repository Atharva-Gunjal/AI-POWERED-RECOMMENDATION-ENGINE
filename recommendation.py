import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample movies dataset
movies = pd.DataFrame({
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Matrix', 'Toy Story', 'Avengers', 'Frozen', 'Inception'],
    'genres': ['Action Sci-Fi', 'Animation Kids', 'Action Adventure', 'Animation Musical', 'Action Thriller']
})

# Sample user ratings
ratings = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3, 4],
    'movie_id': [1, 2, 2, 3, 4, 5, 1],
    'rating': [5, 3, 4, 5, 2, 4, 3]
})

# TF-IDF vectorizer on genres column
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(user_id, top_n=3):
    user_ratings = ratings[ratings['user_id'] == user_id]
    if user_ratings.empty:
        return ['No ratings found for this user']

    sim_scores = pd.Series(0, index=movies.index, dtype=float)
    for _, row in user_ratings.iterrows():
        movie_idx = movies.index[movies['movie_id'] == row['movie_id']][0]
        sim_scores += cosine_sim[movie_idx] * row['rating']

    rated_movie_ids = user_ratings['movie_id'].tolist()
    sim_scores[movies[movies['movie_id'].isin(rated_movie_ids)].index] = 0

    recommended_indices = sim_scores.nlargest(top_n).index
    recommended_titles = movies.loc[recommended_indices, 'title'].tolist()

    return recommended_titles
