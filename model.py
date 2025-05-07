import pandas as pd
from sklearn.decomposition import TruncatedSVD
import faiss
import numpy as np

class RecommenderModel:
    def __init__(self, ratings_path):
        self.ratings = pd.read_csv(ratings_path)
        self.user_embeddings = None
        self.user_ids = None
        self.index = None

    def create_embeddings(self):
        user_movie_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.user_ids = user_movie_matrix.index.tolist()

        svd = TruncatedSVD(n_components=50, random_state=42)
        self.user_embeddings = svd.fit_transform(user_movie_matrix)

    def build_faiss_index(self):
        dimension = self.user_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.user_embeddings).astype('float32'))

    def recommend_movies(self, user_id, top_n=5, genre_filter=None):
        if user_id not in self.user_ids:
            # Cold start - recommend popular movies
            popular_movies = self.ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False)
            top_movie_ids = popular_movies.head(top_n).index.tolist()

            if genre_filter:
                movies = pd.read_csv('movies.csv')
                merged = movies[movies['movieId'].isin(top_movie_ids)]
                merged = merged[merged['genres'].str.contains(genre_filter)]
                return merged['movieId'].tolist()

            return top_movie_ids

        # (Normal flow if user exists)
        user_idx = self.user_ids.index(user_id)
        query_vector = np.array([self.user_embeddings[user_idx]]).astype('float32')
        distances, indices = self.index.search(query_vector, 10)

        similar_user_ids = [self.user_ids[idx] for idx in indices[0] if idx != user_idx]

        user_movie_ratings = self.ratings[self.ratings['userId'].isin(similar_user_ids)]
        good_ratings = user_movie_ratings[user_movie_ratings['rating'] >= 3.5]

        movies_user_watched = self.ratings[self.ratings['userId'] == user_id]['movieId'].tolist()
        filtered_movies = good_ratings[~good_ratings['movieId'].isin(movies_user_watched)]

        if genre_filter:
            movies = pd.read_csv('movies.csv')
            merged = filtered_movies.merge(movies, on='movieId')
            filtered_movies = merged[merged['genres'].str.contains(genre_filter)]

        recommended_movies = filtered_movies['movieId'].value_counts().head(top_n).index.tolist()
        return recommended_movies
