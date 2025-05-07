from flask import Flask, jsonify,request
from model import RecommenderModel
from utils import generate_summary
import pandas as pd

app = Flask(__name__)
recommender = RecommenderModel('ratings.csv')

# Load movie titles
movies = pd.read_csv('movies.csv')

recommender.create_embeddings()
recommender.build_faiss_index()

def get_movie_titles(movie_ids):
    return movies[movies['movieId'].isin(movie_ids)]['title'].tolist()

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    genre = request.args.get('genre')  # get ?genre=Action

    recommended_movie_ids = recommender.recommend_movies(user_id, genre_filter=genre)
    movie_titles = get_movie_titles(recommended_movie_ids)

    gpt_message = generate_summary(movie_titles)

    return jsonify({
        "recommended_movies": movie_titles,
        "gpt_summary": gpt_message
    })
if __name__ == '__main__':
    app.run(debug=True)
