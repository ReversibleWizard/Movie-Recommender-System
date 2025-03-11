from flask import Blueprint, request, jsonify
from movie_recommender import MovieRecommender

api_blueprint = Blueprint("api", __name__)
recommender = MovieRecommender()
recommender.train_model()  # Train all models before handling requests


@api_blueprint.route("/v1/recommend", methods=["POST"])
def recommend():
    """API endpoint to get movie recommendations."""
    try:
        data = request.json
        input_query = " ".join([data.get("actor", ""), data.get("director", ""), str(data.get("genres", ""))])
        recommended_movies = recommender.recommend_movies(input_query)

        if not recommended_movies:
            return jsonify({"error": "No matching movies found"}), 404
        return jsonify(recommended_movies)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
