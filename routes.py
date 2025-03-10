from flask import Blueprint, request, jsonify
from get_data import recommend_movies

api_blueprint = Blueprint("api", __name__)

@api_blueprint.route("/v1/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json()
        recommended_movies = recommend_movies(data)
        return jsonify(recommended_movies)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
