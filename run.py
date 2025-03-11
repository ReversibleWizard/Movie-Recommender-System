from flask import Flask
from routes import api_blueprint
from movie_recommender import MovieRecommender

app = Flask(__name__)

recommender = MovieRecommender()
recommender.train_model()  # Train all models before API starts

app.register_blueprint(api_blueprint)

if __name__ == "__main__":
    app.run(debug=True)
