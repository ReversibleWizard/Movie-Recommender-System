import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input
from tensorflow.keras.models import Model
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from movie_data import MovieDataManager


class MovieRecommender:
    """Hybrid Movie Recommendation System (TF-IDF + SVD + Neural Embeddings)"""

    def __init__(self):
        self.data_manager = MovieDataManager()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.svd_model = None
        self.embeddings_model = None
        self.movie_df = None

    def load_data(self):
        """Loads and processes movie dataset."""
        self.data_manager.ensure_movie_data_exists()
        self.movie_df = pd.read_csv(self.data_manager.movie_file)
        self.movie_df.fillna("", inplace=True)

        self.movie_df["combined_features"] = (
                self.movie_df["title"] + " " +
                self.movie_df["actors"] + " " +
                self.movie_df["director"] + " " +
                self.movie_df["genre_ids"].astype(str)
        )

    def train_tfidf(self):
        """Trains TF-IDF model."""
        self.load_data()
        tfidf_matrix = self.vectorizer.fit_transform(self.movie_df["combined_features"])
        with open("models/vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

    def train_svd(self):
        """Trains SVD collaborative filtering model."""
        ratings_df = pd.read_csv("ratings.csv")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
        trainset, testset = train_test_split(data, test_size=0.2)

        self.svd_model = SVD()
        self.svd_model.fit(trainset)

        with open("models/svd_model.pkl", "wb") as f:
            pickle.dump(self.svd_model, f)

    def train_embeddings(self):
        """Trains Deep Learning Embedding Model."""
        self.load_data()
        movie_ids = self.movie_df["id"].astype("category").cat.codes.values
        num_movies = len(set(movie_ids))

        input_layer = Input(shape=(1,))
        embedding_layer = Embedding(input_dim=num_movies, output_dim=50)(input_layer)
        flatten_layer = Flatten()(embedding_layer)
        output_layer = Dense(1, activation="sigmoid")(flatten_layer)

        self.embeddings_model = Model(inputs=input_layer, outputs=output_layer)
        self.embeddings_model.compile(optimizer="adam", loss="mse")
        self.embeddings_model.fit(movie_ids, np.random.rand(len(movie_ids)), epochs=10, batch_size=32)

        self.embeddings_model.save("models/embeddings_model.h5")

    def train_model(self):
        """Trains all models (TF-IDF, SVD, Embeddings)."""
        print("🚀 Training TF-IDF Model...")
        self.train_tfidf()

        print("🚀 Training SVD Model...")
        self.train_svd()

        print("🚀 Training Deep Learning Embeddings Model...")
        self.train_embeddings()

    def recommend_movies(self, query):
        """Recommends movies based on the hybrid model."""
        self.load_data()

        # TF-IDF Similarity
        input_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(input_vector,
                                              self.vectorizer.transform(self.movie_df["combined_features"])).flatten()
        self.movie_df["tfidf_score"] = similarity_scores

        # SVD Prediction
        with open("models/svd_model.pkl", "rb") as f:
            self.svd_model = pickle.load(f)

        self.movie_df["svd_score"] = self.movie_df["id"].apply(lambda x: self.svd_model.predict(uid=1, iid=x).est)

        # Neural Embeddings Prediction
        self.embeddings_model = keras.models.load_model("models/embeddings_model.h5")
        movie_codes = self.movie_df["id"].astype("category").cat.codes.values
        self.movie_df["nn_score"] = self.embeddings_model.predict(movie_codes)

        # Final Score (Weighted Combination)
        self.movie_df["final_score"] = (0.3 * self.movie_df["tfidf_score"]) + \
                                       (0.4 * self.movie_df["svd_score"]) + \
                                       (0.3 * self.movie_df["nn_score"])

        return self.movie_df.sort_values(by="final_score", ascending=False).head(5).to_dict(orient="records")
