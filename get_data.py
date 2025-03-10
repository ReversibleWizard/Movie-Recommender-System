import ast
import pandas as pd
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# ✅ TMDb API Details
API_KEY = "cd1baff5814003fe88a8d0381b925179"
BASE_URL = "https://api.themoviedb.org/3"

# ✅ Ensure movie dataset exists
def ensure_movie_data_exists():
    """Fetches movies from TMDb if not already available."""
    try:
        movie_df = pd.read_csv("movies.csv")
        print("✅ Movie dataset found. Skipping API fetch.")
    except FileNotFoundError:
        print("🚀 Fetching new movies from TMDb API...")
        movie_list = fetch_movies()
        if not movie_list:
            raise FileNotFoundError("❌ ERROR: No movie data fetched from TMDb API.")

        movie_df = pd.DataFrame(movie_list)
        movie_df.to_csv("movies.csv", index=False)
        print("✅ Movie dataset saved as 'movies.csv'.")


# ✅ Fetch More Movies from TMDb API
def fetch_movies():
    """Fetches a large number of movies from TMDb API (Trending, Popular, Now Playing)."""
    movie_list = []
    total_pages = 15  # Increase the number of pages to get more movies

    categories = ["trending/movie/week", "movie/popular", "movie/now_playing"]

    for category in categories:
        for page in range(1, total_pages + 1):
            url = f"{BASE_URL}/{category}?api_key={API_KEY}&language=en-US&page={page}"
            response = requests.get(url).json()

            if "results" in response:
                for movie in response["results"]:
                    movie_id = movie.get("id")
                    actors = fetch_movie_actors(movie_id)
                    director = fetch_movie_director(movie_id)

                    movie_info = {
                        "id": movie_id,
                        "title": movie.get("title", ""),
                        "genre_ids": movie.get("genre_ids", []),
                        "actors": actors,
                        "director": director,
                        "duration": movie.get("runtime", 0)
                    }

                    movie_list.append(movie_info)
            else:
                print(f"❌ ERROR: No results found for {category}, Page {page}")

    return movie_list

# ✅ Fetch Movies of a Specific Actor
def fetch_actor_movies(actor_name):
    """Fetches all movies of a specific actor from TMDb API."""
    url = f"{BASE_URL}/search/person?api_key={API_KEY}&language=en-US&query={actor_name}"
    response = requests.get(url).json()

    if "results" in response and response["results"]:
        actor_id = response["results"][0]["id"]  # Get the first matching actor
        movie_url = f"{BASE_URL}/person/{actor_id}/movie_credits?api_key={API_KEY}"
        movie_response = requests.get(movie_url).json()

        movie_list = []
        if "cast" in movie_response:
            for movie in movie_response["cast"]:
                movie_id = movie.get("id")
                actors = fetch_movie_actors(movie_id)
                director = fetch_movie_director(movie_id)

                movie_info = {
                    "id": movie_id,
                    "title": movie.get("title", ""),
                    "genre_ids": movie.get("genre_ids", []),
                    "actors": actors,
                    "director": director,
                    "duration": movie.get("runtime", 0)
                }

                movie_list.append(movie_info)
        return movie_list
    return []

# ✅ Fetch Movies of a Specific Director
def fetch_director_movies(director_name):
    """Fetches all movies directed by a specific director from TMDb API."""
    url = f"{BASE_URL}/search/person?api_key={API_KEY}&language=en-US&query={director_name}"
    response = requests.get(url).json()

    if "results" in response and response["results"]:
        director_id = response["results"][0]["id"]
        movie_url = f"{BASE_URL}/person/{director_id}/movie_credits?api_key={API_KEY}"
        movie_response = requests.get(movie_url).json()

        movie_list = []
        if "crew" in movie_response:
            for movie in movie_response["crew"]:
                if movie.get("job") == "Director":
                    movie_id = movie.get("id")
                    actors = fetch_movie_actors(movie_id)

                    movie_info = {
                        "id": movie_id,
                        "title": movie.get("title", ""),
                        "genre_ids": movie.get("genre_ids", []),
                        "actors": actors,
                        "director": director_name,
                        "duration": movie.get("runtime", 0)
                    }

                    movie_list.append(movie_info)
        return movie_list
    return []

# ✅ Fetch Actors for a Movie
def fetch_movie_actors(movie_id):
    """Fetches all actors for a given movie ID from TMDb API."""
    url = f"{BASE_URL}/movie/{movie_id}/credits?api_key={API_KEY}"
    response = requests.get(url).json()
    if "cast" in response:
        return ", ".join([actor["name"] for actor in response["cast"][:10]])  # Fetch up to 10 actors
    return ""

# ✅ Fetch Director for a Movie
def fetch_movie_director(movie_id):
    """Fetches the director for a given movie ID from TMDb API."""
    url = f"{BASE_URL}/movie/{movie_id}/credits?api_key={API_KEY}"
    response = requests.get(url).json()
    if "crew" in response:
        for person in response["crew"]:
            if person["job"] == "Director":
                return person["name"]
    return "Unknown"

# ✅ Fetch TMDb Genre Mapping
def fetch_genres():
    """Fetches genre name-to-ID mapping from TMDb API."""
    url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}&language=en-US"
    response = requests.get(url).json()
    if "genres" in response:
        return {genre["name"].lower(): genre["id"] for genre in response["genres"]}
    return {}

# ✅ Convert Genre Names to Genre IDs
def convert_genres_to_ids(genre_names, genre_map):
    """Converts genre names into corresponding TMDb genre IDs."""
    return [genre_map.get(genre.lower()) for genre in genre_names if genre.lower() in genre_map]

# ✅ Train a Logistic Regression Model for Movie Ranking
def train_model():
    """Train a Logistic Regression model for movie ranking."""
    movie_df = pd.read_csv("movies.csv")

    # ✅ Fill NaN values
    movie_df["title"] = movie_df["title"].fillna("").astype(str)
    movie_df["actors"] = movie_df["actors"].fillna("").astype(str)
    movie_df["director"] = movie_df["director"].fillna("").astype(str)
    movie_df["genre_ids"] = movie_df["genre_ids"].fillna("[]").astype(str)

    # ✅ Convert 'genre_ids' from string to list
    movie_df["genre_ids"] = movie_df["genre_ids"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # ✅ Create "features" column
    movie_df["combined_features"] = (
        movie_df["title"] + " " +
        (movie_df["actors"] + " ") * 3 +
        (movie_df["director"] + " ") * 2 +
        movie_df["genre_ids"].astype(str)
    )

    # ✅ Apply TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movie_df["combined_features"])

    # ✅ Train Logistic Regression Model
    model = LogisticRegression()
    model.fit(tfidf_matrix, movie_df.index % 2)  # Fake labels

    # ✅ Save Model & Vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Model training complete!")


def recommend_movies(input_data):
    """Recommend movies using TF-IDF + Cosine Similarity + Logistic Regression."""

    # ✅ Ensure movie dataset exists
    ensure_movie_data_exists()

    # ✅ Ensure model & vectorizer exist, otherwise train the model
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        print("✅ Model and vectorizer loaded successfully.")
    except FileNotFoundError:
        print("🚀 Training new model...")
        train_model()
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

    # ✅ Load movie dataset
    movie_df = pd.read_csv("movies.csv")

    # ✅ Fill NaN values
    movie_df["title"] = movie_df["title"].fillna("").astype(str)
    movie_df["actors"] = movie_df["actors"].fillna("").astype(str)
    movie_df["director"] = movie_df["director"].fillna("").astype(str)
    movie_df["genre_ids"] = movie_df["genre_ids"].fillna("[]").astype(str)

    # ✅ Convert 'genre_ids' from string to list
    movie_df["genre_ids"] = movie_df["genre_ids"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # ✅ Extract user input
    input_actor = input_data.get("actor", "").strip().lower()
    input_director = input_data.get("director", "").strip().lower()
    input_genres = input_data.get("genres", [])

    # ✅ Filter for movies with the given actor
    if input_actor:
        movie_df = movie_df[movie_df["actors"].str.contains(input_actor, case=False, na=False)]

    # ✅ If no movies match, return an error
    if movie_df.empty:
        return {"error": f"No movies found with actor '{input_actor}'."}

    # ✅ Convert genre names to IDs
    genre_map = fetch_genres()
    input_genres = convert_genres_to_ids(input_genres, genre_map)

    # ✅ Create "features" column
    movie_df["combined_features"] = (
            movie_df["title"] + " " +
            (movie_df["actors"] + " ") * 3 +
            (movie_df["director"] + " ") * 2 +
            movie_df["genre_ids"].astype(str)
    )

    # ✅ Create Input Query for TF-IDF
    input_features = (input_actor + " ") * 3 + \
                     (input_director + " ") * 2 + \
                     str(input_genres)

    # ✅ Apply TF-IDF to Transform Input & Compute Similarity
    input_vector = vectorizer.transform([input_features])
    tfidf_matrix = vectorizer.transform(movie_df["combined_features"])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()

    # ✅ Predict Ranking with Logistic Regression
    predicted_scores = model.predict_proba(tfidf_matrix)[:, 1]

    # ✅ Combine Scores: 50% Cosine Similarity + 50% Logistic Regression
    movie_df["match_score"] = (0.5 * similarity_scores) + (0.5 * predicted_scores)

    # ✅ Sort & Return Top 5 Movies
    movie_df = movie_df.sort_values(by="match_score", ascending=False).head(5)

    return movie_df.to_dict(orient="records")
