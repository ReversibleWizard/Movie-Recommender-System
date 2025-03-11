import ast
import pandas as pd
import time
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

# TMDb API Details
API_KEY = "cd1baff5814003fe88a8d0381b925179"
BASE_URL = "https://api.themoviedb.org/3"

# Ensure movie dataset exists
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


# Fetch More Movies from TMDb API
def fetch_movies():
    """Fetches a large number of movies from TMDb API with improved error handling."""
    movie_list = []
    total_pages = 10  # Fetch more movies

    categories = ["trending/movie/week", "movie/popular", "movie/now_playing"]

    for category in categories:
        for page in range(1, total_pages + 1):
            url = f"{BASE_URL}/{category}?api_key={API_KEY}&language=en-US&page={page}"

            for retry in range(3):  # ✅ Retry up to 3 times if API fails
                try:
                    response = requests.get(url, timeout=20)  # ✅ Increased timeout to 20 seconds
                    response.raise_for_status()  # ✅ Raise error for bad responses
                    data = response.json()
                    break  # ✅ Exit retry loop if request is successful
                except requests.exceptions.Timeout:
                    print(f"⏳ Timeout Error - Retrying ({retry+1}/3)...")
                    time.sleep(5)  # ✅ Wait 5 seconds before retrying
                except requests.exceptions.RequestException as e:
                    print(f"❌ API Error ({e}) - Retrying ({retry+1}/3)...")
                    time.sleep(5)

            if "results" in data:
                for movie in data["results"]:
                    movie_id = movie.get("id")
                    title = movie.get("title", "")

                    # ✅ Fetch full movie details for runtime
                    details_url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US"
                    details_response = requests.get(details_url, timeout=10).json()

                    actors = fetch_movie_actors(movie_id)
                    director = fetch_movie_director(movie_id)
                    duration = details_response.get("runtime", None)  # ✅ Fetch runtime correctly
                    genre_ids = details_response.get("genres", [])
                    genre_ids = [g["id"] for g in genre_ids]

                    movie_info = {
                        "id": movie_id,
                        "title": title,
                        "genre_ids": genre_ids,
                        "actors": actors,
                        "director": director,
                        "duration": duration if duration else 0  # ✅ Ensure duration is not None
                    }

                    movie_list.append(movie_info)

            else:
                print(f"❌ ERROR: No results found for {category}, Page {page}")

    return movie_list

# Fetch Movies of a Specific Actor
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


def fetch_actor_movies(actor_name):
    """Fetches all movies of a specific actor from TMDb API with error handling."""
    search_url = f"{BASE_URL}/search/person?api_key={API_KEY}&language=en-US&query={actor_name}"

    for retry in range(3):
        try:
            response = requests.get(search_url, timeout=20)  # ✅ Increased timeout
            response.raise_for_status()
            response_data = response.json()
            break  # ✅ Exit retry loop if request is successful
        except requests.exceptions.Timeout:
            print(f"⏳ Timeout Error - Retrying ({retry+1}/3)...")
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"❌ API Error ({e}) - Retrying ({retry+1}/3)...")
            time.sleep(5)

    if "results" in response_data and response_data["results"]:
        actor_id = response_data["results"][0]["id"]
        movie_url = f"{BASE_URL}/person/{actor_id}/movie_credits?api_key={API_KEY}&language=en-US"

        for retry in range(3):
            try:
                movie_response = requests.get(movie_url, timeout=20)
                movie_response.raise_for_status()
                movie_data = movie_response.json()
                break
            except requests.exceptions.Timeout:
                print(f"⏳ Timeout Error - Retrying ({retry+1}/3)...")
                time.sleep(5)
            except requests.exceptions.RequestException as e:
                print(f"❌ API Error ({e}) - Retrying ({retry+1}/3)...")
                time.sleep(5)

        movie_list = []
        if "cast" in movie_data:
            for movie in movie_data["cast"]:
                movie_id = movie.get("id")
                title = movie.get("title", "")

                # ✅ Fetch full movie details for runtime
                details_url = f"{BASE_URL}/movie/{movie_id}?api_key={API_KEY}&language=en-US"
                details_response = requests.get(details_url, timeout=10).json()

                actors = fetch_movie_actors(movie_id)
                director = fetch_movie_director(movie_id)
                duration = details_response.get("runtime", None)  # ✅ Fetch duration correctly
                genre_ids = details_response.get("genres", [])
                genre_ids = [g["id"] for g in genre_ids]

                movie_info = {
                    "id": movie_id,
                    "title": title,
                    "genre_ids": genre_ids,
                    "actors": actors,
                    "director": director,
                    "duration": duration if duration else 0
                }

                movie_list.append(movie_info)

        return movie_list

    return []

# Fetch Movies of a Specific Director
def fetch_movie_actors(movie_id):
    """Fetches all actors for a given movie ID from TMDb API."""
    url = f"{BASE_URL}/movie/{movie_id}/credits?api_key={API_KEY}"
    response = requests.get(url).json()
    if "cast" in response:
        return ", ".join([actor["name"] for actor in response["cast"][:10]])  # Fetch up to 10 actors
    return ""

# Fetch Movies director
def fetch_movie_director(movie_id):
    """Fetches the director for a given movie ID from TMDb API."""
    url = f"{BASE_URL}/movie/{movie_id}/credits?api_key={API_KEY}"
    response = requests.get(url).json()
    if "crew" in response:
        for person in response["crew"]:
            if person["job"] == "Director":
                return person["name"]
    return "Unknown"

# Fetch TMDb Genre Mapping
def fetch_genres():
    """Fetches genre name-to-ID mapping from TMDb API."""
    url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}&language=en-US"
    response = requests.get(url).json()
    if "genres" in response:
        return {genre["name"].lower(): genre["id"] for genre in response["genres"]}
    return {}

# Convert Genre Names to Genre IDs
def convert_genres_to_ids(genre_names, genre_map):
    """Converts genre names into corresponding TMDb genre IDs."""
    return [genre_map.get(genre.lower()) for genre in genre_names if genre.lower() in genre_map]

# Train a Logistic Regression Model for Movie Ranking
def train_model():
    """Train a Logistic Regression model for movie ranking."""
    movie_df = pd.read_csv("movies.csv")

    # Fill NaN values
    movie_df["title"] = movie_df["title"].fillna("").astype(str)
    movie_df["actors"] = movie_df["actors"].fillna("").astype(str)
    movie_df["director"] = movie_df["director"].fillna("").astype(str)
    movie_df["genre_ids"] = movie_df["genre_ids"].fillna("[]").astype(str)

    #  Convert 'genre_ids' from string to list
    movie_df["genre_ids"] = movie_df["genre_ids"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Create "features" column
    movie_df["combined_features"] = (
        movie_df["title"] + " " +
        (movie_df["actors"] + " ") * 3 +
        (movie_df["director"] + " ") * 2 +
        movie_df["genre_ids"].astype(str)
    )

    # Apply TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movie_df["combined_features"])

    # Train Logistic Regression Model
    model = LogisticRegression()
    model.fit(tfidf_matrix, movie_df.index % 2)  # Fake labels

    # Save Model & Vectorizer
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Model training complete!")


def recommend_movies(input_data):
    """Recommend movies using TF-IDF + Cosine Similarity + Logistic Regression."""

    ensure_movie_data_exists()

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
        filtered_movies = movie_df[movie_df["actors"].str.contains(input_actor, case=False, na=False)]

        # ✅ If no movies found, fetch movies dynamically
        if filtered_movies.empty:
            print(f"🔍 Fetching movies for actor '{input_actor}' from TMDb API...")
            new_movies = fetch_actor_movies(input_actor)

            if new_movies:
                new_movies_df = pd.DataFrame(new_movies)
                movie_df = pd.concat([movie_df, new_movies_df], ignore_index=True)
                movie_df.to_csv("movies.csv", index=False)  # ✅ Save updated dataset
                print(f"✅ Added {len(new_movies)} movies for actor '{input_actor}' to dataset.")

                # ✅ Retrain the model with new data
                train_model()

                # ✅ Reload dataset
                movie_df = pd.read_csv("movies.csv")

                # ✅ Filter again for the actor
                filtered_movies = movie_df[movie_df["actors"].str.contains(input_actor, case=False, na=False)]

        # ✅ If still no movies found, return an error
        if filtered_movies.empty:
            return {"error": f"No movies found with actor '{input_actor}'."}

        movie_df = filtered_movies  # ✅ Use only relevant movies

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

    # ✅ Load Model & Vectorizer
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

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
