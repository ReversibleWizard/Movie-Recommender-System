import pandas as pd
from tmdb_api import TMDbAPI


class MovieDataManager:
    """Handles movie dataset loading and updating."""

    def __init__(self):
        self.api = TMDbAPI()
        self.movie_file = "movies.csv"

    def ensure_movie_data_exists(self):
        """Ensures the movie dataset is available."""
        try:
            movie_df = pd.read_csv(self.movie_file)
            print("✅ Movie dataset found.")
        except FileNotFoundError:
            print("🚀 Fetching new movies from TMDb API...")
            movie_list = self.fetch_movies()
            movie_df = pd.DataFrame(movie_list)
            movie_df.fillna("", inplace=True)
            movie_df.to_csv(self.movie_file, index=False)

    def fetch_movies(self):
        """Fetches movie data and processes it."""
        movies = self.api.fetch_movies()
        movie_list = []
        for movie in movies:
            details = self.api.fetch_movie_details(movie["id"])
            if not details:
                continue
            movie_info = {
                "id": movie["id"],
                "title": movie["title"],
                "genre_ids": [g["id"] for g in details.get("genres", [])],
                "actors": self.fetch_movie_actors(movie["id"]),
                "director": self.fetch_movie_director(movie["id"]),
                "duration": details.get("runtime", 0)
            }
            movie_list.append(movie_info)
        return movie_list

    def fetch_movie_actors(self, movie_id):
        """Fetches all actors for a given movie."""
        credits = self.api.fetch_movie_credits(movie_id)
        return ", ".join([actor["name"] for actor in credits["cast"][:10]]) if credits else ""

    def fetch_movie_director(self, movie_id):
        """Fetches the director of a given movie."""
        credits = self.api.fetch_movie_credits(movie_id)
        for person in credits["crew"]:
            if person["job"] == "Director":
                return person["name"]
        return "Unknown"
