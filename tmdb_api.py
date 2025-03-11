import requests
import time

class TMDbAPI:
    """Handles all API requests to TMDb."""

    BASE_URL = "https://api.themoviedb.org/3"
    API_KEY = "cd1baff5814003fe88a8d0381b925179"

    def __init__(self):
        self.session = requests.Session()
        self.session.params = {"api_key": self.API_KEY, "language": "en-US"}

    def make_request(self, endpoint, params=None, retries=3):
        """Handles API requests with retries."""
        url = f"{self.BASE_URL}/{endpoint}"
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=20)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"⚠️ API Error: {e}. Retrying {attempt+1}/{retries}...")
                time.sleep(5)
        return None

    def fetch_movies(self, pages=10):
        """Fetches a large number of movies from TMDb API."""
        movies = []
        for page in range(1, pages + 1):
            data = self.make_request(f"trending/movie/week?page={page}")
            if data and "results" in data:
                movies.extend(data["results"])
        return movies

    def fetch_movie_details(self, movie_id):
        """Fetches detailed movie information."""
        return self.make_request(f"movie/{movie_id}")

    def fetch_movie_credits(self, movie_id):
        """Fetches the cast and crew of a movie."""
        return self.make_request(f"movie/{movie_id}/credits")
