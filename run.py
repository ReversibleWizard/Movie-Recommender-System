from flask import Flask
from routes import api_blueprint
from get_data import ensure_movie_data_exists, train_model

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Ensure movie dataset exists
try:
    ensure_movie_data_exists()
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    exit(1)  # Stop execution if movie data cannot be fetched

# ✅ Train model if needed
train_model()

# ✅ Register API Blueprint
app.register_blueprint(api_blueprint)

if __name__ == "__main__":
    app.run(debug=True)
