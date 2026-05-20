import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
CACHE_DIR = BASE_DIR / "cache"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"
DASHBOARDS_DIR = BASE_DIR / "dashboards"

# Ensure directories exist
for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, EMBEDDINGS_DIR, VECTOR_STORE_DIR, CACHE_DIR, VISUALIZATIONS_DIR, DASHBOARDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API Keys
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")

# Redis Cache for high-performance API
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Big Data Configurations
RANDOM_STATE = 42
MAX_RECOMMENDATIONS = 100
NUM_FACTORS_ALS = 150 # Increased factors for better representations on large data
CHUNK_SIZE = 100000
