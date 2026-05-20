import pandas as pd
import json
from configs.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

def parse_genres(x):
    try:
        genres = json.loads(x)
        return " ".join([g['name'] for g in genres])
    except:
        return ""

def parse_keywords(x):
    try:
        keywords = json.loads(x)
        return " ".join([k['name'] for k in keywords])
    except:
        return ""

def parse_cast(x):
    try:
        cast = json.loads(x)
        return ", ".join([c['name'] for c in cast[:4]])
    except:
        return ""

def parse_director(x):
    try:
        crew = json.loads(x)
        for member in crew:
            if member['job'] == 'Director':
                return member['name']
    except:
        pass
    return ""

def run_pipeline():
    logger.info("Starting TMDB Movie Metadata processing pipeline...")
    movies_path = RAW_DATA_DIR / "tmdb" / "tmdb_5000_movies.csv"
    credits_path = RAW_DATA_DIR / "tmdb" / "tmdb_5000_credits.csv"
    
    if not movies_path.exists() or not credits_path.exists():
        raise FileNotFoundError("Raw TMDB 5000 CSV files not found. Please ensure tmdb_5000_movies.csv and tmdb_5000_credits.csv are in data/raw/tmdb/")
        
    logger.info("Loading raw TMDB files...")
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    
    logger.info("Merging datasets...")
    merged = pd.merge(movies, credits, left_on='id', right_on='movie_id', suffixes=('', '_credits'))
    
    logger.info("Extracting attributes...")
    merged['genres_cleaned'] = merged['genres'].apply(parse_genres)
    merged['keywords_cleaned'] = merged['keywords'].apply(parse_keywords)
    merged['cast_cleaned'] = merged['cast'].apply(parse_cast)
    merged['director_cleaned'] = merged['crew'].apply(parse_director)
    
    # Fill NAs
    merged['overview'] = merged['overview'].fillna("")
    merged['tagline'] = merged['tagline'].fillna("")
    
    # Save the processed movies dataset
    processed_path = PROCESSED_DATA_DIR / "movies.parquet"
    logger.info(f"Saving cleaned dataset to {processed_path}...")
    merged.to_parquet(processed_path, index=False)
    
    logger.info("TMDB data pipeline execution complete.")
    return merged

if __name__ == "__main__":
    run_pipeline()
