import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.data_pipeline import run_pipeline
from src.models.recommenders import ContentBasedRecommender
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Starting Simplified TMDB Movie Recommendation Model Training...")
    
    t0 = time.time()
    # Run pipeline to load, merge, and clean the dataset
    movies_df = run_pipeline()
    pipeline_time = time.time() - t0
    logger.info(f"Pipeline finished in {pipeline_time:.2f}s")
    
    # Train/Fit TF-IDF Content-based Recommender
    t1 = time.time()
    cb_model = ContentBasedRecommender()
    cb_model.fit(movies_df)
    cb_model.save()
    cb_time = time.time() - t1
    
    logger.info(f"Content model training completed in {cb_time:.2f}s")
    logger.info("Training pipeline execution successfully finished!")

if __name__ == "__main__":
    main()
