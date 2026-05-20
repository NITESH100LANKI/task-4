import argparse
import subprocess
from configs.settings import BASE_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

def run_pipeline():
    logger.info("Running Data Pipeline and Model Training...")
    subprocess.run(["python", "-m", "src.models.train"], cwd=str(BASE_DIR))

def run_api():
    logger.info("Starting FastAPI Engine...")
    subprocess.run(["uvicorn", "api.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"], cwd=str(BASE_DIR))

def run_frontend():
    logger.info("Starting Streamlit Dashboard...")
    subprocess.run(["streamlit", "run", "frontend/app.py"], cwd=str(BASE_DIR))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CineMind AI Orchestrator")
    parser.add_argument("action", choices=["train", "api", "frontend"], help="Action to perform")
    
    args = parser.parse_args()
    
    if args.action == "train":
        run_pipeline()
    elif args.action == "api":
        run_api()
    elif args.action == "frontend":
        run_frontend()
