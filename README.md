<div align="center">
  <img src="https://via.placeholder.com/800x200/0b0c10/00ff88?text=CineMind+AI+Enterprise+Engine" alt="CineMind AI Banner" />
  
  # 🎬 CineMind AI
  **A High-Performance, Large-Scale Neural Movie Recommendation Ecosystem**
  
  [![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)]()
  [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)]()
  [![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)]()
  [![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)]()
  [![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)]()
  
  *Built to handle 25 Million+ behavioral interactions with sub-15ms inference latency.*
</div>

---

## 🚀 The Architecture (Netflix-Grade Infrastructure)
CineMind AI isn't just a machine learning project; it's a production-grade **AI MLOps Ecosystem** designed to emulate the algorithms powering platforms like Netflix and Amazon Prime.

### Core Engines:
1. **GPU-Accelerated Semantic Vectors**: Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) to map 62,000+ movies into dense hyperspace, indexed by **FAISS (HNSW)** for lightning-fast approximate nearest neighbor search.
2. **Distributed Matrix Factorization**: Processes the **MovieLens 25M** dataset using `Dask` out-of-core chunking and trains a massive Collaborative Filtering model (`AlternatingLeastSquares`) on highly optimized sparse matrices.
3. **Hybrid Ensemble**: Blends behavioral matrix signals with semantic content vectors.
4. **Real-Time TMDB Telemetry**: `FastAPI` + `httpx` dynamically resolves missing metadata and rich media from TMDB in parallel, feeding an elite Streamlit dashboard.

---

## 📊 Benchmark & Performance
| Component | Metric | Performance |
|-----------|--------|-------------|
| **FAISS HNSW** | Search Latency | **~12ms** |
| **ALS Matrix** | Sparsity | **99.8%** |
| **Data Pipeline** | 25M Row Processing | **< 3 mins** (Dask Chunked) |
| **Embeddings** | 62k Movies (CUDA) | **~45s** (Batch 512) |

*Tracked automatically via integrated MLflow MLOps pipeline.*

---

## 💻 Tech Stack
- **Deep Learning**: PyTorch, HuggingFace Transformers
- **Vector DB**: FAISS (Facebook AI Similarity Search)
- **Big Data ETL**: Dask, PyArrow, Parquet
- **Backend API**: FastAPI, Uvicorn, Asyncio
- **Frontend UI**: Streamlit (Custom Cinematic CSS)
- **DevOps/MLOps**: Docker, NGINX, MLflow, GitHub Actions

---

## ⚙️ Quick Start (Local Deployment)

### 1. Configure the Engine
Ensure you have a TMDB API Key.
```bash
cp .env.example .env
# Edit .env and insert your TMDB_API_KEY
```

### 2. Install High-Performance Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize MLOps Pipeline & Train Vectors
```bash
python main.py train
# View telemetry: mlflow ui
```

### 4. Ignite the Engines
**Terminal 1 (Backend API):**
```bash
python main.py api
```
**Terminal 2 (Enterprise UI):**
```bash
python main.py frontend
```

---

## 🐳 Production Deployment (Docker + Nginx)
The system is fully containerized and routed through an **NGINX Reverse Proxy** for scalable serving.
```bash
docker-compose up --build -d
```
- Frontend: `http://localhost:80`
- API Backend: `http://localhost/api/`
- MLflow Tracking: `http://localhost:5000`

---
*Built as a showcase for Advanced Machine Learning, Big Data Engineering, and Scalable Backend Architectures.*
