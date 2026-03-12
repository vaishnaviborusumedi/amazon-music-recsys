# 🎸 Amazon Music Gear Recommendation System

A production-grade hybrid recommendation system built with 
ALS Collaborative Filtering + TF-IDF Content-Based Filtering.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)

---

## 📊 Results

| Metric | ALS | Content | Hybrid |
|--------|-----|---------|--------|
| NDCG@10 | 0.1395 | 0.1479 | **0.1560** |
| Recall@10 | 0.1716 | 0.1755 | **0.1955** |
| Precision@10 | 0.0266 | 0.0288 | **0.0306** |

**Hybrid model outperforms both individual models on every metric.**

---

## 🏗️ Architecture
```
Raw Data (500K reviews)
        ↓
   Data Cleaning
   (72K interactions)
        ↓
   ┌────────────────────────────────┐
   │     Two-Model Approach         │
   │                                │
   │  ALS Collaborative Filtering   │
   │  + TF-IDF Content-Based        │
   │                                │
   │  Weighted Hybrid Combination   │
   └────────────────────────────────┘
        ↓
   FastAPI REST API
        ↓
   Streamlit UI
```

---

## 🚀 Quick Start

### Option 1 — Docker (recommended)
```bash
docker-compose up --build
```
- API: http://localhost:8000
- UI:  http://localhost:8501

### Option 2 — Manual

**Terminal 1 — Start API:**
```bash
cd api
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Start UI:**
```bash
cd ui
streamlit run app.py
```

---

## 📁 Project Structure
```
amazon-recsys/
├── data/
│   ├── clean_ratings.csv
│   ├── clean_ratings_light.csv
│   └── product_metadata.csv
├── models/
│   ├── als_model.pkl
│   ├── combined_matrix.pkl
│   ├── encoders.pkl
│   ├── interaction_matrix.pkl
│   ├── meta_lookup.pkl
│   ├── product_idx_map.pkl
│   ├── product_profiles.pkl
│   └── eval_results.json
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_content_based.ipynb
│   ├── 04_hybrid_model.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_metadata.ipynb
├── src/
│   ├── __init__.py
│   └── recommender.py
├── api/
│   └── main.py
├── ui/
│   └── app.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| GET | `/recommend/{user_id}` | Get recommendations |
| GET | `/users` | List sample users |
| GET | `/products/{product_id}` | Get product details |
| GET | `/similar/{product_id}` | Get similar products |

---

## 📦 Tech Stack

- **ML:** Python, Scikit-learn, Implicit (ALS), Scipy
- **API:** FastAPI, Uvicorn, Pydantic
- **UI:** Streamlit
- **Data:** Pandas, NumPy
- **Deploy:** Docker, Docker Compose

---

## 📈 Dataset

- **Source:** Amazon Musical Instruments Reviews
- **Raw reviews:** 500,000
- **After filtering:** 72,696 interactions
- **Users:** 10,327
- **Products:** 2,717
- **Metadata:** 1,921 enriched products

---

