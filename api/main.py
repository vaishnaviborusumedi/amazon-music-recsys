import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from src.recommender import HybridRecommender

# ── App setup ─────────────────────────────────────────────
app = FastAPI(
    title       = "Amazon Music Instruments RecSys API",
    description = "Hybrid recommendation system using ALS + TF-IDF",
    version     = "1.0.0"
)

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Paths ─────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_PATH = os.path.join(BASE_DIR, 'models')
DATA_PATH   = os.path.join(BASE_DIR, 'data')

# ── Load models on startup ────────────────────────────────
print("Loading models...")

rec = HybridRecommender(
    models_path = MODELS_PATH,
    data_path   = DATA_PATH
)

with open(os.path.join(MODELS_PATH, 'meta_lookup.pkl'), 'rb') as f:
    meta_lookup = pickle.load(f)

df = pd.read_csv(os.path.join(DATA_PATH, 'clean_ratings_light.csv'))

print(f"✅ API ready!")
print(f"   Users    : {df['user_id'].nunique():,}")
print(f"   Products : {df['product_id'].nunique():,}")
print(f"   Metadata : {len(meta_lookup):,} products")

# ── Response models ───────────────────────────────────────
class Product(BaseModel):
    product_id  : str
    title       : str
    brand       : str
    price       : str
    score       : float
    image_url   : str
    description : str
    features    : str
    category    : str

class RecommendationResponse(BaseModel):
    user_id      : str
    count        : int
    strategy     : str
    products     : List[Product]

class HealthResponse(BaseModel):
    status   : str
    users    : int
    products : int
    metadata : int

# ── Helper ────────────────────────────────────────────────
def enrich(product_id: str, score: float) -> Product:
    meta = meta_lookup.get(product_id, {})
    return Product(
        product_id  = product_id,
        title       = meta.get('title',       'Unknown Product'),
        brand       = meta.get('brand',       'Unknown Brand'),
        price       = meta.get('price',       'N/A'),
        score       = score,
        image_url   = meta.get('image_url',   ''),
        description = meta.get('description', ''),
        features    = meta.get('features',    ''),
        category    = meta.get('category',    '')
    )

def get_strategy(user_id: str) -> str:
    count = len(df[df['user_id'] == user_id])
    if count < 5:
        return "content-heavy (new user)"
    elif count < 20:
        return "balanced"
    else:
        return "ALS-heavy (power user)"

# ── Routes ────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "message" : "Amazon Music RecSys API",
        "docs"    : "/docs",
        "health"  : "/health",
        "usage"   : "/recommend/{user_id}"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status   = "healthy",
        users    = int(df['user_id'].nunique()),
        products = int(df['product_id'].nunique()),
        metadata = len(meta_lookup)
    )

@app.get("/recommend/{user_id}",
         response_model=RecommendationResponse,
         tags=["Recommendations"])
def recommend(user_id: str, n: int = 10):
    """
    Get top-N hybrid recommendations for a user.

    - **user_id**: Amazon reviewer ID
    - **n**: number of recommendations (default 10, max 50)
    """
    if user_id not in df['user_id'].values:
        raise HTTPException(
            status_code = 404,
            detail      = f"User '{user_id}' not found in dataset"
        )

    if n < 1 or n > 50:
        raise HTTPException(
            status_code = 400,
            detail      = "n must be between 1 and 50"
        )

    recs     = rec.recommend(user_id, n=n)
    products = [enrich(r['product_id'], r['score']) for r in recs]
    strategy = get_strategy(user_id)

    return RecommendationResponse(
        user_id  = user_id,
        count    = len(products),
        strategy = strategy,
        products = products
    )

@app.get("/users", tags=["Users"])
def list_users(limit: int = 20):
    """Get a sample of user IDs to test with"""
    sample = df['user_id'].unique()[:limit].tolist()
    return {
        "total_users" : int(df['user_id'].nunique()),
        "sample"      : sample
    }

@app.get("/products/{product_id}", tags=["Products"])
def get_product(product_id: str):
    """Get metadata for a specific product"""
    if product_id not in meta_lookup:
        raise HTTPException(
            status_code = 404,
            detail      = f"Product '{product_id}' not found"
        )
    meta = meta_lookup[product_id]
    return {"product_id": product_id, **meta}

@app.get("/similar/{product_id}", tags=["Products"])
def similar_products(product_id: str, n: int = 10):
    """Get products similar to a given product"""
    if product_id not in meta_lookup:
        raise HTTPException(
            status_code = 404,
            detail      = f"Product '{product_id}' not found"
        )

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if product_id not in rec.product_idx_map:
        raise HTTPException(
            status_code = 404,
            detail      = f"Product '{product_id}' not in content model"
        )

    idx   = rec.product_idx_map[product_id]
    sims  = cosine_similarity(
                rec.combined_matrix[idx],
                rec.combined_matrix
            ).flatten()

    top_indices = sims.argsort()[::-1][1:n+1]

    results = []
    for i in top_indices:
        pid  = rec.product_id_map[i]
        meta = meta_lookup.get(pid, {})
        results.append({
            'product_id' : pid,
            'title'      : meta.get('title', 'Unknown'),
            'brand'      : meta.get('brand', ''),
            'price'      : meta.get('price', 'N/A'),
            'similarity' : round(float(sims[i]), 4),
            'image_url'  : meta.get('image_url', '')
        })

    return {
        "product_id" : product_id,
        "title"      : meta_lookup[product_id].get('title', ''),
        "similar"    : results
    }