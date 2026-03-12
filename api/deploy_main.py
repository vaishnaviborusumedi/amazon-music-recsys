import sys
import os
# Fix: add project root to path explicitly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
print(f"Project root: {ROOT}")

import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from src.recommender_deploy import DeployRecommender

# ── App setup ─────────────────────────────────────────────
app = FastAPI(
    title       = "🎸 Music Gear RecSys API",
    description = "Hybrid recommendation system — ALS + TF-IDF",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ── Paths ─────────────────────────────────────────────────
BASE_DIR    = os.path.abspath('.')
DEPLOY_PATH = os.path.join(BASE_DIR, 'models', 'deploy')

# ── Load on startup ───────────────────────────────────────
print("🚀 Starting API...")
rec = DeployRecommender(models_path=DEPLOY_PATH)

with open(os.path.join(DEPLOY_PATH, 'meta_lookup.pkl'), 'rb') as f:
    meta_lookup = pickle.load(f)

df = pd.read_csv(os.path.join(DEPLOY_PATH, 'ratings.csv'))

print(f"✅ API ready!")
print(f"   Users    : {df['user_id'].nunique():,}")
print(f"   Products : {df['product_id'].nunique():,}")
print(f"   Metadata : {len(meta_lookup):,}")

# ── Models ────────────────────────────────────────────────
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
    user_id  : str
    count    : int
    strategy : str
    products : List[Product]

class HealthResponse(BaseModel):
    status   : str
    users    : int
    products : int
    metadata : int

# ── Helpers ───────────────────────────────────────────────
def enrich(product_id, score):
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

def get_strategy(user_id):
    count = len(df[df['user_id'] == user_id])
    if count < 5:
        return "content-heavy (new user)"
    elif count < 20:
        return "balanced"
    else:
        return "ALS-heavy (power user)"

# ── Routes ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message" : "🎸 Music Gear RecSys API",
        "docs"    : "/docs",
        "health"  : "/health"
    }

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status   = "healthy",
        users    = int(df['user_id'].nunique()),
        products = int(df['product_id'].nunique()),
        metadata = len(meta_lookup)
    )

@app.get("/recommend/{user_id}",
         response_model=RecommendationResponse)
def recommend(user_id: str, n: int = 10):
    if user_id not in df['user_id'].values:
        raise HTTPException(
            status_code = 404,
            detail      = f"User '{user_id}' not found"
        )
    if n < 1 or n > 50:
        raise HTTPException(
            status_code = 400,
            detail      = "n must be between 1 and 50"
        )

    recs     = rec.recommend(user_id, n=n)
    products = [enrich(r['product_id'], r['score']) for r in recs]

    return RecommendationResponse(
        user_id  = user_id,
        count    = len(products),
        strategy = get_strategy(user_id),
        products = products
    )

@app.get("/users")
def list_users(limit: int = 20):
    return {
        "total_users" : int(df['user_id'].nunique()),
        "sample"      : df['user_id'].unique()[:limit].tolist()
    }

@app.get("/products/{product_id}")
def get_product(product_id: str):
    if product_id not in meta_lookup:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"product_id": product_id, **meta_lookup[product_id]}

@app.get("/similar/{product_id}")
def similar_products(product_id: str, n: int = 10):
    if product_id not in rec.product_idx_map:
        raise HTTPException(status_code=404, detail="Product not found")

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    idx  = rec.product_idx_map[product_id]
    sims = cosine_similarity(
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
        "title"      : meta_lookup.get(product_id, {}).get('title', ''),
        "similar"    : results
    }