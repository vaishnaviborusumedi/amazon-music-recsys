import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os

class DeployRecommender:
    def __init__(self, models_path="models/deploy"):
        print("Loading deploy models...")

        # Load ratings
        self.df = pd.read_csv(f"{models_path}/ratings.csv")
        print(f"  Ratings: {self.df.shape}")

        # Load pretrained ALS (no retraining needed!)
        with open(f"{models_path}/train_als.pkl", "rb") as f:
            self.train_als = pickle.load(f)

        with open(f"{models_path}/train_matrix.pkl", "rb") as f:
            self.train_matrix = pickle.load(f)

        with open(f"{models_path}/train_encoders.pkl", "rb") as f:
            enc = pickle.load(f)
            self.train_user_encoder    = enc["train_user_encoder"]
            self.train_user_decoder    = enc["train_user_decoder"]
            self.train_product_encoder = enc["train_product_encoder"]
            self.train_product_decoder = enc["train_product_decoder"]

        with open(f"{models_path}/combined_matrix.pkl", "rb") as f:
            self.combined_matrix = pickle.load(f)

        with open(f"{models_path}/product_profiles.pkl", "rb") as f:
            self.product_profiles = pickle.load(f)

        with open(f"{models_path}/product_idx_map.pkl", "rb") as f:
            maps = pickle.load(f)
            self.product_idx_map = maps["product_idx_map"]
            self.product_id_map  = maps["product_id_map"]

        print(f"  ALS user factors : {self.train_als.user_factors.shape}")
        print(f"  Train matrix     : {self.train_matrix.shape}")
        print("✅ Deploy recommender ready — no retraining needed!")

    def recommend(self, user_id, n=10):
        count = len(self.df[self.df["user_id"] == user_id])

        if count < 5:
            als_w, cb_w = 0.2, 0.8
        elif count < 20:
            als_w, cb_w = 0.5, 0.5
        else:
            als_w, cb_w = 0.8, 0.2

        als_r = self._als_recs(user_id, n=50)
        cb_r  = self._content_recs(user_id, n=50)

        als_scores = {p: (1 - i/len(als_r)) for i, p in enumerate(als_r)} if als_r else {}
        cb_scores  = {p: (1 - i/len(cb_r))  for i, p in enumerate(cb_r)}  if cb_r  else {}

        all_products = set(als_scores.keys()) | set(cb_scores.keys())
        if not all_products:
            return []

        hybrid_scores = {
            pid: als_w * als_scores.get(pid, 0) + cb_w * cb_scores.get(pid, 0)
            for pid in all_products
        }
        top = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:n]
        return [{"product_id": pid, "score": round(hybrid_scores[pid], 4)} for pid in top]

    def _als_recs(self, user_id, n=20):
        if user_id not in self.train_user_encoder:
            return []

        uid = self.train_user_encoder[user_id]

        if uid >= self.train_als.user_factors.shape[0]:
            return []

        user_items = csr_matrix(self.train_matrix[uid])
        ids, scores = self.train_als.recommend(
            uid, user_items, N=n,
            filter_already_liked_items=True
        )

        results = []
        for p in ids:
            p = int(p)
            if p in self.train_product_decoder:
                results.append(self.train_product_decoder[p])
        return results

    def _content_recs(self, user_id, n=20):
        liked = self.df[
            (self.df["user_id"] == user_id) &
            (self.df["rating"] >= 4)
        ]
        if liked.empty:
            return []

        acc = np.zeros(len(self.product_profiles))

        for _, row in liked.iterrows():
            if row["product_id"] not in self.product_idx_map:
                continue
            idx  = self.product_idx_map[row["product_id"]]
            sims = cosine_similarity(
                self.combined_matrix[idx],
                self.combined_matrix
            ).flatten()
            acc += sims * row["rating"]

        rated = set(self.df[self.df["user_id"] == user_id]["product_id"])
        for pid in rated:
            if pid in self.product_idx_map:
                acc[self.product_idx_map[pid]] = 0

        top = acc.argsort()[::-1][:n]
        return [self.product_id_map[i] for i in top if acc[i] > 0]