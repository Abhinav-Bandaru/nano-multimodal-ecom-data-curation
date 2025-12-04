# src/07_api_server.py
#
# Final Nano-Orbifold API
#
# Endpoints:
#   ✅ POST /search          -> text → closest images (CLIP retrieval)
#   ✅ GET  /record/{id}     -> inspect single curated record
#   ✅ GET  /stats           -> dataset quality statistics

from pathlib import Path
import sqlite3
from typing import List

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import CLIPModel, CLIPTokenizer

# -----------------------------
# Config
# -----------------------------

DB_PATH = Path("db/curated.db")
EMB_PATH = Path("embeddings/curated_embeddings.npz")
MODEL_NAME = "openai/clip-vit-base-patch32"

# -----------------------------
# Load DB
# -----------------------------

print("[API] Loading SQLite DB...")

if not DB_PATH.exists():
    raise FileNotFoundError("Run src/06_store_db.py first – curated DB not found.")

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
df_products = pd.read_sql_query("SELECT * FROM products", conn)

# -----------------------------
# Load embeddings
# -----------------------------

print("[API] Loading embeddings...")

if not EMB_PATH.exists():
    raise FileNotFoundError("Run src/04_curate.py first – curated embeddings not found.")

data = np.load(EMB_PATH)

image_embs = data["image_embs"]
text_embs = data["text_embs"]

if len(df_products) != image_embs.shape[0]:
    raise ValueError("Mismatch between DB rows and embeddings.")

# Normalize embeddings for cosine similarity
image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)
text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

# -----------------------------
# Load CLIP text encoder
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[API] Using device: {device}")

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
clip_tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
clip_model.eval()

# -----------------------------
# Utilities
# -----------------------------

def encode_text(text: str) -> np.ndarray:
    inputs = clip_tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        vec = clip_model.get_text_features(**inputs)[0]

    vec = vec / vec.norm()
    return vec.cpu().numpy()


def cosine_top_k(query_vec, matrix, k=5):
    q = query_vec / np.linalg.norm(query_vec)
    sims = matrix @ q

    k = min(k, matrix.shape[0])
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    return idx, sims[idx]


def fetch_record(embed_index: int):
    row = df_products[df_products["embed_index"] == embed_index]
    if row.empty:
        raise KeyError
    return row.iloc[0].to_dict()


def make_image_url(local_path: str) -> str:
    """
    Convert a local path like 'data/images/sampled/image_...jpg'
    into a URL served by the /images static mount.
    """
    filename = local_path.replace("\\", "/").split("/")[-1]
    return f"/images/{filename}"


# -----------------------------
# API Models
# -----------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class ProductResponse(BaseModel):
    embed_index: int
    productid: int
    imageid: int
    designation: str
    description: str | None
    prdtypecode: int
    local_image_path: str
    alignment_score: float | None
    score: float | None = None
    image_url: str | None = None

class ProductResponseImageURL(BaseModel):
    score: float | None = None
    image_url: str | None = None

# -----------------------------
# FastAPI setup
# -----------------------------

app = FastAPI(title="Nano-Orbifold Multimodal API")

# Serve images from data/images/sampled under /images
app.mount(
    "/images",
    StaticFiles(directory="data/images/sampled"),
    name="images",
)

# -----------------------------
# ✅ ENDPOINTS
# -----------------------------

# 1) SEARCH  (Text → Images)
@app.post("/search", response_model=List[ProductResponseImageURL])
def search(payload: SearchRequest):

    q = payload.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    # Encode text to CLIP vector
    q_vec = encode_text(q)

    # Compare query against IMAGE embeddings
    idxs, scores = cosine_top_k(q_vec, image_embs, payload.top_k)

    results = []
    for idx, s in zip(idxs, scores):
        p = df_products.iloc[idx]
        local_path = str(p["local_image_path"])
        image_url = make_image_url(local_path)

        # results.append(ProductResponse(
        #     embed_index=int(p["embed_index"]),
        #     productid=int(p["productid"]),
        #     imageid=int(p["imageid"]),
        #     designation=str(p["designation"]),
        #     description=str(p["description"]) if p["description"] else None,
        #     prdtypecode=int(p["prdtypecode"]),
        #     local_image_path=local_path,
        #     alignment_score=float(p["alignment_score"]),
        #     score=float(s),
        #     image_url=image_url,
        # ))

        results.append(ProductResponseImageURL(
            alignment_score=float(p["alignment_score"]),
            score=float(s),
            image_url=image_url,
        ))

    return results


# 2) RECORD INSPECTION
@app.get("/record/{embed_index}", response_model=ProductResponse)
def record(embed_index: int):

    try:
        p = fetch_record(embed_index)
    except KeyError:
        raise HTTPException(status_code=404, detail="Record not found")

    local_path = str(p["local_image_path"])
    image_url = make_image_url(local_path)

    return ProductResponse(
        embed_index=int(p["embed_index"]),
        productid=int(p["productid"]),
        imageid=int(p["imageid"]),
        designation=str(p["designation"]),
        description=str(p["description"]) if p["description"] else None,
        prdtypecode=int(p["prdtypecode"]),
        local_image_path=local_path,
        alignment_score=float(p["alignment_score"]),
        score=None,
        image_url=image_url,
    )


# 3) DATASET HEALTH STATS
@app.get("/stats")
def stats():

    label_dist = df_products["prdtypecode"].value_counts().to_dict()

    align = df_products["alignment_score"]
    align_stats = {
        "min": float(align.min()),
        "max": float(align.max()),
        "mean": float(align.mean()),
    }

    return {
        "num_records": int(len(df_products)),
        "label_distribution": label_dist,
        "alignment_score_stats": align_stats,
    }
