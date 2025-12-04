# src/04_curate.py
#
# Core "distillation" step:
# 1) Filter out low image–text alignment pairs.
# 2) Remove near-duplicates in embedding space.
# 3) Save curated metadata + embeddings.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def main():
    cleaned_csv = Path("data/cleaned/cleaned_metadata.csv")
    emb_path    = Path("embeddings/cleaned_embeddings.npz")

    curated_csv = Path("data/curated/curated_metadata.csv")
    curated_emb = Path("embeddings/curated_embeddings.npz")

    print(f"[INFO] Loading cleaned metadata from: {cleaned_csv}")
    df = pd.read_csv(cleaned_csv)

    print(f"[INFO] Loading embeddings from: {emb_path}")
    data = np.load(emb_path)
    image_embs    = data["image_embs"]       # (N, D)
    text_embs     = data["text_embs"]        # (N, D)
    combined_embs = data["combined_embs"]    # (N, 2D)

    assert len(df) == image_embs.shape[0] == text_embs.shape[0] == combined_embs.shape[0]

    # ---------- 1) Image–text alignment filtering ----------

    # Embeddings were L2-normalized, so dot product = cosine similarity
    alignment_scores = np.sum(image_embs * text_embs, axis=1)

    # Drop the lowest X% of pairs (e.g. bottom 20%) as "low-quality" / misaligned
    keep_quantile = 0.2
    threshold = np.quantile(alignment_scores, keep_quantile)
    keep_mask = alignment_scores >= threshold

    print(f"[INFO] Alignment threshold (keep top {100 - keep_quantile*100:.0f}%): {threshold:.4f}")
    print(f"[INFO] Kept after alignment filter: {keep_mask.sum()} / {len(df)}")

    df_kept          = df[keep_mask].reset_index(drop=True)
    img_kept         = image_embs[keep_mask]
    txt_kept         = text_embs[keep_mask]
    comb_kept        = combined_embs[keep_mask]
    scores_kept      = alignment_scores[keep_mask]

    # ---------- 2) Near-duplicate removal ----------

    # Use combined_embs and cosine distance to find near-duplicates.
    # We treat points whose cosine distance < radius as near-duplicates.
    if len(df_kept) > 1:
        print("[INFO] Running simple duplicate removal in embedding space...")
        nn = NearestNeighbors(
            n_neighbors=2,          # self + nearest neighbor
            metric="cosine",
            algorithm="auto",
        )
        nn.fit(comb_kept)
        distances, indices = nn.kneighbors(comb_kept)

        # distances[:, 0] is 0 (self). Use distances[:, 1] to check nearest neighbor.
        # If distance < dedup_radius → consider it duplicate; drop the later one.
        dedup_radius = 0.03  # cosine distance (~ similarity > 0.97)
        keep_flags = np.ones(len(df_kept), dtype=bool)

        for i in range(len(df_kept)):
            if not keep_flags[i]:
                continue
            # nearest neighbor index (other than self)
            j = indices[i, 1]
            d = distances[i, 1]
            if d < dedup_radius:
                # Mark neighbor as duplicate (drop it)
                keep_flags[j] = False

        print(f"[INFO] Kept after dedup: {keep_flags.sum()} / {len(df_kept)}")

        df_curated     = df_kept[keep_flags].reset_index(drop=True)
        img_curated    = img_kept[keep_flags]
        txt_curated    = txt_kept[keep_flags]
        comb_curated   = comb_kept[keep_flags]
        scores_curated = scores_kept[keep_flags]
    else:
        print("[INFO] Only one sample left after alignment filter; skipping dedup.")
        df_curated     = df_kept
        img_curated    = img_kept
        txt_curated    = txt_kept
        comb_curated   = comb_kept
        scores_curated = scores_kept

    # Attach alignment scores for inspection
    df_curated["alignment_score"] = scores_curated

    # ---------- 3) Save curated metadata + embeddings ----------

    curated_csv.parent.mkdir(parents=True, exist_ok=True)
    df_curated.to_csv(curated_csv, index=False)
    print(f"[INFO] Saved curated metadata to: {curated_csv}")

    curated_emb.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        curated_emb,
        image_embs=img_curated,
        text_embs=txt_curated,
        combined_embs=comb_curated,
        productid=df_curated["productid"].values,
        imageid=df_curated["imageid"].values,
        alignment_score=scores_curated,
    )
    print(f"[INFO] Saved curated embeddings to: {curated_emb}")
    print(f"[INFO] Final curated size: {len(df_curated)}")


if __name__ == "__main__":
    main()