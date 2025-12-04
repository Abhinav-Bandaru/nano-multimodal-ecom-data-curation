# src/05_train_or_eval.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def load_embeddings_and_labels(
    metadata_csv: Path,
    emb_path: Path,
    label_col: str,
):
    df = pd.read_csv(metadata_csv)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {metadata_csv}")

    data = np.load(emb_path)
    X = data["combined_embs"]
    y = df[label_col].values

    if len(df) != X.shape[0]:
        raise ValueError("Row count mismatch between metadata and embeddings")

    return X, y


def train_and_eval(X, y, name):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)

    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="macro")

    print(f"\n====== {name} ======")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {f1:.4f}")
    print(f"Train size: {len(X_tr)}, Test size: {len(X_te)}")


def main():

    label_col = "prdtypecode"

    # RAW (cleaned) dataset
    X_raw, y_raw = load_embeddings_and_labels(
        metadata_csv=Path("data/cleaned/cleaned_metadata.csv"),
        emb_path=Path("embeddings/cleaned_embeddings.npz"),
        label_col=label_col,
    )

    # CURATED dataset
    X_cur, y_cur = load_embeddings_and_labels(
        metadata_csv=Path("data/curated/curated_metadata.csv"),
        emb_path=Path("embeddings/curated_embeddings.npz"),
        label_col=label_col,
    )

    train_and_eval(X_raw, y_raw, "RAW (Cleaned)")
    train_and_eval(X_cur, y_cur, "CURATED (Distilled)")


if __name__ == "__main__":
    main()
