# src/03_embed.py

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel


def load_model(model_name: str = "openai/clip-vit-base-patch32"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return model, processor, device


def build_text(row) -> str:
    """Combine title + description into a single text string."""
    title = str(row.get("designation", "") or "")
    desc = str(row.get("description", "") or "")
    text = (title + " " + desc).strip()
    return text if text else title


def main():
    cleaned_csv = Path("data/cleaned/cleaned_metadata.csv")
    output_path = Path("embeddings/cleaned_embeddings.npz")

    df = pd.read_csv(cleaned_csv)
    print(f"[INFO] Loaded cleaned metadata: {cleaned_csv}, rows={len(df)}")

    model, processor, device = load_model()

    image_emb_list = []
    text_emb_list = []

    batch_size = 32  # safe for 8GB GPU; adjust if needed

    for start in tqdm(range(0, len(df), batch_size), desc="Embedding"):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end]

        image_paths = batch["local_image_path"].tolist()
        texts = [build_text(row) for _, row in batch.iterrows()]

        images = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            images.append(img)

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            img_emb = outputs.image_embeds  # (B, D)
            txt_emb = outputs.text_embeds   # (B, D)

            # L2-normalize (standard for CLIP)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        image_emb_list.append(img_emb.cpu().numpy())
        text_emb_list.append(txt_emb.cpu().numpy())

    image_embs = np.concatenate(image_emb_list, axis=0)
    text_embs = np.concatenate(text_emb_list, axis=0)

    # simple fusion: concatenate image + text embeddings
    combined_embs = np.concatenate([image_embs, text_embs], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image_embs=image_embs,
        text_embs=text_embs,
        combined_embs=combined_embs,
        productid=df["productid"].values,
        imageid=df["imageid"].values,
    )

    print(f"[INFO] Saved embeddings to: {output_path}")
    print(f"[INFO] image_embs shape: {image_embs.shape}")
    print(f"[INFO] text_embs shape: {text_embs.shape}")
    print(f"[INFO] combined_embs shape: {combined_embs.shape}")


if __name__ == "__main__":
    main()
