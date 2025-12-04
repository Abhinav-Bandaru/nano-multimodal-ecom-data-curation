# src/01_download_images.py

import argparse
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

def build_filename(image_id: int, product_id: int) -> str:
    """
    Build the expected filename from image_id and product_id.

    For the Kaggle Multimodal E-Commerce dataset, image files are named like:
        image_<imageid>_product_<productid>.jpg
    """
    return f"image_{image_id}_product_{product_id}.jpg"


def copy_images_for_subset(
    metadata_csv: Path,
    src_images_root: Path,
    dst_images_root: Path,
) -> None:
    """
    For each row in metadata_csv, construct the expected image filename,
    copy it from src_images_root to dst_images_root if it exists,
    and report how many were found / missing.
    """
    print(f"[INFO] Reading subset metadata from: {metadata_csv}")
    df = pd.read_csv(metadata_csv)

    if "imageid" not in df.columns or "productid" not in df.columns:
        raise ValueError(
            "Expected columns 'imageid' and 'productid' in metadata CSV, "
            f"but got: {list(df.columns)}"
        )

    dst_images_root.mkdir(parents=True, exist_ok=True)

    n_rows = len(df)
    n_found = 0
    n_missing = 0

    for _, row in tqdm(df.iterrows(), total=n_rows, desc="Copying images"):
        image_id = row["imageid"]
        product_id = row["productid"]

        filename = build_filename(image_id, product_id)
        src_path = src_images_root / filename
        dst_path = dst_images_root / filename

        if not src_path.exists():
            n_missing += 1
            continue

        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
        n_found += 1

    print(f"[INFO] Total rows in subset: {n_rows}")
    print(f"[INFO] Images found and copied: {n_found}")
    print(f"[INFO] Images missing: {n_missing}")
    print(f"[INFO] Copied images are in: {dst_images_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy sampled images from Kaggle folder into a working directory."
    )

    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="data/sampled/subset_metadata.csv",
        help="Path to the sampled subset metadata CSV.",
    )
    parser.add_argument(
        "--src_images_root",
        type=str,
        default="data/images/images/image_train",
        help="Root folder where the original Kaggle images are stored.",
    )
    parser.add_argument(
        "--dst_images_root",
        type=str,
        default="data/images/sampled",
        help="Destination folder for the images used in this project.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    metadata_csv = Path(args.metadata_csv)
    src_images_root = Path(args.src_images_root)
    dst_images_root = Path(args.dst_images_root)

    copy_images_for_subset(
        metadata_csv=metadata_csv,
        src_images_root=src_images_root,
        dst_images_root=dst_images_root,
    )

