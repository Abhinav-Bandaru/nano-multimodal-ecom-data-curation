import argparse
from pathlib import Path

import pandas as pd


def sample_metadata(
    raw_csv_path: Path,
    output_csv_path: Path,
    sample_size: int = 8000,
    random_state: int = 42,
) -> None:
    """
    Read the full Kaggle Multimodal E-Commerce metadata CSV,
    randomly sample N rows, and save them as a smaller CSV.

    We keep ALL columns as-is; later steps will decide what they need.
    """

    print(f"[INFO] Reading raw metadata from: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)

    n_total = len(df)
    if n_total == 0:
        raise ValueError("Raw metadata CSV is empty.")
    
    n_sample = min(sample_size, n_total)
    print(f"[INFO] Total rows in raw metadata: {n_total}")
    print(f"[INFO] Sampling {n_sample} rows...")

    df_sampled = df.sample(n=n_sample, random_state=random_state).reset_index(drop=True)

    # Make sure output directory exists
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_sampled.to_csv(output_csv_path, index=False)
    print(f"[INFO] Saved sampled metadata to: {output_csv_path}")
    print(df_sampled.head())



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a subset of the Multimodal E-Commerce metadata."
    )

    parser.add_argument(
        "--raw_csv",
        type=str,
        default="data/raw/X_train_update.csv",
        help="Path to the full Kaggle metadata CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/sampled/subset_metadata.csv",
        help="Where to save the sampled subset CSV.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=8000,
        help="Number of rows to sample from the full metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    raw_csv_path = Path(args.raw_csv)
    output_csv_path = Path(args.output_csv)

    sample_metadata(
        raw_csv_path=raw_csv_path,
        output_csv_path=output_csv_path,
        sample_size=args.sample_size,
        random_state=args.seed,
    )