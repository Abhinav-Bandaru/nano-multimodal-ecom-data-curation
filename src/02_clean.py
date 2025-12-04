# src/02_clean.py

from pathlib import Path
import pandas as pd
from tqdm import tqdm


def main():
    metadata_csv = Path("data/sampled/subset_metadata.csv")
    images_dir   = Path("data/images/sampled")
    output_csv  = Path("data/cleaned/cleaned_metadata.csv")

    df = pd.read_csv(metadata_csv)

    valid_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning"):
        image_id   = row["imageid"]
        product_id = row["productid"]

        filename = f"image_{image_id}_product_{product_id}.jpg"
        image_path = images_dir / filename

        if not image_path.exists():
            continue

        # simple text validity check
        title = str(row["designation"]).strip()
        if title == "" or title.lower() == "nan":
            continue

        row["local_image_path"] = str(image_path)
        valid_rows.append(row)

    clean_df = pd.DataFrame(valid_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_csv, index=False)

    print(f"[INFO] Clean rows kept: {len(clean_df)}")
    print(f"[INFO] Saved cleaned metadata to: {output_csv}")


if __name__ == "__main__":
    main()
