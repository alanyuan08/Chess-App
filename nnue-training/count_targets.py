import polars as pl
from pathlib import Path

# Configuration
DATA_DIR = "./data_dedup"          # Folder containing your .parquet files
TARGET_COL = "target"        # Change this if your label column has a different name

def count_binary_targets(data_dir: str, target_col: str):
    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        print(f"No .parquet files found in '{data_dir}'")
        return

    print(f"Scanning {len(files)} Parquet files in '{data_dir}'...")

    stats = (
        pl.scan_parquet(f"{data_dir}/*.parquet")
        .select([
            pl.len().alias("total"),
            pl.col(target_col).cast(pl.Int64).sum().alias("ones")
        ])
        .collect(engine="streaming")
    )

    total = stats["total"][0]
    ones = stats["ones"][0] or 0
    zeros = total - ones

    pct_zeros = (zeros / total * 100) if total > 0 else 0.0
    pct_ones = (ones / total * 100) if total > 0 else 0.0

    print(f"\nTotal rows : {total:,}")
    print(f"target = 0 : {zeros:,} ({pct_zeros:.4f}%)")
    print(f"target = 1 : {ones:,} ({pct_ones:.4f}%)")

if __name__ == "__main__":
    count_binary_targets(DATA_DIR, TARGET_COL)