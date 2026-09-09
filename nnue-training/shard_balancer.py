import os
import glob
import polars as pl

# Directory configurations
INPUT_SHARDS_DIR = "./production_shards"
BALANCED_OUTPUT_DIR = "./balanced_shards"
FINAL_DATA_SIZE = 2_000_000

def get_cyclic_slice(df: pl.DataFrame, shard_idx: int, quota: int, pool_count: int) -> pl.DataFrame:
    """Safely extracts a fixed-length quota from a dataframe, wrapping around to the beginning if necessary."""
    start_idx = (shard_idx * quota) % pool_count
    
    # If the requested quota extends past the end of the dataframe
    if start_idx + quota > pool_count:
        first_part_len = pool_count - start_idx
        second_part_len = quota - first_part_len
        
        # Slice the tail end, then slice the remaining quota from the head
        tail_slice = df.slice(start_idx, first_part_len)
        head_slice = df.slice(0, second_part_len)
        return pl.concat([tail_slice, head_slice])
        
    return df.slice(start_idx, quota)

def run_cyclic_distribution_rebalancer():
    os.makedirs(BALANCED_OUTPUT_DIR, exist_ok=True)
    
    shard_pattern = os.path.join(INPUT_SHARDS_DIR, "data_*.parquet")
    input_files = sorted(glob.glob(shard_pattern))
    
    if not input_files:
        print(f"[ERROR] No input shards discovered in {INPUT_SHARDS_DIR}!")
        return

    print(f"=== Commencing Cyclic Proportional Exporter Across {len(input_files)} Shards ===")
    
    # 1. Scan the full dataset using LazyFrames to separate into independent pools
    lazy_pool = pl.scan_parquet(shard_pattern)
    
    categorized_pool = lazy_pool.with_columns([
        pl.col("target").abs().alias("abs_cp")
    ]).with_columns([
        pl.when(pl.col("abs_cp") <= 150).then(pl.lit("A"))
        .when(pl.col("abs_cp") <= 400).then(pl.lit("B"))
        .when(pl.col("abs_cp") <= 800).then(pl.lit("C"))
        .when(pl.col("abs_cp") <= 1000).then(pl.lit("D"))
        .otherwise(pl.lit("OUT"))
        .alias("bracket")
    ]).filter(pl.col("bracket") != "OUT")

    # 2. Materialize separate clean data frames to act as streaming reservoirs
    print(" -> Pulling bracket datasets into separate memory reservoirs...")
    df_a = categorized_pool.filter(pl.col("bracket") == "A").collect()
    df_b = categorized_pool.filter(pl.col("bracket") == "B").collect()
    df_c = categorized_pool.filter(pl.col("bracket") == "C").collect()
    df_d = categorized_pool.filter(pl.col("bracket") == "D").collect()
    
    pool_counts = {"A": len(df_a), "B": len(df_b), "C": len(df_c), "D": len(df_d)}
    print(f"    Available unique reservoir rows: {pool_counts}")
    
    # 3. Calculate exact row quotas required per 2,000,000-row wave file
    ratios = {"A": 0.45, "B": 0.35, "C": 0.15, "D": 0.05}
    quota = {b: int(FINAL_DATA_SIZE * ratio) for b, ratio in ratios.items()}
    
    # 4. We let the largest category (Bracket A) dictate the total number of shards
    total_exportable_shards = pool_counts["A"] // quota["A"]
    print(f" -> Maximize Mode: Utilizing all Bracket A rows across {total_exportable_shards} perfect files.\n")
    
    if total_exportable_shards == 0:
        print("[ERROR] Bracket A is too small to build even one perfect ratio shard!")
        return

    # 5. Extract, blend, and shuffle file-by-file using modulo cycling for oversampling
    wave_counter = 1
    
    for shard_idx in range(total_exportable_shards):
        # Bracket A does not wrap because total_exportable_shards bounds it perfectly
        slice_a = df_a.slice(shard_idx * quota["A"], quota["A"])
        
        # Use our safe cyclic slicing wrapper for oversampled brackets
        slice_b = get_cyclic_slice(df_b, shard_idx, quota["B"], pool_counts["B"])
        slice_c = get_cyclic_slice(df_c, shard_idx, quota["C"], pool_counts["C"])
        slice_d = get_cyclic_slice(df_d, shard_idx, quota["D"], pool_counts["D"])
        
        # Interleave blend them together into a unified frame
        blended_shard = pl.concat([slice_a, slice_b, slice_c, slice_d])
        
        # Execute an in-memory shuffle so that the brackets don't sit in blocks inside the file
        shuffled_shard = blended_shard.sample(fraction=1.0, shuffle=True, seed=42 + shard_idx)
        
        # Clean up tracking columns before hitting disk storage
        final_shard = shuffled_shard.drop(["abs_cp", "bracket"])
        
        # Export file wave
        output_path = os.path.join(BALANCED_OUTPUT_DIR, f"data_{wave_counter}.parquet")
        final_shard.write_parquet(output_path, compression="snappy")
        
        print(f"  [EXPORT SYSTEM] data_{wave_counter}.parquet [A:45% | B:35% | C:15% | D:5%] -> {len(final_shard):,} rows.")
        
        wave_counter += 1
        
    print(f"\n[SUCCESS] Cyclic Proportional Shard Compilation Complete!")
    print(f"Total Perfectly Balanced Files Standing in Production Pool: {wave_counter - 1}")

if __name__ == "__main__":
    run_cyclic_distribution_rebalancer()
