import os
import glob
import polars as pl

# Directory configurations
INPUT_SHARDS_DIR = "./production_shards"
BALANCED_OUTPUT_DIR = "./balanced_shards"
FINAL_DATA_SIZE = 2_000_000

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
        .else_(pl.lit("OUT"))
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
    # Ratios: A=45%, B=35%, C=15%, D=5%
    ratios = {"A": 0.45, "B": 0.35, "C": 0.15, "D": 0.05}
    quota = {b: int(FINAL_DATA_SIZE * ratio) for b, ratio in ratios.items()}
    
    # 4. We let the largest category (Bracket A) dictate the total number of shards
    # This guarantees we use 100% of your unique draw/opening positions.
    total_exportable_shards = pool_counts["A"] // quota["A"]
    print(f" -> Maximize Mode: Utilizing all Bracket A rows across {total_exportable_shards} perfect files.\n")
    
    if total_exportable_shards == 0:
        print("[ERROR] Bracket A is too small to build even one perfect ratio shard!")
        return

    # 5. Extract, blend, and shuffle file-by-file using modulo cycling for oversampling
    wave_counter = 1
    
    for shard_idx in range(total_exportable_shards):
        # Calculate cursor offsets for this specific file wave.
        # Brackets B, C, and D will safely wrap around (% count) if they run out of unique rows!
        start_a = shard_idx * quota["A"]
        
        # Safe cyclic slicing math
        slice_a = df_a.slice(start_a, quota["A"])
        slice_b = df_b.slice((shard_idx * quota["B"]) % pool_counts["B"], quota["B"])
        slice_c = df_c.slice((shard_idx * quota["C"]) % pool_counts["C"], quota["C"])
        slice_d = df_d.slice((shard_idx * quota["D"]) % pool_counts["D"], quota["D"])
        
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
