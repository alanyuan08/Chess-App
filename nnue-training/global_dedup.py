import os
import sys
import glob
import polars as pl

# Updated configuration variables
CLEAN_BINARY_DIR = "./data"
PRODUCTION_DIR = "./data_dedup"
FINAL_DATA_SIZE = 2_000_000 

def run_global_deduplication():
    os.makedirs(PRODUCTION_DIR, exist_ok=True)
    
    input_pattern = os.path.join(CLEAN_BINARY_DIR, "*.parquet")
    all_shards = sorted(glob.glob(input_pattern))
    
    if not all_shards:
        print(f"[ERROR] No clean shards found in {CLEAN_BINARY_DIR} matching pattern *.parquet!")
        sys.exit(1)
        
    # --- PRE-FLIGHT CORRUPTION CHECK ---
    print(f" -> Verifying structural integrity of {len(all_shards)} shards...")
    cleaned_shards = []
    for path in all_shards:
        try:
            # Quickly scan metadata to check if the PAR1 footer exists
            pl.scan_parquet(path).slice(0, 1).collect()
            cleaned_shards.append(path)
        except Exception as e:
            print(f"    [WARNING] Skipping corrupt file (Missing PAR1 footer): {os.path.basename(path)}")
            
    if not cleaned_shards:
        print("[ERROR] No structurally valid Parquet files found!")
        sys.exit(1)
        
    print(f"=== Commencing Ultra-Safe Streaming Deduplication Across {len(cleaned_shards)} Healthy Shards ===")
    
    # =========================================================================
    # PHASE 1: High-efficiency global max-depth mapping using Polars
    # =========================================================================
    print(" -> Phase 1: Constructing global unique depth map...")
    
    # We map using native Polars expressions to bypass slow Python loops entirely
    lazy_frames = []
    for path in cleaned_shards:
        lf = (
            pl.scan_parquet(path)
            .select(["fen", "depth"])
            .with_columns(pl.col("fen").hash().alias("fen_hash"))
            .select(["fen_hash", "depth"])
        )
        lazy_frames.append(lf)
        
    # Resolve the entire global depth map out-of-core
    print("    [POLARS ENGINE] Aggregating global max-depth across all sharded tables...")
    global_max_depth = (
        pl.concat(lazy_frames)
        .group_by("fen_hash")
        .agg(pl.col("depth").max().alias("max_depth"))
        .collect(engine="streaming")
    )
    
    print(f" -> Mapping complete. Unique positions tracked: {len(global_max_depth):,}")

    # =========================================================================
    # PHASE 2 & 3: Global Row Selection, Micro-Filtering, and Shuffling
    # =========================================================================
    print("\n -> Phase 2 & 3: Streaming rows via out-of-core join maps...")
    
    production_wave_counter = 1
    collected_shards_buffer = []

    for i, path in enumerate(cleaned_shards, 1):
        print(f"    [{i}/{len(cleaned_shards)}] Filtering & Mapping {os.path.basename(path)}...")
        
        # Stream the shard, match it against our pre-calculated global max depth map
        df_filtered = (
            pl.scan_parquet(path)
            .with_columns(pl.col("fen").hash().alias("fen_hash"))
            # Left join to find rows matching our max-depth criteria
            .join(global_max_depth.lazy(), on="fen_hash", how="left")
            .filter(pl.col("depth") == pl.col("max_depth"))
            # Execute global multi-file duplicate protection by dropping remaining transpositions
            .unique(subset=["fen_hash"], keep="first")
            .drop(["fen_hash", "max_depth"])
            .collect(engine="streaming")
        )
        
        if len(df_filtered) == 0:
            continue
            
        collected_shards_buffer.append(df_filtered)
        current_buffer_rows = sum(len(x) for x in collected_shards_buffer)
        
        # Write out when we gather enough unique rows for production waves
        while current_buffer_rows >= FINAL_DATA_SIZE:
            blended_pool = pl.concat(collected_shards_buffer)
            
            # Slice exactly one wave chunk
            production_shard = blended_pool.slice(0, FINAL_DATA_SIZE)
            
            # Deep random shuffle to shatter consecutive game sequence blocks completely
            production_shard = production_shard.sample(fraction=1.0, shuffle=True, seed=42 + production_wave_counter)
            
            output_path = os.path.join(PRODUCTION_DIR, f"data_{production_wave_counter}.parquet")
            production_shard.write_parquet(output_path, compression="snappy")
            print(f"       [PRODUCTION EXPORT {production_wave_counter}] Written {FINAL_DATA_SIZE:,} globally unique rows.")
            
            # Maintain leftovers via clean out-of-core slice indexing
            leftover_shard = blended_pool.slice(FINAL_DATA_SIZE, None)
            collected_shards_buffer = [leftover_shard] if len(leftover_shard) > 0 else []
            current_buffer_rows = len(leftover_shard)
            production_wave_counter += 1

    # Flush final remaining trailing elements
    if collected_shards_buffer:
        final_shard = pl.concat(collected_shards_buffer)
        if len(final_shard) > 0:
            final_shard = final_shard.sample(fraction=1.0, shuffle=True, seed=999)
            output_path = os.path.join(PRODUCTION_DIR, f"data_{production_wave_counter}.parquet")
            final_shard.write_parquet(output_path, compression="snappy")
            production_wave_counter += 1

    print(f"\n[SUCCESS] Global Deduplication & Serialization Engine Complete!")
    print(f"Total Unique Production Waves Compiled: {production_wave_counter - 1}")

if __name__ == "__main__":
    run_global_deduplication()
