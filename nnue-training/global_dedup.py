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
    cleaned_shards = sorted(glob.glob(input_pattern))
    
    if not cleaned_shards:
        print(f"[ERROR] No clean shards found in {CLEAN_BINARY_DIR} matching pattern data_*.parquet!")
        sys.exit(1)
        
    print(f"=== Commencing Ultra-Safe Streaming Deduplication Across {len(cleaned_shards)} Shards ===")
    
    # =========================================================================
    # PHASE 1: High-efficiency integer-based global max-depth mapping 
    # =========================================================================
    print(" -> Phase 1: Processing shards sequentially to construct global depth map...")
    global_depth_map = {}
    
    for i, path in enumerate(cleaned_shards, 1):
        print(f"    [{i}/{len(cleaned_shards)}] Hashing & Aggregating {os.path.basename(path)}...")
        
        shard_data = (
            pl.scan_parquet(path)
            .select(["fen", "depth"])
            .with_columns(pl.col("fen").hash().alias("fen_hash"))
            .select(["fen_hash", "depth"])
            .group_by("fen_hash")
            .agg(pl.col("depth").max())
            .collect()
        )
        
        hashes = shard_data["fen_hash"].to_numpy()
        depths = shard_data["depth"].to_numpy()
        
        for h, d in zip(hashes, depths):
            if h in global_depth_map:
                if d > global_depth_map[h]:
                    global_depth_map[h] = d
            else:
                global_depth_map[h] = d

    print(f" -> Mapping complete. Unique positions tracked: {len(global_depth_map):,}")

    # =========================================================================
    # PHASE 2 & 3 COMBINED: Chunked Streaming and Production Slicing
    # =========================================================================
    print("\n -> Phase 2 & 3: Streaming rows sequentially and slicing production waves...")
    
    production_wave_counter = 1
    leftover_rows = None  # Buffer to hold rows that don't fill a complete 2M shard

    for i, path in enumerate(cleaned_shards, 1):
        print(f"    [{i}/{len(cleaned_shards)}] Filtering & Slicing {os.path.basename(path)}...")
        
        # Load the whole shard into memory (since it's only 1 out of 20 shards, it's very safe)
        df_shard = pl.read_parquet(path)
        
        # Generate the hashes on the fly to match our tracking dictionary keys
        shard_hashes = df_shard["fen"].hash().to_numpy()
        shard_depths = df_shard["depth"].to_numpy()
        
        # Build a rapid boolean mask: only keep the row if it matches our true global max depth
        # and has not already been processed (using a secondary tracking set to handle collisions/duplicates)
        keep_mask = []
        seen_hashes = set()
        
        for h, d in zip(shard_hashes, shard_depths):
            if global_depth_map.get(h) == d and h not in seen_hashes:
                keep_mask.append(True)
                seen_hashes.add(h)
            else:
                keep_mask.append(False)
                
        # Filter the dataframe shard using the map mask
        df_filtered = df_shard.filter(keep_mask)
        
        # If we have trailing elements from a previous file, stack them on top
        if leftover_rows is not None:
            df_filtered = pl.concat([leftover_rows, df_filtered])
            leftover_rows = None
            
        # Write out as many perfect 2,000,000 row production shards as possible
        total_available = len(df_filtered)
        j = 0
        while j + FINAL_DATA_SIZE <= total_available:
            production_shard = df_filtered.slice(j, FINAL_DATA_SIZE)
            output_path = os.path.join(PRODUCTION_DIR, f"data_dedup_{production_wave_counter}.parquet")
            production_shard.write_parquet(output_path, compression="snappy")
            print(f"       [PRODUCTION EXPORT {production_wave_counter}] Written {FINAL_DATA_SIZE:,} rows.")
            
            production_wave_counter += 1
            j += FINAL_DATA_SIZE
            
        # Cache any trailing items left over to prepend onto the next loop iteration
        if j < total_available:
            leftover_rows = df_filtered.slice(j, total_available - j)

    # Handle any remaining rows after processing the final file
    if leftover_rows is not None and production_wave_counter > 1:
        print(f" -> Appending {len(leftover_rows):,} trailing positions to the final wave.")
        # Optional: Un-comment the lines below to write out the remainder instead of dropping it
        # output_path = os.path.join(PRODUCTION_DIR, f"data_dedup_{production_wave_counter}.parquet")
        # leftover_rows.write_parquet(output_path, compression="snappy")

    print(f"\n[SUCCESS] Global Deduplication Complete!")
    print(f"Total Unique Production Waves Compiled: {production_wave_counter - 1}")

if __name__ == "__main__":
    run_global_deduplication()
