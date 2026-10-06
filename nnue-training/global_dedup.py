import os
import sys
import glob
import polars as pl

# Configuration variables
INPUT_DIR = "./processed_shards"
PRODUCTION_DIR = "./data_dedup"
FINAL_DATA_SIZE = 2_000_000 
NUM_BUCKETS = 64  # Divides memory footprint by 64 out-of-core
TEMP_DIR = "./dedup_temp_buckets"

def get_bucket_id(fen_series: pl.Series) -> pl.Series:
    # Use native fast hashing to generate absolute bucket assignments
    return (fen_series.hash(seed=42) % NUM_BUCKETS).abs()

def run_partitioned_deduplication():
    os.makedirs(PRODUCTION_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    input_pattern = os.path.join(INPUT_DIR, "*.parquet")
    all_shards = sorted(glob.glob(input_pattern))
    
    if not all_shards:
        print(f"[ERROR] No shards found in {INPUT_DIR} matching pattern *.parquet!")
        sys.exit(1)
        
    print(f"=== Phase 1: Streaming Partition Bucketing Across {len(all_shards)} Shards ===")
    
    # Process each raw shard out-of-core and dump them into append-only bucket streams
    for i, path in enumerate(all_shards, 1):
        try:
            print(f"    [{i}/{len(all_shards)}] Splitting {os.path.basename(path)} into temporary disk buckets...")
            lazy_shard = pl.scan_parquet(path)
            
            # Map the rows to their respective bucket id chunk in streaming fashion
            df_partitioned = (
                lazy_shard
                .with_columns(bucket=get_bucket_id(pl.col("fen")))
                .collect(engine="streaming")
            )
            
            # Append rows into their respective hash bucket directory structure
            for bucket_idx in range(NUM_BUCKETS):
                df_bucket = df_partitioned.filter(pl.col("bucket") == bucket_idx).drop("bucket")
                if len(df_bucket) == 0:
                    continue
                
                bucket_dir = os.path.join(TEMP_DIR, f"b_{bucket_idx}")
                os.makedirs(bucket_dir, exist_ok=True)
                
                # Write an isolated partition slice file
                part_file = os.path.join(bucket_dir, f"part_{i}.parquet")
                df_bucket.write_parquet(part_file, compression="snappy")
                
        except Exception as e:
            print(f"    [WARNING] Error splitting shard {os.path.basename(path)}: {str(e)}")

    print("\n=== Phase 2: Isolated In-Bucket Deduplications ===")
    
    wave_counter = 1
    collected_buffer = None
    
    for bucket_idx in range(NUM_BUCKETS):
        bucket_dir = os.path.join(TEMP_DIR, f"b_{bucket_idx}")
        bucket_pattern = os.path.join(bucket_dir, "*.parquet")
        bucket_files = glob.glob(bucket_pattern)
        
        if not bucket_files:
            continue
            
        print(f"    Processing isolated Bucket {bucket_idx}/{NUM_BUCKETS - 1}...")
        
        # Deduplicate exactly one bucket partition. 
        # Safe to run unique operations because duplicates are trapped in the same bucket!
        df_bucket_clean = (
            pl.scan_parquet(bucket_files)
            .sort("depth", descending=True)
            .unique(subset=["fen"], keep="first")
            .collect(engine="streaming")
        )
        
        if len(df_bucket_clean) == 0:
            continue
            
        if collected_buffer is None:
            collected_buffer = df_bucket_clean
        else:
            collected_buffer = pl.concat([collected_buffer, df_bucket_clean])
            
        # Flush full production chunks when buffer hits target
        while len(collected_buffer) >= FINAL_DATA_SIZE:
            print(f"       [BUFFER MATCH] Exporting production Wave {wave_counter}...")
            production_shard = collected_buffer.slice(0, FINAL_DATA_SIZE)
            production_shard = production_shard.sample(fraction=1.0, shuffle=True, seed=42 + wave_counter)
            
            output_path = os.path.join(PRODUCTION_DIR, f"data_{wave_counter}.parquet")
            production_shard.write_parquet(output_path, compression="snappy")
            
            collected_buffer = collected_buffer.slice(FINAL_DATA_SIZE, None)
            wave_counter += 1

    # Flush any remaining fragments hanging out in the buffer
    if collected_buffer is not None and len(collected_buffer) > 0:
        print(f"    Flushing residual buffer of {len(collected_buffer):,} elements...")
        final_shard = collected_buffer.sample(fraction=1.0, shuffle=True, seed=999)
        output_path = os.path.join(PRODUCTION_DIR, f"data_{wave_counter}.parquet")
        final_shard.write_parquet(output_path, compression="snappy")
        wave_counter += 1

    # Clean up temporary disk files
    print("\n -> Cleaning up disk space footprints...")
    import shutil
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    print(f"\n[SUCCESS] Partitioned Deduplication Complete! Compiled Waves: {wave_counter - 1}")

if __name__ == "__main__":
    run_partitioned_deduplication()
