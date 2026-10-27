import os
import sys
import glob
import shutil
import polars as pl
import pyarrow.parquet as pq 

# Configuration variables
INPUT_DIR = "./processed_shards"
PRODUCTION_DIR = "./data_dedup"
FINAL_DATA_SIZE = 2_000_000 
NUM_BUCKETS = 64  
TEMP_DIR = "./dedup_temp_buckets"
BUFFER_DIR = "./buffer_temp"  # NEW: Disk-backed buffer to prevent RAM buildup

def run_partitioned_deduplication():
    os.makedirs(PRODUCTION_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(BUFFER_DIR, exist_ok=True)
        
    input_pattern = os.path.join(INPUT_DIR, "*.parquet")
    all_shards = sorted(glob.glob(input_pattern))
        
    if not all_shards:
        print(f"[ERROR] No shards found in {INPUT_DIR} matching pattern *.parquet!")
        sys.exit(1)
            
    print(f"=== Phase 1: Streaming Partition Bucketing Across {len(all_shards)} Shards ===")
        
    for i, path in enumerate(all_shards, 1):
        try:
            print(f"    [{i}/{len(all_shards)}] Splitting {os.path.basename(path)} into temporary disk buckets...")

            lazy_shard = pl.scan_parquet(path).with_columns(
                bucket=(pl.col("fen").hash(seed=42) % NUM_BUCKETS).abs()
            )
            
            df_partitioned = lazy_shard.collect(engine="streaming")
                        
            for bucket_idx in range(NUM_BUCKETS):
                df_bucket = df_partitioned.filter(pl.col("bucket") == bucket_idx).drop("bucket")
                if len(df_bucket) == 0:
                    continue
                                
                bucket_dir = os.path.join(TEMP_DIR, f"b_{bucket_idx}")
                os.makedirs(bucket_dir, exist_ok=True)
                                
                part_file = os.path.join(bucket_dir, f"part_{i}.parquet")
                df_bucket.write_parquet(part_file, compression="snappy")
                
            # Clear memory immediately after each shard iteration
            del df_partitioned
                        
        except Exception as e:
            print(f"    [WARNING] Error splitting shard {os.path.basename(path)}: {str(e)}")

    print("\n=== Phase 2: Isolated In-Bucket Deduplications ===")
        
    wave_counter = 1
    buffer_file_counter = 0
        
    for bucket_idx in range(NUM_BUCKETS):
        bucket_dir = os.path.join(TEMP_DIR, f"b_{bucket_idx}")
        bucket_pattern = os.path.join(bucket_dir, "*.parquet")
        bucket_files = glob.glob(bucket_pattern)
                
        if not bucket_files:
            continue
                    
        print(f"    Processing isolated Bucket {bucket_idx}/{NUM_BUCKETS - 1}...")
                
        # Deduplicate exactly one bucket partition.
        df_bucket_clean = (
            pl.scan_parquet(bucket_files)
            .sort("depth", descending=True)
            .unique(subset=["fen"], keep="first")
            .collect(engine="streaming")
        )
                
        shutil.rmtree(bucket_dir, ignore_errors=True)
                
        if len(df_bucket_clean) == 0:
            continue
                    
        buf_file = os.path.join(BUFFER_DIR, f"buf_{buffer_file_counter}.parquet")
        df_bucket_clean.write_parquet(buf_file, compression="snappy")
        buffer_file_counter += 1
        
        # Free memory reference immediately
        del df_bucket_clean

        # Check if the cumulative buffered files on disk are ready to yield a full shard
        buffer_pattern = os.path.join(BUFFER_DIR, "*.parquet")
        current_buffer_files = glob.glob(buffer_pattern)
        
        if current_buffer_files:
            # Check size without fully materializing dataframes            
            total_buffered_rows = sum(pq.read_metadata(f).num_rows for f in current_buffer_files)

            while total_buffered_rows >= FINAL_DATA_SIZE:
                print(f"       [BUFFER MATCH] Exporting production Wave {wave_counter}...")
                
                # Stream directly from disk to slice out our wave size safely
                df_stream = pl.scan_parquet(current_buffer_files).collect(engine="streaming")
                
                production_shard = df_stream.slice(0, FINAL_DATA_SIZE)
                production_shard = production_shard.sample(fraction=1.0, shuffle=True, seed=42 + wave_counter)
                
                output_path = os.path.join(PRODUCTION_DIR, f"data_{wave_counter}.parquet")
                production_shard.write_parquet(output_path, compression="snappy")
                
                # Keep the leftovers, save back to disk, wipe old files
                leftovers = df_stream.slice(FINAL_DATA_SIZE, None)
                shutil.rmtree(BUFFER_DIR, ignore_errors=True)
                os.makedirs(BUFFER_DIR, exist_ok=True)
                
                if len(leftovers) > 0:
                    leftovers.write_parquet(os.path.join(BUFFER_DIR, "buf_0.parquet"), compression="snappy")
                    buffer_file_counter = 1
                else:
                    buffer_file_counter = 0
                    
                wave_counter += 1
                
                # Recalculate loop boundaries based on new disk states
                current_buffer_files = glob.glob(os.path.join(BUFFER_DIR, "*.parquet"))
                total_buffered_rows = len(leftovers)
                
                del df_stream, production_shard, leftovers

    # Flush any remaining fragments hanging out on disk
    final_buffer_files = glob.glob(os.path.join(BUFFER_DIR, "*.parquet"))
    if final_buffer_files:
        df_final = pl.scan_parquet(final_buffer_files).collect(engine="streaming")
        if len(df_final) > 0:
            print(f"    Flushing residual buffer of {len(df_final):,} elements...")
            final_shard = df_final.sample(fraction=1.0, shuffle=True, seed=999)
            output_path = os.path.join(PRODUCTION_DIR, f"data_{wave_counter}.parquet")
            final_shard.write_parquet(output_path, compression="snappy")
            wave_counter += 1
        del df_final

    # Clean up temporary disk files
    print("\n -> Cleaning up disk space footprints...")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    shutil.rmtree(BUFFER_DIR, ignore_errors=True)
        
    print(f"\n[SUCCESS] Partitioned Deduplication Complete! Compiled Waves: {wave_counter - 1}")

if __name__ == "__main__":
    run_partitioned_deduplication()
