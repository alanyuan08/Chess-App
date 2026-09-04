import os
import sys
import glob
import polars as pl

# Configure your incoming quiet/mirrored paths here
DEDUP_DATA = "./data_dedup"
MIXED_PRODUCTION_DIR = "./data_dedup_mixed" 
TEMP_MIX_DIR = "./temp_mixer_shards" 
FINAL_DATA_SIZE = 2_000_000

def run_global_mixer():
    os.makedirs(MIXED_PRODUCTION_DIR, exist_ok=True)
    os.makedirs(TEMP_MIX_DIR, exist_ok=True)
    
    # 1. Discover all Parquet files produced by dedup
    input_pattern = os.path.join(DEDUP_DATA, "*.parquet")
    input_shards = sorted(glob.glob(input_pattern))
    
    if not input_shards:
        print(f"[ERROR] No parquet shards found in {DEDUP_DATA}!")
        sys.exit(1)
        
    print(f"=== Commencing Global Mixer Across {len(input_shards)} Clean Shards ===")
    
    # =========================================================================
    # PASS 1: Assign Pseudo-Random Keys & Distribute into Temp Buckets
    # =========================================================================
    print(" -> Pass 1: Assigning randomized keys and partitioning shards...")
    
    # We will split data randomly across 16 bucket files to keep subsequent sorting chunks small
    NUM_BUCKETS = 16
    
    for i, path in enumerate(input_shards, 1):
        print(f"    [{i}/{len(input_shards)}] Partitioning {os.path.basename(path)}...")
        
        # Read the shard
        df_shard = pl.read_parquet(path)
        
        # Generate an incredibly fast deterministic pseudo-random key by mapping a hash to a bucket index.
        # This breaks up chronological game biases while using zero extra memory.
        df_partitioned = df_shard.with_columns(
            pl.col("fen").hash().alias("mix_hash")
        ).with_columns(
            (pl.col("mix_hash") % NUM_BUCKETS).alias("bucket_id")
        )
        
        # Append partition chunks to our local temp buckets
        for bucket_id in range(NUM_BUCKETS):
            chunk = df_partitioned.filter(pl.col("bucket_id") == bucket_id).drop(["mix_hash", "bucket_id"])
            if len(chunk) > 0:
                bucket_path = os.path.join(TEMP_MIX_DIR, f"bucket_{bucket_id}.parquet")
                
                # If bucket file already exists, read and append, otherwise write new
                if os.path.exists(bucket_path):
                    existing = pl.read_parquet(bucket_path)
                    pl.concat([existing, chunk]).write_parquet(bucket_path, compression="snappy")
                else:
                    chunk.write_parquet(bucket_path, compression="snappy")
                    
    # =========================================================================
    # PASS 2: Locally Shuffle Buckets & Slice into 2M Row Production Waves
    # =========================================================================
    print("\n -> Pass 2: Executing local shuffles and writing production waves...")
    
    production_wave_counter = 1
    leftover_rows = None
    
    bucket_files = glob.glob(os.path.join(TEMP_MIX_DIR, "bucket_*.parquet"))
    
    for i, b_path in enumerate(bucket_files, 1):
        print(f"    [{i}/{len(bucket_files)}] Final Mixing Bucket {os.path.basename(b_path)}...")
        
        df_bucket = pl.read_parquet(b_path)
        
        # Execute an in-memory shuffle on the scaled bucket using a fast random sampler fraction
        # This completely randomizes the position sequence.
        df_shuffled = df_bucket.sample(fraction=1.0, shuffle=True, seed=42)
        
        # Combine with trailing records from the previous bucket if applicable
        if leftover_rows is not None:
            df_shuffled = pl.concat([leftover_rows, df_shuffled])
            leftover_rows = None
            
        # Slice into perfect production sizes
        total_available = len(df_shuffled)
        j = 0
        while j + FINAL_DATA_SIZE <= total_available:
            production_shard = df_shuffled.slice(j, FINAL_DATA_SIZE)
            output_path = os.path.join(MIXED_PRODUCTION_DIR, f"production_wave_{production_wave_counter}.parquet")
            production_shard.write_parquet(output_path, compression="snappy")
            print(f"       [MIX EXPORT {production_wave_counter}] Written {FINAL_DATA_SIZE:,} randomized rows.")
            
            production_wave_counter += 1
            j += FINAL_DATA_SIZE
            
        # Retain trailing components for the next block iteration
        if j < total_available:
            leftover_rows = df_shuffled.slice(j, total_available - j)
            
    # Save absolute remainder into final wave file if valid
    if leftover_rows is not None and production_wave_counter > 1:
        print(f" -> Appending final trailing {len(leftover_rows):,} records to complete the pipeline.")
        output_path = os.path.join(MIXED_PRODUCTION_DIR, f"data_dedup_mixed_{production_wave_counter}.parquet")
        leftover_rows.write_parquet(output_path, compression="snappy")
        production_wave_counter += 1

    # Cleanup temp workspace file allocations 
    print("\n -> Performing system disk cleanup of temporary files...")
    for f in bucket_files:
        try: os.remove(f)
        except: pass
    try: os.rmdir(TEMP_MIX_DIR)
    except: pass
        
    print(f"\n[SUCCESS] Global Position Mixing Complete!")
    print(f"Total Completely Randomized Production Waves Compiled: {production_wave_counter - 1}")

if __name__ == "__main__":
    run_global_mixer()
