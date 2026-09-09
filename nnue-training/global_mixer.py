import os
import sys
import glob
import numpy as np
import polars as pl

# Configure your incoming paths here
DEDUP_DATA = "./data_mirrored"
MIXED_PRODUCTION_DIR = "./production_shards" 
TEMP_MIX_DIR = "./temp_mixer_shards" 
FINAL_DATA_SIZE = 2_000_000

def run_global_mixer():
    os.makedirs(MIXED_PRODUCTION_DIR, exist_ok=True)
    os.makedirs(TEMP_MIX_DIR, exist_ok=True)
    
    input_pattern = os.path.join(DEDUP_DATA, "*.parquet")
    input_shards = sorted(glob.glob(input_pattern))
    
    if not input_shards:
        print(f"[ERROR] No parquet shards found in {DEDUP_DATA}!")
        sys.exit(1)
        
    print(f"=== Commencing True Global Mixer Across {len(input_shards)} Clean Shards ===")
    
    # 64 buckets splits 550M rows into ~8.5M row blocks, which are hyper-fast to sort in RAM
    NUM_BUCKETS = 64
    
    # =========================================================================
    # PASS 1: Buffered Global Hashing & Partitioning into Temp Buckets
    # =========================================================================
    print(" -> Pass 1: Assigning randomized keys and partitioning shards...")
    
    # Initialize an in-memory buffer pool to cache rows before hitting storage
    bucket_buffers = {b_id: [] for b_id in range(NUM_BUCKETS)}
    
    # Flush data down to disk every 20 shards to keep RAM usage lean and safe
    FLUSH_INTERVAL = 20 

    for i, path in enumerate(input_shards, 1):
        print(f"    [{i}/{len(input_shards)}] Partitioning {os.path.basename(path)}...")
        
        df_shard = pl.read_parquet(path)
        
        # Generate a cryptographic-strength hash of the FEN to serve as a sorting index.
        # This completely untethers a position from its original game structure.
        df_partitioned = df_shard.with_columns(
            pl.col("fen").hash().alias("mix_hash")
        ).with_columns(
            (pl.col("mix_hash").abs() % NUM_BUCKETS).alias("bucket_id")
        )
        
        # Append data chunks into the in-memory RAM buffers instead of rewriting files immediately
        for bucket_id in range(NUM_BUCKETS):
            chunk = df_partitioned.filter(pl.col("bucket_id") == bucket_id).drop(["bucket_id"])
            if len(chunk) > 0:
                bucket_buffers[bucket_id].append(chunk)
                
        # --- BATCH FLUSH LOGIC ---
        # Every 20 shards, or on the absolute final shard, we drop the RAM cache down to disk
        if i % FLUSH_INTERVAL == 0 or i == len(input_shards):
            print(f"       [DISK I/O] Flushing accumulated memory cache into bucket files...")
            for bucket_id in range(NUM_BUCKETS):
                chunks_list = bucket_buffers[bucket_id]
                if not chunks_list:
                    continue
                
                # Merge everything collected over the batch cycle in a single operation
                merged_chunk = pl.concat(chunks_list)
                bucket_path = os.path.join(TEMP_MIX_DIR, f"bucket_{bucket_id}.parquet")
                
                if os.path.exists(bucket_path):
                    # Combine existing bucket file with new data block
                    existing = pl.read_parquet(bucket_path)
                    pl.concat([existing, merged_chunk]).write_parquet(bucket_path, compression="snappy")
                else:
                    merged_chunk.write_parquet(bucket_path, compression="snappy")
            
            # Flush memory allocations clean to prevent RAM leaks
            bucket_buffers = {b_id: [] for b_id in range(NUM_BUCKETS)}
                    
    # =========================================================================
    # PASS 2: Explicit Sort on Mix Hash & Slice into 2M Row Waves
    # =========================================================================
    print("\n -> Pass 2: Executing global sort interleaving and writing production waves...")
    
    production_wave_counter = 1
    leftover_rows = None
    
    bucket_files = sorted(glob.glob(os.path.join(TEMP_MIX_DIR, "bucket_*.parquet")))
    
    for i, b_path in enumerate(bucket_files, 1):
        print(f"    [{i}/{len(bucket_files)}] Final Mixing Bucket {os.path.basename(b_path)}...")
        
        df_bucket = pl.read_parquet(b_path)
        
        # Because positions are sorted by their hash values rather than their original file sequence,
        # positions from File 1, File 50, and File 200 interleave perfectly in memory!
        df_shuffled = df_bucket.sort("mix_hash").drop("mix_hash")
        
        # Combine with trailing records from the previous bucket if applicable
        if leftover_rows is not None:
            df_shuffled = pl.concat([leftover_rows, df_shuffled])
            leftover_rows = None
            
        # Slice into perfect production sizes
        total_available = len(df_shuffled)
        j = 0
        while j + FINAL_DATA_SIZE <= total_available:
            production_shard = df_shuffled.slice(j, FINAL_DATA_SIZE)
            output_path = os.path.join(MIXED_PRODUCTION_DIR, f"data_{production_wave_counter}.parquet")
            production_shard.write_parquet(output_path, compression="snappy")
            print(f"       [MIX EXPORT {production_wave_counter}] Written {FINAL_DATA_SIZE:,} randomized rows.")
            
            production_wave_counter += 1
            j += FINAL_DATA_SIZE
            
        if j < total_available:
            leftover_rows = df_shuffled.slice(j, total_available - j)
            
    # Save absolute remainder into final wave file if valid
    if leftover_rows is not None and production_wave_counter > 1:
        print(f" -> Appending final trailing {len(leftover_rows):,} records to complete the pipeline.")
        output_path = os.path.join(MIXED_PRODUCTION_DIR, f"data_{production_wave_counter}.parquet")
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
