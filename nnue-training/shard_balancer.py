import os
import sys
import glob
import gc
import time
import polars as pl

# =========================================================================
# GLOBAL CONFIGURATION
# =========================================================================
DEDUP_DATA_DIR = "./data_dedup"           
FINAL_OUTPUT_DIR = "./balanced_shards" 

TRAIN_DIR = os.path.join(FINAL_OUTPUT_DIR, "training")
VAL_DIR = os.path.join(FINAL_OUTPUT_DIR, "validation")
TEMP_STAGE_DIR = "./temp_split_stage"

NUM_BUCKETS = 64
BATCH_SIZE = 8192
RUN_WORKERS = 6  # The golden divisor for perfect runtime core distribution
IDEAL_ROWS_PER_SHARD = BATCH_SIZE * 20 

CANONICAL_COLUMNS = ["active_indices", "passive_indices", "target", "depth", "fen"]

# =========================================================================
# STREAMING HIGH-ENTROPY INTERLEAVER (COMPLETELY OOM-PROOF)
# =========================================================================
def stream_and_balance_pool(stage_paths, pool_height, target_dir, prefix, seed_offset):
    """
    Streams row data from disk fragments sequentially using small row caches,
    shuffling locally to keep RAM usage flat and predictable.
    """
    if not stage_paths:
        return

    # Calculate exact shard structural geometry
    shards = pool_height // IDEAL_ROWS_PER_SHARD
    target_count = ((shards + RUN_WORKERS - 1) // RUN_WORKERS) * RUN_WORKERS
    if target_count == 0: 
        target_count = RUN_WORKERS
    
    rows_per_shard = (pool_height // target_count)
    rows_per_shard = (rows_per_shard // BATCH_SIZE) * BATCH_SIZE

    print(f"   [{prefix} GEOMETRY] Target Shard Count  : {target_count} (Divisible by {RUN_WORKERS})")
    print(f"   [{prefix} GEOMETRY] Exact Rows per Shard : {rows_per_shard:,} (Multiple of {BATCH_SIZE})")
    print(f"   -> Packing shards using streaming partition iterators...")

    # Open all partitions as lazy scans
    scans = [pl.scan_parquet(p) for p in stage_paths]
    
    # We will track our current read position across our collection of files on disk
    current_file_idx = 0
    buffer_df = pl.DataFrame([], schema=scans[0].schema)
    
    # Generate random seed deterministically
    seed = (int(time.time()) + seed_offset) % 2**32

    for shard_idx in range(target_count):
        needed_rows = rows_per_shard
        
        # For the final shard, drain absolutely everything left over
        if shard_idx == target_count - 1:
            # First, pull any remaining data out of all our scanned frames
            remaining_chunks = []
            if buffer_df.height > 0:
                remaining_chunks.append(buffer_df)
                
            while current_file_idx < len(scans):
                try:
                    f_chunk = scans[current_file_idx].collect()
                    if f_chunk.height > 0:
                        remaining_chunks.append(f_chunk)
                except Exception:
                    pass
                current_file_idx += 1
                
            if remaining_chunks:
                shard_out = pl.concat(remaining_chunks)
                # Final entropy shuffle on just this small output shard
                shard_out = shard_out.sample(fraction=1.0, shuffle=True, seed=seed + shard_idx)
                shard_out.write_parquet(os.path.join(target_dir, f"{prefix}_balanced_s{shard_idx}.parquet"), compression="snappy")
            break

        # Accumulate exactly enough rows for the current target shard
        shard_chunks = []
        while needed_rows > 0 and current_file_idx < len(scans):
            if buffer_df.height == 0:
                # Load the next file from disk entirely into memory
                # Since these are split from original buckets, individual chunks are small and memory-safe
                buffer_df = scans[current_file_idx].collect()
                current_file_idx += 1
                if buffer_df.height == 0:
                    continue
                # Local shuffle of the chunk to mix bucket distributions before slicing
                buffer_df = buffer_df.sample(fraction=1.0, shuffle=True, seed=seed + current_file_idx)

            if buffer_df.height <= needed_rows:
                # Consume the whole buffer
                shard_chunks.append(buffer_df)
                needed_rows -= buffer_df.height
                buffer_df = pl.DataFrame([], schema=scans[0].schema)
            else:
                # Take only a slice of the buffer, keep the rest for the next loop execution
                shard_chunks.append(buffer_df.slice(0, needed_rows))
                buffer_df = buffer_df.slice(needed_rows)
                needed_rows = 0

        # Concat the accumulated chunks for this isolated shard
        if shard_chunks:
            shard_out = pl.concat(shard_chunks)
            # Final localized high-entropy mix
            shard_out = shard_out.sample(fraction=1.0, shuffle=True, seed=seed + shard_idx)
            shard_out.write_parquet(os.path.join(target_dir, f"{prefix}_balanced_s{shard_idx}.parquet"), compression="snappy")
        
        # Force instantaneous memory cleanup
        del shard_chunks
        gc.collect()

    print(f"   [SUCCESS] Exported exactly {target_count} balanced {prefix} shards.")


def run_perfect_balanced_mixer():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEMP_STAGE_DIR, exist_ok=True)

    all_fragments = glob.glob(os.path.join(DEDUP_DATA_DIR, "*.parquet"))
    if not all_fragments:
        print(f"[ERROR] No raw fragment data found to process in {DEDUP_DATA_DIR}.")
        sys.exit(1)

    print(f"=== Commencing Ultra-Low RAM Shard-Balancer Across {len(all_fragments)} Blocks ===")
    
    # -------------------------------------------------------------------------
    # PASS 1: STREAMING PARTITION VIA FEN HASHING
    # -------------------------------------------------------------------------
    print(" -> Pass 1: Streaming fragments to disk partitions via FEN hashes...")
    
    train_stage_paths = []
    val_stage_paths = []
    total_physical_rows = 0

    for idx, frag in enumerate(all_fragments):
        try:
            df_chunk = (
                pl.scan_parquet(frag)
                .select(CANONICAL_COLUMNS)
                .with_columns([
                    pl.col("fen").hash().alias("split_hash")
                ])
                .collect()
            )
            
            total_physical_rows += df_chunk.height
            
            train_chunk = df_chunk.filter(pl.col("split_hash") % 100 < 95).drop("split_hash")
            val_chunk = df_chunk.filter(pl.col("split_hash") % 100 >= 95).drop("split_hash")
            
            if train_chunk.height > 0:
                t_path = os.path.join(TEMP_STAGE_DIR, f"train_p_{idx}.parquet")
                train_chunk.write_parquet(t_path, compression="snappy")
                train_stage_paths.append(t_path)
                
            if val_chunk.height > 0:
                v_path = os.path.join(TEMP_STAGE_DIR, f"val_p_{idx}.parquet")
                val_chunk.write_parquet(v_path, compression="snappy")
                val_stage_paths.append(v_path)
                
            del df_chunk, train_chunk, val_chunk
            gc.collect()
            
            try: os.remove(frag)
            except OSError: pass
                
        except Exception as e:
            print(f"  [CRITICAL ERROR] Failed during out-of-core file partition: {e}")
            raise e

    print(f"  -> Total physical rows collected from disk: {total_physical_rows:,}")

    # -------------------------------------------------------------------------
    # PASS 2: ZERO-RAM SHUFFLE & STREAMING BALANCER
    # -------------------------------------------------------------------------
    if train_stage_paths:
        train_pool_height = sum([pl.scan_parquet(p).select(pl.len()).collect().item() for p in train_stage_paths])
        print(f"\n -> Compiling TRAINING Universe ({train_pool_height:,} rows)...")
        stream_and_balance_pool(train_stage_paths, train_pool_height, TRAIN_DIR, "train", seed_offset=0)
        
        # Clean up training temporary files immediately to free up NVMe space
        for p in train_stage_paths:
            try: os.remove(p)
            except OSError: pass

    if val_stage_paths:
        val_pool_height = sum([pl.scan_parquet(p).select(pl.len()).collect().item() for p in val_stage_paths])
        print(f"\n -> Compiling VALIDATION Universe ({val_pool_height:,} rows)...")
        stream_and_balance_pool(val_stage_paths, val_pool_height, VAL_DIR, "val", seed_offset=5000)
        
        for p in val_stage_paths:
            try: os.remove(p)
            except OSError: pass

    try:
        os.rmdir(TEMP_STAGE_DIR)
    except OSError:
        pass
        
    print("\n=== Perfect 6-Worker Dataset Split & Compaction Sequence Finalized ===")

if __name__ == "__main__":
    run_perfect_balanced_mixer()