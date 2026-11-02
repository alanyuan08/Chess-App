import os
import sys
import glob
import gc
import numpy as np
import polars as pl
import multiprocessing as mp

# =========================================================================
# GLOBAL CONFIGURATION
# =========================================================================
DEDUP_DATA_DIR = "./data_dedup"           
FINAL_OUTPUT_DIR = "./balanced_shards" 
TEMP_UNIFIED_DIR = "./temp_training_dir"

TRAIN_DIR = os.path.join(FINAL_OUTPUT_DIR, "training")
VAL_DIR = os.path.join(FINAL_OUTPUT_DIR, "validation")

NUM_BUCKETS = 64
BATCH_SIZE = 8192
RUN_WORKERS = 6  # The golden divisor for perfect runtime core distribution
IDEAL_ROWS_PER_SHARD = BATCH_SIZE * 20 

CANONICAL_COLUMNS = ["active_indices", "passive_indices", "target", "depth", "fen",]

# =========================================================================
# HIGH-ENTROPY GLOBAL SHARD BALANCER & SPLITTER
# =========================================================================
def run_perfect_balanced_mixer():
    """
    Consolidates raw bucket fragments, executes a leak-proof 95/5 train/val 
    partition via FEN hashing, shuffles each subset globally, and scales the 
    chunk sizes so that BOTH output directories land exactly on a multiple of 6.
    """
    import time
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    # Step 1: Discover and collect ALL localized fragments across all 64 buckets
    all_fragments = glob.glob(os.path.join(TEMP_UNIFIED_DIR, "bucket_*_raw_*.parquet"))
    if not all_fragments:
        print(f"[ERROR] No raw fragment data found to process in {TEMP_UNIFIED_DIR}.")
        sys.exit(1)

    print(f"=== Commencing Master Shard-Balancer & 95/5 Split Engine Across {len(all_fragments)} Blocks ===")
    print(" -> Consolidating fragments into a unified out-of-core memory pool...")
    
    # Read raw fragments using Polars lazy frames to keep memory flat
    lazy_frames = [pl.scan_parquet(frag).select(CANONICAL_COLUMNS) for frag in all_fragments]
    
    try:
        unified_frames = pl.collect_all(lazy_frames)
        flat_master_df = pl.concat(unified_frames)
        
        # Immediate memory and disk space reclamation
        del lazy_frames, unified_frames
        gc.collect()
        for frag in all_fragments:
            try: os.remove(frag)
            except OSError: pass
    except Exception as e:
        print(f"  [CRITICAL ERROR] Failed during global dataset consolidation: {e}")
        raise e

    print(f"  -> Total physical rows collected from disk: {flat_master_df.height:,}")

    # Step 2: Execute the deterministic 95% / 5% split directly in the graph
    print(" -> Executing deterministic 95/5 partition split based on FEN hashes...")
    flat_master_df = flat_master_df.with_columns([
        pl.col("fen").hash().alias("split_hash")
    ])
    
    train_pool_df = flat_master_df.filter(pl.col("split_hash") % 100 < 95).drop("split_hash")
    val_pool_df = flat_master_df.filter(pl.col("split_hash") % 100 >= 95).drop("split_hash")
    
    del flat_master_df
    gc.collect()

    # Step 3: Compile and Aligns the Training Pool (95%)
    print(f"\n -> Compiling TRAINING Universe ({train_pool_df.height:,} rows)...")
    train_shards = train_pool_df.height // IDEAL_ROWS_PER_SHARD
    train_target_count = ((train_shards + RUN_WORKERS - 1) // RUN_WORKERS) * RUN_WORKERS
    if train_target_count == 0: train_target_count = RUN_WORKERS
    
    train_rows_per_shard = (train_pool_df.height // train_target_count)
    train_rows_per_shard = (train_rows_per_shard // BATCH_SIZE) * BATCH_SIZE

    print(f"   [TRAIN GEOMETRY] Target Shard Count  : {train_target_count} (Divisible by 6)")
    print(f"   [TRAIN GEOMETRY] Exact Rows per Shard : {train_rows_per_shard:,} (Multiple of 8192)")

    print("   -> Executing master global shuffle pass across Training pool...")
    train_seed = int(time.time() * 1000) % 2**32
    shuffled_train_df = train_pool_df.sample(fraction=1.0, shuffle=True, seed=train_seed)
    del train_pool_df
    gc.collect()

    for idx in range(train_target_count):
        offset = idx * train_rows_per_shard
        if idx == train_target_count - 1:
            chunk = shuffled_train_df.slice(offset)
        else:
            chunk = shuffled_train_df.slice(offset, train_rows_per_shard)
            
        chunk.write_parquet(os.path.join(TRAIN_DIR, f"train_balanced_s{idx}.parquet"), compression="snappy")
        
    del shuffled_train_df
    gc.collect()
    print(f"   [SUCCESS] Exported exactly {train_target_count} training shards to {TRAIN_DIR}.")

    # Step 4: Compile and Aligns the Validation Pool (5%)
    print(f"\n -> Compiling VALIDATION Universe ({val_pool_df.height:,} rows)...")
    val_shards = val_pool_df.height // IDEAL_ROWS_PER_SHARD
    val_target_count = ((val_shards + RUN_WORKERS - 1) // RUN_WORKERS) * RUN_WORKERS
    if val_target_count == 0: val_target_count = RUN_WORKERS
    
    val_rows_per_shard = (val_pool_df.height // val_target_count)
    val_rows_per_shard = (val_rows_per_shard // BATCH_SIZE) * BATCH_SIZE

    print(f"   [VAL GEOMETRY] Target Shard Count  : {val_target_count} (Divisible by 6)")
    print(f"   [VAL GEOMETRY] Exact Rows per Shard : {val_rows_per_shard:,} (Multiple of 8192)")

    print("   -> Executing master global shuffle pass across Validation pool...")
    val_seed = (int(time.time() * 1000) % 2**32) + 1
    shuffled_val_df = val_pool_df.sample(fraction=1.0, shuffle=True, seed=val_seed)
    del val_pool_df
    gc.collect()

    for idx in range(val_target_count):
        offset = idx * val_rows_per_shard
        if idx == val_target_count - 1:
            chunk = shuffled_val_df.slice(offset)
        else:
            chunk = shuffled_val_df.slice(offset, val_rows_per_shard)
            
        chunk.write_parquet(os.path.join(VAL_DIR, f"val_balanced_s{idx}.parquet"), compression="snappy")
        
    del shuffled_val_df
    gc.collect()
    print(f"   [SUCCESS] Exported exactly {val_target_count} validation shards to {VAL_DIR}.")
    
    print("\n=== Perfect 6-Worker Dataset Split & Compaction Sequence Finalized ===")

if __name__ == "__main__":
    run_perfect_balanced_mixer()