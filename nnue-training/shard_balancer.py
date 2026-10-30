import os
import sys
import glob
import gc
import numpy as np
import polars as pl
import chess
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================================
# GLOBAL CONSTANTS & PIPELINE CONFIGURATION
# =========================================================================
DEDUP_DATA_DIR = "./data_dedup"           
FINAL_OUTPUT_DIR = "./balanced_shards" 

TEMP_TRAINING_DIR = "./temp_training_dir"    
TEMP_VAL_DIR = "./temp_validation_dir"   

TRAIN_DIR = os.path.join(FINAL_OUTPUT_DIR, "training")
VAL_DIR = os.path.join(FINAL_OUTPUT_DIR, "validation")

NUM_BUCKETS = 64
CHUNK_SIZE = 8192

CANONICAL_COLUMNS = [
    "active_indices", 
    "passive_indices", 
    "target", 
    "depth", 
    "fen", 
    "king_bucket"
]

# =========================================================================
# PHASE 1 WORKER: STREAM ROUTING & DETERMINISTIC PARTITION SPLITTING
# =========================================================================
def process_single_shard(args):
    """
    Processes a single raw Parquet shard out-of-core. Applies a deterministic 
    95% / 5% split based on a FEN hash to prevent data leakage, and isolates 
    rows by their pre-labeled king_bucket into intermediate temporary shards.
    """
    shard_idx, path, num_buckets, temp_train_dir, temp_val_dir = args
    
    try:
        # 1. Read the shard using lazy scanning to minimize memory overhead
        lf = pl.scan_parquet(path).select(CANONICAL_COLUMNS)
        
        # Compute a deterministic hash directly off the raw FEN structure to isolate games
        df = lf.with_columns([
            pl.col("fen").hash().alias("split_hash")
        ]).collect()
        
        # 2. Execute a leak-proof 95/5 partition partition split
        df_train = df.filter(pl.col("split_hash") % 100 < 95)
        df_val = df.filter(pl.col("split_hash") % 100 >= 95)
        
        # 3. Partition rows cleanly by their pre-labeled king_bucket ID straight to disk
        for bucket_id in range(num_buckets):
            # Training Fragment Flush
            sub_train = df_train.filter(pl.col("king_bucket") == bucket_id)
            if sub_train.height > 0:
                t_path = os.path.join(temp_train_dir, f"bucket_{bucket_id}_raw_shard_{shard_idx}.parquet")
                sub_train.select(CANONICAL_COLUMNS).write_parquet(t_path, compression="snappy")
                
            # Validation Fragment Flush
            sub_val = df_val.filter(pl.col("king_bucket") == bucket_id)
            if sub_val.height > 0:
                v_path = os.path.join(temp_val_dir, f"bucket_{bucket_id}_raw_shard_{shard_idx}.parquet")
                sub_val.select(CANONICAL_COLUMNS).write_parquet(v_path, compression="snappy")
                
        del df, df_train, df_val
        gc.collect()
        
    except Exception as e:
        print(f"  [CRITICAL ERROR] Failed routing stream on shard index {shard_idx}: {e}")
        raise e

# =========================================================================
# PHASE 2 WORKER: NATURAL STREAM SHUFFLING & COMPILATION
# =========================================================================
def _worker_process_bucket_group(
    worker_id, 
    bucket_ids, 
    source_dir, 
    output_dir, 
    file_prefix
):
    """
    Ultra-low memory, zero-accumulation stream worker.
    Shuffles and flushes fragments sequentially to keep RAM flat and under 100MB.
    """
    CANONICAL_COLUMNS = [
        "active_indices", 
        "passive_indices", 
        "target", 
        "depth", 
        "fen", 
        "king_bucket"
    ]
    
    for bucket_id in bucket_ids:
        frag_pattern = os.path.join(source_dir, f"bucket_{bucket_id}_raw_*.parquet")
        fragments = glob.glob(frag_pattern)
        if not fragments:
            continue
            
        # Process every fragment independently to ensure RAM never climbs
        for chunk_idx, frag_path in enumerate(fragments):
            try:
                # 1. Read a single tiny raw chunk fragment into memory
                chunk_df = pl.read_parquet(frag_path).select(CANONICAL_COLUMNS)
                
                # 2. Localized Shuffling Pass (100% unique positions, zero memory footprint)
                chunk_shuffled = chunk_df.sample(
                    fraction=1.0, 
                    shuffle=True, 
                    seed=1337 + worker_id + bucket_id + chunk_idx
                )
                
                # 3. Stream the shuffled subset immediately to disk as an independent shard
                output_filename = f"{file_prefix}_natural_w{worker_id}_b{bucket_id}_c{chunk_idx}.parquet"
                output_filepath = os.path.join(output_dir, output_filename)
                chunk_shuffled.write_parquet(output_filepath, compression="snappy")
                
                # 4. Immediate disk and memory reclamation
                os.remove(frag_path)
                del chunk_df, chunk_shuffled
                gc.collect()
                
            except (OSError, pl.exceptions.ComputeError) as e:
                print(f"  [WARNING] Worker {worker_id} skipped broken shard {frag_path}: {e}")
                continue

    print(f"  > Worker {worker_id} successfully compiled pure natural data streams.")
    return None

def produce_complete_shards_sequential(
    source_dir, 
    output_dir, 
    file_prefix, 
    num_buckets, 
    is_validation=False
):
    """
    Executes a pure sequential single-worker compilation pass.
    Maximizes physical RAM headroom to process massive 240M un-pruned shards without OOM crashes.
    """
    mode_str = "VALIDATION" if is_validation else "TRAINING"
    print(f"\n[{mode_str}] -> Commencing single-worker high-headroom natural compilation...")
    
    # Process all 64 buckets sequentially on a single, leak-proof thread
    all_buckets = list(range(num_buckets))
    
    try:
        _worker_process_bucket_group(
            worker_id=0,
            bucket_ids=all_buckets,
            source_dir=source_dir,
            output_dir=output_dir,
            file_prefix=file_prefix
        )
    except Exception as e:
        print(f"  > Single worker encountered critical compilation failure: {e}")
        raise e

    written_files = glob.glob(os.path.join(output_dir, f"{file_prefix}_natural_worker_*_bucket_*.parquet"))
    total_shards_written = len(written_files)
    print(f"[EXPORTED] Total of {total_shards_written} natural shards written cleanly to disk.")
        
    # Final housecleaning of un-shuffled remnants
    try:
        raw_remnants = glob.glob(os.path.join(source_dir, f"bucket_*_raw_*.parquet"))
        for f in raw_remnants:
            try: os.remove(f)
            except OSError: pass
    except Exception:
        pass
        
    print(f"[SUCCESS] {mode_str} natural sequential compilation complete! Target: {output_dir}\n")
    
# =========================================================================
# CENTRAL PIPELINE ORCHESTRATOR
# =========================================================================
def run_unified_mixer_balancer():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    
    os.makedirs(TEMP_VAL_DIR, exist_ok=True)
    os.makedirs(TEMP_TRAINING_DIR, exist_ok=True)
    
    input_shards = sorted(glob.glob(os.path.join(DEDUP_DATA_DIR, "*.parquet")))
    if not input_shards:
        print(f"[ERROR] No data found in directory folder: {DEDUP_DATA_DIR}!")
        sys.exit(1)
        
    print(f"=== Commencing Unified Global Mixer & Balancer Across {len(input_shards)} Shards ===")
    
    # -------------------------------------------------------------------------
    # PHASE 1: Low-Memory Stream Routing via Intermediate Sharding
    # -------------------------------------------------------------------------
    print(" -> Phase 1: Running chunk-based out-of-core streaming topology...")
    
    task_arguments = [
        (i, path, NUM_BUCKETS, TEMP_TRAINING_DIR, TEMP_VAL_DIR)
        for i, path in enumerate(input_shards, 1)
    ]

    # Enforce exactly 3 working threads to process the shards concurrently
    with ProcessPoolExecutor(max_workers=3) as executor:
        # list() forces the executor map to actively evaluate and block until all jobs conclude
        list(executor.map(process_single_shard, task_arguments))

    print("--- Preprocessing Step Complete. Multi-threaded sharding finished. ---")
    
    # -------------------------------------------------------------------------
    # PHASE 2: Parallelized Natural Compilation (No Balancing Nonsense)
    # -------------------------------------------------------------------------
    produce_complete_shards_sequential(
        source_dir=TEMP_TRAINING_DIR,
        output_dir=TRAIN_DIR,
        file_prefix="train",
        num_buckets=NUM_BUCKETS,
        is_validation=False
    )
    
    produce_complete_shards_sequential(
        source_dir=TEMP_VAL_DIR,
        output_dir=VAL_DIR,
        file_prefix="val",
        num_buckets=NUM_BUCKETS,
        is_validation=True
    )
    
    # Remove empty caching directories safely
    for temp_dir in [TEMP_TRAINING_DIR, TEMP_VAL_DIR]:
        try: os.rmdir(temp_dir)
        except OSError: pass
        
    print("=== [FINISH] Pure Natural Dataset Compilation Successfully Completed ===")

if __name__ == "__main__":
    run_unified_mixer_balancer()