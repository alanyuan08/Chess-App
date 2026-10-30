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
    Memory-optimized parallel stream worker. Consolidated isolated bucket file segments
    and runs a 100% unique natural random shuffle pass without any cloning or dropping.
    """
    for bucket_id in bucket_ids:
        frag_pattern = os.path.join(source_dir, f"bucket_{bucket_id}_raw_*.parquet")
        fragments = glob.glob(frag_pattern)
        if not fragments:
            continue
            
        # Parallel read of fragment arrays
        df_processed = pl.read_parquet(frag_pattern).select(CANONICAL_COLUMNS)
        
        # Instantly discard transient intermediates to protect disk footprint
        for frag_path in fragments:
            try: os.remove(frag_path)
            except OSError: pass
                
        # Pure natural shuffle execution (Fraction=1.0 guarantees zero sampling drops)
        df_final_shuffled = df_processed.sample(
            fraction=1.0, 
            shuffle=True, 
            seed=1337 + worker_id + bucket_id
        )
        
        # Output clean finalized parquet production shard
        output_filename = f"{file_prefix}_natural_worker_{worker_id}_bucket_{bucket_id}.parquet"
        output_filepath = os.path.join(output_dir, output_filename)
        df_final_shuffled.write_parquet(output_filepath, compression="snappy")

        # Explicit release of memory allocations
        del df_processed, df_final_shuffled
        gc.collect()

    print(f"  > Worker {worker_id} successfully compiled pure natural data streams.")
    return None

def produce_complete_shards_parallel(
    source_dir, 
    output_dir, 
    file_prefix, 
    num_buckets, 
    is_validation=False, 
    num_workers=5
):
    """Orchestrates pool executors to process unique bucket sharding lists across cores."""
    mode_str = "VALIDATION" if is_validation else "TRAINING"
    print(f"\n[{mode_str}] -> Commencing {num_workers}-worker natural compilation...")
    
    all_buckets = list(range(num_buckets))
    avg = len(all_buckets) // num_workers
    worker_groups = [all_buckets[i * avg : (i + 1) * avg] for i in range(num_workers)]
    
    for r_idx in range(num_workers * avg, len(all_buckets)):
        worker_groups[r_idx % num_workers].append(all_buckets[r_idx])
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _worker_process_bucket_group, 
                w_id, worker_groups[w_id], source_dir, output_dir, file_prefix
            ): w_id for w_id in range(num_workers)
        }
        
        for future in as_completed(futures):
            w_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  > Worker {w_id} encountered critical failure: {e}")
                raise e

    written_files = glob.glob(os.path.join(output_dir, f"{file_prefix}_natural_worker_*_bucket_*.parquet"))
    total_shards_written = len(written_files)
    print(f"[EXPORTED] Total of {total_shards_written} natural shards written across all workers.")
        
    try:
        raw_remnants = glob.glob(os.path.join(source_dir, f"bucket_*_raw_*.parquet"))
        for f in raw_remnants:
            try: os.remove(f)
            except OSError: pass
    except Exception:
        pass
        
    print(f"[SUCCESS] {mode_str} natural parallel compilation complete! Target: {output_dir}\n")

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

    # Enforce exactly 5 working threads to process the shards concurrently
    with ProcessPoolExecutor(max_workers=5) as executor:
        # list() forces the executor map to actively evaluate and block until all jobs conclude
        list(executor.map(process_single_shard, task_arguments))

    print("--- Preprocessing Step Complete. Multi-threaded sharding finished. ---")
    
    # PHASE 2: Parallelized Natural Compilation (No Balancing Nonsense)
    # -------------------------------------------------------------------------
    produce_complete_shards_parallel(
        source_dir=TEMP_TRAINING_DIR,
        output_dir=TRAIN_DIR,
        file_prefix="train",
        num_buckets=NUM_BUCKETS,
        is_validation=False,
        num_workers=5
    )
    
    produce_complete_shards_parallel(
        source_dir=TEMP_VAL_DIR,
        output_dir=VAL_DIR,
        file_prefix="val",
        num_buckets=NUM_BUCKETS,
        is_validation=True,
        num_workers=5
    )
    
    # Remove empty caching directories safely
    for temp_dir in [TEMP_TRAINING_DIR, TEMP_VAL_DIR]:
        try: os.rmdir(temp_dir)
        except OSError: pass
        
    print("=== [FINISH] Pure Natural Dataset Compilation Successfully Completed ===")

if __name__ == "__main__":
    run_unified_mixer_balancer()