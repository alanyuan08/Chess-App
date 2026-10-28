import os
import sys
import glob
import gc
import numpy as np
import polars as pl
import chess

from threading import Lock
from concurrent.futures import ProcessPoolExecutor, as_completed

DEDUP_DATA_DIR = "./data_dedup"           
FINAL_OUTPUT_DIR = "./balanced_shards" 

TEMP_TRAINING_DIR = "./temp_training_dir"    
TEMP_VAL_DIR = "./temp_validation_dir"   

TRAIN_DIR = os.path.join(FINAL_OUTPUT_DIR, "training")
VAL_DIR = os.path.join(FINAL_OUTPUT_DIR, "validation")

NUM_BUCKETS = 64
FLUSH_INTERVAL = 20 
FINAL_DATA_SIZE = 2_007_040 
CHUNK_SIZE = 8192
STREAM_STEP = 500_000

def get_strata_expressions():
    piece_count_expr = (
        pl.col("fen")
        .str.split(" ")
        .list.get(0)
        .str.replace_all(r"[\d/kK]", "")
        .str.len_chars()
    )
    abs_score_expr = pl.col("target").abs()

    phase_expr = (
        pl.when(piece_count_expr >= 26).then(pl.lit("early"))
        .when(piece_count_expr >= 13).then(pl.lit("mid"))
        .otherwise(pl.lit("late"))                            
    )

    score_expr = (
        pl.when(abs_score_expr <= 0.45).then(pl.lit("dead_equal"))
        .when(abs_score_expr <= 0.95).then(pl.lit("slight_pull"))
        .when(abs_score_expr <= 1.75).then(pl.lit("micro_advantage"))
        .when(abs_score_expr <= 3.50).then(pl.lit("solid_edge"))
        .when(abs_score_expr <= 6.00).then(pl.lit("clear_dominance"))    
        .otherwise(pl.lit("dropped")) 
    )

    return (phase_expr + "_" + score_expr).alias("strata_key")

def to_canonical_fen(fen_string: str) -> str:
    """
    Finds the horizontal mirror twin of a FEN string.
    Returns min(original_fen, mirrored_fen) alphabetically to form a solid master key.
    """
    try:
        # standard FEN strings usually have 4 components separated by spaces
        # (board, active player, castling rights, en passant target)
        board = chess.Board(fen_string)
        mirrored_board = board.mirror().transform(chess.flip_horizontal)
        fen_mirror = mirrored_board.fen()
        return min(fen_string, fen_mirror)
    except Exception:
        # Fallback safeguard in case an incomplete or custom format FEN enters the pipe
        return fen_string

telemetry_lock = Lock()
global_strata_counts = {}

def process_single_shard(args):
    """Worker function that processes a single shard path using out-of-core Polars."""
    i, path, strata_key_expr, num_buckets, temp_train_dir, temp_val_dir = args
    
    df_shard = pl.read_parquet(path)
    if len(df_shard) == 0: 
        return
        
    df_filtered = df_shard.with_columns(
        pl.col("fen")
        .map_elements(to_canonical_fen, return_dtype=pl.String, strategy="thread_local")
        .alias("canonical_fen")
    )
    
    df_filtered = df_filtered.with_columns([
        strata_key_expr,
        pl.col("canonical_fen").hash(seed=999).abs().alias("split_hash"),
        pl.col("canonical_fen").hash(seed=777).alias("mix_hash")
    ])

    df_filtered = df_filtered.with_columns(
        (pl.col("mix_hash").abs() % num_buckets).alias("bucket_id")
    )

    df_train_chunk = df_filtered.filter(pl.col("split_hash") % 100 < 98)
    df_val_chunk = df_filtered.filter(pl.col("split_hash") % 100 >= 98)

    # Stream Training row allocations directly to raw disk bucket files
    if len(df_train_chunk) > 0:
        for (b_id,), df_sub_bucket in df_train_chunk.group_by("bucket_id"):
            if len(df_sub_bucket) == 0: continue
            frag_path = os.path.join(temp_train_dir, f"bucket_{b_id}_raw_{i}.parquet")
            df_sub_bucket.drop(["bucket_id", "split_hash", "mix_hash"]).write_parquet(frag_path, compression="snappy")

    # Stream Validation row allocations directly to separate raw disk bucket files
    if len(df_val_chunk) > 0:
        for (b_id,), df_sub_bucket in df_val_chunk.group_by("bucket_id"):
            if len(df_sub_bucket) == 0: continue
            frag_path = os.path.join(temp_val_dir, f"bucket_{b_id}_raw_{i}.parquet")
            df_sub_bucket.drop(["bucket_id", "split_hash", "mix_hash"]).write_parquet(frag_path, compression="snappy")

def _worker_process_bucket_group(
    worker_id, 
    bucket_ids, 
    source_dir, 
    output_dir, 
    file_prefix, 
    is_validation, 
    num_buckets, 
    retention_fractions, 
    target_ratios,
    strata_key_expr
):
    """Memory-optimized streaming worker process. Balances and writes entire buckets without drops."""
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
            
        # 1. Native Parallel Read for all matching bucket fragments
        df_processed = pl.read_parquet(frag_pattern)
        df_processed = df_processed.select(CANONICAL_COLUMNS)
        df_processed = df_processed.with_columns([strata_key_expr])
        
        # Immediate disk cleanup of raw fragments to preserve storage space
        for frag_path in fragments:
            try: os.remove(frag_path)
            except OSError: pass
                
        # 2. Stratification and Proportional Extraction Processing
        bucket_strata = []
        for (strata_k,), sub_df in df_processed.group_by("strata_key"):
            fraction = retention_fractions.get(strata_k, 0.0)
            if fraction <= 0.0 or strata_k not in target_ratios:
                continue
                
            target_rows = int(len(sub_df) * fraction)
            
            if fraction <= 1.0:
                target_per_bucket = max(1, target_rows // 10)
                sub_df_slice = (
                    sub_df
                    .sort(by=["king_bucket", "depth"], descending=[False, True])
                    .group_by("king_bucket")
                    .head(target_per_bucket)
                )
                
                dynamic_seed = 1337 + sub_df_slice.shape[0] + worker_id
                sub_df_shuffled = sub_df_slice.sample(fraction=1.0, shuffle=True, seed=dynamic_seed)
                bucket_strata.append(sub_df_shuffled.select(CANONICAL_COLUMNS))
            else:
                sub_df_starved = (
                    sub_df
                    .sort(by=["king_bucket", "depth"], descending=[False, True])
                    .unique(subset=["fen"], keep="first")
                )
                
                dynamic_seed = 1337 + sub_df_starved.shape[0] + worker_id
                sub_df_shuffled = sub_df_starved.sample(fraction=1.0, shuffle=True, seed=dynamic_seed)
                sub_df_shuffled = sub_df_shuffled.select(CANONICAL_COLUMNS)
                
                full_replications = int(fraction)
                remainder_fraction = fraction - full_replications
                
                for _ in range(full_replications):
                    bucket_strata.append(sub_df_shuffled)
                
                remainder_count = int(len(sub_df_shuffled) * remainder_fraction)
                if remainder_count > 0:
                    bucket_strata.append(sub_df_shuffled.head(remainder_count))

        if not bucket_strata:
            del df_processed
            gc.collect()
            continue

        # 3. Consolidate and instantly shuffle this specific bucket group
        df_bucket_pool = pl.concat(bucket_strata)
        df_final_shuffled = df_bucket_pool.sample(fraction=1.0, shuffle=True, seed=42 + worker_id)
        
        # 4. Stream whole bucket shards straight to disk (No rows dropped)
        output_filename = f"{file_prefix}_worker_{worker_id}_bucket_{bucket_id}.parquet"
        output_filepath = os.path.join(output_dir, output_filename)
        
        # Write out every single row generated by the stratification weights
        df_final_shuffled.write_parquet(output_filepath, compression="snappy")

        # Explicitly release memory allocations before processing the next bucket loop
        del df_processed, df_bucket_pool, df_final_shuffled, bucket_strata
        gc.collect()

    print(f"  > Worker {worker_id} executed successfully and flushed whole bucket shards to disk.")
    return None

def produce_complete_shards_parallel(
    source_dir, 
    output_dir, 
    file_prefix, 
    num_buckets, 
    retention_fractions, 
    target_ratios, 
    strata_key_expr,
    is_validation=False, 
    num_workers=4
):
    """Orchestrates parallel workers to output entire stratified buckets without dropping data."""
    mode_str = "VALIDATION" if is_validation else "TRAINING"
    print(f"\n[{mode_str}] -> Commencing {num_workers}-worker leak-proof compilation...")
    
    # 1. Distribute bucket shards evenly across workers
    all_buckets = list(range(num_buckets))
    avg = len(all_buckets) // num_workers
    worker_groups = [all_buckets[i * avg : (i + 1) * avg] for i in range(num_workers)]
    
    for r_idx in range(num_workers * avg, len(all_buckets)):
        worker_groups[r_idx % num_workers].append(all_buckets[r_idx])
    
    # 2. Fire multi-processed worker executors
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _worker_process_bucket_group, 
                w_id, worker_groups[w_id], source_dir, output_dir, file_prefix, 
                is_validation, num_buckets, retention_fractions, target_ratios,
                strata_key_expr
            ): w_id for w_id in range(num_workers)
        }
        
        for future in as_completed(futures):
            w_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  > Worker {w_id} encountered critical failure: {e}")
                raise e

    # 3. Post-Process Scan for telemetry tracking
    written_files = glob.glob(os.path.join(output_dir, f"{file_prefix}_worker_*_bucket_*.parquet"))
    total_shards_written = len(written_files)

    print(f"[EXPORTED] Total of {total_shards_written} dynamic bucket shards written across all workers.")
        
    # 4. Clean up source directory now that workers have detached
    try:
        raw_remnants = glob.glob(os.path.join(source_dir, "*.parquet"))
        for f in raw_remnants:
            os.remove(f)
        os.rmdir(source_dir)
    except OSError:
        pass
        
    print(f"[SUCCESS] {mode_str} stream parallel compilation complete! Target: {output_dir}\n")

def run_unified_mixer_balancer():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    
    os.makedirs(TEMP_VAL_DIR, exist_ok=True)
    os.makedirs(TEMP_TRAINING_DIR, exist_ok=True)
    
    input_shards = sorted(glob.glob(os.path.join(DEDUP_DATA_DIR, "*.parquet")))
    if not input_shards:
        print(f"[ERROR] No data found in {DEDUP_DATA_DIR}!")
        sys.exit(1)
        
    print(f"=== Commencing Unified Global Mixer & Balancer Across {len(input_shards)} Shards ===")
    
    # -------------------------------------------------------------------------
    # PHASE 1: Low-Memory Stream Routing via Intermediate Sharding
    # -------------------------------------------------------------------------
    print(" -> Phase 1: Running chunk-based out-of-core streaming topology...")
        
    # Pre-calculated expression to dynamically parse piece counts excluding kings
    piece_count_expr = (
        pl.col("fen")
        .str.split(" ")
        .list.get(0)
        .str.replace_all(r"[\d/kK]", "")
        .str.len_chars()
    )
    
    strata_key_expr = get_strata_expressions()
    # Bundle execution constants to pass safely into the thread workers
    task_arguments = [
        (i, path, strata_key_expr, NUM_BUCKETS, TEMP_TRAINING_DIR, TEMP_VAL_DIR)
        for i, path in enumerate(input_shards, 1)
    ]

    # Enforce exactly 4 working threads to process the shards concurrently
    with ProcessPoolExecutor(max_workers=4) as executor:
        # list() forces the executor map to actively evaluate and block until all jobs conclude
        list(executor.map(process_single_shard, task_arguments))

    print("--- Preprocessing Step Complete. Multi-threaded sharding finished. ---")
    
    # =========================================================================
    # PHASE 2: Dynamic Capacity Scaling & Stratification Reporting
    # Bridges global bucketing matrices with the downstream block exporter
    # =========================================================================
    print("\n -> Phase 2: Recalculating true strata counts and optimizing global scale ceilings...")
    
    # 1. Re-scan the freshly deduplicated training buckets to clear out-of-core telemetry
    true_strata_counts = {}
    
    for bucket_id in range(NUM_BUCKETS):
        frag_pattern = os.path.join(TEMP_TRAINING_DIR, f"bucket_{bucket_id}_raw_*.parquet")
        fragments = glob.glob(frag_pattern)
        if not fragments: continue
        
        # Read the files natively, compute the strata_key column 
        # on the fly in memory via expressions, and then run your group aggregations!
        df_counts = (
            pl.concat([pl.read_parquet(f) for f in fragments])
            .with_columns([strata_key_expr])
            .group_by("strata_key")
            .len("count")
        )
        
        for row in df_counts.iter_rows(named=True):
            k = row["strata_key"]
            if "_dropped" in k: continue
            true_strata_counts[k] = true_strata_counts.get(k, 0) + row["count"]

        del df_counts

    # 2. Define stratified amplification limits by phase to eliminate over-fitting risks
    AMPLIFICATION_LIMITS = {
        "early": 1.0,
        "mid": 1.5,
        "late": 1.5
    }

    TARGET_RATIOS = {
        "early_dead_equal": 0.095,
        "early_slight_pull": 0.075,
        "early_micro_advantage": 0.045,
        "early_solid_edge": 0.018, 
        "early_clear_dominance": 0.010,
        
        "mid_dead_equal": 0.185,
        "mid_slight_pull": 0.165,
        "mid_micro_advantage": 0.110,
        "mid_solid_edge": 0.055, 
        "mid_clear_dominance": 0.015,
        
        "late_dead_equal": 0.125, 
        "late_slight_pull": 0.045, 
        "late_micro_advantage": 0.022, 
        "late_solid_edge": 0.025, 
        "late_clear_dominance": 0.010,
    }
        
    max_possible_scale = float('inf')
    missing_strata = []
    
    for k, target_pct in TARGET_RATIOS.items():
        actual_count = true_strata_counts.get(k, 0)
        
        # Track if a required stratum is missing to raise an operational warning
        if actual_count == 0: 
            missing_strata.append(k)
            continue
            
        # --- Isolate the first element 'early', 'mid', or 'late' ---
        phase_prefix = k.split('_')[0]
        max_amp = AMPLIFICATION_LIMITS.get(phase_prefix, 3.0)
            
        # Dynamically scale global scale limits around the clean phase metrics
        scale_limit = (actual_count * max_amp) / target_pct
        if scale_limit < max_possible_scale:
            max_possible_scale = scale_limit

    # Handle the catastrophic empty data edge-case gracefully
    if max_possible_scale == float('inf'):
        print("[CRITICAL ERROR] All evaluated strata shapes returned 0 rows! Verify Phase 1 file loading paths.")
        sys.exit(1)
        
    if missing_strata:
        print(f"    [WARNING] The following strata targets were entirely absent from data pool: {missing_strata}")

    DYNAMIC_TOTAL_DATASET = int(max_possible_scale)
    retention_fractions = {}
    
    for k, target_pct in TARGET_RATIOS.items():
        desired = int(target_pct * DYNAMIC_TOTAL_DATASET)
        actual = true_strata_counts.get(k, 0)
        # Type-safe fraction generation
        retention_fractions[k] = desired / actual if actual > 0 else 0.0

    print(f"    [CEILING SOLVED] Target dataset optimized to {DYNAMIC_TOTAL_DATASET:,} rows based on limiting stratum capacity.")
    
    # --- AUTOMATED TELEMETRY METRICS DASHBOARD ---
    print("\n======================= STRATIFICATION METRICS REPORT (POST-REDUCE) =======================")
    print(f"{'Strata Key':<30} | {'Raw Count':>12} | {'Target Pct':>10} | {'Target Count':>12} | {'Factor':>8}")
    print("-" * 81)
    
    for k, target_pct in TARGET_RATIOS.items():
        raw_cnt = true_strata_counts.get(k, 0)
        tgt_cnt = int(target_pct * DYNAMIC_TOTAL_DATASET)
        factor = retention_fractions.get(k, 0.0)
        print(f"{k:<30} | {raw_cnt:>12,} | {target_pct*100:>9.1f}% | {tgt_cnt:>12,} | {factor:>7.2f}x")

    # =========================================================================
    # PHASE 3: Streaming Bucket Balancing & Strict Block Production
    # =========================================================================
    
    print("\n -> Phase 3: Stream Bucket Balancing...")
    produce_complete_shards_parallel(
        source_dir=TEMP_TRAINING_DIR,
        output_dir=TRAIN_DIR,
        file_prefix="nnue_train_shard",
        num_buckets=NUM_BUCKETS,
        retention_fractions=retention_fractions,
        target_ratios=TARGET_RATIOS,
        strata_key_expr=strata_key_expr,
        is_validation=False,
        num_workers=4
    )
    
    # Run Part B: Materialize validation shards
    produce_complete_shards_parallel(
        source_dir=TEMP_VAL_DIR,
        output_dir=VAL_DIR,
        file_prefix="nnue_val_shard",
        num_buckets=NUM_BUCKETS,
        retention_fractions=retention_fractions,
        target_ratios=TARGET_RATIOS,
        strata_key_expr=strata_key_expr,
        is_validation=True,
        num_workers=4
    )
    
    print(f"\n[PIPELINE SYSTEM COMPLETE] Co-balanced training sets are ready for the GPU!")

if __name__ == "__main__":
    run_unified_mixer_balancer()