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

def get_strata_expressions():
    """
    Returns a unified out-of-core Polars expression for canonical stratification keys,
    perfectly synchronized with your KING_BUCKET_MAP and protecting the geometric 
    purity of your endgame experts.
    """
    # 1. Native popcount replacement for counting remaining pieces
    piece_count_expr = (
        pl.col("fen")
        .str.split(" ")
        .list.get(0)
        .str.replace_all(r"[\d/kK]", "")
        .str.len_chars()
    )
    
    # 2. Extract your un-warped fail-soft Pawn target
    abs_score_expr = pl.col("target").abs()

    # 3. ARCHITECTURAL PHASE ENFORCEMENT:
    # "late" is strictly Bucket 9 AND an active low-piece count endgame phase.
    phase_expr = (
        pl.when((pl.col("king_bucket") == 9) | (piece_count_expr < 8)).then(pl.lit("late"))
        .when((pl.col("king_bucket") == 8) | (piece_count_expr < 26)).then(pl.lit("mid"))
        .otherwise(pl.lit("early"))                            
    )

    # 4. Corrected Monotonic Mapping of Pawn Targets to Real Data Densities
    score_expr = (
        pl.when(abs_score_expr <= 0.30).then(pl.lit("dead_equal"))
        .when(abs_score_expr <= 0.75).then(pl.lit("micro_advantage"))
        .when(abs_score_expr <= 1.50).then(pl.lit("slight_pull"))
        .when(abs_score_expr <= 2.50).then(pl.lit("solid_edge")) 
        .when(abs_score_expr <= 6.00).then(pl.lit("clear_dominance"))    
        .otherwise(pl.lit("dropped")) 
    )

    return (phase_expr + "_" + score_expr).alias("strata_key")

# =========================================================================
# PHASE 1 WORKER: DIRECT MULTI-THREADED SHARD ROUTING
# =========================================================================
def process_single_shard(args):
    """
    Worker function that processes a single shard path using pure vectorized out-of-core Polars.
    Bypasses python map_elements bottlenecks to run entirely at compiled speed.
    """
    i, path, strata_key_expr, num_buckets, temp_train_dir, temp_val_dir = args
    
    # Lazily scan or read the parquet file directly into memory arrays
    df_filtered = pl.read_parquet(path)
    if len(df_filtered) == 0: 
        return
        
    # NATIVE OPTIMIZATION: Extract only the base board placement string component from the FEN 
    # (The first space-separated token) to execute pure vectorized hashing without python-chess overhead.
    base_board_layout = pl.col("fen").str.split(" ").list.get(0)
    
    # Compile hashes natively inside the core engine layout bounds
    df_filtered = df_filtered.with_columns([
        strata_key_expr,
        base_board_layout.hash(seed=999).abs().alias("split_hash"),
        base_board_layout.hash(seed=777).alias("mix_hash")
    ])

    # Assign target hardware expert bucket routes natively
    df_filtered = df_filtered.with_columns(
        (pl.col("mix_hash").abs() % num_buckets).alias("bucket_id")
    )

    # Execute deterministic 98% / 2% split partition filtering
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

# =========================================================================
# PHASE 3 WORKER: STRUCTURAL PROPORTIONAL EXTRACTION
# =========================================================================
def _worker_process_bucket_group(
    worker_id, 
    bucket_ids, 
    source_dir, 
    output_dir, 
    file_prefix, 
    retention_fractions, 
    target_ratios,
    strata_key_expr
):
    """
    Memory-optimized streaming worker process. Dynamically balances both evaluation
    strata and individual King Buckets natively via Polars to ensure uniform feature maps.
    """
    CANONICAL_COLUMNS = [
        "active_indices", 
        "passive_indices", 
        "target", 
        "depth", 
        "fen", 
        "king_bucket"
    ]
    
    # Intended uniform target distribution across your 10 specialized halfKA king buckets
    TARGET_BUCKET_DISTRIBUTION = {i: 0.10 for i in range(10)}
    
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
                
        # 2. Dual-Layer Stratification (Strata Key + King Bucket Re-Balancing)
        bucket_strata = []
        for (strata_k,), sub_df in df_processed.group_by("strata_key"):
            global_fraction = retention_fractions.get(strata_k, 0.0)
            if global_fraction <= 0.0 or strata_k not in target_ratios:
                continue
            
            # Compute the actual localized distribution of your 10 king buckets within this stratum
            bucket_counts = sub_df.group_by("king_bucket").len("count")
            total_sub_rows = sub_df.height
            
            # Convert counts to a fast-lookup dictionary
            local_dist = {
                row["king_bucket"]: row["count"] / total_sub_rows 
                for row in bucket_counts.iter_rows(named=True) if row["king_bucket"] is not None
            }
            
            # Sub-group loop ensures less-visited buckets are protected/boosted
            for (b_id,), bucket_sub_df in sub_df.group_by("king_bucket"):
                if b_id is None or not (0 <= b_id < 10):
                    continue
                
                # Spatial Balancing Logic: Calculate how starved this specific king bucket is
                actual_pct = local_dist.get(b_id, 0.0)
                target_pct = TARGET_BUCKET_DISTRIBUTION.get(b_id, 0.10)
                
                # If actual prevalence is lower than the target 10%, boost its retention fraction
                bucket_modifier = (target_pct / actual_pct) if actual_pct > 0 else 1.0
                
                # Compound fraction merges global evaluation targets with localized bucket balancing
                final_fraction = global_fraction * bucket_modifier
                dynamic_seed = 1337 + bucket_sub_df.height + worker_id + int(b_id)
                
                if final_fraction <= 1.0:
                    # Clean down-sampling via random selection for over-represented king placements
                    bucket_resampled = bucket_sub_df.sample(
                        fraction=final_fraction, 
                        shuffle=True, 
                        seed=dynamic_seed
                    )
                else:
                    # Safe up-sampling via fast native replacement for rare king buckets (e.g. edge files)
                    bucket_resampled = bucket_sub_df.sample(
                        fraction=final_fraction, 
                        with_replacement=True, 
                        shuffle=True, 
                        seed=dynamic_seed
                    )
                    
                bucket_strata.append(bucket_resampled.select(CANONICAL_COLUMNS))
            
        if not bucket_strata:
            del df_processed
            gc.collect()
            continue

        # 3. Consolidate and instantly shuffle this specific bucket group
        df_bucket_pool = pl.concat(bucket_strata)
        df_final_shuffled = df_bucket_pool.sample(fraction=1.0, shuffle=True, seed=42 + worker_id)
        
        # 4. Stream whole bucket shards straight to disk
        output_filename = f"{file_prefix}_worker_{worker_id}_bucket_{bucket_id}.parquet"
        output_filepath = os.path.join(output_dir, output_filename)
        
        df_final_shuffled.write_parquet(output_filepath, compression="snappy")

        # Explicitly release memory allocations before processing the next bucket loop
        del df_processed, df_bucket_pool, df_final_shuffled, bucket_strata
        gc.collect()

    print(f"  > Worker {worker_id} executed successfully with Spatial Bucket Balancing.")
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
    """Orchestrates parallel workers to output entire stratified buckets safely."""
    mode_str = "VALIDATION" if is_validation else "TRAINING"
    print(f"\n[{mode_str}] -> Commencing {num_workers}-worker leak-proof compilation...")
    
    all_buckets = list(range(num_buckets))
    avg = len(all_buckets) // num_workers
    worker_groups = [all_buckets[i * avg : (i + 1) * avg] for i in range(num_workers)]
    
    for r_idx in range(num_workers * avg, len(all_buckets)):
        worker_groups[r_idx % num_workers].append(all_buckets[r_idx])
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                _worker_process_bucket_group, 
                w_id, worker_groups[w_id], source_dir, output_dir, file_prefix, 
                retention_fractions, target_ratios,
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

    written_files = glob.glob(os.path.join(output_dir, f"{file_prefix}_worker_*_bucket_*.parquet"))
    total_shards_written = len(written_files)
    print(f"[EXPORTED] Total of {total_shards_written} dynamic bucket shards written across all workers.")
        
    try:
        raw_remnants = glob.glob(os.path.join(source_dir, f"bucket_*_raw_*.parquet"))
        for f in raw_remnants:
            try: os.remove(f)
            except OSError: pass
    except Exception:
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
    
    strata_key_expr = get_strata_expressions()
    task_arguments = [
        (i, path, strata_key_expr, NUM_BUCKETS, TEMP_TRAINING_DIR, TEMP_VAL_DIR)
        for i, path in enumerate(input_shards, 1)
    ]

    # Enforce exactly 5 working threads to process the shards concurrently
    with ProcessPoolExecutor(max_workers=5) as executor:
        # list() forces the executor map to actively evaluate and block until all jobs conclude
        list(executor.map(process_single_shard, task_arguments))

    print("--- Preprocessing Step Complete. Multi-threaded sharding finished. ---")
    
    # =========================================================================
    # PHASE 2: Dynamic Capacity Scaling & Stratification Reporting (Bucket-Aware)
    # =========================================================================
    print("\n -> Phase 2: Recalculating true strata counts balanced off the 10 Pre-Labeled King Buckets...")
        
    TARGET_RATIOS = {
        # --- EARLY GAME (Openings/Early Middlegame): 15% Total ---
        "early_dead_equal": 0.050, 
        "early_micro_advantage": 0.040, 
        "early_slight_pull": 0.030, 
        "early_solid_edge": 0.020, 
        "early_clear_dominance": 0.010,
        
        # --- MID GAME (Complex/Active Tactical Middlegames): 64.5% Total ---
        "mid_dead_equal": 0.150, 
        "mid_micro_advantage": 0.165,
        "mid_slight_pull": 0.170,      # +1.0% Shifted to consume unique tactical lines
        "mid_solid_edge": 0.120,       
        "mid_clear_dominance": 0.040,
        
        # --- LATE GAME (Pure Geometric Endgame Experts): 20.5% Total ---
        "late_dead_equal": 0.130, 
        "late_micro_advantage": 0.040,
        "late_slight_pull": 0.010,
        "late_solid_edge": 0.023, 
        "late_clear_dominance": 0.002,
    }

    # 1. Gather all file fragments natively into an ultra-fast out-of-core memory tracking layer
    lazy_scans = []
    for bucket_id in range(NUM_BUCKETS):
        frag_pattern = os.path.join(TEMP_TRAINING_DIR, f"bucket_{bucket_id}_raw_*.parquet")
        fragments = glob.glob(frag_pattern)
        if not fragments: continue
        lazy_scans.extend([pl.scan_parquet(f) for f in fragments])

    if not lazy_scans:
        print("[CRITICAL ERROR] No temporary raw files detected inside cache directory!")
        sys.exit(1)

    # 2. High-speed Polars reduction mapping: Consolidates lazy plans and evaluates group counters
    df_global_universe = (
        pl.concat(lazy_scans)
        .with_columns([strata_key_expr])
        .filter(pl.col("strata_key") != "dropped")
        .group_by(["strata_key", "king_bucket"])
        .len("count")
        .collect() 
    )

    # Convert the clean Polars matrix into stable python tracking dictionaries
    bucket_level_telemetry = {k: {i: 0 for i in range(10)} for k in TARGET_RATIOS.keys()}
    true_strata_counts = {k: 0 for k in TARGET_RATIOS.keys()}

    for row in df_global_universe.iter_rows(named=True):
        k = row["strata_key"]
        b_id = row["king_bucket"]  
        cnt = row["count"]
        
        if k in bucket_level_telemetry and b_id is not None and 0 <= b_id < 10:
            bucket_level_telemetry[k][b_id] += cnt
            true_strata_counts[k] += cnt

    del df_global_universe, lazy_scans
    gc.collect()
    
    max_possible_scale = float('inf')
    missing_strata = []
    
    # 4. Flattened Proportional Ceiling Solver Engine (Global Headroom Tracking)
    for k, target_pct in TARGET_RATIOS.items():
        actual_total = true_strata_counts.get(k, 0)
        if actual_total == 0:
            missing_strata.append(k)
            continue
            
        phase_prefix = k.split('_')[0]
        
        # Determine maximum allowed amplification ceiling per game phase
        if phase_prefix == "late":
            max_amp = 1.4
        elif phase_prefix == "mid":
            max_amp = 1.0
        else: # early
            max_amp = 1.0  
        
        # Calculate the ultimate global capacity limit supported by this stratum
        global_scale_limit = (actual_total * max_amp) / target_pct
        if global_scale_limit < max_possible_scale:
            max_possible_scale = global_scale_limit

    # Safe fallback if extreme layout boundaries trigger division edge cases
    if max_possible_scale in (float('inf'), 0) or max_possible_scale < 100_000:
        print("    [WARNING] Fallback engaged. Restricting capacity calculation to strict non-amplified limits...")
        max_possible_scale = float('inf')
        for k, target_pct in TARGET_RATIOS.items():
            actual_total = true_strata_counts.get(k, 0)
            if actual_total == 0: continue
            
            global_scale_limit = actual_total / target_pct
            if global_scale_limit < max_possible_scale:
                max_possible_scale = global_scale_limit

    DYNAMIC_TOTAL_DATASET = int(max_possible_scale)
    
    # Compute precise retention ratios for downstream multi-threaded sampling passes
    retention_fractions = {
        k: (float(target_pct * DYNAMIC_TOTAL_DATASET) / true_strata_counts[k] if true_strata_counts.get(k, 0) > 0 else 0.0)
        for k, target_pct in TARGET_RATIOS.items()
    }

    print(f"    [CEILING SOLVED] Target dataset optimized to {DYNAMIC_TOTAL_DATASET:,} rows based on limiting Key Bucket capacities.")

    # --- AUTOMATED TELEMETRY METRICS DASHBOARD ---
    print("\n======================= STRATIFICATION METRICS REPORT (BUCKET-BALANCED) =======================")
    print(f"{'Strata Key':<30} | {'Raw Count':>12} | {'Target Pct':>10} | {'Target Count':>12} | {'Factor':>8}")
    print("-" * 81)
    
    for k, target_pct in TARGET_RATIOS.items():
        raw_cnt = true_strata_counts.get(k, 0)
        tgt_cnt = int(target_pct * DYNAMIC_TOTAL_DATASET)
        factor = retention_fractions.get(k, 0.0)
        print(f"{k:<30} | {raw_cnt:>12,} | {target_pct*100:>9.1f}% | {tgt_cnt:>12,} | {factor:>7.2f}x")

    # =========================================================================
    # PHASE 3: Streaming Bucket Balancing & Full Block Production
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
        num_workers=5
    )

    produce_complete_shards_parallel(
        source_dir=TEMP_VAL_DIR,
        output_dir=VAL_DIR,
        file_prefix="nnue_val_shard",
        num_buckets=NUM_BUCKETS,
        retention_fractions=retention_fractions,
        target_ratios=TARGET_RATIOS,
        strata_key_expr=strata_key_expr,
        is_validation=True,
        num_workers=5
    )
    print(f"\n[PIPELINE SYSTEM COMPLETE] Co-balanced training sets are ready for the GPU!")

if __name__ == "__main__":
    run_unified_mixer_balancer()