import os
import glob
import polars as pl
import hashlib
import shutil

INPUT_SHARDS_DIR = "./production_shards"
BALANCED_OUTPUT_DIR = "./balanced_shards"
NUM_OUTPUT_SHARDS = 200

def get_strata_and_filter_expressions():
    """
    Translates the updated centipawn stratification rules natively into Polars expressions.
    Filters out extreme blunders (>1000 cp) and scales categories appropriately.
    """
    # Optimized string parsing expression for piece counts
    piece_count_expr = (
        pl.col("fen")
        .str.split(" ")
        .list.get(0)
        .str.replace_all(r"[\d/kK]", "")
        .str.len_chars()
    )
    abs_score_expr = pl.col("target").abs()
    
    # Establish a high-fidelity depth floor to completely filter out shallow engine noise
    is_valid_depth = pl.col("depth") >= 24
    
    phase_expr = (
        pl.when(piece_count_expr >= 26).then(pl.lit("early"))
        .when(piece_count_expr <= 14).then(pl.lit("late"))
        .otherwise(pl.lit("mid"))
    )
    
    score_expr = (
        pl.when(abs_score_expr <= 0.4).then(pl.lit("dead_equal"))
        .when(abs_score_expr <= 1.2).then(pl.lit("slight_pull"))
        .when(abs_score_expr <= 2.2).then(pl.lit("solid_edge"))
        .when(abs_score_expr <= 4.0).then(pl.lit("clear_dominance"))
        .when(abs_score_expr <= 6.0).then(pl.lit("decisive_minor"))
        .when(abs_score_expr <= 10.0).then(pl.lit("decisive_major"))
        .otherwise(pl.lit("dropped"))
    )

    strata_key_expr = (phase_expr + "_" + score_expr).alias("strata_key")
    
    return is_valid_depth, strata_key_expr

def run_precision_matrix_shuffler_low_mem():
    os.makedirs(BALANCED_OUTPUT_DIR, exist_ok=True)
    
    shard_pattern = os.path.join(INPUT_SHARDS_DIR, "data_*.parquet")
    input_files = sorted(glob.glob(shard_pattern))
    
    if not input_files:
        print("[ERROR] No input shards discovered!")
        return

    print("=== Commencing Depth-Saturated Stability Stratification Router ===")
    
    TEMP_BUFFER_DIR = "./temp_shard_buffers"
    shutil.rmtree(TEMP_BUFFER_DIR, ignore_errors=True)
    os.makedirs(TEMP_BUFFER_DIR, exist_ok=True)
    
    for b_idx in range(NUM_OUTPUT_SHARDS):
        os.makedirs(os.path.join(TEMP_BUFFER_DIR, f"shard_{b_idx}"), exist_ok=True)

    is_valid_depth, strata_key_expr = get_strata_and_filter_expressions()
    
    # =========================================================================
    # STEP 1: INCREMENTAL METADATA ANALYSIS (TRACK BUCKET + DEPTH)
    # =========================================================================
    print("-> Analyzing dataset bucket and depth distributions incrementally...")
    global_depth_counts = {}  # Key: strata_key, Value: {depth: count}
    
    for file_path in input_files:
        chunk_counts = (
            pl.scan_parquet(file_path)
            .filter((pl.col("target").abs() <= 1000) & is_valid_depth)
            .select([strata_key_expr, pl.col("depth"), pl.col("fen")])
            .unique(subset=["fen"], keep="first")
            .group_by(["strata_key", "depth"])
            .len("count")
            .collect(engine="streaming") 
        )
        
        for row in chunk_counts.iter_rows(named=True):
            k = row["strata_key"]
            d = row["depth"]
            c = row["count"]
            if k == "early_dropped" or k == "mid_dropped" or k == "late_dropped":
                continue
            if k not in global_depth_counts:
                global_depth_counts[k] = {}
            global_depth_counts[k][d] = global_depth_counts[k].get(d, 0) + c

    # Performance-first distribution targets (Total: 100M positions baseline reference)
    target_ratios = {
        # --- PHASE 1: EARLY GAME (Total: 0.300) ---
        "early_dead_equal":       0.045,  # 0 - 40 cp
        "early_slight_pull":      0.090,  # 40 - 120 cp
        "early_solid_edge":       0.075,  # 120 - 220 cp
        "early_clear_dominance":  0.054,  # 220 - 400 cp
        "early_decisive_minor":   0.024,  # 400 - 600 cp
        "early_decisive_major":   0.012,  # 600 - 1000 cp

        # --- PHASE 2: MID GAME (Total: 0.400) ---
        "mid_dead_equal":         0.060,
        "mid_slight_pull":        0.120,
        "mid_solid_edge":         0.100,
        "mid_clear_dominance":    0.072,
        "mid_decisive_minor":     0.032,
        "mid_decisive_major":     0.016,

        # --- PHASE 3: LATE GAME (Total: 0.300) ---
        "late_dead_equal":        0.045,
        "late_slight_pull":       0.090,
        "late_solid_edge":        0.075,
        "late_clear_dominance":   0.054,
        "late_decisive_minor":    0.024,
        "late_decisive_major":    0.012,
    }

    # Dynamic baseline adjustments targeting maximum depth saturation 
    # instead of constraining abundant strata volume down to starving buckets
    TARGET_TOTAL_DATASET = 100_000_000
    depth_cutoffs = {}
    
    print("\n=== Saturated Depth-Stability Cutoffs per Strata ===")
    for k, target_pct in target_ratios.items():
        desired_count = int(target_pct * TARGET_TOTAL_DATASET)
        actual_dict = global_depth_counts.get(k, {})
        actual_count = sum(actual_dict.values())
        
        # If the category is data-starved, retain 100% of data to maximize training signals
        if actual_count <= desired_count or actual_count == 0:
            depth_cutoffs[k] = {"min_depth": 0, "boundary_fraction": 1.0}
            print(f" -> {k.ljust(22)}: Data Deficit | Retaining 100% (Available: {actual_count:,})")
        else:
            # Sort evaluations to strictly keep deep calculations and discard unstable/shallow evaluations
            sorted_depths = sorted(actual_dict.keys(), reverse=True)
            running_sum = 0
            cutoff_depth = 0
            fraction_needed = 0.0
            
            for d in sorted_depths:
                count_at_d = actual_dict[d]
                if running_sum + count_at_d >= desired_count:
                    cutoff_depth = d
                    needed_from_this_depth = desired_count - running_sum
                    fraction_needed = needed_from_this_depth / count_at_d
                    running_sum += needed_from_this_depth
                    break
                else:
                    running_sum += count_at_d
            
            depth_cutoffs[k] = {"min_depth": cutoff_depth, "boundary_fraction": fraction_needed}
            print(f" -> {k.ljust(22)}: Saturated Clamped | Drop depths < {cutoff_depth} (Keep {running_sum:,}/{actual_count:,})")

    # =========================================================================
    # STEP 2: DETERMINISTIC ROUTING PASS WITH DEPTH FILTERING
    # =========================================================================
    print("\n=== Step 2: Downsampling by Depth and Routing to Target Shards ===")
    post_sampled_total = 0
    
    for idx, file_path in enumerate(input_files):
        print(f" -> Processing input chunk {idx + 1}/{len(input_files)}: {os.path.basename(file_path)}")
        
        chunk_df = (
            pl.scan_parquet(file_path)
            .filter((pl.col("target").abs() <= 1000) & is_valid_depth)
            .with_columns(strata_key_expr)
            .filter(pl.col("strata_key") != "dropped")
            .collect(engine="streaming")
            .unique(subset=["fen"], keep="first")
        )
        
        if chunk_df.is_empty():
            continue

        sampled_blocks = []
        for strata_key, group in chunk_df.partition_by("strata_key", as_dict=True).items():
            s_key = strata_key[0] if isinstance(strata_key, tuple) else strata_key
            if s_key not in depth_cutoffs:
                continue
                
            cutoff_info = depth_cutoffs[s_key]
            min_d = cutoff_info["min_depth"]
            b_frac = cutoff_info["boundary_fraction"]
            
            # Keep evaluations strictly higher than the stability floor cutoff depth
            high_depth_df = group.filter(pl.col("depth") > min_d)
            sampled_blocks.append(high_depth_df)
            
            # Deterministically down-sample at the specific threshold edge boundary
            boundary_df = group.filter(pl.col("depth") == min_d)
            if not boundary_df.is_empty() and b_frac > 0.0:
                # Add deterministic row selection using hashes to prevent memory leaks
                boundary_df = boundary_df.filter(
                    (pl.col("fen").map_elements(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 10000, return_dtype=pl.Int64) / 10000.0) <= b_frac
                )
                sampled_blocks.append(boundary_df)
        
        if not sampled_blocks:
            continue
            
        chunk_filtered = pl.concat(sampled_blocks)
        post_sampled_total += len(chunk_filtered)
        
        # Route the deep rows out to temporary buffered target files
        if not chunk_filtered.is_empty():
            chunk_filtered = chunk_filtered.with_columns(
                (pl.col("fen").map_elements(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % NUM_OUTPUT_SHARDS, return_dtype=pl.Int64)).alias("shard_id")
            )
            
            for s_id, group in chunk_filtered.partition_by("shard_id", as_dict=True).items():
                actual_id = s_id[0] if isinstance(s_id, tuple) else s_id
                out_path = os.path.join(TEMP_BUFFER_DIR, f"shard_{actual_id}", f"part_{idx}.parquet")
                group.drop(["strata_key", "shard_id"]).write_parquet(out_path)
                
    print(f"\n[SUCCESS] Route complete. Total stable positions consolidated: {post_sampled_total:,}")
    
    # =========================================================================
    # STEP 3: CONSOLIDATION & SHUFFLE
    # =========================================================================
    print("\n=== Step 3: Consolidating and Shuffling Final Buffered Shards ===")

    for b_idx in range(NUM_OUTPUT_SHARDS):
        print(f" -> Applying final shuffle and saving organic balanced shard {b_idx + 1}/{NUM_OUTPUT_SHARDS}...")
        
        shard_chunks = glob.glob(os.path.join(TEMP_BUFFER_DIR, f"shard_{b_idx}", "*.parquet"))
        if not shard_chunks:
            continue
            
        shard_df = pl.read_parquet(shard_chunks).sample(fraction=1.0, shuffle=True)
        output_path = os.path.join(BALANCED_OUTPUT_DIR, f"data_{b_idx + 1}.parquet")
        shard_df.write_parquet(output_path, compression="snappy")
        
        del shard_df

    shutil.rmtree(TEMP_BUFFER_DIR, ignore_errors=True)
    print(f"\n[SUCCESS] Extracted {post_sampled_total:,} rows. Low-depth positions were pruned first to maintain maximum evaluation quality!")

if __name__ == "__main__":
    run_precision_matrix_shuffler_low_mem()
