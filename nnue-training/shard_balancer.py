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
    Translates the bucketing rules natively into Polars expressions.
    Defines the original 3x4 matrix layout (12 buckets).
    """
    piece_count_expr = (
        pl.col("fen")
        .str.split(" ")
        .list.get(0)
        .str.replace_all(r"[\d/kK]", "")
        .str.len_chars()
    )
    abs_score_expr = pl.col("target").abs()
    
    is_valid_depth = pl.lit(True)
    
    phase_expr = (
        pl.when(piece_count_expr >= 26).then(pl.lit("early"))
        .when(piece_count_expr <= 14).then(pl.lit("late"))
        .otherwise(pl.lit("mid"))
    )
    
    score_expr = (
        pl.when(abs_score_expr <= 1.5).then(pl.lit("quiet"))
        .when(abs_score_expr <= 4).then(pl.lit("advantage"))
        .when(abs_score_expr <= 8).then(pl.lit("decisive"))
        .otherwise(pl.lit("blunder"))
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

    print("=== Commencing Depth-Prioritized Stratified Target Ratio Router ===")
    
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
    global_depth_counts = {} # Key: strata_key, Value: {depth: count}
    
    for file_path in input_files:
        chunk_counts = (
            pl.scan_parquet(file_path)
            .filter((pl.col("target").abs() <= 1000) & is_valid_depth)
            .sort("depth", descending=True)
            .unique(subset=["fen"], keep="first")
            .select([strata_key_expr, pl.col("depth")])
            .group_by(["strata_key", "depth"])
            .len("count")
            .collect(engine="streaming") 
        )
        
        for row in chunk_counts.iter_rows(named=True):
            k = row["strata_key"]
            d = row["depth"]
            c = row["count"]
            if k not in global_depth_counts:
                global_depth_counts[k] = {}
            global_depth_counts[k][d] = global_depth_counts[k].get(d, 0) + c

    # Target Matrix weights optimizing for balanced Win-Loss Sigmoid distribution
    target_ratios = {
        "early_quiet": 0.100, "mid_quiet": 0.200, "late_quiet": 0.100,
        "early_advantage": 0.070, "mid_advantage": 0.180, "late_advantage": 0.100,
        "early_decisive": 0.025, "mid_decisive": 0.100, "late_decisive": 0.075,
        "early_blunder": 0.005, "mid_blunder": 0.020, "late_blunder": 0.025
    }

    # Calculate global counts per bucket to determine anchor scale
    global_counts = {k: sum(depths.values()) for k, depths in global_depth_counts.items()}
    
    # Anchor to mid_advantage to avoid over-squeezing good tactical data
    anchor_multiplier = global_counts.get("mid_advantage", 1) / target_ratios["mid_advantage"]
    
    # Determine absolute depth cutoffs and sampling fractions for the boundary depth
    depth_cutoffs = {}
    total_rows = sum(global_counts.values())

    print("\n=== Calculated Depth Cutoffs per Strata ===")
    for k, target_pct in target_ratios.items():
        desired_count = int(target_pct * anchor_multiplier)
        actual_dict = global_depth_counts.get(k, {})
        actual_count = sum(actual_dict.values())
        
        if actual_count <= desired_count or actual_count == 0:
            # Keep 100% of the data in this bucket across all depths
            depth_cutoffs[k] = {"min_depth": 0, "boundary_fraction": 1.0}
            print(f" -> {k.ljust(18)}: Keep ALL positions (Available: {actual_count:,})")
        else:
            # Sort depths descending to prioritize keeping deeper calculations
            sorted_depths = sorted(actual_dict.keys(), reverse=True)
            running_sum = 0
            cutoff_depth = 0
            fraction_needed = 0.0
            
            for d in sorted_depths:
                count_at_d = actual_dict[d]
                if running_sum + count_at_d >= desired_count:
                    cutoff_depth = d
                    # Calculate how much of this specific boundary depth we need to hit our exact target
                    needed_from_this_depth = desired_count - running_sum
                    fraction_needed = needed_from_this_depth / count_at_d
                    running_sum += needed_from_this_depth
                    break
                else:
                    running_sum += count_at_d
            
            depth_cutoffs[k] = {"min_depth": cutoff_depth, "boundary_fraction": fraction_needed}
            print(f" -> {k.ljust(18)}: Drop depths below {cutoff_depth} (Keep {running_sum:,}/{actual_count:,})")

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
            .sort("depth", descending=True)
            .unique(subset=["fen"], keep="first")
            .with_columns(strata_key_expr)
            .collect(engine="streaming")
        )
        
        if chunk_df.is_empty():
            continue

        sampled_blocks = []
        for strata_key, group in chunk_df.partition_by("strata_key", as_dict=True).items():
            s_key = strata_key if isinstance(strata_key, tuple) else strata_key
            cutoff_info = depth_cutoffs.get(s_key, {"min_depth": 0, "boundary_fraction": 1.0})
            
            min_d = cutoff_info["min_depth"]
            b_frac = cutoff_info["boundary_fraction"]
            
            # 1. Keep rows strictly higher than the floor cutoff depth
            high_depth_df = group.filter(pl.col("depth") > min_d)
            if not high_depth_df.is_empty():
                sampled_blocks.append(high_depth_df)
                
            # 2. Sample the boundary floor depth to meet the exact distribution requirement
            if b_frac > 0.0:
                boundary_df = group.filter(pl.col("depth") == min_d)
                if not boundary_df.is_empty():
                    if b_frac >= 1.0:
                        sampled_blocks.append(boundary_df)
                    else:
                        sampled_blocks.append(boundary_df.sample(fraction=b_frac, shuffle=False))
                        
        if not sampled_blocks:
            continue
            
        chunk_df = pl.concat(sampled_blocks)
        post_sampled_total += chunk_df.height

        # Map FEN uniformly across the 200 output shards
        chunk_df = chunk_df.with_columns(
            pl.col("fen").map_elements(
                lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % NUM_OUTPUT_SHARDS, 
                return_dtype=pl.Int64
            ).alias("target_shard")
        )
        
        # Fixed: Explicitly grab the first integer scalar value out of the Series column map
        for _, target_sub_df in chunk_df.partition_by("target_shard", as_dict=True).items():
            if target_sub_df.is_empty():
                continue
            
            s_idx = int(target_sub_df["target_shard"][0])
                
            out_chunk_path = os.path.join(TEMP_BUFFER_DIR, f"shard_{s_idx}", f"part_{idx}.parquet")
            target_sub_df.drop(["strata_key", "target_shard"]).write_parquet(out_chunk_path, compression="snappy")
            
        del chunk_df

    # =========================================================================
    # STEP 3: CONSOLIDATE & RE-SHUFFLE SHARDS
    # =========================================================================
    print("\n=== Step 3: Finalizing and Shuffling Balanced Output Shards ===")
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
