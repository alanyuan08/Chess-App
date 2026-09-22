import os
import sys
import glob
import numpy as np
import polars as pl
import chess

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
        .when(abs_score_expr <= 15.00).then(pl.lit("decisive_zone"))
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
    global_strata_counts = {}

    # Read and filter each file eagerly one by one to keep active RAM footprint tiny
    for i, path in enumerate(input_shards, 1):
        df_shard = pl.read_parquet(path)
        if len(df_shard) == 0: continue
            
        # 1. Apply strict stratified depth floors instantly on the small local chunk
        df_filtered = df_shard.filter(
            ((piece_count_expr >= 26) & (pl.col("depth") >= 26)) |
            ((piece_count_expr.is_between(13, 25)) & (pl.col("depth") >= 28)) |
            ((piece_count_expr < 13) & (pl.col("depth") >= 32))
        )
        if len(df_filtered) == 0: continue

        # 2. Add features and split hashes immediately on the clean local subset
        df_filtered = df_filtered.with_columns(
            pl.col("fen")
            .map_elements(to_canonical_fen, return_dtype=pl.String)
            .alias("canonical_fen")
        )
        
        df_filtered = df_filtered.with_columns([
            strata_key_expr,
            pl.col("fen").hash(seed=999).abs().alias("split_hash"),
            pl.col("fen").hash(seed=777).alias("mix_hash")
        ])

        # Calculate routing bucket IDs based on your mixing entropy
        df_filtered = df_filtered.with_row_index(name="row_idx")
        df_filtered = df_filtered.with_columns(
            (pl.col("mix_hash").abs() % NUM_BUCKETS).alias("bucket_id")
        )

        # 3. Secure Hash Split: Divert the 98% Train and 2% Validation rows cleanly
        df_train_chunk = df_filtered.filter(pl.col("split_hash") % 100 < 98)
        df_val_chunk = df_filtered.filter(pl.col("split_hash") % 100 >= 98)

        # 4. Stream Training row allocations directly to raw disk bucket files
        if len(df_train_chunk) > 0:
            # Accumulate strata telemetry directly from the raw chunk
            counts = df_train_chunk.group_by("strata_key").len("count")
            for row in counts.iter_rows(named=True):
                k = row["strata_key"]
                if "_dropped" in k: continue
                global_strata_counts[k] = global_strata_counts.get(k, 0) + row["count"]

            for (b_id,), df_sub_bucket in df_train_chunk.group_by("bucket_id"):
                if len(df_sub_bucket) == 0: continue
                frag_path = os.path.join(TEMP_TRAINING_DIR, f"bucket_{b_id}_raw_{i}.parquet")
                df_sub_bucket.drop(["bucket_id", "row_idx", "split_hash"]).write_parquet(frag_path, compression="snappy")

        # 5. Stream Validation row allocations directly to separate raw disk bucket files
        if len(df_val_chunk) > 0:
            for (b_id,), df_sub_bucket in df_val_chunk.group_by("bucket_id"):
                if len(df_sub_bucket) == 0: continue
                frag_path = os.path.join(TEMP_VAL_DIR, f"bucket_{b_id}_raw_{i}.parquet")
                df_sub_bucket.drop(["bucket_id", "row_idx", "split_hash"]).write_parquet(frag_path, compression="snappy")

        del df_shard, df_filtered, df_train_chunk, df_val_chunk

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
        
        # Read only the strata column from the clean fragment to save speed
        df_counts = (
            pl.concat([pl.read_parquet(f, columns=["strata_key"]) for f in fragments])
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
        "early": 2.0,   # Avoids over-memorizing specific engine opening lines
        "mid": 3.5,     # Protects middlegame tactical volatility horizons
        "late": 6.0     # Safely upscales scarce, pure depth-32+ endgames
    }

    TARGET_RATIOS = {
        "early_dead_equal": 0.080, "early_slight_pull": 0.060, "early_micro_advantage": 0.035,
        "early_solid_edge": 0.015, "early_clear_dominance": 0.007, "early_decisive_zone": 0.003,
        
        "mid_dead_equal": 0.140, "mid_slight_pull": 0.120, "mid_micro_advantage": 0.100,
        "mid_solid_edge": 0.065, "mid_clear_dominance": 0.020, "mid_decisive_zone": 0.005,  
        
        "late_dead_equal": 0.110, "late_slight_pull": 0.100, "late_micro_advantage": 0.085,
        "late_solid_edge": 0.040, "late_clear_dominance": 0.010, "late_decisive_zone": 0.005,
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
    
    def produce_strict_blocks(source_dir, output_dir, file_prefix, is_validation=False):
        """
        Processes pre-deduplicated fragments from a source cache directory, 
        balances them symmetrically against retention fractions, shuffles, 
        and writes block-aligned Parquet shards to the target destination.
        """
        mode_str = "VALIDATION" if is_validation else "TRAINING"
        print(f"\n 3 [{mode_str}] -> Processing individual buckets with strict 8192 out-of-core block production...")
        
        CHUNK_MULTIPLE = 8192
        shard_counter = 0
        df_overflow_buffer = None

        for bucket_id in range(NUM_BUCKETS):
            # Look for the finalized fragment file generated in Phase 2
            frag_pattern = os.path.join(source_dir, f"bucket_{bucket_id}_raw_*.parquet")
            fragments = glob.glob(frag_pattern)
            if not fragments: continue

            print(f"    Processing pre-deduplicated {mode_str.lower()} bucket {bucket_id}/{NUM_BUCKETS - 1}...")
            
            # 1. Concat all intermediate shard fragments
            df_raw = pl.concat([pl.read_parquet(f) for f in fragments])
            
            # 2. Tag each position with its canonical FEN key and determine its mirror status
            df_processed = df_raw.with_columns(
                pl.col("fen")
                .map_elements(to_canonical_fen, return_dtype=pl.String)
                .alias("canonical_fen")
            ).with_columns(
                # is_mirror is True (1) if it's a mirrored transformation, False (0) if it's the original FEN
                is_mirror=(pl.col("fen") != pl.col("canonical_fen"))
            )
            
            # 3. Clean up input fragments immediately to free up MacBook SSD space
            for frag_path in fragments:
                try: os.remove(frag_path)
                except: pass

            processed_strata = []
            
            # 4. Group by strata key to evaluate target ratios before dropping any data
            for (strata_k,), sub_df in df_processed.group_by("strata_key"):
                fraction = retention_fractions.get(strata_k, 0.0)
                if fraction <= 0.0 or strata_k not in TARGET_RATIOS: continue
                
                # Check target sufficiency: If fraction <= 1.0, we have more than enough data
                if fraction <= 1.0:
                    # COPIOUS DATA STRATEGY: Prioritize unique originals to maximize structural diversity.
                    # Sort unique native positions to the top, then by deepest search value.
                    sub_df_optimized = (
                        sub_df.sort(by=["is_mirror", "depth"], descending=[False, True])
                        .unique(subset=["canonical_fen"], keep="first")
                        .drop(["canonical_fen", "is_mirror"])
                    )
                    
                    # Apply multi-seeded shuffle on the high-diversity, un-mirrored subset
                    sub_df_shuffled = sub_df_optimized.sample(fraction=1.0, shuffle=True, seed=1337 + bucket_id)
                    keep_count = int(len(sub_df_shuffled) * fraction)
                    if keep_count > 0:
                        processed_strata.append(sub_df_shuffled.head(keep_count))
                        
                else:
                    # DATA STARVATION STRATEGY: We don't have enough positions to meet the target ratio.
                    # Keep both original AND mirrored perspectives to build critical mass.
                    # Deduplicate only on exact text matching ("fen") so distinct mirror versions survive.
                    sub_df_starved = (
                        sub_df.sort("depth", descending=True)
                        .unique(subset=["fen"], keep="first")
                        .drop(["canonical_fen", "is_mirror"])
                    )
                    
                    sub_df_shuffled = sub_df_starved.sample(fraction=1.0, shuffle=True, seed=1337 + bucket_id)
                    
                    # Execute symmetrical upscaling replication loop (applies up to 6.0x to BOTH streams)
                    full_replications = int(fraction)
                    remainder_fraction = fraction - full_replications
                    
                    for _ in range(full_replications):
                        processed_strata.append(sub_df_shuffled)
                        
                    remainder_count = int(len(sub_df_shuffled) * remainder_fraction)
                    if remainder_count > 0:
                        processed_strata.append(sub_df_shuffled.head(remainder_count))
                    
            if not processed_strata: 
                continue

            # Strip metadata fields no longer needed by the active NNUE network graph
            df_balanced = pl.concat(processed_strata).drop(["mix_hash", "strata_key"])
            
            # --- STITCH WITH MOVING OVERFLOW SLIDING WINDOW ---
            if df_overflow_buffer is not None and len(df_overflow_buffer) > 0:
                df_working = pl.concat([df_overflow_buffer, df_balanced])
            else:
                df_working = df_balanced
                
            # --- HIGH-ENTROPY INTERLEAVING SHUFFLE ---
            # Completely shatters the sequential upscaled rows to protect AdamW gradients
            df_working = df_working.sample(fraction=1.0, shuffle=True, seed=777 + bucket_id)
                
            total_available = len(df_working)
            num_chunks_to_write = total_available // CHUNK_MULTIPLE
            rows_to_write = num_chunks_to_write * CHUNK_MULTIPLE
            
            if rows_to_write > 0:
                df_production_shard = df_working.head(rows_to_write)
                
                final_out_path = os.path.join(output_dir, f"{file_prefix}_{shard_counter}.parquet")
                df_production_shard.write_parquet(final_out_path, compression="snappy")
                
                print(f"      [EXPORTED] Production file {shard_counter} written with {rows_to_write} rows.")
                shard_counter += 1
                
                df_overflow_buffer = df_working.slice(rows_to_write, total_available - rows_to_write)
            else:
                df_overflow_buffer = df_working
                
            del df_bucket, df_balanced, df_working, processed_strata

        # --- PARITY BLOCK FLUSH ---
        if df_overflow_buffer is not None and len(df_overflow_buffer) >= CHUNK_MULTIPLE:
            total_available = len(df_overflow_buffer)
            num_chunks_to_write = total_available // CHUNK_MULTIPLE
            rows_to_write = num_chunks_to_write * CHUNK_MULTIPLE
            
            df_production_shard = df_overflow_buffer.head(rows_to_write)
            final_out_path = os.path.join(output_dir, f"{file_prefix}_{shard_counter}.parquet")
            df_production_shard.write_parquet(final_out_path, compression="snappy")
            print(f"    [FINAL FLUSH] Production file {shard_counter} written with {rows_to_write} rows.")
            
            remainder_dropped = total_available - rows_to_write
            if remainder_dropped > 0:
                print(f"    [TRUNCATED] Dropped {remainder_dropped} trailing rows for block parity alignment.")
        elif df_overflow_buffer is not None and len(df_overflow_buffer) > 0:
            print(f"    [TRUNCATED] Dropped final {len(df_overflow_buffer)} leftover rows to preserve strict block stride bounds.")
                
        # Clean up the localized raw intermediate directory cache folder completely
        try: os.rmdir(source_dir)
        except: pass
        
        print(f"[SUCCESS] {mode_str} stream block compilation complete! Target: {output_dir}")

    # =========================================================================
    # EXECUTE DUAL-STREAM MIRRORED COMPILATION
    # =========================================================================
    
    # Run Part A: Materialize balanced training shards
    produce_strict_blocks(
        source_dir=TEMP_TRAINING_DIR,
        output_dir=TRAIN_DIR,
        file_prefix="nnue_train_shard",
        is_validation=False
    )

    # Run Part B: Materialize balanced validation shards
    produce_strict_blocks(
        source_dir=TEMP_VAL_DIR,
        output_dir=VAL_DIR,
        file_prefix="nnue_val_shard",
        is_validation=True
    )
    
    print(f"\n[PIPELINE SYSTEM COMPLETE] Co-balanced training sets are ready for the GPU!")

if __name__ == "__main__":
    run_unified_mixer_balancer()
