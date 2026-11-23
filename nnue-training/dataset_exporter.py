import os
import sys
import glob
import numpy as np

import chess
import multiprocessing
import pandas as pd
import pyarrow.parquet as pq

from concurrent.futures import ProcessPoolExecutor, as_completed
os.environ.setdefault("RAYON_NUM_THREADS", "2")

import rust_compute

# --- INITIALIZATION ENGINE CONSTANTS ---
DATA_SIZE = 2_000_000
MAX_ACTIVE_FEATURES = 35

# 11 Active Pices * 64 Squares = 704
FEATURES_PER_BUCKET = 11 * 64 # 704

# 32 Buckets * 11 Active Pieces (excluding own King) * 64 Squares = 22528
CASTLING_BASE = 32 * 11 * 64 # 22528

INPUT_FEATURES = 32 * 11 * 64 + 4  # Dual-Perspective HalfKA Dimension (22532)

# Map FEN character to an integer type 0-11
PIECE_MAP = {
    'P': 0, 'B': 1, 'N': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'b': 7, 'n': 8, 'r': 9, 'q': 10, 'k': 11
}

QSEARCH_TOLERANCE_CP = 60
NUM_WORKERS = 5

# --- DIRECTORY PATH AUTO-RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
BINARY_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_shards")
INPUT_DIR = "data"

# Compute Attack Tables
rust_compute.init_attack_tables()

# The 32-bucket coordinate mapping layout array with strict horizontal mirroring.
KING_BUCKET_MAP = [
    0,  1,  2,  3,    3,  2,  1,  0,
    4,  5,  6,  7,    7,  6,  5,  4,
    8,  9,  10, 11,   11, 10, 9,  8,
    12, 13, 14, 15,   15, 14, 13, 12,
    16, 17, 18, 19,   19, 18, 17, 16,
    20, 21, 22, 23,   23, 22, 21, 20,
    24, 25, 26, 27,   27, 26, 25, 24,
    28, 29, 30, 31,   31, 30, 29, 28
]

def parse_fen_to_features(fen_string):
    """
    Parses a standard FEN string into sparse categorical feature indices for a
    10-bucket, horizontally mirrored HalfKA NNUE layout using the custom PIECE_MAP.
    
    Appends 4 traditional castling right nodes to the end of the feature transformer
    array (Indices 7040-7043)

    Returns:
        tuple: (active_side_indices, passive_side_indices)
               Each list contains integer indices ranging from 0 to 7043.
    """
    parts = fen_string.split()
    board_part = parts[0]
    turn_part = parts[1]
    castling_part = parts[2] if len(parts) > 2 else "-"
    is_black_turn = (turn_part == 'b')
        
    # Process FEN ranks from top to bottom (Rank 8 down to Rank 1)
    rows = board_part.split('/')
    rows.reverse()  # Reverse so Square 0 = A1, Square 63 = H8
    
    clean_board = ""
    for row in rows:
        for char in row:
            if char.isdigit():
                clean_board += '.' * int(char)
            else:
                clean_board += char
                    
    pieces = []  
    white_king_sq = 0
    black_king_sq = 0
        
    # Scan the board to collect pieces and isolate King positions
    for sq in range(64):
        char = clean_board[sq]
        if char == '.':
            continue
        piece_type_id = PIECE_MAP[char]
        pieces.append((piece_type_id, char, sq))
        if char == 'K':
            white_king_sq = sq
        elif char == 'k':
            black_king_sq = sq

    # --------------------------------------------------------------------------
    # --- RESOLVE WHITE KING BUCKET & HORIZONTAL MIRRORING FLAG ---
    # --------------------------------------------------------------------------
    white_king_file = white_king_sq % 8
    mirror_white = (white_king_file >= 4)  # True if King is on files E, F, G, or H
    
    # If the king is on the right wing, look up bucket using its mirrored position
    w_king_lookup_sq = white_king_sq ^ 7 if mirror_white else white_king_sq
    white_king_bucket = KING_BUCKET_MAP[w_king_lookup_sq]

    # --------------------------------------------------------------------------
    # --- RESOLVE BLACK KING BUCKET & HORIZONTAL MIRRORING FLAG ---
    # --------------------------------------------------------------------------
    # Rotate square 180 degrees vertically so Black tracks the board from its own side
    black_king_sq_rotated = black_king_sq ^ 63
    black_king_file = black_king_sq_rotated % 8
    mirror_black = (black_king_file >= 4)
    
    b_king_lookup_sq = black_king_sq_rotated ^ 7 if mirror_black else black_king_sq_rotated
    black_king_bucket = KING_BUCKET_MAP[b_king_lookup_sq]

    white_perspective_indices = []
    black_perspective_indices = []
        
    for p_type, char, p_sq in pieces:
        # ----------------------------------------------------------------------
        # --- WHITE PERSPECTIVE ---
        # ----------------------------------------------------------------------
        if p_type != 5:  # Skip own King ('K' feature index 5 is omitted)
            # Shift types above omitted King down by 1 to compress features to 0..10 range
            w_compressed_type = p_type if p_type < 5 else p_type - 1
            
            # If White King requires horizontal mirroring, flip this piece square horizontally
            final_w_sq = p_sq ^ 7 if mirror_white else p_sq
            
            w_idx = (white_king_bucket * FEATURES_PER_BUCKET) + (w_compressed_type * 64) + final_w_sq
            white_perspective_indices.append(w_idx)
                
        # ----------------------------------------------------------------------
        # --- BLACK PERSPECTIVE (Flipped & Rotated) ---
        # ----------------------------------------------------------------------
        # Invert side mapping using your custom mapping indices (P<->p, B<->b, etc.)
        b_orig_type = (p_type + 6) % 12
        
        if b_orig_type != 5:  # Skip own King ('k' became index 5 after side inversion)
            # Compress feature types to 0..10 range
            b_compressed_type = b_orig_type if b_orig_type < 5 else b_orig_type - 1
            
            # Flip square 180 degrees vertically for baseline black perspective
            b_sq_rotated = p_sq ^ 63
            
            # If Black King requires horizontal mirroring, flip this square horizontally too
            final_b_sq = b_sq_rotated ^ 7 if mirror_black else b_sq_rotated
            
            b_idx = (black_king_bucket * FEATURES_PER_BUCKET) + (b_compressed_type * 64) + final_b_sq
            black_perspective_indices.append(b_idx)

    # --------------------------------------------------------------------------
    # --- TRADITIONAL CASTLING INFLATION LAYER (Indices 7040 - 7043) ---
    # --------------------------------------------------------------------------
    has_w_king  = 'K' in castling_part
    has_w_queen = 'Q' in castling_part
    has_b_king  = 'k' in castling_part
    has_b_queen = 'q' in castling_part

    if mirror_white:
        w_k_final = has_w_queen
        w_q_final = has_w_king
    else:
        w_k_final = has_w_king
        w_q_final = has_w_queen

    if mirror_black:
        b_k_final = has_b_queen
        b_q_final = has_b_king
    else:
        b_k_final = has_b_king
        b_q_final = has_b_queen

    # White Indexes (Us = White, Them = Black)
    if w_k_final: white_perspective_indices.append(CASTLING_BASE + 0)
    if w_q_final: white_perspective_indices.append(CASTLING_BASE + 1)
    if b_k_final: white_perspective_indices.append(CASTLING_BASE + 2)
    if b_q_final: white_perspective_indices.append(CASTLING_BASE + 3)

    # Black Indexes (Us = Black, Them = White)
    if b_k_final: black_perspective_indices.append(CASTLING_BASE + 0)
    if b_q_final: black_perspective_indices.append(CASTLING_BASE + 1)
    if w_k_final: black_perspective_indices.append(CASTLING_BASE + 2)
    if w_q_final: black_perspective_indices.append(CASTLING_BASE + 3)

    # Return indices organized cleanly by active player turn
    if is_black_turn:
        return black_perspective_indices, white_perspective_indices
    else:
        return white_perspective_indices, black_perspective_indices


def is_invalid_training_row(depth_val: int, knodes: int) -> bool:
    """
    Elite Quality Gate for Middlegame Positions (> 8 Pieces).
    Completely eliminates the lenient Depth 21 core pool. Enforces a strict
    minimum baseline of Depth 26+ and massive node verification to guarantee 
    universal positional stability.
    """
    try:
        # 1. Immediate rejection of absolute low-resource or corrupted rows
        if knodes <= 0:
            return True

        # 2. THE CLASSICAL WALL: 
        # Over 1 Billion nodes (1,000,000 KNodes) is too specific/inaccurate for general training
        if knodes > 1_000_000:
            return True

        # 3. THE ANALYTICAL WALL: 
        # Ban anything below Depth 26 entirely for shallow depth
        if depth_val < 26:
            return True
            
        # Guardrail for Moderate High-Depth (Depths 26 to 32)
        # 400 KNodes = 400,000 nodes searched. Keeps your elite depth 30 transpositions.
        if 26 <= depth_val <= 32:
            return knodes < 400
            
        # Guardrail for Master-Class Analysis (Depth 33+)
        # 1200 KNodes = 1.2 Million nodes searched. Safely saves the 42M node / Depth 42 gem!
        else:
            return knodes < 1200
        
    except (ValueError, TypeError):
        # Defensively mark corrupt structural strings as invalid
        return True

def pad_indices(a_idx, p_idx):
    a_pad = np.full(MAX_ACTIVE_FEATURES, INPUT_FEATURES, dtype=np.int16)
    p_pad = np.full(MAX_ACTIVE_FEATURES, INPUT_FEATURES, dtype=np.int16)
    a_pad[:min(len(a_idx), MAX_ACTIVE_FEATURES)] = a_idx[:min(len(a_idx), MAX_ACTIVE_FEATURES)]
    p_pad[:min(len(p_idx), MAX_ACTIVE_FEATURES)] = p_idx[:min(len(p_idx), MAX_ACTIVE_FEATURES)]
    return a_pad.tolist(), p_pad.tolist()

def save_parquet_shard(batch_records, output_dir, base_name, shard_idx):
    """Helper function to build a structured dataframe and write to Parquet format."""
    output_path = os.path.join(output_dir, f"{base_name}_shard_{shard_idx}.parquet")
    df = pd.DataFrame(batch_records)
    df.to_parquet(output_path, compression="snappy", index=False)
    print(f"[Worker] Saved {len(df)} clean positions -> {output_path}")

# =====================================================================
# WORKER EXECUTION TASK
# =====================================================================
def run_parquet_cleaning_pass(
    file_path: str,
    output_dir: str,
    samples_per_file: int,
    tolerance: int = QSEARCH_TOLERANCE_CP,
) -> int:
    """
    Processes a single raw Parquet file in 65,536-row batches using vectorized
    Pandas pre-filters + multi-threaded Rust 30-cp NNUE qsearch validation.
    """
    rust_compute.init_attack_tables()

    base_name = os.path.splitext(os.path.basename(file_path))
    parquet_file = pq.ParquetFile(file_path)

    batch_records = []
    local_shard_counter = 0
    file_processed_count = 0

    for pyarrow_batch in parquet_file.iter_batches(
        batch_size=65536, columns=["fen", "cp", "depth", "mate", "knodes"]
    ):
        chunk_df = pyarrow_batch.to_pandas()

        # 1. Vectorized structural FEN & workload pre-filtering
        chunk_df = chunk_df[chunk_df["fen"].notna()].copy()
        chunk_df["fen"] = chunk_df["fen"].astype(str).str.strip()

        valid_fen_mask = (
            ~chunk_df["fen"].str.contains("\x00", regex=False)
            & (chunk_df["fen"].str.count("/") == 7)
            & (chunk_df["fen"].str.split().str.len() >= 4)
        )
        chunk_df = chunk_df[valid_fen_mask]
        if chunk_df.empty:
            continue

        depth_arr = chunk_df["depth"].fillna(0).astype(int).to_numpy()
        knodes_arr = chunk_df["knodes"].fillna(0).astype(int).to_numpy()

        valid_work_mask = np.array(
            [not is_invalid_training_row(d, k) for d, k in zip(depth_arr, knodes_arr)],
            dtype=bool,
        )
        chunk_df = chunk_df[valid_work_mask]
        depth_arr = depth_arr[valid_work_mask]
        if chunk_df.empty:
            continue

        # 2. Vectorized Mate & Positional Centipawn Scoring
        has_mate = chunk_df["mate"].notna() & (chunk_df["mate"] != 0)
        has_cp = ~has_mate & chunk_df["cp"].notna()
        valid_score_mask = (has_mate | has_cp).to_numpy()

        chunk_df = chunk_df[valid_score_mask]
        depth_arr = depth_arr[valid_score_mask]
        if chunk_df.empty:
            continue

        mate_vals = chunk_df["mate"].to_numpy(dtype=np.float64)
        cp_vals = chunk_df["cp"].to_numpy(dtype=np.float64)

        is_mate_row = ~np.isnan(mate_vals)
        mate_sign = np.sign(mate_vals)
        mate_scores = mate_sign * (30000.0 - ((np.abs(mate_vals) - 1.0) * 20.0))
        cp_scores = np.round(cp_vals)

        active_pawn_scores = np.where(is_mate_row, mate_scores, cp_scores)

        # 3. Terminal State Gate (Check / Stalemate / Insufficient Material)
        candidate_fens = []
        candidate_boards = []
        candidate_scores = []
        candidate_depths = []

        for fen_str, score, depth_val in zip(
            chunk_df["fen"].tolist(), active_pawn_scores.tolist(), depth_arr.tolist()
        ):
            board = chess.Board(fen_str)

            # 1. Standard-chess piece count limits (blocks 35-piece stress-test mazes)
            if (
                board.occupied.bit_count() > 32
                or board.occupied_co[chess.WHITE].bit_count() > 16
                or board.occupied_co[chess.BLACK].bit_count() > 16
                or board.pawns.bit_count() > 16
            ):
                continue

            # 2. Full python-chess legality validation (kings, backrank pawns, castling, ep, opposite check)
            if not board.is_valid():
                continue

            if (
                board.is_check()
                or board.is_stalemate()
                or board.is_insufficient_material()
            ):
                continue

            candidate_fens.append(fen_str)
            candidate_boards.append(board)
            candidate_scores.append(score)
            candidate_depths.append(int(depth_val))

        if not candidate_fens:
            continue

        # 4. Batched Rust 30-cp NNUE Q-Search Safety Gate
        NNUE_MODEL_PATH = os.path.abspath(
            os.path.join(SCRIPT_DIR, "..", "nnue-training", "nnue_weights.bin")
        )

        # Inside run_parquet_cleaning_pass:
        unsafe_flags = rust_compute.is_unsafe_batch(
            candidate_fens, tolerance, NNUE_MODEL_PATH
        )

        # 5. Perspective Feature Generation & Shard Batching
        for fen_str, board, active_pawn_score, depth_val, is_unsafe in zip(
            candidate_fens,
            candidate_boards,
            candidate_scores,
            candidate_depths,
            unsafe_flags,
        ):
            if is_unsafe:
                continue

            is_black_turn = board.turn == chess.BLACK
            active_player_target = (
                -active_pawn_score if is_black_turn else active_pawn_score
            )

            # Original perspective
            active_indices, passive_indices = parse_fen_to_features(fen_str)
            active_orig, passive_orig = pad_indices(active_indices, passive_indices)

            # Mirrored + horizontally flipped perspective
            rotated_board = board.mirror().transform(chess.flip_horizontal)
            rotated_fen = rotated_board.fen()

            active_rot, passive_rot = parse_fen_to_features(rotated_fen)
            active_rot_pad, passive_rot_pad = pad_indices(active_rot, passive_rot)

            if len(batch_records) + 2 > samples_per_file:
                save_parquet_shard(
                    batch_records, output_dir, base_name, local_shard_counter
                )
                file_processed_count += len(batch_records)
                batch_records = []
                local_shard_counter += 1

            # Append rotated perspective
            clean_rotated_fen = " ".join(rotated_fen.split()[:3])
            batch_records.append(
                {
                    "active_indices": active_rot_pad,
                    "passive_indices": passive_rot_pad,
                    "target": active_player_target,
                    "depth": depth_val,
                    "fen": clean_rotated_fen,
                }
            )

            # Append original perspective
            clean_orig_fen = " ".join(fen_str.split()[:3])
            batch_records.append(
                {
                    "active_indices": active_orig,
                    "passive_indices": passive_orig,
                    "target": active_player_target,
                    "depth": depth_val,
                    "fen": clean_orig_fen,
                }
            )

    # Flush remaining records
    if len(batch_records) > 0:
        save_parquet_shard(batch_records, output_dir, base_name, local_shard_counter)
        file_processed_count += len(batch_records)

    print(
        f"[Finished] Shard {base_name} complete. Total cleaned: {file_processed_count}"
    )
    return file_processed_count

# =====================================================================
# GLOBAL ENVELOPE EXECUTION MAIN FUNCTION
# =====================================================================
def main():
    os.makedirs(BINARY_OUTPUT_DIR, exist_ok=True)

    raw_parquet_pattern = os.path.join(SCRIPT_DIR, INPUT_DIR, "data_*.parquet")
    raw_files = sorted(glob.glob(raw_parquet_pattern))

    if not raw_files:
        print("[CRITICAL ERROR] No source Parquet files detected matching *.parquet")
        print("Please run your shell download script to seed the data_dedup/ directory first!")
        sys.exit(1)

    print(f"Detected {len(raw_files)} raw shards ready for feature cleaning.")
    print(f"Spawning {NUM_WORKERS} worker processes")

    total_global_processed = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(
                run_parquet_cleaning_pass, file_path, BINARY_OUTPUT_DIR, DATA_SIZE
            ): file_path
            for file_path in raw_files
        }

        for future in as_completed(futures):
            file_path = futures[future]
            try:
                count = future.result()
                total_global_processed += count
            except Exception as exc:
                print(
                    f"[CRITICAL WORKER EXCEPTION] {file_path} generated an exception: {exc}"
                )

    print(f"\n[ALL COMPLETE] Total cleaned positions generated: {total_global_processed:,}")

                
    print("\n=====================================================================")
    print("   ALL RAW PARQUET SHARDS FULLY GUARDRAILED, CLEANED, AND EXPORTED   ")
    print(f"   Grand Total Clean Positions Serialized Across Workers: {total_global_processed}")
    print("=====================================================================")

if __name__ == "__main__":
    if sys.platform == "darwin":
        multiprocessing.set_start_method("spawn", force=True)
        
    main()
