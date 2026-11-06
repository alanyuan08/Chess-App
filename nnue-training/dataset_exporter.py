import os
import sys
import glob
import numpy as np
import shutil

import chess
import multiprocessing
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- INITIALIZATION ENGINE CONSTANTS ---
DATA_SIZE = 2_000_000
# 10 Buckets * 11 Active Pieces (excluding own King) * 64 Squares + 4 Castling Squares = 7044
INPUT_FEATURES = 10 * 11 * 64 + 4  # Dual-Perspective HalfKA Dimension (7044)
MAX_PIECES = 32  

# Map FEN character to an integer type 0-11
PIECE_MAP = {
    'P': 0, 'B': 1, 'N': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'b': 7, 'n': 8, 'r': 9, 'q': 10, 'k': 11
}

# --- DIRECTORY PATH AUTO-RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
BINARY_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_shards")
INPUT_DIR = "data"

# Highly standardized baseline values matching your pipeline configuration
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def static_evaluate(b: chess.Board) -> int:
    """Side-to-move pure material evaluation optimized using raw bitboards."""
    score = 0
    stm = b.turn
    opp = not stm
    
    # Fast bitboard counts with zero dictionary allocations
    score += len(b.pieces(chess.PAWN, stm)) * 100
    score -= len(b.pieces(chess.PAWN, opp)) * 100
    
    score += len(b.pieces(chess.KNIGHT, stm)) * 300
    score -= len(b.pieces(chess.KNIGHT, opp)) * 300
    
    score += len(b.pieces(chess.BISHOP, stm)) * 300
    score -= len(b.pieces(chess.BISHOP, opp)) * 300
    
    score += len(b.pieces(chess.ROOK, stm)) * 500
    score -= len(b.pieces(chess.ROOK, opp)) * 500
    
    score += len(b.pieces(chess.QUEEN, stm)) * 900
    score -= len(b.pieces(chess.QUEEN, opp)) * 900

    score += len(b.pieces(chess.KING, stm)) * 20000
    score -= len(b.pieces(chess.KING, opp)) * 20000
            
    return score

def is_unsafe_for_training(board: chess.Board, tolerance: int = 40) -> bool:
    """
    Validates training safety by verifying tactical deadness for both sides.
    Returns True if either player can forcefully break the material balance.
    Ensures the board is returned in its exact original state for downstream calculations.
    """
    # Guardrail 0: Active checks are inherently chaotic and volatile
    if board.is_check():
        return True
    
    # Capture the exact state signature before any mutations or searches occur
    root_state_hash = board.fen()

    # -------------------------------------------------------------------------
    # 1. Standing Pat (Your Turn): Verify no capture chain improves your side
    # -------------------------------------------------------------------------
    static_A = static_evaluate(board)
    q_score_A = q_search(board, -float('inf'), float('inf'))
    
    # If q-search discovers a sequence that actively wins material for you
    if (q_score_A - static_A) > tolerance:
        return True

    # -------------------------------------------------------------------------
    # 2. Null Move (Opponent's Turn): Verify no chain can destroy you
    # -------------------------------------------------------------------------
    board.push(chess.Move.null())
    try:
        static_B = static_evaluate(board)
        q_score_B = q_search(board, -float('inf'), float('inf'))
    except BaseException as e:
        # Prints the exact error type and trace if q_search crashes
        print(f"\n[!] Problem detected during null move branch evaluation: {type(e).__name__} -> {e}")
        raise
    finally:
        board.pop()

    # If the opponent finds a forcing sequence to destroy your position after a pass
    if (q_score_B - static_B) > tolerance:
        return True
    
    # Quick Aduit
    if board.fen() != root_state_hash:
        raise RuntimeError(
            f"CRITICAL MOVE-STACK DRIFT DETECTED AT FUNCTION END!\n"
            f"The board state failed to unwind cleanly to its original root signature.\n"
            f"Corrupted FEN Context: {board.fen()}"
        )

    # The position is perfectly stable for both sides
    return False

def q_search(b: chess.Board, alpha: float, beta: float, depth: int = 0, max_depth: int = 24) -> int:
    """
    Fail-Soft Alpha-Beta Quiescence Search with explicit terminal state resolution.
    """
    if depth > 0 and b.is_repetition(2):
        return 0

    if depth >= max_depth:
        return static_evaluate(b)
    
    in_check = b.is_check()

    if not in_check:
        best_score = static_evaluate(b)
        if best_score > alpha:
            alpha = best_score
            if alpha >= beta:
                return best_score
    else:
        best_score = -float('inf')

    moves_played = 0

    # Generate and filter forcing moves
    for move in b.generate_pseudo_legal_moves(b.turn):
        if in_check or b.is_capture(move) or move.promotion:
            if not b.is_legal(move):
                continue

            moves_played += 1
            b.push(move)
            try:
                score = -q_search(b, -beta, -alpha, depth + 1, max_depth)
            except BaseException as e:
                # Triggers only when a crash occurs somewhere down this sub-tree branch
                print(f"\n[!] Problem inside q_search at depth {depth} following move {move}: {type(e).__name__} -> {e}")
                raise
            finally:
                b.pop()

            if score > best_score:
                best_score = score
            
            if score > alpha:
                alpha = score
                # FAIL-SOFT LOWER BOUND CUTOFF:
                if alpha >= beta:
                    return best_score

    # --- TERMINAL RULES LAYER ---
    if moves_played == 0:
        if in_check:
            return -20000 + depth
        
        if b.is_stalemate():
            return 0 
    
    return best_score


# The 10-bucket coordinate mapping layout array with horizontal mirroring.
KING_BUCKET_MAP = [
    # Rank 1: Full file resolution (Buckets 0-3)
    0,  1,  2,  3,    3,  2,  1,  0,
    
    # Rank 2: Full file resolution (Buckets 4-7)
    4,  5,  6,  7,    7,  6,  5,  4,
    
    # Ranks 3, 4, 5: Consolidated Midgame Core Block (24 squares) (Bucket 8)
    8,  8,  8,  8,    8,  8,  8,  8,
    8,  8,  8,  8,    8,  8,  8,  8,
    8,  8,  8,  8,    8,  8,  8,  8,
    
    # Ranks 6, 7, 8: High-Density Active Attacking & Endgame Pool (24 squares) (Bucket 9)
    9,  9,  9,  9,    9,  9,  9,  9,
    9,  9,  9,  9,    9,  9,  9,  9,
    9,  9,  9,  9,    9,  9,  9,  9
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
            
            w_idx = (white_king_bucket * 704) + (w_compressed_type * 64) + final_w_sq
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
            
            b_idx = (black_king_bucket * 704) + (b_compressed_type * 64) + final_b_sq
            black_perspective_indices.append(b_idx)

    # --------------------------------------------------------------------------
    # --- TRADITIONAL CASTLING INFLATION LAYER (Indices 7040 - 7043) ---
    # --------------------------------------------------------------------------
    CASTLING_BASE = 7040
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
    a_pad = np.full(MAX_PIECES, INPUT_FEATURES, dtype=np.int32)
    p_pad = np.full(MAX_PIECES, INPUT_FEATURES, dtype=np.int32)
    a_pad[:min(len(a_idx), MAX_PIECES)] = a_idx[:min(len(a_idx), MAX_PIECES)]
    p_pad[:min(len(p_idx), MAX_PIECES)] = p_idx[:min(len(p_idx), MAX_PIECES)]
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
def run_parquet_cleaning_pass(parquet_path, output_dir, samples_per_file=DATA_SIZE):
    """
    Processes a single parquet file in an isolated worker process using 
    memory-bounded row-group streaming to flatline RAM utilization.
    """
    base_name = os.path.splitext(os.path.basename(parquet_path))[0]
    os.makedirs(output_dir, exist_ok=True)
        
    batch_records = []
    file_processed_count = 0
    local_shard_counter = 0

    print(f"[Start] Worker streaming shard: {parquet_path}")
        
    try:
        # Open file handler without pulling any data into RAM yet
        parquet_file = pq.ParquetFile(parquet_path)
    except Exception as e:
        print(f"[Error] Failed to open {parquet_path}: {e}")
        return 0

    # Stream the file in manageable memory chunks (65,536 rows at a time)
    # Only loads the requested 4 columns into memory per batch
    for pyarrow_batch in parquet_file.iter_batches(batch_size=65536, columns=['fen', 'cp', 'depth', 'mate', 'knodes']):
        
        # Zero-copy conversion of just this specific chunk to a lightweight DataFrame
        chunk_df = pyarrow_batch.to_pandas()

        for idx, row in chunk_df.iterrows():
            # --- Robust Data Extraction & Type Safety ---
            fen = row.get("fen")
            
            # Use safe type conversion fallback to avoid pandas float/NaN vs string type issues
            fen_str = str(fen).strip() if fen is not None and not pd.isna(fen) else ""

            # Standard Python/Pandas NaN conversions to safe float/int/None primitives
            raw_score = None if pd.isna(row.get("cp")) else float(row.get("cp"))
            mate_val = None if pd.isna(row.get("mate")) else int(row.get("mate"))

            depth_val = 0 if pd.isna(row.get("depth")) else int(row.get("depth"))
            knodes = 0 if pd.isna(row.get("knodes")) else int(row.get("knodes"))

            # --- Structural FEN Validation (Fast Failure Guards) ---
            # 1. Base Type check & string cleaning
            if not isinstance(fen, str) or '\x00' in fen_str:
                continue

            # 2. Basic layout sanity (8 rank slots separated by 7 slashes)
            if fen_str.count('/') != 7:
                continue

            # 3. Component check: A legal FEN string must have at least 4 components 
            #    (Placement, Active Color, Castling Rights, En Passant Target)
            fen_parts = fen_str.split()
            if len(fen_parts) < 4:
                continue

            # 4. Invalid Line if Mate and Score
            if mate_val and raw_score:
                continue

            # =========================================================================
            # 1. THE MATE ENGINE: TARGETED ENDGAME FILTERING
            # =========================================================================
            if mate_val is not None:
                mate_count = float(mate_val)
                mate_priority = abs(mate_count)

                if mate_count == 0:
                    continue

                # --- ORIGINAL TARGET SCALING (2.00 - 3.00 Pawns / 200.0 - 300.0 CP) ---
                mate_sign = 1.0 if mate_count > 0 else -1.0
                score_target = mate_sign * (250.0 + (50.0 / max(1.0, abs(mate_count))))
                active_pawn_score = score_target / 100.0

            # =========================================================================
            # 2. THE POSITION ENGINE (Reached if it's a normal row OR a valid trained mate)
            # =========================================================================
            else:
                if raw_score is None:
                    continue

                mate_priority = 1000

                score_target = float(raw_score)
                active_pawn_score = score_target / 100.0

            # =========================================================================
            # 3.  GLOBAL QUALITY GATE & VALIDATION FIREWALL
            # Centralized scope forces ALL entries to satisfy high-integrity parameters
            # =========================================================================
            # Validate chess constraints
            board = chess.Board(fen)
            if board.is_check() or board.is_stalemate() or board.is_insufficient_material():
                continue

            # Check workload limits and analytical constraints
            if is_invalid_training_row(depth_val, knodes):
                continue

            # Filter out highly volatile tactical configurations via Q-Search
            if is_unsafe_for_training(board):
                continue

            # =========================================================================
            # 4. PERSPECTIVE FEATURE GENERATION & SHARD BATCHING Matrix
            # =========================================================================
            is_black_turn = (board.turn == chess.BLACK)
            active_player_target = -active_pawn_score if is_black_turn else active_pawn_score

            active_indices, passive_indices = parse_fen_to_features(fen)
            active_orig, passive_orig = pad_indices(active_indices, passive_indices)

            rotated_board = board.mirror().transform(chess.flip_horizontal)
            rotated_fen = rotated_board.fen()

            active_rot, passive_rot = parse_fen_to_features(rotated_fen)
            active_rot_pad, passive_rot_pad = pad_indices(active_rot, passive_rot)

            # Check capacity constraints before extending the list
            if len(batch_records) + 2 > samples_per_file:
                save_parquet_shard(batch_records, output_dir, base_name, local_shard_counter)
                file_processed_count += len(batch_records)
                batch_records = []
                local_shard_counter += 1
                    
            # Append rotated perspective
            fen_parts_rot = rotated_fen.split()
            clean_rotated_fen = " ".join(fen_parts_rot[0:3])
            batch_records.append({
                'active_indices': active_rot_pad,
                'passive_indices': passive_rot_pad,
                'target': active_player_target,
                'mate_priority': mate_priority,
                'depth': depth_val,
                'fen': clean_rotated_fen,
            })

            # Append original perspective
            fen_parts_orig = fen.split()
            clean_orig_fen = " ".join(fen_parts_orig[0:3])
            batch_records.append({
                'active_indices': active_orig,
                'passive_indices': passive_orig,
                'target': active_player_target,
                'mate_priority': mate_priority, 
                'depth': depth_val,
                'fen': clean_orig_fen,
            })

    # Flush remaining records from memory
    if len(batch_records) > 0:
        save_parquet_shard(batch_records, output_dir, base_name, local_shard_counter)
        file_processed_count += len(batch_records)

    print(f"[Finished] Shard {base_name} complete. Total cleaned: {file_processed_count}")
    return file_processed_count

# =====================================================================
# GLOBAL ENVELOPE EXECUTION MAIN FUNCTION
# =====================================================================
def main():
    print("=====================================================================")
    print("       COMMENCING MULTI-THREADED MULTI-PARQUET EXPORT PASS           ")
    print("=====================================================================")
        
    raw_parquet_pattern = os.path.join(SCRIPT_DIR, INPUT_DIR, "data_*.parquet")
    raw_files = sorted(glob.glob(raw_parquet_pattern))
        
    if not raw_files:
        print(f"[CRITICAL ERROR] No source Parquet files detected matching *.parquet")
        print("Please run your shell download script to seed the data_dedup/ directory first!")
        sys.exit(1)
            
    print(f"Detected {len(raw_files)} raw shards ready for feature cleaning.")
    print(f"Spawning 5 worker processes")
    
    total_global_processed = 0

    # Process Pool Executor manages CPU-bound processing scales automatically
    with ProcessPoolExecutor(max_workers=5) as executor:
        # Submit all tasks to the process pool
        futures = {
            executor.submit(run_parquet_cleaning_pass, file_path, BINARY_OUTPUT_DIR, DATA_SIZE): file_path 
            # Submitting tasks as generators
            for file_path in raw_files
        }
        
           # As workers finish execution, grab metrics dynamically
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                count = future.result()
                total_global_processed += count
            except Exception as exc:
                print(f"[CRITICAL WORKER EXCEPTION] {file_path} generated an exception: {exc}")
                
    print("\n=====================================================================")
    print("   ALL RAW PARQUET SHARDS FULLY GUARDRAILED, CLEANED, AND EXPORTED   ")
    print(f"   Grand Total Clean Positions Serialized Across Workers: {total_global_processed}")
    print("=====================================================================")

if __name__ == "__main__":
    if sys.platform == "darwin":
        multiprocessing.set_start_method("spawn", force=True)
        
    main()
