import os
import sys
import glob
import math
import numpy as np
import chess
import multiprocessing
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- INITIALIZATION ENGINE CONSTANTS ---
DATA_SIZE = 2_000_000
# 10 Buckets * 11 Active Pieces (excluding own King) * 64 Squares + 4 Castling Squares = 7040
INPUT_FEATURES = 10 * 11 * 64 + 4  # Dual-Perspective HalfKA Dimension (7040)
MAX_PIECES = 32  

# Map FEN character to an integer type 0-11
PIECE_MAP = {
    'P': 0, 'B': 1, 'N': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'b': 7, 'n': 8, 'r': 9, 'q': 10, 'k': 11
}

PIECE_VALUES = {
    chess.PAWN: 100, 
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500, 
    chess.QUEEN: 900, 
    chess.KING: 0
}

# --- DIRECTORY PATH AUTO-RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
BINARY_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_shards")
INPUT_DIR = "data"

# =====================================================================
# CHESS LOGIC AND FILTERING SYSTEM
# =====================================================================
def static_evaluate(b):
    """Side-to-move material evaluation."""
    score = 0
    current_player = b.turn
    for square, piece in b.piece_map().items():
        val = PIECE_VALUES[piece.piece_type]
        if piece.color == current_player:
            score += val
        else:
            score -= val
    return score

def q_search(b, alpha, beta, depth=0, max_depth=24):
    """
    Absolute-Safe, Non-Pruned Negamax Quiescence Search.
    Completely eliminates the stand-pat cutoff to prevent horizon traps.
    Forces the engine to calculate every single capture chain to absolute resolution.
    """
    if depth > 0 and b.is_repetition(2):
        return 0

    if depth >= max_depth:
        return static_evaluate(b)

    in_check = b.is_check()
    if not in_check:
        static_eval = static_evaluate(b)
        if static_eval > alpha:
            alpha = static_eval

    moves_played = 0

    # Force the engine to evaluate every single capture and promotion line
    for move in b.generate_pseudo_legal_moves(b.turn):
        if in_check or b.is_capture(move) or move.promotion:
            if not b.is_legal(move):
                continue

            moves_played += 1
            b.push(move)
            score = -q_search(b, -beta, -alpha, depth + 1, max_depth)
            b.pop()

            if score > alpha:
                alpha = score

    # --- TERMINAL RULES LAYER ---
    # If no captures, promotions, or king evasions are legally possible:
    if moves_played == 0:
        if in_check:
            return -20000

    return alpha

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
    # Base offset boundary right where the 10 king buckets end (10 * 704 = 7040)
    CASTLING_BASE = 7040

    has_w_king  = 'K' in castling_part
    has_w_queen = 'Q' in castling_part
    has_b_king  = 'k' in castling_part
    has_b_queen = 'q' in castling_part

    # Castling Flags are only relevant of the White king is on E1 and the Black king is on E8
    # If they are not, these flags are auto set to False

    # White needed to Mirrored due to King on E1
    w_k, w_q = has_w_queen, has_w_king 

    # Black does not need to be mirrored
    b_k, b_q = has_b_king, has_b_queen

    # ==========================================================================
    # 2. ASSIGN INDICES DIRECTLY BY SIDE TO MOVE
    # ==========================================================================
    # White King_Castle / Queen Castle | Black King_Castle / Queen Castle 
    if w_k: white_perspective_indices.append(CASTLING_BASE + 0)
    if w_q: white_perspective_indices.append(CASTLING_BASE + 1)
    if b_k: white_perspective_indices.append(CASTLING_BASE + 2)
    if b_q: white_perspective_indices.append(CASTLING_BASE + 3)
    
    if b_k: black_perspective_indices.append(CASTLING_BASE + 0)
    if b_q: black_perspective_indices.append(CASTLING_BASE + 1)
    if w_k: black_perspective_indices.append(CASTLING_BASE + 2)
    if w_q: black_perspective_indices.append(CASTLING_BASE + 3)

    # Return indices organized cleanly by active player turn
    if is_black_turn:
        return black_perspective_indices, white_perspective_indices
    else:
        return white_perspective_indices, black_perspective_indices


def is_invalid_training_row(depth_str: str, knodes_str: str) -> bool:
    """
    Elite Quality Gate for Middlegame Positions (> 8 Pieces).
    Completely eliminates the lenient Depth 21 core pool. Enforces a strict
    minimum baseline of Depth 26+ and massive node verification to guarantee 
    universal positional stability.
    """
    try:
        depth = int(depth_str)
        knodes = int(knodes_str) if knodes_str and knodes_str != "None" else 0
        # 1. Immediate rejection of absolute low-resource or corrupted rows
        if knodes <= 0:
            return True

        # 2. THE ANALYTICAL WALL: Ban anything below Depth 26 entirely
        if depth < 26:
            return True
            
        # Guardrail for Moderate High-Depth (Depths 26 to 32)
        # 400 KNodes = 400,000 nodes searched. Keeps your elite depth 30 transpositions.
        if 26 <= depth <= 32:
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
            fen = row.get("fen")
            raw_score = row.get("cp")
            depth_val = row.get("depth")

            knodes = row.get("knodes")
            mate_val = None if pd.isna(row.get("mate")) else row.get("mate")

            if not isinstance(fen, str) or '\x00' in fen or fen.count('/') != 7:
                continue

            if pd.isna(raw_score) and mate_val is None:
                continue

            board = chess.Board(fen)

            # --- CHESS RULES: Terminal and Illegal Verification (Always Run) ---
            if board.is_check() or board.is_stalemate() or board.is_insufficient_material():
                continue

            # Calculate total pieces remaining on the board
            piece_count = bin(int(board.occupied)).count("1")
            active_pawn_score = 0
            
            # =================================================================
            # STRATEGIC PATH A: ENDGAME PIPELINE (<= 8 Pieces) + Mate Sequences
            # Assume 90%+ of 8 Piece positions are computed via endgame
            # =================================================================
            if piece_count <= 8 and mate_val is not None and not pd.isna(mate_val):
                # DEFENSIVE SHIELD: Drop shallow mobile app or low-resource bursts.
                knodes_val = int(knodes) if knodes and knodes != "None" else 0
                if knodes_val < 200:
                    continue

                static_score = static_evaluate(board)
                q_score = q_search(board, -float('inf'), float('inf'))
                if abs(static_score - q_score) > 40:
                    continue
                
                mate_count = float(mate_val)
                
                # Maximum terminal ceiling right below overflow boundaries
                MATE_MAX = 300.0
                MATE_BASE = 250.0
                
                if mate_count > 0:
                    # Mate-in-1  -> 250.0 + (50.0 / 1.0)  = 300.0 (Perfect Win Target)
                    # Mate-in-10 -> 250.0 + (50.0 / 10.0) = 255.0 (Lower priority win)
                    score_target = min(MATE_MAX, MATE_BASE + (50.0 / max(1.0, mate_count)))
                else:
                    # Mate-in--1  -> -250.0 + (50.0 / -1.0)  = -300.0 (Maximum Loss Penalty)
                    # Mate-in--10 -> -250.0 + (50.0 / -10.0) = -255.0 (Rewarded for resistance)
                    score_target = max(-MATE_MAX, -MATE_BASE + (50.0 / min(-1.0, mate_count)))

                active_pawn_score = score_target / 100.0
            # =================================================================
            # STRATEGIC PATH B: MIDDLEGAME PIPELINE
            # =================================================================
            else:
                # Filter out explicit engine-calculated checkmates for middlegames
                if mate_val is not None:
                    continue
                if is_invalid_training_row(depth_val, knodes):
                    continue

                score_target = float(raw_score)
                # 3. Apply the 600 Guardrail to eliminate deep glitched engine values
                if abs(score_target) >= 600:
                    continue

                # 4. Filter out highly volatile tactical configurations via Q-Search
                static_score = static_evaluate(board)
                q_score = q_search(board, -float('inf'), float('inf'))
                if abs(static_score - q_score) > 40:
                    continue
                    
                active_pawn_score = score_target / 100.0

            # --- PERSPECTIVE A: Original Board Orientation ---
            is_black_turn = (board.turn == chess.BLACK)
            active_player_target = -active_pawn_score if is_black_turn else active_pawn_score

            active_indices, passive_indices = parse_fen_to_features(fen)
            active_orig, passive_orig = pad_indices(active_indices, passive_indices)

            # --- PERSPECTIVE B: Mirrored Board ---
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
