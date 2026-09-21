import os
import sys
import glob
import numpy as np
import chess
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- INITIALIZATION ENGINE CONSTANTS ---
DATA_SIZE = 2_000_000
INPUT_FEATURES = 64 * 64 * 12  # Dual-Perspective HalfKA Dimension (49,152)
MAX_PIECES = 32  

# Map FEN character to an integer type 0-11
PIECE_MAP = {
    'P': 0, 'B': 1, 'N': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'b': 7, 'n': 8, 'r': 9, 'q': 10, 'k': 11
}

PIECE_VALUES = {
    chess.PAWN: 100, 
    chess.KNIGHT: 320, 
    chess.BISHOP: 330, 
    chess.ROOK: 500, 
    chess.QUEEN: 900, 
    chess.KING: 20000
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

def q_search(b, alpha, beta, depth=0, max_depth=12):
    """Highly stable, fast pseudo-legal Negamax Quiescence Search."""
    if depth >= max_depth:
        return static_evaluate(b)

    in_check = b.is_check()
    static_eval = static_evaluate(b)
        
    if not in_check:
        if static_eval >= beta:
            return static_eval
        if static_eval > alpha:
            alpha = static_eval

    for move in b.generate_pseudo_legal_moves(b.turn):
        if in_check or b.is_capture(move) or move.promotion:
            if not b.is_legal(move):
                continue

            b.push(move)
            score = -q_search(b, -beta, -alpha, depth + 1, max_depth)
            b.pop()

            if score >= beta:
                return score
            if score > alpha:
                alpha = score

    return alpha

def parse_fen_to_features(fen_string):
    """Parses a standard FEN string into sparse categorical indices."""
    parts = fen_string.split()
    board_part = parts[0]
    turn_part = parts[1]
    is_black_turn = (turn_part == 'b')
        
    rows = board_part.split('/')
    rows.reverse()
    
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
        
    for sq in range(64):
        char = clean_board[sq]
        if char == '.':
            continue
        piece_type_id = PIECE_MAP[char]
        pieces.append((piece_type_id, sq))
        if char == 'K':
            white_king_sq = sq
        elif char == 'k':
            black_king_sq = sq

    white_perspective_indices = []
    black_perspective_indices = []
        
    for p_type, p_sq in pieces:
        # --- WHITE PERSPECTIVE ---
        w_idx = (white_king_sq * 768) + (p_type * 64) + p_sq
        white_perspective_indices.append(w_idx)
                
        # --- BLACK PERSPECTIVE (Flipped & Rotated) ---
        b_type = (p_type + 6) % 12
        b_sq = p_sq ^ 63
        b_king_sq_rotated = black_king_sq ^ 63
                
        b_idx = (b_king_sq_rotated * 768) + (b_type * 64) + b_sq
        black_perspective_indices.append(b_idx)
            
    if is_black_turn:
        return black_perspective_indices, white_perspective_indices
    else:
        return white_perspective_indices, black_perspective_indices

def is_invalid_training_row(depth_str) -> bool:
    return int(depth_str) < 20

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
    """Processes a single parquet file in an isolated worker process."""
    base_name = os.path.splitext(os.path.basename(parquet_path))[0]
    os.makedirs(output_dir, exist_ok=True)
        
    batch_records = []
    file_processed_count = 0
    local_shard_counter = 0

    print(f"[Start] Worker processing shard: {parquet_path}")
        
    try:
        parquet_df = pd.read_parquet(parquet_path, columns=['fen', 'cp', 'depth', 'mate'])
    except Exception as e:
        print(f"[Error] Failed to read {parquet_path}: {e}")
        return 0

    for idx, row in parquet_df.iterrows():
        fen = row.get("fen")
        raw_score = row.get("cp")
        depth_val = row.get("depth")
        mate_val = row.get("mate")

        if not isinstance(fen, str) or '\x00' in fen or fen.count('/') != 7:
            print(f"[Warning] Skipped corrupted FEN at row {idx} in {base_name}")
            continue

        # 1. Skip explicit text mates or shallow searches
        if mate_val is not None and not pd.isna(mate_val):
            continue
        if depth_val is not None and is_invalid_training_row(depth_val):
            continue
        if raw_score is None or pd.isna(raw_score):
            continue

        board = chess.Board(fen)

        # 2. Skip for Check / Stalemate / Insufficient Material
        if board.is_check() or board.is_stalemate() or board.is_insufficient_material():
            continue

        # 3. Apply the 1200 Guardrail to eliminate deep glitched engine values
        score_target = float(raw_score)
        if abs(score_target) >= 1200:
            continue

        # 4. Filter out highly volatile tactical configurations via Q-Search
        static_score = static_evaluate(board)
        q_score = q_search(board, -float('inf'), float('inf'))
        if abs(static_score - q_score) > 40:
            continue
                
        active_pawn_score = float(raw_score) / 100.0

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

        if len(batch_records) + 2 > samples_per_file:
            save_parquet_shard(batch_records, output_dir, base_name, local_shard_counter)
            file_processed_count += len(batch_records)
            batch_records = []
            local_shard_counter += 1
                
        # Append rotated perspective
        batch_records.append({
            'active_indices': active_rot_pad,
            'passive_indices': passive_rot_pad,
            'target': active_player_target,
            'depth': depth_val,
            'fen': rotated_fen
        })

        # Append original perspective
        batch_records.append({
            'active_indices': active_orig,
            'passive_indices': passive_orig,
            'target': active_player_target,
            'depth': depth_val,
            'fen': fen
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
    print(f"Spawning 4 worker processes")
    
    total_global_processed = 0
    
    # Process Pool Executor manages CPU-bound processing scales automatically
    with ProcessPoolExecutor(max_workers=4) as executor:
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
    main()
