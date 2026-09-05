# dataset_exporter.py
import os
import sys
import numpy as np
import chess
import pandas as pd
import hashlib

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
BINARY_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "production_shards")

# --- GLOBAL TRACKERS ---
file_counter = 0
total_processed = 0

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
    """
    Parses a standard FEN string into sparse categorical indices 
    representing active features for White and Black perspective.
    """
    parts = fen_string.split()
    board_part = parts[0]

    turn_part = parts[1]  # 'w' or 'b'
    is_black_turn = (turn_part == 'b')
    
    # Expand numeric spaces in FEN to empty string dots for alignment
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
        
    # Dynamically return based on active player turn context
    if is_black_turn:
        return black_perspective_indices, white_perspective_indices
    else:
        return white_perspective_indices, black_perspective_indices

def get_endgame_piece_count(fen_string: str) -> int:
    board_part = fen_string.split()[0]
    target_pieces = set("pnbrqPNBRQ")
    total_pieces = sum(1 for char in board_part if char in target_pieces)
    return total_pieces

def is_invalid_training_row(depth_str, fen_string: str) -> bool:
    depth = int(depth_str)
    fen_pieces_count = get_endgame_piece_count(fen_string)
    if fen_pieces_count <= 12:
        return depth < 32
    else:
        return depth < 20

def save_parquet_shard(batch_records, output_dir):
    """Helper function to build a structured dataframe and write to Parquet format."""
    global file_counter
    prefix = f"production_data_{file_counter}"
    output_path = os.path.join(output_dir, f"{prefix}.parquet")
    
    df = pd.DataFrame(batch_records)
    df.to_parquet(output_path, compression="snappy", index=False)
    print(f"[Exported Shard {file_counter}] Saved {len(df)} clean positions to {output_path}.")

        
def pad_indices(a_idx, p_idx):
    a_pad = np.full(MAX_PIECES, INPUT_FEATURES, dtype=np.int32)
    p_pad = np.full(MAX_PIECES, INPUT_FEATURES, dtype=np.int32)
    a_pad[:min(len(a_idx), MAX_PIECES)] = a_idx[:min(len(a_idx), MAX_PIECES)]
    p_pad[:min(len(p_idx), MAX_PIECES)] = p_idx[:min(len(p_idx), MAX_PIECES)]
    return a_pad.tolist(), p_pad.tolist()

def run_parquet_cleaning_pass(parquet_path, output_dir, samples_per_file=DATA_SIZE):
    os.makedirs(output_dir, exist_ok=True)
    
    batch_records = []
    global total_processed
    global file_counter

    print(f"\n--- Commencing High-Speed Parquet Scan: {parquet_path} ---")
    
    # Using low-RAM chunk reading to sweep columns seamlessly
    # Parquet parsing handles streaming columns with zero serialization lag
    parquet_df = pd.read_parquet(parquet_path, columns=['fen', 'cp', 'depth', 'mate'])
    
    for idx, row in parquet_df.iterrows():
        fen = row.get("fen")
        raw_score = row.get("cp")
        depth_val = row.get("depth")
        mate_val = row.get("mate")

        # 1. Skip explicit text mates or shallow searches
        if mate_val is not None and not pd.isna(mate_val):
            continue
        if depth_val is not None and is_invalid_training_row(depth_val, fen):
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
        if abs(static_score - q_score) > 120:
            continue
        
        active_pawn_score = float(raw_score) / 100.0

        # --- PERSPECTIVE A: Original Board Orientation ---
        active_indices, passive_indices = parse_fen_to_features(fen)

        # Scale raw score to a pawn target
        is_black_turn = (board.turn == chess.BLACK)
        active_player_target = -active_pawn_score if is_black_turn else active_pawn_score

        active_orig, passive_orig = pad_indices(active_indices, passive_indices)
        batch_records.append({
            'active_indices': active_orig,
            'passive_indices': passive_orig,
            'target': active_player_target,
        })

        # --- PERSPECTIVE B: Mirrored Board ---
        rotated_board = board.mirror().transform(chess.flip_horizontal)
        rotated_fen = rotated_board.fen()

        active_rot, passive_rot = parse_fen_to_features(rotated_fen)
        active_rot_pad, passive_rot_pad = pad_indices(active_rot, passive_rot)

        is_black_turn_rot = (rotated_board.turn == chess.BLACK)
        target_rot = -active_pawn_score if is_black_turn_rot else active_pawn_score
        
        batch_records.append({
            'active_indices': active_rot_pad,
            'passive_indices': passive_rot_pad,
            'target': target_rot,
        })
        
        if len(batch_records) >= samples_per_file:
            save_parquet_shard(batch_records, output_dir)
            total_processed += len(batch_records)
            batch_records = []
            file_counter += 1

    # Flush remaining records from memory
    if len(batch_records) > 0:
        save_parquet_shard(batch_records, output_dir)
        total_processed += len(batch_records)

    print(f"\n[SUCCESS] Processing Complete! Total clean positions serialized: {total_processed}")

# =====================================================================
# GLOBAL ENVELOPE EXECUTION MAIN FUNCTION
# =====================================================================
def main():
    print("=====================================================================")
    print("          COMMENCING MASSIVE MULTI-PARQUET EXPORT PASS               ")
    print("=====================================================================")
    
    import glob
    
    # Dynamically discover all downloaded raw parquet files inside your data/ folder
    raw_parquet_pattern = os.path.join(SCRIPT_DIR, "data_dedup_mixed", "*.parquet")
    raw_files = sorted(glob.glob(raw_parquet_pattern))
    
    if not raw_files:
        print(f"[CRITICAL ERROR] No source Parquet files detected matching *.parquet")
        print("Please run your shell download script to seed the data/ directory first!")
        sys.exit(1)
        
    print(f"Detected {len(raw_files)} raw shards ready for feature cleaning.")
    
    # Loop over every single downloaded raw file sequentially
    for file_idx, parquet_path in enumerate(raw_files):
        print(f"\n[Processing Raw File {file_idx+1}/{len(raw_files)}]")
        try:
            # We pass the file_counter to the function using the index, or let it accumulate globally
            run_parquet_cleaning_pass(
                parquet_path=parquet_path, 
                output_dir=BINARY_OUTPUT_DIR,
                samples_per_file=DATA_SIZE
            )
        except KeyboardInterrupt:
            print("\n[PROCESS PAUSED] Pipeline halted cleanly by request.")
            break
            
    print("\n=====================================================================")
    print("   ALL RAW PARQUET SHARDS FULLY GUARDRAILED, CLEANED, AND EXPORTED   ")
    print("=====================================================================")

if __name__ == "__main__":
    main()