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
PARQUET_FILE = os.path.join(SCRIPT_DIR, "data", "data_0000.parquet")
BINARY_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "clean_binary_shards")

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

    active_indices = []
    passive_indices = []
    
    for p_type, p_sq in pieces:
        # --- WHITE PERSPECTIVE ---
        w_idx = (white_king_sq * 768) + (p_type * 64) + p_sq
        active_indices.append(w_idx)
        
        # --- BLACK PERSPECTIVE (Flipped & Rotated) ---
        b_type = (p_type + 6) % 12        
        b_sq = p_sq ^ 63
        b_king_sq_rotated = black_king_sq ^ 63
        
        b_idx = (b_king_sq_rotated * 768) + (b_type * 64) + b_sq
        passive_indices.append(b_idx)
        
    return active_indices, passive_indices

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

def save_parquet_shard(batch_records, output_dir, file_counter):
    """Helper function to build a structured dataframe and write to Parquet format."""
    prefix = f"clean_wave_{file_counter}"
    output_path = os.path.join(output_dir, f"{prefix}.parquet")
    
    df = pd.DataFrame(batch_records)
    df.to_parquet(output_path, compression="snappy", index=False)
    print(f"[Exported Shard {file_counter}] Saved {len(df)} clean positions to {output_path}.")

def run_parquet_cleaning_pass(parquet_path, output_dir, samples_per_file=DATA_SIZE):
    os.makedirs(output_dir, exist_ok=True)
    
    batch_records = []
    file_counter = 1
    total_processed = 0

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

        # 5. Standardize target perspective to match your Side-To-Move network design
        y_pawn_target = score_target / 100.0
        is_black_turn = (board.turn == chess.BLACK)
        if is_black_turn:
            y_pawn_target = -y_pawn_target

        # 6. Extract HalfKA indices and enforce uniform padding layout
        w_indices, b_indices = parse_fen_to_features(fen)
        fen_hash = int(hashlib.md5(fen.encode('utf-8')).hexdigest(), 16) % (10**10)
        
        def pad_indices(w_idx, b_idx):
            w_pad = np.full(MAX_PIECES, -1, dtype=np.int32)
            b_pad = np.full(MAX_PIECES, -1, dtype=np.int32)
            w_pad[:min(len(w_idx), MAX_PIECES)] = w_idx[:min(len(w_idx), MAX_PIECES)]
            b_pad[:min(len(b_idx), MAX_PIECES)] = b_idx[:min(len(b_idx), MAX_PIECES)]
            return w_pad.tolist(), b_pad.tolist()

        # Scale raw score to a pawn target
        y_pawn_target = float(raw_score) / 100.0
        is_black_turn = (board.turn == chess.BLACK)

        # ----------------------------------------------------
        # PERSPECTIVE A: Original Board Orientation
        # ----------------------------------------------------
        w_orig, b_orig = pad_indices(w_indices, b_indices)
        batch_records.append({
            'white_indices': w_orig,
            'black_indices': b_orig,
            'is_black_turn': 1.0 if is_black_turn else 0.0,
            'target': -y_pawn_target if is_black_turn else y_pawn_target,
            'position_hash': fen_hash
        })

        # ----------------------------------------------------
        # PERSPECTIVE B: Mirrored Board.
        # ----------------------------------------------------
        batch_records.append({
            'white_indices': b_orig, 
            'black_indices': w_orig, 
            'is_black_turn': 0.0 if is_black_turn else 1.0,
            'target': y_pawn_target if is_black_turn else -y_pawn_target,
            'position_hash': fen_hash
        })
        
        if len(batch_records) >= samples_per_file:
            save_parquet_shard(batch_records, output_dir, file_counter)
            total_processed += len(batch_records)
            batch_records = []
            file_counter += 1

    # Flush remaining records from memory
    if len(batch_records) > 0:
        save_parquet_shard(batch_records, output_dir, file_counter)
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
    raw_parquet_pattern = os.path.join(SCRIPT_DIR, "data", "data_*.parquet")
    raw_files = sorted(glob.glob(raw_parquet_pattern))
    
    if not raw_files:
        print(f"[CRITICAL ERROR] No source Parquet files detected matching data_*.parquet")
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