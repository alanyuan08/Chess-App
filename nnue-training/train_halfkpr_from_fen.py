import re
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers, Model
from datasets import load_dataset

# --- CONSTANTS ---
INPUT_FEATURES = 64 * 64 * 12  # 49,152
HIDDEN_SIZE = 256
SCALE_MAX = 127.0

BATCH_SIZE = 256

# Map FEN character to an integer type 0-11
PIECE_MAP = {
    'P': 0, 'B': 1, 'N': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'b': 7, 'n': 8, 'r': 9, 'q': 10, 'k': 11
}

# --- 1. FEN TO NNUE ARCHITECTURE PARSER ---
def parse_fen_to_features(fen_string):
    """
    Parses a standard FEN string into sparse categorical indices 
    representing active features for White and Black perspective.
    """
    parts = fen_string.split()
    board_part = parts[0]
    
    # Expand numeric spaces in FEN to empty string dots for alignment
    rows = board_part.split('/')
    clean_board = ""
    for row in rows:
        for char in row:
            if char.isdigit():
                clean_board += '.' * int(char)
            else:
                clean_board += char
                
    # Track piece properties and locate Kings
    pieces = []  # List of tuples: (piece_type_id, square_idx)
    white_king_sq = 0
    black_king_sq = 0
    
    for sq in range(64):
        char = clean_board[sq]
        if char == '.':
            continue
        piece_type = PIECE_MAP[char]
        pieces.append((piece_type, sq))
        if char == 'K':
            white_king_sq = sq
        elif char == 'k':
            black_king_sq = sq

    # Build active features arrays for White and Black
    white_features = np.zeros(INPUT_FEATURES, dtype=np.float32)
    black_features = np.zeros(INPUT_FEATURES, dtype=np.float32)
    
    for p_type, p_sq in pieces:
        # --- WHITE PERSPECTIVE ---
        # Formula: PieceType + (12 * PieceSquare) + (12 * 64 * KingSquare)
        w_idx = p_type + (12 * p_sq) + (12 * 64 * white_king_sq)
        white_features[w_idx] = 1.0
        
        # --- BLACK PERSPECTIVE (Flipped) ---
        # 1. Flip colors: invert white ids (0..5) to black ids (6..11) and vice versa
        b_type = (p_type + 6) % 12
        # 2. Mirror squares vertically/horizontally via bitwise XOR 56 (A1 becomes A8)
        b_sq = p_sq ^ 56
        b_king_sq_flipped = black_king_sq ^ 56
        
        b_idx = b_type + (12 * b_sq) + (12 * 64 * b_king_sq_flipped)
        black_features[b_idx] = 1.0
        
    return white_features, black_features

# --- 2. PIPELINE GENERATOR FOR TENSORFLOW ---
def dataset_generator(hf_dataset, batch_size=128, min_depth=20):
    """
    Streams and processes rows from the Lichess Hugging Face dataset.
    Extracts FEN tokens, normalizes scores relative to the side to move, 
    and yields structured mini-batches for TensorFlow.
    """
    while True:
        w_batch, b_batch, stm_batch, y_batch = [], [], [], []
        for row in hf_dataset:
            fen = row.get("fen")
            raw_score = row.get("cp")
            depth_str = row.get("depth") # Depth field from Lichess dataset
            
            # Check for depth first
            if not fen or depth_str is None:
                continue

            try:  
                # 1. Filter by minimum search depth
                if int(depth_str) < min_depth:
                    continue

                # 2. Extract Active Turn
                fen_tokens = fen.split()
                is_black_turn = (fen_tokens[1] == 'b')
                
                # 3. Scale and normalize perspective
                raw_score = float(raw_score)
                if is_black_turn:
                    raw_score *= -1.0

                score = np.clip(raw_score / 400.0, -3.0, 3.0)
            except (ValueError, TypeError, IndexError):
                continue
            
            w_feats, b_feats = parse_fen_to_features(fen)
            w_batch.append(w_feats)
            b_batch.append(b_feats)
            stm_batch.append([is_black_turn])
            y_batch.append(score)
            
            if len(w_batch) == batch_size:
                yield (
                    {
                        "white_features": np.array(w_batch), 
                        "black_features": np.array(b_batch),
                        "side_to_move": np.array(stm_batch, dtype=bool)
                    },
                    np.array(y_batch, dtype=np.float32).reshape(-1, 1)
                )
                # Reset all trackers completely for the next iteration step
                w_batch, b_batch, stm_batch, y_batch = [], [], [], []

# --- 4. MODEL DESIGN & TRAINING RUNNER ---
def train_nnue_on_fens():
    # Inputs
    white_input = layers.Input(shape=(INPUT_FEATURES,), name="white_features")
    black_input = layers.Input(shape=(INPUT_FEATURES,), name="black_features")
    stm_input = layers.Input(shape=(1,), dtype="bool", name="side_to_move")

    # Shared Accumulator Layer
    transformer = layers.Dense(256, activation=None, name="accumulator_layer") 
    w_acc = transformer(white_input)
    b_acc = transformer(black_input)

    # Clipped ReLU Activation: clamp(0.0, 1.0)
    w_act = keras.ops.clip(w_acc, 0.0, SCALE_MAX)
    b_act = keras.ops.clip(b_acc, 0.0, SCALE_MAX)

    # Cast boolean mask to float for safe, broadcastable mathematical selection
    # Black's turn, stm_float == 1.0, White's Turn stm_float == 0.0
    stm_float = keras.ops.cast(stm_input, dtype="float32")
    
    # Multiplex perspectives seamlessly
    first_half = stm_float * b_act + (1.0 - stm_float) * w_act
    second_half = stm_float * w_act + (1.0 - stm_float) * b_act

    # Concat and output feed forward structure
    merged = layers.Concatenate()([first_half, second_half])       # Shape: (Batch, 512)
    
    # Hidden Layer 2 (Standard NNUE)
    x = layers.Dense(32, activation="relu")(merged)     # Shape: (Batch, 32)
    
    # Hidden Layer 3 (The Missing Layer)
    x = layers.Dense(32, activation="relu")(x)          # Shape: (Batch, 32)
    
    # Output Layer
    raw_eval = layers.Dense(1, activation="linear", name="chess_eval")(x)

    output = keras.ops.clip(raw_eval, -3.0, 3.0)

    model = Model(inputs=[white_input, black_input, stm_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    # Load the streaming dataset directly from Hugging Face

   # Factory function to fresh-load the stream every time from_generator requests it
    def create_fresh_generator():
        # NOTE: dataset_generator function must accept a factory/fresh instance
        dset = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
        return dataset_generator(dset, batch_size=BATCH_SIZE)

    train_dataset = tf.data.Dataset.from_generator(
        create_fresh_generator,
        output_signature=(
            {
                "white_features": tf.TensorSpec(shape=(BATCH_SIZE, INPUT_FEATURES), dtype=tf.float32),
                "black_features": tf.TensorSpec(shape=(BATCH_SIZE, INPUT_FEATURES), dtype=tf.float32),
                "side_to_move": tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.bool),
            },
            tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.float32)
        )
    )

    print("\n--- Model compilation complete. Commencing Training Step ---")
    # Steps per epoch represents: Total Records / Batch Size
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', 
        factor=0.5, 
        patience=3, 
        verbose=1, 
        min_lr=1e-5
    )

    # Pass the callback into your fit runner
    model.fit(train_dataset, steps_per_epoch=15000, epochs=30, callbacks=[lr_scheduler])
    
    return model

def export_dense_nnue_for_rust(model, file_path="model.nnue"):
    with open(file_path, "wb") as f:
        print("--- Commencing Weight Quantization & Serialization for Rust ---")
        
        # 1. First Layer (Accumulator Layer) -> Quantize to i16
        # Matches shape (49152, 256) and bias (256,)
        acc_layer = model.get_layer("accumulator_layer")
        w1, b1 = acc_layer.get_weights()
        
        # Scale factor 127.0 matches your clamp(0.0, 1.0) normalized range
        f.write(np.round(w1 * 127.0).astype(np.int16).tobytes())
        f.write(np.round(b1 * 127.0).astype(np.int16).tobytes())
        print(f"-> Accumulator Layer weights serialized. Shape: {w1.shape}")

        # 2. Hidden Layer 2 -> Quantize to i8 weights / i32 biases
        # Matches shape (512, 32) and bias (32,)
        # Note: We look up by index or custom name to avoid default naming collisions
        layer2 = [l for l in model.layers if isinstance(l, layers.Dense) and l.name != "accumulator_layer" and l.name != "chess_eval"][0]
        w2, b2 = layer2.get_weights()
        f.write(np.round(w2 * 64.0).astype(np.int8).tobytes())
        f.write(np.round(b2 * 64.0).astype(np.int32).tobytes())
        print(f"-> Hidden Layer 2 weights serialized. Shape: {w2.shape}")

        # 3. Hidden Layer 3 -> Quantize to i8 weights / i32 biases
        # Matches shape (32, 32) and bias (32,)
        layer3 = [l for l in model.layers if isinstance(l, layers.Dense) and l.name != "accumulator_layer" and l.name != "chess_eval"][1]
        w3, b3 = layer3.get_weights()
        f.write(np.round(w3 * 64.0).astype(np.int8).tobytes())
        f.write(np.round(b3 * 64.0).astype(np.int32).tobytes())
        print(f"-> Hidden Layer 3 weights serialized. Shape: {w3.shape}")

        # 4. Output Layer -> Quantize to i8 weights / i32 biases
        # Matches shape (32, 1) and bias (1,)
        output_layer = model.get_layer("chess_eval")
        w4, b4 = output_layer.get_weights()
        f.write(np.round(w4 * 64.0).astype(np.int8).tobytes())
        f.write(np.round(b4 * 64.0).astype(np.int32).tobytes())
        print(f"-> Output Layer weights serialized. Shape: {w4.shape}")

    print(f"\n[SUCCESS] NNUE file successfully compiled and written to: {file_path}")


if __name__ == "__main__":
    # Load the streaming dataset directly from Hugging Face
    trained_model = train_nnue_on_fens()

    export_dense_nnue_for_rust(trained_model, "nnue_weights.bin")