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
DATASET_NAME = "mateuszgrzyb/lichess-stockfish-normalized"
VAL_SAMPLE_SIZE = 30000

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
def dataset_generator(get_dataset_fn):
    """
    Streams and processes rows from the Lichess Hugging Face dataset.
    Extracts FEN tokens, normalizes scores relative to the side to move, 
    and yields structured mini-batches for TensorFlow.
    """
    while True:
        # Simply load the stream once
        hf_dataset = get_dataset_fn()

        for row in hf_dataset:
            fen = row.get("fen")
            raw_score = row.get("cp")
            mate = row.get("mate")
            
            try:  
                # 1. Extract Active Turn
                fen_tokens = fen.split()
                is_black_turn = (fen_tokens[1] == 'b')
                
                # 2. Assign absolute White-relative score
                if raw_score is not None:
                    raw_score = float(raw_score)
                elif mate is not None:
                    mate_val = int(mate)
                    # Calculate baseline penalty purely based on distance, ignoring sign
                    distance_penalty = abs(mate_val) * 10.0
                    
                    if mate_val > 0:
                        raw_score = 25000.0 - distance_penalty   # White forces mate
                    else:
                        raw_score = -25000.0 + distance_penalty  # Black forces mate
                else:
                    continue 

                # 2. Scale and normalize perspective
                if is_black_turn:
                    raw_score *= -1.0

                score = np.tanh(raw_score / 410.0)
            except (ValueError, TypeError, IndexError):
                continue
            
            w_feats, b_feats = parse_fen_to_features(fen)
            
            w_feats_flat = np.array(w_feats, dtype=np.float32).flatten()
            b_feats_flat = np.array(b_feats, dtype=np.float32).flatten()
            
            yield (
                {
                    "white_features": w_feats_flat, 
                    "black_features": b_feats_flat,
                    "side_to_move": np.array([is_black_turn], dtype=bool)
                },
                np.array([score], dtype=np.float32).flatten()
            )


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
    merged = layers.Concatenate(name="perspective_multiplex")([first_half, second_half])       # Shape: (Batch, 512)
    
    # Hidden Layer 2 (Standard NNUE)
    x = layers.Dense(32, activation=None, name="hidden_layer_2")(merged)     # Shape: (Batch, 32)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)

    # Hidden Layer 3 (The Missing Layer)
    x = layers.Dense(32, activation=None, name="hidden_layer_3")(x)          # Shape: (Batch, 32)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)

    # Output Layer
    raw_eval = layers.Dense(1, activation=None, name="chess_eval")(x)
    output = layers.Activation("tanh", name="normalized_eval")(raw_eval)

    model = Model(
        inputs=[white_input, black_input, stm_input],
        outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003), loss="mse")

    def load_train_stream():
        # 1. Load the main training stream
        dset = load_dataset(DATASET_NAME, split="train", streaming=True)
        # Shuffle the training stream independently
        return dset.shuffle(seed=42, buffer_size=300000)

    def load_val_stream():
        # 2. Load an independent validation stream instance
        dset = load_dataset(DATASET_NAME, split="train", streaming=True)
        # Skip the first 5,000,000 rows to ensure zero overlap with training data [2, 3]
        # Then lock down a clean, fixed evaluation sample size [2, 3]
        return dset.skip(5000000).take(VAL_SAMPLE_SIZE)

    # --- Train Dataset ---
    train_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(load_train_stream),
        output_signature=(
            {
                "white_features": tf.TensorSpec(shape=(INPUT_FEATURES,), dtype=tf.float32),
                "black_features": tf.TensorSpec(shape=(INPUT_FEATURES,), dtype=tf.float32),
                "side_to_move": tf.TensorSpec(shape=(1,), dtype=tf.bool),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    )

    # --- Validation Dataset ---
    val_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(load_val_stream),
        output_signature=(
            {
                "white_features": tf.TensorSpec(shape=(INPUT_FEATURES,), dtype=tf.float32),
                "black_features": tf.TensorSpec(shape=(INPUT_FEATURES,), dtype=tf.float32),
                "side_to_move": tf.TensorSpec(shape=(1,), dtype=tf.bool),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    )

    val_dataset = val_dataset.take(15360)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.batch(1024)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    print("\n--- Model compilation complete. Commencing Training Step ---")
    # Steps per epoch represents: Total Records / Batch Size
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=1, 
        verbose=1, 
        min_lr=1e-5
    )

    # Pass the callback into your fit runner
    model.fit(
        train_dataset, 
        steps_per_epoch=15000, 
        epochs=30, 
        validation_data=val_dataset,
        validation_steps=VAL_SAMPLE_SIZE // 1024 ,
        callbacks=[lr_scheduler])
    
    # Ideal Error is between 0.040 and 0.055
    return model

def export_dense_nnue_for_rust(model, file_path="model.nnue"):
    with open(file_path, "wb") as f:
        print("--- Commencing Weight Quantization & Serialization for Rust ---")
        
        # 1. First Layer (Accumulator Layer) -> Quantize to i16
        # Matches shape (49152, 256) and bias (256,)
        acc_layer = model.get_layer("accumulator_layer")
        w1, b1 = acc_layer.get_weights()
        
        # Scale factor 128.0 enables quick bit-shifts in Rust (>> 7)
        f.write(np.round(w1.T * 128.0).astype(np.int16).tobytes())
        f.write(np.round(b1 * 128.0).astype(np.int16).tobytes())
        print(f"-> Accumulator Layer weights serialized. Shape: {w1.shape}")

        # 2. Hidden Layer 2 -> Quantize to i8 weights / i32 biases
        # Matches shape (512, 32) and bias (32,)
        # Note: We look up by index or custom name to avoid default naming collisions
        layer2 = model.get_layer("hidden_layer_2") 
        w2, b2 = layer2.get_weights()
        f.write(np.round(w2.T * 64.0).astype(np.int8).tobytes())
        f.write(np.round(b2 * 64.0).astype(np.int32).tobytes())
        print(f"-> Hidden Layer 2 weights serialized. Shape: {w2.shape}")

        # 3. Hidden Layer 3 -> Quantize to i8 weights / i32 biases
        # Matches shape (32, 32) and bias (32,)
        layer3 = model.get_layer("hidden_layer_3")
        w3, b3 = layer3.get_weights()
        f.write(np.round(w3.T * 64.0).astype(np.int8).tobytes())
        f.write(np.round(b3 * 64.0).astype(np.int32).tobytes())
        print(f"-> Hidden Layer 3 weights serialized. Shape: {w3.shape}")

        # 4. Output Layer -> Quantize to i8 weights / i32 biases
        # Matches shape (32, 1) and bias (1,)
        output_layer = model.get_layer("chess_eval")
        w4, b4 = output_layer.get_weights()
        f.write(np.round(w4.T * 127.0).astype(np.int16).tobytes())
        f.write(np.round(b4 * 127.0).astype(np.int32).tobytes())
        print(f"-> Output Layer weights serialized. Shape: {w4.shape}")

    print(f"\n[SUCCESS] NNUE file successfully compiled and written to: {file_path}")


if __name__ == "__main__":
    # Load the streaming dataset directly from Hugging Face
    trained_model = train_nnue_on_fens()

    export_dense_nnue_for_rust(trained_model, "nnue_weights.bin")