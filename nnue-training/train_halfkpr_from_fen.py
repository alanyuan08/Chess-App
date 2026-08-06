import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers, Model
from datasets import load_dataset
import math

# --- CONSTANTS ---
INPUT_FEATURES = 64 * 64 * 12  # 49,152
HIDDEN_SIZE = 256
SCALE_MAX = 1.0

BATCH_SIZE = 256
DATASET_NAME = "mateuszgrzyb/lichess-stockfish-normalized"
BIN_SAVE_PATH = "nnue_weights.bin"

VAL_SAMPLE_SIZE = 15360
VAL_BATCH_SIZE = 1024
VAL_START_ROW = 10000000 

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
        # Fixed Dimension Strides: (KingSq * 768) + (PieceType * 64) + PieceSq
        # 12 piece types * 64 squares = 768 slots per king square position block
        w_idx = (white_king_sq * 768) + (p_type * 64) + p_sq
        white_features[w_idx] = 1.0
        
        # --- BLACK PERSPECTIVE (Flipped & Rotated) ---
        # 1. Flip color associations cleanly: White (0..5) <-> Black (6..11)
        b_type = (p_type + 6) % 12
        
        # 2. pply a full 180-degree board rotation using bitwise XOR 63
        # This simultaneously mirrors the ranks vertically AND the files horizontally (A1 <-> H8)
        b_sq = p_sq ^ 63
        b_king_sq_rotated = black_king_sq ^ 63
        
        b_idx = (b_king_sq_rotated * 768) + (b_type * 64) + b_sq
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
                cp_val = float(raw_score) if (raw_score is not None and not math.isnan(raw_score)) else None
                mate_val = int(mate) if (mate is not None and not math.isnan(mate)) else None
                
                # 4. Assign objective, White-relative baseline evaluation scores
                if cp_val is not None:
                    score_target = cp_val
                elif mate_val is not None and mate_val != 0:
                    # Penalize slow mates. 25000 sits safely above standard CP limits.
                    distance_penalty = abs(mate_val) * 10.0
                    if mate_val > 0:
                        score_target = 25000.0 - distance_penalty  # White forces mate
                    else:
                        score_target = -25000.0 + distance_penalty # Black forces mate
                else:
                    continue

                if is_black_turn:
                    score_target = -score_target

                score = np.tanh(score_target / 410.0)
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
    # 1. Inputs
    white_input = layers.Input(shape=(INPUT_FEATURES,), name="white_features")
    black_input = layers.Input(shape=(INPUT_FEATURES,), name="black_features")
    stm_input = layers.Input(shape=(1,), dtype="bool", name="side_to_move")

    # 2. Shared Accumulator Layer (HalfK Virtual Weights)
    accumulator = layers.Dense(256, activation=None, name="accumulator_layer") 
    w_acc = accumulator(white_input)
    b_acc = accumulator(black_input)

    # 3. Clipped ReLU Activation (ReLU1 / Bounded ReLU)
    w_act = keras.ops.clip(w_acc, 0.0, SCALE_MAX)
    b_act = keras.ops.clip(b_acc, 0.0, SCALE_MAX)

    # 4. Cast boolean mask to float for branchless tensor operations
    # Black's turn -> stm_float = 1.0 | White's Turn -> stm_float = 0.0
    stm_float = keras.ops.cast(stm_input, dtype="float32")
    
    # 5. Concat into the final accumulator vector (Shape: Batch, 512)
    first_half = stm_float * b_act + (1.0 - stm_float) * w_act
    second_half = stm_float * w_act + (1.0 - stm_float) * b_act
    merged = layers.Concatenate(name="perspective_multiplex")([first_half, second_half])       # Shape: (Batch, 512)
    
    # 6. Hidden Layer 2 with ReLU1 activation
    x = layers.Dense(64, activation=None, name="hidden_layer_2")(merged)     # Shape: (Batch, 64)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)

    # 7. Hidden Layer 3 with ReLU1 activation
    x = layers.Dense(32, activation=None, name="hidden_layer_3")(x)          # Shape: (Batch, 32)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)

    # 8. Output Layer mapped to Tanh range [-1.0, 1.0]
    raw_eval = layers.Dense(1, activation=None, name="chess_eval")(x)
    output = layers.Activation("tanh", name="normalized_eval")(raw_eval)

    model = Model(
        inputs=[white_input, black_input, stm_input],
        outputs=output)
    
    total_training_steps = int(15000 * 30 * 1.1)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=total_training_steps,
        alpha=0.0100
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), 
        loss="mse",
        metrics=["mae"]
    )

    def load_train_stream():
        # 1. Load the main training stream
        dset = load_dataset(DATASET_NAME, split="train", streaming=True)
        # Shuffle the training stream independently
        return dset.shuffle(seed=42, buffer_size=400000)

    def load_val_stream():
        dset = load_dataset(DATASET_NAME, split="train", streaming=True)
        
        # Move 10 million rows deep to build a bulletproof wall against training data leakage
        dset = dset.skip(VAL_START_ROW)
        return dset.take(VAL_SAMPLE_SIZE)

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
    ).repeat()

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

    val_dataset = val_dataset.take(VAL_SAMPLE_SIZE)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.batch(VAL_BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    print("\n--- Model compilation complete. Commencing Training Step ---")
    checkpoint_path = "best_chess_nnue.keras"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # Pass the callback into your fit runner
    model.fit(
        train_dataset, 
        steps_per_epoch=15000, 
        epochs=3, 
        validation_data=val_dataset,
        validation_steps=VAL_SAMPLE_SIZE // VAL_BATCH_SIZE,
        callbacks=[checkpoint_cb])
    
    # Ideal Error is between 0.040 and 0.055
    return model

def export_dense_nnue_for_rust(model, file_path="model.nnue"):
    with open(file_path, "wb") as f:
        print("--- Commencing Weight Quantization & Serialization for Rust ---")
        
        # 1. Accumulator Layer (49152 -> 256)
        acc_layer = model.get_layer("accumulator_layer")
        w1, b1 = acc_layer.get_weights()
        
        # Scale by 128.0. Since activations are capped at 1.0, 
        f.write(np.round(w1.T * 128.0).astype(np.int16).tobytes())
        f.write(np.round(b1 * 128.0).astype(np.int16).tobytes())
        print(f"-> Accumulator Layer serialized. Shape: {w1.shape} (i16)")

        # 2. Hidden Layer 2 (512 -> 64) - Dual Accumlator
        # [Active Player - 256 / Passive Player - 256] Inputs
        layer2 = model.get_layer("hidden_layer_2") 
        w2, b2 = layer2.get_weights()
        
        # Bias scale = Accumulator weight scale (128) * Hidden 2 weight scale (32) = 4096
        w2_quant = np.clip(np.round(w2.T * 32.0), -128, 127).astype(np.int8)
        b2_quant = np.round(b2 * 4096.0).astype(np.int32)
        
        f.write(w2_quant.tobytes())
        f.write(b2_quant.tobytes())
        print(f"-> Hidden Layer 2 serialized. Shape: {w2.shape} (i8 / i32)")

        # 3. Hidden Layer 3 (64 -> 32)
        # Shift >> 7 / Div 128 
        layer3 = model.get_layer("hidden_layer_3")
        w3, b3 = layer3.get_weights()
        
        # Bias scale = Layer 2 output scale (32) * Hidden 3 weight scale (32) = 1024
        w3_quant = np.clip(np.round(w3.T * 32.0), -128, 127).astype(np.int8)
        b3_quant = np.round(b3 * 1024.0).astype(np.int32)
        
        f.write(w3_quant.tobytes())
        f.write(b3_quant.tobytes())
        print(f"-> Hidden Layer 3 serialized. Shape: {w3.shape} (i8 / i32)")

        # 4. Output Layer (32 -> 1)
        # Shift >> 5 / Div 32 
        output_layer = model.get_layer("chess_eval")
        w4, b4 = output_layer.get_weights()
        
        # (Scale x 127) + Bias(4064)
        w4_quant = np.clip(np.round(w4.T * 127.0), -128, 127).astype(np.int8)
        b4_quant = np.round(b4 * 4064.0).astype(np.int32)
        
        f.write(w4_quant.tobytes())
        f.write(b4_quant.tobytes())
        print(f"-> Output Layer serialized. Shape: {w4.shape} (i8 / i32)")

    print(f"\n[SUCCESS] Safe NNUE file successfully compiled and written to: {file_path}")

if __name__ == "__main__":
    # Load the streaming dataset directly from Hugging Face
    trained_model = train_nnue_on_fens()

    export_dense_nnue_for_rust(trained_model, BIN_SAVE_PATH)