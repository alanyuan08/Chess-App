import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers, Model
from huggingface_hub import HfApi
from datasets import load_dataset
import math
import chess

# --- CONSTANTS ---
INPUT_FEATURES = 64 * 64 * 12  # 49,152
HIDDEN_SIZE = 256
SCALE_MAX = 1.0

BATCH_SIZE = 256
DATASET_NAME = "Lichess/chess-position-evaluations"
BIN_SAVE_PATH = "nnue_weights.bin"

VAL_SAMPLE_SIZE = 15360
VAL_BATCH_SIZE = 256
VAL_START_ROW = 10000 

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
        # Flip color associations cleanly: White (0..5) <-> Black (6..11)
        b_type = (p_type + 6) % 12
        
        # 2. Apply a full 180-degree board rotation using bitwise XOR 63
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
                # 1. Filter for Quiescent Data (Tactical Silence)
                board = chess.Board(fen)
                
                # Rule A: Skip if a forced tactical checkmate sequence is imminent
                if mate is not None and not math.isnan(mate):
                    continue

                # Rule B: Skip if the side-to-move is currently under an active check
                if board.is_check():
                    continue

                # Rule C: Skip if there are immediate tactical, forcing, or promoting legal options
                # This ensures the network learns static values, not temporary tactical spikes
                is_tactical = False
                for move in board.legal_moves:
                    # Check for regular captures and en passant captures
                    if board.is_capture(move):
                        is_tactical = True
                        break
                    
                    # Check for pawn promotions (e.g., promoting to a Queen)
                    if move.promotion is not None:
                        is_tactical = True
                        break

                if is_tactical:
                    continue

                # 2. Extract Active Turn
                fen_tokens = fen.split()
                is_black_turn = (fen_tokens[1] == 'b')
                score_target = 0.0
                
                # 3. Drop Positions with Known Mate 
                if raw_score is not None and not math.isnan(float(raw_score)):
                    score_target = float(raw_score) 
                else:
                    continue 

                # 4. Invert for Side to move
                if is_black_turn:
                    score_target = -score_target

                # 5. Clip Score
                if score_target > 1000.0:  
                    score_target = 1000.0
                elif score_target < -1000.0: 
                    score_target = -1000.0

                # 6. Convert to Pawns and Apply Sigmoid Transformation
                # This naturally handles high scores asymptotically without a hard 1000 CP cap.
                # Alpha=0.6 maps a +1.0 pawn advantage to ~65% win probability.
                pawn_units = score_target / 100.0
                alpha = 0.6
                win_probability = 1.0 / (1.0 + math.exp(-alpha * pawn_units))
                w_feats, b_feats = parse_fen_to_features(fen)
                
                w_feats_flat = np.array(w_feats, dtype=np.float32).flatten()
                b_feats_flat = np.array(b_feats, dtype=np.float32).flatten()
                
                yield (
                    {
                        "white_features": w_feats_flat, 
                        "black_features": b_feats_flat,
                        "side_to_move": np.array([is_black_turn], dtype=bool)
                    },
                    np.array([win_probability], dtype=np.float32).flatten()
                )
                
            except (ValueError, TypeError, IndexError):
                continue


def get_lichess_shards():
    """
    Dynamically fetches the underlying parquet filenames from the Hugging Face hub repository.
    Allows us to cleanly isolate validation files from training files without downloading them.
    """
    api = HfApi()
    # Fetch all data files inside the dataset repository
    files = api.list_repo_files(repo_id="Lichess/chess-position-evaluations", repo_type="dataset")
    
    # Filter for the core parquet data shards
    parquet_files = sorted([f for f in files if f.endswith(".parquet")])
    
    # Reserve the final 2 files exclusively for validation testing (~10 million rows)
    # The remaining files are allocated strictly for training
    train_files = parquet_files[:-2]
    val_files = parquet_files[-2:]
    
    return train_files, val_files

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
    merged = layers.Concatenate(name="perspective_multiplex")([first_half, second_half]) 
    
    # 6. Hidden Layer 2 with ReLU1 activation
    x = layers.Dense(64, activation=None, name="hidden_layer_2")(merged)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)

    # 7. Hidden Layer 3 with ReLU1 activation
    x = layers.Dense(32, activation=None, name="hidden_layer_3")(x)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)

    # 8. Output Layer 
    output = layers.Dense(1, activation=None, name="chess_eval")(x)

    model = Model(
        inputs=[white_input, black_input, stm_input],
        outputs=output
    )
    
    def chess_probability_mae(y_true, y_pred):
        # Pass the raw network logits through a sigmoid to get a 0-1 probability
        pred_probs = tf.nn.sigmoid(y_pred)
        # Now compare apples to apples (0-1 prediction vs 0-1 target)
        return tf.reduce_mean(tf.abs(y_true - pred_probs))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
        metrics=[chess_probability_mae]
    )

    # Retrieve the file listings before launching our data loops
    TRAIN_SHARDS, VAL_SHARDS = get_lichess_shards()

    def load_train_stream():
        # Only streams from our dedicated training data shard files
        return load_dataset(
            "Lichess/chess-position-evaluations",
            data_files={"train": TRAIN_SHARDS},
            split="train",
            streaming=True
        ).shuffle(seed=42, buffer_size=20000)

    def load_val_stream():
        # Only streams from our dedicated validation data shard files
        # Bypasses percentage strings and sequential skipping freezes instantly!
        return load_dataset(
            "Lichess/chess-position-evaluations",
            data_files={"train": VAL_SHARDS}, # Must specify 'train' key mapping to match HF structure
            split="train",
            streaming=True
        ).shuffle(seed=999, buffer_size=10000)

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

    lr_scheduler_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )

    # Pass the callback into your fit runner
    model.fit(
        train_dataset, 
        steps_per_epoch=15000, 
        epochs=1, 
        validation_data=val_dataset,
        validation_steps=VAL_SAMPLE_SIZE // VAL_BATCH_SIZE,
        callbacks=[checkpoint_cb, lr_scheduler_cb])
    
    return model

def export_dense_nnue_for_rust(model, file_path="model.nnue"):
    with open(file_path, "wb") as f:
        print("--- Commencing Weight Quantization & Serialization for Rust ---")
        
        # 1. Accumulator Layer (49152 -> 256)
        # (i16) -> Scale up by 128 (2^7)
        acc_layer = model.get_layer("accumulator_layer")
        w1, b1 = acc_layer.get_weights()
        w1_quant = np.ascontiguousarray(np.round(w1 * 128.0)).astype(np.int16)
        b1_quant = np.round(b1 * 128.0).astype(np.int16)
        f.write(w1_quant.tobytes())
        f.write(b1_quant.tobytes())
        print(f"-> Accumulator Layer serialized. Shape: {w1.shape} (i16)")

        # 2. Hidden Layer 2 (512 -> 64)
        # (i8 weights / i32 bias) -> Scale up by 32 (2^5)
        layer2 = model.get_layer("hidden_layer_2") 
        w2, b2 = layer2.get_weights()
        w2_quant = np.ascontiguousarray(np.clip(np.round(w2.T * 32.0), -128, 127).astype(np.int8))
        b2_quant = np.round(b2 * 32.0).astype(np.int32) 
        f.write(w2_quant.tobytes())
        f.write(b2_quant.tobytes())
        print(f"-> Hidden Layer 2 serialized. Shape: {w2.shape} (i8 / i32)")

        # 3. Hidden Layer 3 (64 -> 32)
        # (i8 weights / i32 bias) -> Scale up by 32 (2^5)
        layer3 = model.get_layer("hidden_layer_3")
        w3, b3 = layer3.get_weights()
        w3_quant = np.ascontiguousarray(np.clip(np.round(w3.T * 32.0), -128, 127).astype(np.int8))
        b3_quant = np.round(b3 * 32.0).astype(np.int32)
        f.write(w3_quant.tobytes())
        f.write(b3_quant.tobytes())
        print(f"-> Hidden Layer 3 serialized. Shape: {w3.shape} (i8 / i32)")

        # 4. Output Layer (32 -> 1)
        # (i16) -> Scale up by 128 (2^7)
        output_layer = model.get_layer("chess_eval")
        w4, b4 = output_layer.get_weights()
        w4_quant = np.ascontiguousarray(np.clip(np.round(w4.T * 128.0), -128, 127).astype(np.int8))
        b4_quant = np.round(b4 * 128.0).astype(np.int32)
        f.write(w4_quant.tobytes())
        f.write(b4_quant.tobytes())
        print(f"-> Output Layer serialized. Shape: {w4.shape} (i8 / i32)")

    print(f"\n[SUCCESS] Safe NNUE file successfully compiled and written to: {file_path}")

if __name__ == "__main__":
    # Load the streaming dataset directly from Hugging Face
    trained_model = train_nnue_on_fens()

    export_dense_nnue_for_rust(trained_model, BIN_SAVE_PATH)