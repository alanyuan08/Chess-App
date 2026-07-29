import re
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers, Model
import struct

# --- CONSTANTS ---
INPUT_FEATURES = 64 * 64 * 12  # 49,152
HIDDEN_SIZE = 256

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
def dataset_generator(file_path, batch_size=128):
    """
    Reads lines of text formatted as "FEN,Score" or "FEN;Score".
    Parses and yields mini-batches efficiently.
    """
    while True:
        with open(file_path, 'r') as f:
            # Skip header line if present
            next(f, None)
            
            w_batch, b_batch, stm_batch, y_batch = [], [], [], []
            for line in f:
                if not line.strip(): 
                    continue

                parts = re.split(r'[;,]', line.strip())
                if len(parts) < 2: 
                    continue
                
                fen = parts[0].strip()
                fen_tokens = fen.split()
                if len(fen_tokens) < 2:
                    continue

                # Determine active side: False for White to move, True for Black to move
                is_black_turn = (fen_tokens[1] == 'b')

                try:
                    raw_score = float(parts[1])
                    if is_black_turn:
                        raw_score *= -1.0

                    score = np.clip(raw_score / 400.0, -3.0, 3.0)
                except ValueError:
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

# --- 3. CUSTOM NNUE TRANSFORMER LAYER ---
class FeatureTransformer(layers.Layer):
    def __init__(self, hidden_size, **kwargs):
        super(FeatureTransformer, self).__init__(**kwargs)
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(INPUT_FEATURES, self.hidden_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
            name="nnue_weights"
        )
        self.bias = self.add_weight(
            shape=(self.hidden_size,),
            initializer="zeros",
            trainable=True,
            name="nnue_biases"
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias

# --- 4. MODEL DESIGN & TRAINING RUNNER ---
def train_nnue_on_fens(data_file_path):
    # Inputs
    white_input = layers.Input(shape=(INPUT_FEATURES,), name="white_features")
    black_input = layers.Input(shape=(INPUT_FEATURES,), name="black_features")
    stm_input = layers.Input(shape=(1,), dtype="bool", name="side_to_move")

    # Shared Accumulator Layer
    transformer = FeatureTransformer(HIDDEN_SIZE, name="accumulator_layer")
    w_acc = transformer(white_input)
    b_acc = transformer(black_input)

    # Clipped ReLU Activation: clamp(0.0, 1.0)
    w_act = keras.ops.clip(w_acc, 0.0, 1.0)
    b_act = keras.ops.clip(b_acc, 0.0, 1.0)

    # Cast boolean mask to float for safe, broadcastable mathematical selection
    # Black's turn, stm_float == 1.0, White's Turn stm_float == 0.0
    stm_float = 0.0
    
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
    output = layers.Dense(1, activation="linear", name="chess_eval")(x)

    model = Model(inputs=[white_input, black_input, stm_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")

    # Wrap the generator in a tf.data runtime pipeline
    batch_size = 256
    train_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator(data_file_path, batch_size),
        output_signature=(
            {
                "white_features": tf.TensorSpec(shape=(batch_size, INPUT_FEATURES), dtype=tf.float32),
                "black_features": tf.TensorSpec(shape=(batch_size, INPUT_FEATURES), dtype=tf.float32),
                "side_to_move": tf.TensorSpec(shape=(batch_size, 1), dtype=tf.bool),
            },
            tf.TensorSpec(shape=(batch_size, 1), dtype=tf.float32)
        )
    )

    print("\n--- Model compilation complete. Commencing Training Step ---")
    # Steps per epoch represents: Total Records / Batch Size
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', 
        factor=0.5, 
        patience=2, 
        verbose=1, 
        min_lr=1e-5
    )

    # Pass the callback into your fit runner
    model.fit(train_dataset, steps_per_epoch=2000, epochs=20, callbacks=[lr_scheduler])
    
    return model

def export_nnue_to_rust(model, filename="nnue_weights.bin"):
    print(f"\n--- Exporting NNUE Model to {filename} ---")
    
    # 1. Extract Layers
    ft_layer = model.get_layer("accumulator_layer")
    dense_32 = model.layers[-2]  # The intermediate Dense(32) layer
    output_layer = model.layers[-1]  # The final Dense(1) layer
    
    # 2. Extract Floating-Point Arrays
    w_ft, b_ft = ft_layer.kernel.numpy(), ft_layer.bias.numpy()
    w_d32, b_d32 = dense_32.kernel.numpy(), dense_32.bias.numpy()
    w_out, b_out = output_layer.kernel.numpy(), output_layer.bias.numpy()

    # 3. Define Quantization Factors (Scaling floats to integers)
    # Standard NNUE uses these standard quantization scales:
    SCALE_FT = 255      # Scale factor for accumulator layer
    SCALE_OUT = 64      # Scale factor for subsequent layers

    print("Quantizing weights into integer formats...")
    # Quantize to int16 (Accumulator layer matches int16)
    q_w_ft = np.clip(w_ft * SCALE_FT, -32768, 32767).astype(np.int16)
    q_b_ft = np.clip(b_ft * SCALE_FT, -32768, 32767).astype(np.int16)
    
    # Quantize subsequent layers to int8 or int16. For simplicity in Rust, we use int16 here.
    q_w_d32 = np.clip(w_d32 * SCALE_OUT, -32768, 32767).astype(np.int16)
    q_b_d32 = np.clip(b_d32 * SCALE_FT * SCALE_OUT, -32768, 32767).astype(np.int32) # Biases scale quadratically
    
    q_w_out = np.clip(w_out * SCALE_OUT, -32768, 32767).astype(np.int16)
    q_b_out = np.clip(b_out * SCALE_FT * SCALE_OUT * SCALE_OUT, -2147483648, 2147483647).astype(np.int32)

    # 4. Stream sequentially out to raw bytes
    with open(filename, "wb") as f:
        # Layer 1: Feature Transformer
        f.write(q_w_ft.tobytes())   # Shape: [49152, 256] -> 25,165,824 bytes
        f.write(q_b_ft.tobytes())   # Shape: [256]        -> 512 bytes
        
        # Layer 2: Hidden Dense Layer
        f.write(q_w_d32.tobytes())  # Shape: [512, 32]    -> 32,768 bytes
        f.write(q_b_d32.tobytes())  # Shape: [32]         -> 128 bytes (int32)
        
        # Layer 3: Output Neuron
        f.write(q_w_out.tobytes())  # Shape: [32, 1]      -> 64 bytes
        f.write(q_b_out.tobytes())  # Shape: [1]          -> 4 bytes (int32)

    print(f"Success! Saved completed binary model framework to {filename}")


if __name__ == "__main__":
    DATA_PATH = "FilteredEvals.csv" 

    trained_model = train_nnue_on_fens(DATA_PATH)

    export_nnue_to_rust(trained_model, "nnue_weights.bin")