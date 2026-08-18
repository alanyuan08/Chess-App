import numpy as np
import tensorflow as tf
import keras
import multiprocessing as mp
import queue
import gc
from tensorflow.keras import layers, Model
from huggingface_hub import HfApi
import math

# --- CONSTANTS ---
INPUT_FEATURES = 64 * 64 * 12  # 49,152
HIDDEN_SIZE = 256
SCALE_MAX = 1.0

DATASET_NAME = "Lichess/chess-position-evaluations"
BIN_SAVE_PATH = "nnue_weights.bin"

# Training / Validation Per Epoch
BATCH_SIZE = 8192  # 4096 * 244
VAL_BATCH_SIZE = BATCH_SIZE * 2
SHUFFLE_BUFFER = BATCH_SIZE * 4

NUM_THREADS = 8

# Map FEN character to an integer type 0-11
PIECE_MAP = {
    'P': 0, 'B': 1, 'N': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'b': 7, 'n': 8, 'r': 9, 'q': 10, 'k': 11
}

# --- Isolated Shard Worker Method used for preprocessing ---
def process_shard_worker(stream_fn, data_queue, stop_event, dataset_name, shards_list):
    """
    An isolated OS process that reads from the stream, handles python-chess overhead,
    verifies position quietness using an embedded Q-search, and pushes processed
    numpy arrays into a shared queue.
    """
    import queue
    import math
    import chess
    import numpy as np
    import gc

    # Centipawn values matching standard NNUE training scaling targets
    PIECE_VALUES = {
        chess.PAWN: 100, 
        chess.KNIGHT: 320, 
        chess.BISHOP: 330, 
        chess.ROOK: 500, 
        chess.QUEEN: 900, 
        chess.KING: 20000
    }

    # --- FEN TO NNUE ARCHITECTURE PARSER ---
    def parse_fen_to_features(fen_string):
        """
        Parses a standard FEN string into sparse categorical indices 
        representing active features for White and Black perspective.
        """
        parts = fen_string.split()
        board_part = parts[0]
        
        # Expand numeric spaces in FEN to empty string dots for alignment
        rows = board_part.split('/')

        # Rank 1 (A1) is 0
        rows.reverse()

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

    def static_evaluate(b):
        """Computes a quick static material evaluation from the current player's perspective."""
        score = 0
        # Fast piece map iteration is faster than scanning all 64 squares sequentially
        for square, piece in b.piece_map().items():
            val = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                score += val
            else:
                score -= val
        return score if b.turn == chess.WHITE else -score

    def q_search(b, alpha, beta, depth=0, max_depth=12):
        """
        Quiescence Search that resolves checks by searching all moves, 
        but uses a strict depth limit to prevent infinite recursion.
        """
        # Force a cutoff if the tactics go too deep
        if depth >= max_depth:
            return static_evaluate(b)

        in_check = b.is_check()
        static_eval = static_evaluate(b)
        
        # Standing pat is only allowed if we are NOT in check
        if not in_check:
            if static_eval >= beta:
                return beta
            if static_eval > alpha:
                alpha = static_eval

        # Rule: If in check, generate ALL moves. If safe, generate only tactical moves.
        for move in b.generate_pseudo_legal_moves(b.turn):
            is_tactical = b.is_capture(move) or move.promotion
            
            # If we are in check, we must look at ALL moves to find an evasion.
            # If we are not in check, we only look at tactical moves.
            if in_check or is_tactical:
                
                if not b.is_legal(move):
                    continue
                    
                b.push(move)
                # Pass depth + 1 to keep track of the search depth
                score = -q_search(b, -beta, -alpha, depth + 1, max_depth)
                b.pop()

                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
                    
        return alpha
    
    def is_invalid_training_row(depth_str, fen_string: str) -> bool:
        # Split the FEN string into its component fields
        fen_parts = fen_string.split()
        
        # 1. Castling Rights Check: If any active castling flag exists, discard the row
        castling_field = fen_parts[2]
        if castling_field != "-":
            return True 
        
        # 2. En Passant Check: If an en passant target square exists, discard the row
        en_passant_field = fen_parts[3]
        if en_passant_field != "-":
            return True

        # 3. Depth & Phase Check
        depth = int(depth_str)
        fen_pieces_count = get_endgame_piece_count(fen_string)
        if fen_pieces_count <= 12:
            return depth < 32
        else:
            # Normal middlegame
            return depth < 20
        
    def get_endgame_piece_count(fen_string: str) -> int:
        board_part = fen_string.split()[0]
        target_pieces = set("pnbrqPNBRQ")
        total_pieces = sum(1 for char in board_part if char in target_pieces)
        
        return total_pieces
    
    hf_dataset = stream_fn(dataset_name, shards_list)
    loop_counter = 0

    for row in hf_dataset:
        if stop_event.is_set():
            break

        fen = row.get('fen')
        raw_score = row.get('cp')
        mate = row.get('mate')
        depth = row['depth']

        if fen is None:
            continue

        try:
            if mate is not None and not math.isnan(mate):
                continue
            if depth is not None and is_invalid_training_row(depth, fen):
                continue
            if raw_score is None or math.isnan(float(raw_score)):
                continue

            first_space = fen.find(' ')
            if first_space == -1:
                continue
            is_black_turn = (fen[first_space + 1] == 'b')

            board = chess.Board(fen)
            
            # Rule 1: Skip if the side to move is currently under check
            if board.is_check():
                continue

            # Rule 2: Run a Q-search to calculate tactical resolution
            # Both functions output absolute, White-centric scores.
            static_score = static_evaluate(board)
            q_score = q_search(board, -float('inf'), float('inf'))
            
            # If the score swings by more than 15 centipawns during tactical resolution,
            # it means there are pending captures/promotions. Skip the position.
            if abs(static_score - q_score) > 15:
                continue

            # 3. Score Normalization & Sigmoid Target Mapping
            score_target = float(raw_score)
            if is_black_turn:
                score_target = -score_target
                
            # Smooth out Pawn Scores for extreme winning/ losing positons
            # pawn_units = score_target / 100.0
            # smooth_pawns = 10.0 * tf.math.tanh(pawn_units / 10.0)
            # win_probability = 1.0 / (1.0 + tf.math.exp(-0.41  * smooth_pawns))
            pawn_units = score_target / 100.0
            
            # 4. Feature Extraction & Flattening
            w_feats, b_feats = parse_fen_to_features(fen)
            w_feats_flat = np.array(w_feats, dtype=np.float32).flatten()
            b_feats_flat = np.array(b_feats, dtype=np.float32).flatten()

            payload = (
                {
                    'white_features': w_feats_flat,
                    'black_features': b_feats_flat,
                    'side_to_move': np.array([is_black_turn], dtype=bool)
                },
                np.array([pawn_units], dtype=np.float32).flatten()
            )

            # 5. Push to Bounded Multi-Processing Queue
            while not stop_event.is_set():
                try:
                    data_queue.put(payload, timeout=0.1)
                    break
                except queue.Full:
                    continue

        except (ValueError, TypeError, IndexError):
            continue
            
        finally:
            loop_counter += 1
            if loop_counter % 500 == 0:  # Crank this down from 2500 to 500
                if 'board' in locals():
                    board.clear()  # Wipes the internal python-chess bitboards
                    del board
                gc.collect()

# --- Load Training / Validation Datasets --- 
def load_train_stream_global(dataset_name, train_shards):
    return load_lichess_stream(dataset_name, train_shards, True)

def load_val_stream_global(dataset_name, val_shards):
    return load_lichess_stream(dataset_name, val_shards, False)

def load_lichess_stream(dataset_name, shards_list, is_training):
    """
    A single, unified loader function used for both Training and Validation data streams.
    """
    from datasets import load_dataset
    # 1. Connect to the specific shards passed into the function
    dataset = load_dataset(
        dataset_name,
        data_files={"train": shards_list},
        split="train",
        streaming=True
    )
    
    # 2. Only shuffle if it's the training stream!
    # Shuffling validation data is a waste of CPU power and RAM.
    if is_training:
        dataset = dataset.shuffle(seed=42, buffer_size=BATCH_SIZE * 4)
        
    return dataset

def get_lichess_shards():
    """
    Dynamically fetches the underlying parquet filenames from the Hugging Face hub repository.
    Strides across the files to ensure train and validation splits both contain a balanced
    mix of historical and modern engine evaluations.
    """
    api = HfApi()
    files = api.list_repo_files(repo_id="Lichess/chess-position-evaluations", repo_type="dataset")
    
    # Filter for the core parquet data shards
    parquet_files = sorted([f for f in files if f.endswith(".parquet")])
    
    train_files = []
    val_files = []
    
    # Stride by 10: allocate 1 out of every 10 files to validation (~10% split)
    # distributed perfectly across the entire dataset timeline
    for i, file_path in enumerate(parquet_files):
        if i % 10 == 0:
            val_files.append(file_path)
        else:
            train_files.append(file_path)
            
    return train_files, val_files

# --- Safe Generator Thread Hook ---
class PermanentDatasetManager:
    def __init__(self, stream_fn, dataset_name, shards_list, num_workers=4, queue_size=20):
        # 1. Create a single, permanent queue and stop flag for the entire script run
        self.data_queue = mp.Queue(maxsize=queue_size)
        self.stop_event = mp.Event()
        self.workers = []
        
        chunk_size = max(1, len(shards_list) // num_workers)
        
        # 2. Spawn the background workers ONCE at initialization
        for i in range(num_workers):
            worker_shards = shards_list[i * chunk_size : (i + 1) * chunk_size]
            if not worker_shards:
                worker_shards = shards_list
                
            p = mp.Process(
                target=process_shard_worker, 
                args=(stream_fn, self.data_queue, self.stop_event, dataset_name, worker_shards)
            )
            p.daemon = True
            p.start()
            self.workers.append(p)
            
    def generator_fn(self):
        """This is the clean function you pass directly to TensorFlow."""
        while True:
            try:
                # Yield data continuously from our permanent, non-leaking queue
                yield self.data_queue.get(timeout=0.1)
            except queue.Empty:
                # If all workers unexpectedly died, stop the loop
                if all(not p.is_alive() for p in self.workers) and self.data_queue.empty():
                    break
                continue
                
    def shutdown(self):
        """Call this at the very end of your script to clean up the OS processes."""
        self.stop_event.set()
        while not self.data_queue.empty():
            try: self.data_queue.get_nowait()
            except queue.Empty: break
        for p in self.workers:
            p.join(timeout=1.0)
            if p.is_alive(): p.terminate()

# --- Memory Clean up ---
class AggressiveMemoryCleanup(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 1. Clear the Keras C++ graph session entirely
        tf.keras.backend.clear_session()
        
        # 2. Force Python to run an immediate, deep garbage collection sweep
        gc.collect()
        
        print(f"\n[System Guard] Epoch {epoch+1} ended. C++ Graph and Python RAM cleared cleanly.")

# --- Model deesign & Training Runner ---
def train_nnue_on_fens():
    # 1. Inputs
    white_input = layers.Input(shape=(INPUT_FEATURES,), sparse=True, name="white_features")
    black_input = layers.Input(shape=(INPUT_FEATURES,), sparse=True, name="black_features")
    stm_input = layers.Input(shape=(1,), dtype="bool", name="side_to_move")

    # 2. Shared Accumulator Layer (HalfK Virtual Weights)
    accumulator = layers.Dense(256, activation=None, kernel_initializer='random_normal', name="accumulator_layer") 
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
    
    # 8. Output Layer (Linear, Pawn-Scale Output)
    output = layers.Dense(1, activation=None, name="chess_eval")(x)

    model = Model(
        inputs=[white_input, black_input, stm_input],
        outputs=output
    )

    def stockfish_nnue_probability_mse_loss(y_true, y_pred):
        """
        MSE loss computed on stockfish win probabilities.
        
        y_true: Target score from your dataset (Pawn units, e.g., +1.50)
        y_pred: Network output (Model evaluation in Pawn units, e.g., +1.42)
        """
        # 1. STOCK_FISH CONSTANT SCALING
        STOCK_FISH = 0.575653

        # 2. CONVERT TRUE SCORES TO TARGET PROBABILITIES (0.0 to 1.0)
        target_probability = 1.0 / (1.0 + tf.math.exp(-STOCK_FISH * y_true))

        # 3. CONVERT NETWORK PREDICTIONS TO PREDICTED PROBABILITIES (0.0 to 1.0)
        # This keeps the sigmoid behavior active during the training pass
        predicted_probability = 1.0 / (1.0 + tf.math.exp(-STOCK_FISH * y_pred))

        # 4. COMPUTE MEAN SQUARED ERROR ON THE PROBABILITIES
        # Linear, steady gradients that won't stall on equal/drawish positions
        return tf.reduce_mean(tf.square(target_probability - predicted_probability))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=stockfish_nnue_probability_mse_loss
    )

    # 90 / 10 Split for Training / Validation Shards
    TRAIN_SHARDS, VAL_SHARDS = get_lichess_shards()

    # Create the permanent managers ONCE. They spawn background processes that live forever.
    train_manager = PermanentDatasetManager(
        load_train_stream_global, DATASET_NAME, TRAIN_SHARDS, num_workers=4, queue_size=20
    )
    val_manager = PermanentDatasetManager(
        load_val_stream_global, DATASET_NAME, VAL_SHARDS, num_workers=1, queue_size=10
    )

    # --- Train Dataset ---
    train_dataset = tf.data.Dataset.from_generator(
        train_manager.generator_fn,
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
        val_manager.generator_fn,
        output_signature=(
            {
                "white_features": tf.TensorSpec(shape=(INPUT_FEATURES,), dtype=tf.float32),
                "black_features": tf.TensorSpec(shape=(INPUT_FEATURES,), dtype=tf.float32),
                "side_to_move": tf.TensorSpec(shape=(1,), dtype=tf.bool),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    )

    # Shuffles Filtered Data Sets
    train_dataset = train_dataset.shuffle(buffer_size=SHUFFLE_BUFFER, reshuffle_each_iteration=True)

    # Train / Val Dataset Batch Size 
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(1)
    val_dataset = val_dataset.batch(VAL_BATCH_SIZE).prefetch(1)

    print("\n--- Model compilation complete. Commencing Training Step ---")
    checkpoint_path = "best_chess_nnue.keras"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # Adjust Learning Rate
    def lr_schedule(epoch):
        """
        Smooth Cosine Decay learning rate schedule for NNUE training.
        """
        initial_lr = 0.001
        min_lr = 0.00001      # The lowest learning rate you want to reach
        total_epochs = 20     # Update this to match your total number of planned epochs
        
        # Boundary check to prevent progress from going past 1.0
        if epoch >= total_epochs:
            return min_lr
            
        # Calculate how far along we are in training (0.0 to 1.0)
        progress = epoch / total_epochs
        
        # Calculate the smooth cosine multiplier
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Return the decayed rate, bounded by your minimum learning rate floor
        return min_lr + (initial_lr - min_lr) * cosine_decay

    lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

    # Instantiate the new cleanup guard
    cleanup_cb = AggressiveMemoryCleanup()

    model.fit(
        train_dataset, 
        steps_per_epoch=976,
        epochs=25, 
        validation_data=val_dataset,
        validation_steps=40,
        callbacks=[checkpoint_cb, lr_scheduler_cb, cleanup_cb]
    )

     # At the very bottom of your script after model.fit() finishes all 25 epochs:
    print("\nTraining complete. Terminating background workers cleanly...")
    train_manager.shutdown()
    val_manager.shutdown()

    return model

# --- Export NNuE Weights for Rust ---
def export_dense_nnue_for_rust(model, file_path="model.nnue"):
    with open(file_path, "wb") as f:
        print("--- Commencing Weight Quantization & Serialization for Rust ---")
        
        # 1. Accumulator Layer (49152 -> 256)
        # Input: Binary (0/1) | Weights: i16 | Bias/Output Accumulator: i32
        acc_layer = model.get_layer("accumulator_layer")
        w1, b1 = acc_layer.get_weights()
        w1_quant = np.ascontiguousarray(np.round(w1 * 128.0)).astype(np.int16)
        b1_quant = np.round(b1 * 128.0).astype(np.int32) 
        f.write(w1_quant.tobytes())
        f.write(b1_quant.tobytes())
        print(f"-> Accumulator Layer serialized. Shape: {w1.shape} (Weights: i16 / Bias: i32)")

        # 2. Hidden Layer 2 (512 -> 64)
        # Input: i16 (Clipped from Accumulator) | Weights: i8 | Bias/Output: i32
        # Shift Right by 7 (>> 7) before clipping to next input scale.
        layer2 = model.get_layer("hidden_layer_2") 
        w2, b2 = layer2.get_weights()
        w2_quant = np.ascontiguousarray(np.clip(np.round(w2.T * 32.0), -128, 127).astype(np.int8))
        b2_quant = np.round(b2 * 32.0).astype(np.int32) 
        f.write(w2_quant.tobytes())
        f.write(b2_quant.tobytes())
        print(f"-> Hidden Layer 2 serialized. Shape: {w2.shape} (Weights: i8 / Bias: i32) [Rust -> Shift >> 7]")

        # 3. Hidden Layer 3 (64 -> 32)
        # Input: i16 | Weights: i8 | Bias/Output: i32 
        # Shift Right by 5 (>> 5) before clipping to next input scale.
        layer3 = model.get_layer("hidden_layer_3")
        w3, b3 = layer3.get_weights()
        w3_quant = np.ascontiguousarray(np.clip(np.round(w3.T * 32.0), -128, 127).astype(np.int8))
        b3_quant = np.round(b3 * 32.0).astype(np.int32)
        f.write(w3_quant.tobytes())
        f.write(b3_quant.tobytes())
        print(f"-> Hidden Layer 3 serialized. Shape: {w3.shape} (Weights: i8 / Bias: i32) [Rust -> Shift >> 5]")

        # 4. Output Layer (32 -> 1)
        # Input: i16 | Weights: i8 | Bias/Output: i32 
        # Shift Right by 5 (>> 5) to get final scaled score.
        output_layer = model.get_layer("chess_eval")
        w4, b4 = output_layer.get_weights()
        w4_quant = np.ascontiguousarray(np.clip(np.round(w4.T * 128.0), -128, 127).astype(np.int8))
        b4_quant = np.round(b4 * 128.0).astype(np.int32)
        f.write(w4_quant.tobytes())
        f.write(b4_quant.tobytes())
        print(f"-> Output Layer serialized. Shape: {w4.shape} (Weights: i8 / Bias: i32) [Rust -> Shift >> 5]")

    print(f"\n[SUCCESS] Safe NNUE file successfully compiled and written to: {file_path}")

if __name__ == "__main__":
    trained_model = train_nnue_on_fens()

    export_dense_nnue_for_rust(trained_model, BIN_SAVE_PATH)