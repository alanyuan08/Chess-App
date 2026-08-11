import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers, Model
from huggingface_hub import HfApi

# --- CONSTANTS ---
INPUT_FEATURES = 64 * 64 * 12  # 49,152
HIDDEN_SIZE = 256
SCALE_MAX = 1.0

BATCH_SIZE = 256
DATASET_NAME = "Lichess/chess-position-evaluations"
BIN_SAVE_PATH = "nnue_weights.bin"

VAL_SAMPLE_SIZE = 15360
VAL_BATCH_SIZE = 256

NUM_THREADS = 8

# Map FEN character to an integer type 0-11
PIECE_MAP = {
    'P': 0, 'B': 1, 'N': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'b': 7, 'n': 8, 'r': 9, 'q': 10, 'k': 11
}


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
    
    hf_dataset = stream_fn(dataset_name, shards_list)
    loop_counter = 0

    for row in hf_dataset:
        if stop_event.is_set():
            break

        fen = row.get('fen')
        raw_score = row.get('cp')
        mate = row.get('mate')

        if fen is None:
            continue

        try:
            if mate is not None and not math.isnan(mate):
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
                
            score_target = max(-1000.0, min(1000.0, score_target))
            pawn_units = score_target / 100.0
            win_probability = 1.0 / (1.0 + math.exp(-0.6 * pawn_units))

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
                np.array([win_probability], dtype=np.float32).flatten()
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
            # Prevent memory exhaustion across spawned worker cycles
            loop_counter += 1
            if loop_counter % 2500 == 0:
                if 'board' in locals():
                    del board
                gc.collect()

# Load Training / Val Data Sets
def load_train_stream_global(dataset_name, train_shards):
    """Global un-nested loader for pickling across worker processes."""
    from datasets import load_dataset  # Keep imports inside or at top level
    return load_dataset(
        dataset_name,
        data_files={"train": train_shards},
        split="train",
        streaming=True
    ).shuffle(seed=42, buffer_size=4000)

def load_val_stream_global(dataset_name, val_shards):
    """Global un-nested loader for pickling across worker processes."""
    from datasets import load_dataset
    return load_dataset(
        dataset_name,
        data_files={"train": val_shards}, # Matching HuggingFace's key mapping structure
        split="train",
        streaming=True
    ).shuffle(seed=999, buffer_size=4000)

# --- New Safe Generator Thread Hook ---
def parallel_dataset_generator(stream_fn, dataset_name, shards_list, num_workers=4, queue_size=20):
    """
    Spawns background processes using clean global functions to handle python-chess,
    yielding clean data structures safely to TensorFlow.
    """
    import multiprocessing as mp
    import queue

    data_queue = mp.Queue(maxsize=queue_size)
    stop_event = mp.Event()
    workers = []
    
    chunk_size = max(1, len(shards_list) // num_workers)

    # Launch truly parallel CPU workers passing global function refs and pickleable lists
    for i in range(num_workers):
        # Slice the global shard array so each worker gets its own segment
        worker_shards = shards_list[i * chunk_size : (i + 1) * chunk_size]
        
        # Guard clause: if there are fewer shards than workers, pass the full list to remaining workers
        if not worker_shards:
            worker_shards = shards_list
            
        p = mp.Process(
            target=process_shard_worker, 
            args=(stream_fn, data_queue, stop_event, dataset_name, worker_shards)
        )
        p.daemon = True
        p.start()
        workers.append(p)
        
    try:
        while True:
            try:
                yield data_queue.get(timeout=0.1)
            except queue.Empty:
                if all(not p.is_alive() for p in workers) and data_queue.empty():
                    break
                continue
    finally:
        stop_event.set()
        while not data_queue.empty():
            try: data_queue.get_nowait()
            except queue.Empty: break
        for p in workers:
            p.join(timeout=1.0)
            if p.is_alive(): p.terminate()

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
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
    )

    # Retrieve the file listings before launching our data loops
    TRAIN_SHARDS, VAL_SHARDS = get_lichess_shards()

    # --- Train Dataset ---
    train_dataset = tf.data.Dataset.from_generator(
        lambda: parallel_dataset_generator(
            stream_fn=load_train_stream_global,
            dataset_name=DATASET_NAME,
            shards_list=TRAIN_SHARDS,
            num_workers=NUM_THREADS,
            queue_size=10
        ),
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
        lambda: parallel_dataset_generator(
            stream_fn=load_val_stream_global,
            dataset_name=DATASET_NAME,
            shards_list=VAL_SHARDS,
            num_workers=2,
            queue_size=10
        ),
        output_signature=(
            {
                "white_features": tf.TensorSpec(shape=(INPUT_FEATURES,), dtype=tf.float32),
                "black_features": tf.TensorSpec(shape=(INPUT_FEATURES,), dtype=tf.float32),
                "side_to_move": tf.TensorSpec(shape=(1,), dtype=tf.bool),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    )

    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(2)
    val_dataset = val_dataset.batch(VAL_BATCH_SIZE).prefetch(2)

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
        patience=4,
        min_lr=1e-5,
        verbose=1
    )

    # Pass the callback into your fit runner
    model.fit(
        train_dataset, 
        steps_per_epoch=15000, 
        epochs=45, 
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