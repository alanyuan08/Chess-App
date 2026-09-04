import os
import math
import tensorflow as tf
import keras
from keras import layers, Model
import numpy as np

# Import your newly written modular classes
from permanent_dataset_manager import PermanentDatasetManager

# --- Global Configurations ---
INPUT_FEATURES = 64 * 64 * 12  # Dual-Perspective HalfKA Dimension (24,576)
MAX_PIECES = 32                # Uniform layout array padding bound
SCALE_MAX = 1.0                # Bounded Clipped ReLU limit
BATCH_SIZE = 16384             # Standard massive NNUE training batch size
VAL_BATCH_SIZE = 4096          # Validation tracking step batch size
SHUFFLE_BUFFER = 50000         # Buffer allocation size for secondary tf.data mix

# Mixed Data Sets
CLEAN_DATASET_DIR = "./production_shards" 
BIN_SAVE_PATH = "nnue_weights.bin"

# Fallback structures for metric and loop calculations
class AggressiveMemoryCleanup(keras.callbacks.Callback):
    """Triggers forced garbage collection loops to maintain training VRAM stability."""
    def on_epoch_end(self, epoch, logs=None):
        import gc
        gc.collect()
        keras.backend.clear_session()

def get_local_shard_directories():
    """
    Defines the storage locations for your exported clean Parquet files.
    Modify these strings to point directly to your dataset output directories.
    """
    train_dir = "./production_shards/training/"
    val_dir = "./production_shards/validation/"
    return train_dir, val_dir

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


def train_nnue_on_fens():
    # Value token allocated to safely represent masking rows in the weights matrix
    PADDING_INDEX_VALUE = INPUT_FEATURES 

    # 1. Inputs: Sequences of active feature index tokens matching your Parquet shapes
    white_input = layers.Input(shape=(MAX_PIECES,), dtype="int32", name="white_features")
    black_input = layers.Input(shape=(MAX_PIECES,), dtype="int32", name="black_features")
    stm_input = layers.Input(shape=(1,), dtype="bool", name="side_to_move")

    # 2. Shared Accumulator Layer (Using Embedding Reduction to replace the old sparse matrix bottleneck)
    nnue_accumulator_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.005)
    
    # We dimension the lookup array to INPUT_FEATURES + 1 to house the positive padding index row
    embedding_layer = layers.Embedding(
        input_dim=INPUT_FEATURES + 1,
        output_dim=256,
        embeddings_initializer=nnue_accumulator_init,
        name="shared_accumulator_embedding"
    )

    # 3. Pull dense weights representations for all slots
    w_embed = embedding_layer(white_input) # Target shape: (Batch, 32, 256)
    b_embed = embedding_layer(black_input) # Target shape: (Batch, 32, 256)

    # 4. Synthesize masking vectors to isolate and zero-out padding weight contributions
    w_mask = keras.ops.cast(keras.ops.not_equal(white_input, PADDING_INDEX_VALUE), dtype="float32")
    b_mask = keras.ops.cast(keras.ops.not_equal(black_input, PADDING_INDEX_VALUE), dtype="float32")
    
    # Expand to allow broadcasting dimensions across the 256 embedding properties
    w_mask = keras.ops.expand_dims(w_mask, axis=-1) # Target shape: (Batch, 32, 1)
    b_mask = keras.ops.expand_dims(b_mask, axis=-1)

    # Execute masked pool aggregation to compile the 256 accumulator vectors
    w_acc = keras.ops.sum(w_embed * w_mask, axis=1) # Target shape: (Batch, 256)
    b_acc = keras.ops.sum(b_embed * b_mask, axis=1) # Target shape: (Batch, 256)

    # 5. Clipped ReLU Activation (ReLU1 / Bounded ReLU)
    w_act = keras.ops.clip(w_acc, 0.0, SCALE_MAX)
    b_act = keras.ops.clip(b_acc, 0.0, SCALE_MAX)

    # 6. Cast boolean side-to-move mask to float for branchless evaluations
    stm_float = keras.ops.cast(stm_input, dtype="float32")
    
    # 7. Perspective Multiplexing Layer (Shape: Batch, 512)
    first_half = stm_float * b_act + (1.0 - stm_float) * w_act
    second_half = stm_float * w_act + (1.0 - stm_float) * b_act
    merged = layers.Concatenate(name="perspective_multiplex")([first_half, second_half]) 
    
    # 8. Hidden Layer 2 with ReLU1 activation
    x = layers.Dense(64, activation=None, name="hidden_layer_2")(merged)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)

    # 9. Hidden Layer 3 with ReLU1 activation
    x = layers.Dense(32, activation=None, name="hidden_layer_3")(x)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)
    
    # 10. Output Layer (Linear, Pawn-Scale Evaluation Output)
    output = layers.Dense(1, activation=None, name="chess_eval")(x)

    model = Model(
        inputs=[white_input, black_input, stm_input],
        outputs=output
    )

    def stockfish_nnue_pure_loss(y_true, y_pred):
        """
        Official Stockfish Style NNUE Loss Function.
        Calculates Mean Squared Error in WDL (probability) space.
        """
        y_true_cp = y_true * 100.0
        y_pred_cp = y_pred * 100.0
        
        SF_SCALE = 0.0075
        
        # Pass both target and prediction through a sigmoid to map to WDL space
        target_wdl = tf.math.sigmoid(y_true_cp * SF_SCALE)
        pred_wdl = tf.math.sigmoid(y_pred_cp * SF_SCALE)
        
        loss = tf.math.squared_difference(target_wdl, pred_wdl)
        return tf.reduce_mean(loss)

    def close_position_error_3(y_true, y_pred):
        """
        Tracks micro-accuracy in close, quiet positions. 
        Caps errors at 3.0 pawns so large material blunders don't distort evaluation metrics.
        """
        SCALE_THRESHOLD = 3.0
        raw_error = tf.abs(y_true - y_pred)
        return tf.reduce_mean(tf.math.tanh(raw_error / SCALE_THRESHOLD) * SCALE_THRESHOLD)

    def general_position_error_10(y_true, y_pred):
        """
        Tracks macro-accuracy across general play, including material differences.
        Caps errors at 10.0 pawns to capture full piece values up to a Queen.
        """
        SCALE_THRESHOLD = 10.0
        raw_error = tf.abs(y_true - y_pred)
        return tf.reduce_mean(tf.math.tanh(raw_error / SCALE_THRESHOLD) * SCALE_THRESHOLD)

    # --- Compile the model ---
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=stockfish_nnue_pure_loss,
        metrics=[close_position_error_3, general_position_error_10]
    )

    # Track file storage roots for local Parquet files
    train_dir, val_dir = get_local_shard_directories()

    # Create the permanent managers ONCE. They spawn background processes that live forever.
    train_manager = PermanentDatasetManager(
        shard_directory=train_dir, shard_pattern="production_data_*.parquet", num_workers=4, queue_size=50000
    )
    val_manager = PermanentDatasetManager(
        shard_directory=val_dir, shard_pattern="production_data_*.parquet", num_workers=1, queue_size=10000
    )

    # --- Train Dataset (Updated to accept dense list tokens signatures) ---
    train_dataset = tf.data.Dataset.from_generator(
        train_manager.generator_fn,
        output_signature=(
            {
                "white_features": tf.TensorSpec(shape=(MAX_PIECES,), dtype=tf.int32),
                "black_features": tf.TensorSpec(shape=(MAX_PIECES,), dtype=tf.int32),
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
                "white_features": tf.TensorSpec(shape=(MAX_PIECES,), dtype=tf.int32),
                "black_features": tf.TensorSpec(shape=(MAX_PIECES,), dtype=tf.int32),
                "side_to_move": tf.TensorSpec(shape=(1,), dtype=tf.bool),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    )

    # Apply standard streaming shuffles and dynamic background caching buffers
    train_dataset = train_dataset.shuffle(buffer_size=SHUFFLE_BUFFER, reshuffle_each_iteration=True)
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

    def lr_schedule(epoch):
        """Smooth Cosine Decay learning rate schedule for NNUE training."""
        initial_lr = 0.001
        min_lr = 0.00001      
        total_epochs = 25     
        
        if epoch >= total_epochs:
            return min_lr
            
        progress = epoch / total_epochs
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (initial_lr - min_lr) * cosine_decay

    lr_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    cleanup_cb = AggressiveMemoryCleanup()

    # Train model execution call
    model.fit(
        train_dataset, 
        steps_per_epoch=1200,
        epochs=25, 
        validation_data=val_dataset,
        validation_steps=120,
        callbacks=[checkpoint_cb, lr_scheduler_cb, cleanup_cb]
    )

    print("\nTraining complete. Terminating background workers cleanly...")
    train_manager.shutdown()
    val_manager.shutdown()

    return model


if __name__ == "__main__":
    # Ensure system multiprocessing primitives initialize cleanly across Windows/Linux architectures
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Method already assigned by context execution

    # Execute full pipeline initialization
    trained_model = train_nnue_on_fens()

    # Export the Model
    export_dense_nnue_for_rust(trained_model, BIN_SAVE_PATH)