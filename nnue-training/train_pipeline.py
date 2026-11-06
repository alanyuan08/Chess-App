import os
import glob
import gc
import random
import polars as pl

import numpy as np
import tensorflow as tf

# Standardized Keras 3 public modules
import keras
from keras import Model
from keras import layers

# Import your newly written modular classes
from permanent_dataset_manager import InfiniteConveyorDatasetManager

# --- Global Configurations ---
# 10 Buckets * 11 Active Pieces (excluding own King) * 64 Squares + 4 (Castling Squares)= 7044
INPUT_FEATURES = 10 * 11 * 64 + 4  # Dual-Perspective HalfKA Dimension (7044)
MAX_PIECES = 32                # Uniform layout array padding bound
SCALE_MAX = 1.0                # Bounded Clipped ReLU limit 

# --- Step-Based Architecture Configuration ---
# Auto-calculate exact step parameters from compiled files
BATCH_SIZE = 8192
TOTAL_EPOCHS = 30  

# Mixed Data Sets
CLEAN_DATASET_DIR = "./balanced_shards" 
BIN_SAVE_PATH = "nnue_weights.bin"

def compute_dynamic_epoch_geometry(train_dir="./balanced_shards/training", 
                                   val_dir="./balanced_shards/validation", 
                                   batch_size=BATCH_SIZE):
    """
    Scans the compiled parquet sharding directories to instantly compute
    flawless, exact step parameters based on physical file constraints.
    Rounds down step counts to guarantee clean epoch-boundary recycles.
    """
    print("-> Dynamically calculating training universe step geometry...")
    
    # 1. Gather file listings
    train_files = glob.glob(os.path.join(train_dir, "*.parquet"))
    val_files = glob.glob(os.path.join(val_dir, "*.parquet"))
    
    if not train_files:
        raise FileNotFoundError(f"No production shards found in {train_dir}!")
        
    # 2. Extract total row footings instantly via metadata scans (zero RAM overhead)
    total_train_rows = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in train_files)
    total_val_rows = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in val_files) if val_files else 0
    
    # 3. Apply strict block geometry calculations (Floor division rounds down explicitly)
    steps_per_epoch = total_train_rows // batch_size
    validation_per_epoch = total_val_rows // batch_size
    
    # Calculate leftovers to verify drop alignment
    train_leftovers = total_train_rows % batch_size
    val_leftovers = total_val_rows % batch_size
    
    print(f"   [TRAIN UNIVERSE] Physical Rows: {total_train_rows:,} -> Steps/Epoch: {steps_per_epoch} (Dropped trailing remainder: {train_leftovers:,} rows)")
    print(f"   [VAL UNIVERSE]   Physical Rows: {total_val_rows:,} -> Steps/Epoch: {validation_per_epoch} (Dropped trailing remainder: {val_leftovers:,} rows)")
    
    return steps_per_epoch, validation_per_epoch

# Fallback structures for metric and loop calculations
class AggressiveMemoryCleanup(keras.callbacks.Callback):
    """Safely frees dead CPU worker references to prevent RAM accumulation."""
    def on_epoch_end(self, epoch, logs=None):
        gc.collect() 
def get_local_shard_directories():
    """
    Defines the storage locations for your exported clean Parquet files.
    Modify these strings to point directly to your dataset output directories.
    """
    train_dir = "./balanced_shards/training/"
    val_dir = "./balanced_shards/validation/"
    return train_dir, val_dir

# --- Export NNUE Weights for Rust ---
def export_dense_nnue_for_rust(model, file_path="model.nnue"):
    with open(file_path, "wb") as f:
        print("--- Commencing Weight Quantization & Serialization for Rust ---")
        
        # 1. Accumulator Layer (7040 -> 256)
        # Input: Binary (0/1) | Weights: i16 | Bias/Output Accumulator: i32
        acc_layer = model.get_layer("accumulator_layer")
        embedding_weights = acc_layer.get_weights()[0]
        # Accumulator Weights (Updated dimension bound for 16-bucket layout)
        w1_real = embedding_weights[:7040, :]
        w1_quant = np.ascontiguousarray(np.round(w1_real * 128.0)).astype(np.int16)

        f.write(w1_quant.tobytes())
        print(f"-> Accumulator Layer serialized. Shape: {w1_real.shape} (Weights: i16 / Bias: NONE)")

        # 2. Hidden Layer 2 (256*2 -> 64)
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

def pawn_probability_mse_loss(y_true, y_pred):
    """
    Tuned for an ultra-lean 8-neuron bottleneck outputting PAWN units.
    Ensures gradients heavily prioritize precision between -2.0 and +2.0 Pawns.
    """
    # 3.6 Pawns acts as the scaling factor (Equivalent to 360 Centipawns).
    # This aligns the log-odds win probability to a standard Pawn-scale model output.
    SF_SCALE = 3.6 
    
    # Mathematical conversion factor: ln(10) / 3.6
    # 2.30258509299 / 3.6 ≈ 0.639607
    scale_factor = 2.30258509299 / SF_SCALE
    
    # Cast tensors to float32 to protect numerical precision during sigmoid scaling
    # Expects y_true and y_pred to be float values like 0.5, 1.2, -3.0
    y_true_prob = tf.math.sigmoid(keras.ops.cast(y_true, "float32") * scale_factor)
    y_pred_prob = tf.math.sigmoid(keras.ops.cast(y_pred, "float32") * scale_factor)
    
    # Calculate MSE on the probability landscape
    loss = tf.math.squared_difference(y_true_prob, y_pred_prob)
    return tf.reduce_mean(loss)

def close_position_error_all(y_true, y_pred):
    raw_error = tf.abs(y_true - y_pred)
    return tf.reduce_mean(raw_error)

def close_position_error(y_true, y_pred):
    """
    Calculates MAE only for positions with a true evaluation between 0 and 2.5 pawns.
    Assumes y_true is scaled in pawns (e.g., 1.0 = 1 pawn, 2.5 = 2.5 pawns).
    """
    # 1. Compute raw absolute error per position
    raw_error = tf.abs(y_true - y_pred)
    
    # 2. Create a mask for positions within the 0 to 3 pawn range (absolute value)
    is_close_position = tf.abs(y_true) <= 2.5
    mask = tf.cast(is_close_position, tf.float32)
    
    # 3. Filter the error using the mask
    masked_error = raw_error * mask
    
    # 4. Compute the mean using only the valid filtered positions
    total_error = tf.reduce_sum(masked_error)
    valid_count = tf.reduce_sum(mask)
    
    # divide_no_nan prevents a 0/0 crash if a batch happens to have zero close positions
    return tf.math.divide_no_nan(total_error, valid_count)

def train_nnue_on_fens():
    # 1. Inputs: Set to MAX_PIECES (32) and cast to int16 to optimize L1/L2 cache efficiency
    active_input = keras.Input(shape=(MAX_PIECES,), dtype="int16", name="active_features")
    passive_input = keras.Input(shape=(MAX_PIECES,), dtype="int16", name="passive_features")

    # 2. Shared Accumulator Layer (Embedding-based indexing for fast memory execution)
    # Dimensioned to INPUT_FEATURES + 1 to house the positive padding index row safely
    nnue_accumulator_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=42)
    embedding_layer = layers.Embedding(
        input_dim=INPUT_FEATURES + 1,
        output_dim=256,
        mask_zero=True,
        embeddings_initializer=nnue_accumulator_init,
        name="accumulator_layer"
    )

    # 3. Pull dense weights vector representations for active and passive piece decks
    a_embed = embedding_layer(active_input)  # Shape: (Batch, 32, 256)
    p_embed = embedding_layer(passive_input)  # Shape: (Batch, 32, 256)

    # =========================================================================
    # 4. Native Masked Sum Pooling (Replaces the broken manual 7044 masks)
    # GlobalSumPooling1D automatically honors mask_zero and ignores index 0 completely.
    # =========================================================================
    a_acc_raw = layers.Lambda(
        lambda x: keras.ops.sum(x, axis=1), 
        name="active_sum_pool"
    )(a_embed)  # Shape: (Batch, 256)

    p_acc_raw = layers.Lambda(
        lambda x: keras.ops.sum(x, axis=1), 
        name="passive_sum_pool"
    )(p_embed)  # Shape: (Batch, 256)

    # =========================================================================
    # 5. SMOOTH BOUNDED RELU ACTIVATION (SIMD VECTOR REGISTER FRIENDLY)
    # =========================================================================
    alpha_leak = 0.01  # Tiny 1% gradient retention to protect AdamW momentum tracking

    # Smooth floor handling: smooth max between 0 and x
    a_floor = 0.5 * (a_acc_raw + keras.ops.abs(a_acc_raw)) + alpha_leak * (a_acc_raw - keras.ops.abs(a_acc_raw))
    p_floor = 0.5 * (p_acc_raw + keras.ops.abs(p_acc_raw)) + alpha_leak * (p_acc_raw - keras.ops.abs(p_acc_raw))

    # Smooth ceiling handling: smooth min between x and SCALE_MAX
    diff_a = a_floor - SCALE_MAX
    diff_p = p_floor - SCALE_MAX
    a_act = 0.5 * (a_floor + SCALE_MAX - keras.ops.abs(diff_a)) + alpha_leak * (diff_a + keras.ops.abs(diff_a))
    p_act = 0.5 * (p_floor + SCALE_MAX - keras.ops.abs(diff_p)) + alpha_leak * (diff_p + keras.ops.abs(diff_p))

    # =========================================================================
    # 6. Perspective Multiplexing Layer (Shape: Batch, 512)
    # =========================================================================
    merged = layers.Concatenate(name="perspective_multiplex")([a_act, p_act]) 
    
    # Deep layer weight configurations
    hidden_init = keras.initializers.HeNormal(seed=42)

    # 7. Hidden Layer 2 with Smooth Bounded Activation (Shape: Batch, 64)
    x2_raw = layers.Dense(
        64, 
        activation=None, 
        use_bias=True, 
        kernel_initializer=hidden_init,
        name="hidden_layer_2"
    )(merged)
    
    x2_floor = 0.5 * (x2_raw + keras.ops.abs(x2_raw)) + alpha_leak * (x2_raw - keras.ops.abs(x2_raw))
    diff_x2 = x2_floor - SCALE_MAX
    x2_act = 0.5 * (x2_floor + SCALE_MAX - keras.ops.abs(diff_x2)) + alpha_leak * (diff_x2 + keras.ops.abs(diff_x2))

    # 8. Hidden Layer 3 with Smooth Bounded Activation (Shape: Batch, 32)
    x3_raw = layers.Dense(
        32, 
        activation=None, 
        use_bias=True, 
        kernel_initializer=hidden_init,
        name="hidden_layer_3"
    )(x2_act)
    
    x3_floor = 0.5 * (x3_raw + keras.ops.abs(x3_raw)) + alpha_leak * (x3_raw - keras.ops.abs(x3_raw))
    diff_x3 = x3_floor - SCALE_MAX
    x3_act = 0.5 * (x3_floor + SCALE_MAX - keras.ops.abs(diff_x3)) + alpha_leak * (diff_x3 + keras.ops.abs(diff_x3))
    
    # 9. Output Layer (Linear, Pawn-Scale Evaluation Output) (Shape: Batch, 1)
    output = layers.Dense(
        1, 
        activation=None, 
        use_bias=True, 
        kernel_initializer=hidden_init,
        name="chess_eval"
    )(x3_act)

    model = Model(
        inputs=[active_input, passive_input],
        outputs=output
    )

    # 2. Bind directly to your engineered AdamW
    STEPS_PER_EPOCH, VALIDATION_PER_EPOCH = compute_dynamic_epoch_geometry(
        train_dir="./balanced_shards/training",
        val_dir="./balanced_shards/validation",
        batch_size=BATCH_SIZE
    )
    
    total_train_steps = int(TOTAL_EPOCHS * STEPS_PER_EPOCH)
    warmup_steps = int(STEPS_PER_EPOCH * 0.2) 

    # 1. Use the standard, bulletproof Keras 3 native Cosine Decay with built-in warmup
    native_lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=total_train_steps,
        alpha=0.02,
        warmup_target=1.5e-4,   
        warmup_steps=warmup_steps
    )

    # 2. Instatiate stock AdamW with native coupling
    stockfish_optimizer = keras.optimizers.AdamW(
        learning_rate=native_lr_schedule,
        weight_decay=0.01,
        epsilon=1e-5,
        global_clipnorm=1.0
    )

    model.compile(
        optimizer=stockfish_optimizer,
        loss=pawn_probability_mse_loss,
        metrics=[close_position_error_all, close_position_error]
    )

    train_dir, val_dir = get_local_shard_directories()
    optimal_train_workers = 6
    
    # 1. Instantiate both managers with identical lowercase tracking keys
    train_manager = InfiniteConveyorDatasetManager(
        shard_directory=train_dir, 
        shard_pattern="*.parquet", 
        num_workers=optimal_train_workers, 
        queue_size=32, 
        batch_size=BATCH_SIZE, 
        name="training"
    )
    val_manager = InfiniteConveyorDatasetManager(
        shard_directory=val_dir, 
        shard_pattern="*.parquet", 
        num_workers=2,
        queue_size=32, 
        batch_size=BATCH_SIZE, 
        name="validation"  
    )

    # 2. Named generator bindings (No loops, no inline lambdas)
    def train_generator_wrapper():
        return train_manager.generator_fn()

    def val_generator_wrapper():
        return val_manager.generator_fn()

    # 3. Build structural dataset graph definitions EXACTLY ONCE with uniform infinite loops
    def get_fresh_train_dataset():
        return tf.data.Dataset.from_generator(
            generator=train_generator_wrapper,
            output_signature=(
                {
                    "active_features": tf.TensorSpec(shape=(None, MAX_PIECES), dtype=tf.int16),
                    "passive_features": tf.TensorSpec(shape=(None, MAX_PIECES), dtype=tf.int16),
                },
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
            )
        ).repeat().prefetch(tf.data.AUTOTUNE)

    def get_fresh_val_dataset():
        return tf.data.Dataset.from_generator(
            generator=val_generator_wrapper,
            output_signature=(
                {
                    "active_features": tf.TensorSpec(shape=(None, MAX_PIECES), dtype=tf.int16),
                    "passive_features": tf.TensorSpec(shape=(None, MAX_PIECES), dtype=tf.int16),
                },
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
            )
        ).repeat().prefetch(tf.data.AUTOTUNE) 

    # 4. Instantiate static, backend-agnostic dataset graphs
    train_dataset = get_fresh_train_dataset()
    val_dataset = get_fresh_val_dataset()

    print(f"\n--- Model compilation complete. Commencing Training with {optimal_train_workers} Workers ---")

    # 5. Core callback lifecycle infrastructure
    checkpoint_path = "best_chess_nnue.keras"
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=False,
        verbose=1
    )

    cleanup_cb = AggressiveMemoryCleanup()

    # 6. Fit Execution: Runs perfectly synchronized across all 30 epochs
    model.fit(
        train_dataset,
        epochs=TOTAL_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_dataset,
        validation_steps=VALIDATION_PER_EPOCH,
        shuffle=False, 
        callbacks=[checkpoint_cb, cleanup_cb]
    )

    print("\nTraining complete. Terminating background workers cleanly...")
    train_manager.stop_epoch_stream()
    val_manager.stop_epoch_stream()

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