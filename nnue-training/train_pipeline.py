import os
import glob
import polars as pl

import numpy as np
import tensorflow as tf

# Standardized Keras 3 public modules
import keras
from keras import Model
from keras import layers

# Import your newly written modular classes
from permanent_dataset_manager import EpochSynchronizedDatasetManager

# --- Global Configurations ---
# 10 Buckets * 11 Active Pieces (excluding own King) * 64 Squares = 7040
INPUT_FEATURES = 10 * 11 * 64  # Dual-Perspective HalfKA Dimension (7040)
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
    """Triggers forced garbage collection loops to maintain training VRAM stability."""
    def on_epoch_end(self, epoch, logs=None):
        import gc
        gc.collect()
        keras.backend.clear_session()

# --- THE TELEMETRY TRACKER CALLBACK ---
@keras.utils.register_keras_serializable(package="CustomSchedules")
class CustomWarmupCosineSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_steps, alpha=0.0167, initial_lr=1e-6, weight_decay_ratio=None):
        super().__init__()
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = int(decay_steps)
        self.warmup_steps = int(warmup_steps)
        self.alpha = float(alpha)
        self.initial_lr = float(initial_lr)
        self.weight_decay_ratio = float(weight_decay_ratio) if weight_decay_ratio is not None else None

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        warmup_steps_f = tf.cast(self.warmup_steps, tf.float32)
        decay_steps_f = tf.cast(self.decay_steps, tf.float32)
        peak_lr_f = tf.cast(self.initial_learning_rate, tf.float32)
        alpha_f = tf.cast(self.alpha, tf.float32)

        # 1. Linear Warmup Ramp
        # Avoid division-by-zero if warmup_steps is configured to 0
        warmup_denom = tf.where(warmup_steps_f > 0.0, warmup_steps_f, 1.0)
        warmup_lr = self.initial_lr + (peak_lr_f - self.initial_lr) * (step_f / warmup_denom)

        # 2. Native Cosine Decay Calculation (Manually written for graph safety and config isolation)
        adjusted_decay_steps = tf.maximum(1.0, decay_steps_f - warmup_steps_f)
        decay_step = tf.maximum(0.0, step_f - warmup_steps_f)
        
        # Bound the cosine step to max adjusted decay steps to prevent reverse loop upswing
        decay_step = tf.minimum(decay_step, adjusted_decay_steps)
        
        cosine_decay = 0.5 * (1.0 + tf.cos(3.141592653589793 * decay_step / adjusted_decay_steps))
        decayed_lr = (1.0 - alpha_f) * cosine_decay + alpha_f
        cosine_lr = peak_lr_f * decayed_lr

        # 3. Dynamic Selection Matrix
        current_lr = tf.where(step_f < warmup_steps_f, warmup_lr, cosine_lr)

        # 4. Decoupled Weight Decay Scaling
        if self.weight_decay_ratio is not None:
            return current_lr * self.weight_decay_ratio
            
        return current_lr

    def get_config(self):
        # Explicitly return names matching __init__ signatures exactly for fault-free deserialization
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
            "initial_lr": self.initial_lr,
            "weight_decay_ratio": self.weight_decay_ratio,
        }

class WeightDecaySchedulerCallback(keras.callbacks.Callback):
    """
    Step-level Keras callback that evaluates the dynamic schedule 
    and manually injects the correct weight decay value into the 
    AdamW slot, bypassing the Keras object-casting limitation.
    """
    def __init__(self, schedule_instance):
        super().__init__()
        self.schedule_instance = schedule_instance

    def on_train_batch_begin(self, batch, logs=None):
        # 1. Safely extract the raw scalar value to bypass symbolic tensor casting errors
        if hasattr(self.model.optimizer, 'iterations'):
            global_step_var = self.model.optimizer.iterations
            if hasattr(global_step_var, 'numpy'):
                global_step = int(global_step_var.numpy())
            else:
                global_step = int(self.model.optimizer.learning_rate.step.numpy())
        else:
            return

        # 2. Evaluate your custom schedule class with a clean, raw integer scalar
        current_wd = self.schedule_instance(global_step)
        
        # 3. Safely update the optimizer's weight decay value based on its type
        if hasattr(self.model.optimizer, 'weight_decay'):
            wd_var = self.model.optimizer.weight_decay
            if hasattr(wd_var, 'assign'):
                # Keras 2 / Variable tracking style
                wd_var.assign(current_wd)
            else:
                # Keras 3 / Pure native float assignment style
                self.model.optimizer.weight_decay = float(current_wd)

class EpochSynchronizedResetCallback(tf.keras.callbacks.Callback):
    """
    Keras callback designed to reset, re-shuffle, and clear queues 
    at epoch boundaries by recycling background multiprocessing pools.
    """
    def __init__(self, train_manager, val_manager):
        super().__init__()
        self.train_manager = train_manager
        self.val_manager = val_manager

    def on_train_begin(self, logs=None):
        """
        Natively executed by Keras immediately upon entering model.fit().
        Primes all background multiprocessing streams before Epoch 1 starts.
        """
        print("\n[Training Loop Commenced] Automatically priming background multiprocessing streams...")
        self.train_manager.start_epoch_stream()
        self.val_manager.start_epoch_stream()
        print("[Training Loop Commenced] Streams successfully primed for Epoch 1.")

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n[Epoch {epoch + 1} Complete] Recycling background multiprocessing streams...")
        
        # 1. Force terminate old workers and completely drop old multiprocess queues
        self.train_manager.stop_epoch_stream()
        self.val_manager.stop_epoch_stream()
        
        # 2. Re-spawn background workers with a fresh master file shuffle.
        # The infinite generator wrapper ensures that the active C++ graph handle
        # doesn't close—it will simply block and wait for these new queues to fill up.
        self.train_manager.start_epoch_stream()
        self.val_manager.start_epoch_stream()
            
        print(f"[Epoch {epoch + 1}] Streams successfully refreshed at epoch end.")

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

def close_position_error_3(y_true, y_pred):
    """
    Calculates MAE only for positions with a true evaluation between 0 and 3 pawns.
    Assumes y_true is scaled in pawns (e.g., 1.0 = 1 pawn, 3.0 = 3 pawns).
    """
    # 1. Compute raw absolute error per position
    raw_error = tf.abs(y_true - y_pred)
    
    # 2. Create a mask for positions within the 0 to 3 pawn range (absolute value)
    is_close_position = tf.abs(y_true) <= 3.0
    mask = tf.cast(is_close_position, tf.float32)
    
    # 3. Filter the error using the mask
    masked_error = raw_error * mask
    
    # 4. Compute the mean using only the valid filtered positions
    total_error = tf.reduce_sum(masked_error)
    valid_count = tf.reduce_sum(mask)
    
    # divide_no_nan prevents a 0/0 crash if a batch happens to have zero close positions
    return tf.math.divide_no_nan(total_error, valid_count)

def train_nnue_on_fens():
    # 1. Inputs: Sequences of active feature index tokens matching your Parquet shapes
    active_input = keras.Input(shape=(MAX_PIECES,), dtype="int32", name="active_features")
    passive_input = keras.Input(shape=(MAX_PIECES,), dtype="int32", name="passive_features")

    # 2. Shared Accumulator Layer (Using Embedding Reduction to replace the old sparse matrix bottleneck)
    nnue_accumulator_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=42)
    
    # We dimension the lookup array to INPUT_FEATURES + 1 to house the positive padding index row
    embedding_layer = layers.Embedding(
        input_dim=INPUT_FEATURES + 1,
        output_dim=256,
        embeddings_initializer=nnue_accumulator_init,
        mask_zero=True,
        name="accumulator_layer"
    )
    # 3. Pull dense weights representations for all slots
    a_embed = embedding_layer(active_input) # Target shape: (Batch, 16, 256)
    p_embed = embedding_layer(passive_input) # Target shape: (Batch, 16, 256)

    # 4. Synthesize masking vectors to isolate and zero-out padding weight contributions
    a_mask = keras.ops.cast(keras.ops.not_equal(active_input, INPUT_FEATURES), dtype="float32")
    p_mask = keras.ops.cast(keras.ops.not_equal(passive_input, INPUT_FEATURES), dtype="float32")
    
    # Expand to allow broadcasting dimensions across the 256 embedding properties
    a_mask = keras.ops.expand_dims(a_mask, axis=-1) 
    p_mask = keras.ops.expand_dims(p_mask, axis=-1)

    # Execute masked pool aggregation to compile the 256 accumulator vectors
    a_acc_raw = keras.ops.sum(a_embed * a_mask, axis=1)  # Target shape: (Batch, 256)
    p_acc_raw = keras.ops.sum(p_embed * p_mask, axis=1)  # Target shape: (Batch, 256)

    # 5. Clipped ReLU Activation (ReLU1 / Bounded ReLU)
    a_act = keras.ops.clip(a_acc_raw, 0.0, SCALE_MAX)
    p_act = keras.ops.clip(p_acc_raw, 0.0, SCALE_MAX)
    
    # 6. Perspective Multiplexing Layer (Shape: Batch, 256*2)
    merged = layers.Concatenate(name="perspective_multiplex")([a_act, p_act]) 
    
    # --- DEEP LAYER INITIALIZATION BLUEPRINT ---
    sub_layer_init = keras.initializers.RandomUniform(minval=-0.04, maxval=0.04, seed=42)
    hidden_bias_init = keras.initializers.Constant(value=0.005)

    # 7. Hidden Layer 2 with ReLU1 activation
    x = layers.Dense(
        64, 
        activation=None, 
        use_bias=True, 
        kernel_initializer=sub_layer_init,
        bias_initializer=hidden_bias_init,
        name="hidden_layer_2"
    )(merged)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)

    # 8. Hidden Layer 3 with ReLU1 activation
    x = layers.Dense(
        32, 
        activation=None, 
        use_bias=True, 
        kernel_initializer=sub_layer_init,
        bias_initializer=hidden_bias_init,
        name="hidden_layer_3"
    )(x)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)
    
    # 9. Output Layer (Linear, Pawn-Scale Evaluation Output)
    output = layers.Dense(
        1, 
        activation=None, 
        use_bias=True, 
        kernel_initializer=sub_layer_init,
        bias_initializer=keras.initializers.Zeros(),
        name="chess_eval"
    )(x)

    model = Model(
        inputs=[active_input, passive_input],
        outputs=output
    )

    # 2. Bind directly to your engineered AdamW
    decay_steps_calculated = TOTAL_EPOCHS * STEPS_PER_EPOCH

    # 1. Maintain your optimized learning rate schedule
    lr_schedule = CustomWarmupCosineSchedule(
        initial_learning_rate=3e-4,
        decay_steps=decay_steps_calculated,
        warmup_steps=int(STEPS_PER_EPOCH * 0.05) ,
        alpha=0.0167,
        initial_lr=1e-6
    )

    # 2. Maintain your custom weight decay schedule class instance
    wd_schedule = CustomWarmupCosineSchedule(
        initial_learning_rate=3e-4,
        decay_steps=decay_steps_calculated,
        warmup_steps=STEPS_PER_EPOCH,
        alpha=0.0167,
        initial_lr=1e-6,
        weight_decay_ratio=(0.01 / 3e-4)
    )

    # Initialize AdamW with a simple float fallback value for weight_decay
    stockfish_optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01,
        epsilon=1e-8,
        global_clipnorm=1.0
    )

    model.compile(
        optimizer=stockfish_optimizer,
        loss=pawn_probability_mse_loss,
        metrics=[close_position_error_all, close_position_error_3]
    )

    train_dir, val_dir = get_local_shard_directories()
    optimal_train_workers = 6
    
    train_manager = EpochSynchronizedDatasetManager(
        shard_directory=train_dir, shard_pattern="*.parquet", num_workers=optimal_train_workers, 
        queue_size=32, batch_size=BATCH_SIZE
    )
    val_manager = EpochSynchronizedDatasetManager(
        shard_directory=val_dir, shard_pattern="*.parquet", num_workers=1,
        queue_size=16, batch_size=BATCH_SIZE
    )

    STEPS_PER_EPOCH, VALIDATION_PER_EPOCH = compute_dynamic_epoch_geometry(
        train_dir="./balanced_shards/training",
        val_dir="./balanced_shards/validation",
        batch_size=BATCH_SIZE
    )

    # 1. Clean, named generator bindings (No loops, no inline lambdas)
    def train_generator_wrapper():
        return train_manager.generator_fn(total_epoch_steps=STEPS_PER_EPOCH)

    def val_generator_wrapper():
        return val_manager.generator_fn(total_epoch_steps=VALIDATION_PER_EPOCH)

    # 2. Build structural dataset graph definitions EXACTLY ONCE.
    # The .repeat() loop ensures Keras never runs out of data at epoch limits,
    # while the static name prevents graph retracing memory leaks.
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

    # Instantiate the static dataset references
    train_dataset = get_fresh_train_dataset()
    val_dataset = get_fresh_val_dataset()

    print(f"\n--- Model compilation complete. Commencing Training with {optimal_train_workers} Workers ---")

    checkpoint_path = "best_chess_nnue.keras"
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=False,
        verbose=1
    )

    cleanup_cb = AggressiveMemoryCleanup()

    # 3. Your callback that cleans up and resets the managers at epoch end
    epoch_reset_cb = EpochSynchronizedResetCallback(
        train_manager=train_manager, 
        val_manager=val_manager
    )

    wd_callback = WeightDecaySchedulerCallback(schedule_instance=wd_schedule)

    # Pass the stable dataset references directly into fit execution
    model.fit(
        train_dataset,
        epochs=TOTAL_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_dataset,
        validation_steps=VALIDATION_PER_EPOCH,
        shuffle=False, 
        callbacks=[checkpoint_cb, cleanup_cb, epoch_reset_cb, wd_callback]
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