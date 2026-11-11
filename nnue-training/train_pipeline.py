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
# 32 Buckets * 11 Active Pieces (excluding own King) * 64 Squares + 4 Castling Squares = 22532
INPUT_FEATURES = 32 * 11 * 64 + 4  # Dual-Perspective HalfKA Dimension (22532)
MAX_ACTIVE_FEATURES = 35               # Uniform layout array padding bound
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
        
def get_local_shard_directories():
    """
    Defines the storage locations for your exported clean Parquet files.
    Modify these strings to point directly to your dataset output directories.
    """
    train_dir = "./balanced_shards/training/"
    val_dir = "./balanced_shards/validation/"
    return train_dir, val_dir

# Fallback structures for metric and loop calculations
class AggressiveMemoryCleanup(keras.callbacks.Callback):
    """Safely frees dead CPU worker references to prevent RAM accumulation."""
    def on_epoch_end(self, epoch, logs=None):
        gc.collect() 

class TieredPositionDecayCallback(keras.callbacks.Callback):
    """
    Three-tier curriculum decay calibrated for SF_SCALE = 360.0:
      - |eval| <= 150 cp:        Always 1.00 (peak WDL gradient zone)
      - 150 < |eval| <= 300 cp:  Decays 1.00 -> 0.70 (Slight)
      - 300 < |eval| <= 500 cp:  Decays 1.00 -> 0.40 (Mid)
      - |eval| > 500 cp:         Decays 1.00 -> 0.10 (Extreme)
    """
    def __init__(
        self,
        slight_prob_var,
        mid_prob_var,
        extreme_prob_var,
        warmup_epochs=3,
        slight_end_prob=0.70,
        mid_end_prob=0.40,
        extreme_end_prob=0.10,
        decay_epochs=15,
        verbose=1
    ):
        super().__init__()
        self.slight_prob_var = slight_prob_var
        self.mid_prob_var = mid_prob_var
        self.extreme_prob_var = extreme_prob_var
        self.warmup_epochs = warmup_epochs
        self.slight_end_prob = slight_end_prob
        self.mid_end_prob = mid_end_prob
        self.extreme_end_prob = extreme_end_prob
        self.decay_epochs = decay_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            slight_p = 1.0
            mid_p = 1.0
            ext_p = 1.0
        else:
            progress = min((epoch - self.warmup_epochs) / max(self.decay_epochs, 1), 1.0)
            slight_p = 1.0 - progress * (1.0 - self.slight_end_prob)
            mid_p = 1.0 - progress * (1.0 - self.mid_end_prob)
            ext_p = 1.0 - progress * (1.0 - self.extreme_end_prob)

        self.slight_prob_var.assign(slight_p)
        self.mid_prob_var.assign(mid_p)
        self.extreme_prob_var.assign(ext_p)

        if self.verbose > 0:
            print(
                f"\n[Curriculum] Epoch {epoch + 1}: "
                f"Slight(150-300cp)={slight_p:.2f} | "
                f"Mid(300-500cp)={mid_p:.2f} | "
                f"Extreme(>500cp)={ext_p:.2f}"
            )

class StockfishNNUE(Model):
    def __init__(self, input_features, scale_max=1.0, alpha_leak=0.01, **kwargs):
        super().__init__(**kwargs)
        self.input_features = input_features
        self.scale_max = scale_max
        self.alpha_leak = alpha_leak
        self.max_active_features = MAX_ACTIVE_FEATURES

        # --- Architecture Layer Instantiations ---
        nnue_accumulator_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)
        
        self.embedding_layer = layers.Embedding(
            input_dim=self.input_features + 1,
            output_dim=256, 
            mask_zero=True, 
            embeddings_initializer=nnue_accumulator_init,
            name="accumulator_layer"
        )
        
        self.concat = layers.Concatenate(name="perspective_multiplex")
        
        hidden_init = keras.initializers.HeNormal(seed=42)
        self.dense2 = layers.Dense(32, activation=None, use_bias=True, kernel_initializer=hidden_init, name="hidden_layer_2")
        self.dense3 = layers.Dense(32, activation=None, use_bias=True, kernel_initializer=hidden_init, name="hidden_layer_3")
        self.chess_eval = layers.Dense(1, activation=None, use_bias=True, kernel_initializer=hidden_init, name="chess_eval")

    def get_config(self):
        """
        Enables Keras to cleanly serialize and recreate this exact model configuration 
        during ModelCheckpoint saving and loading operations.
        """
        config = super().get_config()
        config.update({
            "input_features": self.input_features,
            "scale_max": self.scale_max,
            "alpha_leak": self.alpha_leak,
        })
        return config

    def safe_leaky_clipped_relu(self, x):
        floor_handled = keras.ops.maximum(x, 0.0) + self.alpha_leak * keras.ops.minimum(x, 0.0)
        capped = keras.ops.minimum(floor_handled, self.scale_max)
        overshoot = keras.ops.maximum(floor_handled - self.scale_max, 0.0)
        return capped + (self.alpha_leak * overshoot)

    def call(self, inputs):
        active_input = inputs["active_features"]
        passive_input = inputs["passive_features"]

        # Create binary masks (1.0 for valid features, 0.0 for padding index 0)
        a_mask = keras.ops.expand_dims(keras.ops.cast(active_input != 0, "float32"), -1)
        p_mask = keras.ops.expand_dims(keras.ops.cast(passive_input != 0, "float32"), -1)

        # Look up embeddings and zero out padded entries before summing
        a_embed = self.embedding_layer(active_input) * a_mask
        p_embed = self.embedding_layer(passive_input) * p_mask

        a_acc_raw = keras.ops.sum(a_embed, axis=1)
        p_acc_raw = keras.ops.sum(p_embed, axis=1)

        a_act = self.safe_leaky_clipped_relu(a_acc_raw)
        p_act = self.safe_leaky_clipped_relu(p_acc_raw)

        merged = self.concat([a_act, p_act])
        
        x2_raw = self.dense2(merged)
        x2_act = self.safe_leaky_clipped_relu(x2_raw)

        x3_raw = self.dense3(x2_act)
        x3_act = self.safe_leaky_clipped_relu(x3_raw)
        
        raw_eval = self.chess_eval(x3_act)
        
        return raw_eval * 360.0
    
    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # 1. Manually update your custom centipawn evaluation metrics
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
                
        # 2. Build the output tracker and explicitly inject the computed loss scalar
        results = {m.name: m.result() for m in self.metrics if m.name != "loss"}
        results["loss"] = loss
        return results

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
        
        # 1. Update validation metric logs
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)
                
        # 2. Build the output tracker and explicitly inject the validation loss
        results = {m.name: m.result() for m in self.metrics if m.name != "loss"}
        results["loss"] = loss
        return results

# =========================================================================
# COMPLETED REWRITTEN NEWBOB SCHEDULER CALLBACK
# =========================================================================
class NewbobLearningRateScheduler(keras.callbacks.Callback):
    """
    Classic Chess NNUE Newbob Learning Rate Scheduler.
    """
    def __init__(self, initial_lr=1.0e-3, decay_factor=0.5, threshold=1.0e-4, min_lr=5.0e-5, verbose=1):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.threshold = threshold  
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_loss = np.inf
        self.current_lr = initial_lr

    def on_train_begin(self, logs=None):
        if hasattr(self.model.optimizer, "learning_rate"):
            self.model.optimizer.learning_rate.assign(self.initial_lr)
            self.current_lr = self.initial_lr
        if self.verbose > 0:
            print(f"\n[Newbob] Training initiated. Initial learning rate configured to: {self.current_lr}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")

        if val_loss is None:
            if self.verbose > 0:
                print(f"\n[Newbob Warning] 'val_loss' not found in logs for epoch {epoch + 1}. Skipping step.")
            return

        if self.best_loss == np.inf:
            self.best_loss = val_loss
            if self.verbose > 0:
                print(f"\n[Newbob] Baseline validation loss established: {self.best_loss:.6f}")
            return

        improvement = self.best_loss - val_loss

        if improvement > self.threshold:
            self.best_loss = val_loss
            if self.verbose > 0:
                print(f"\n[Newbob] Validation loss improved by {improvement:.6f} (Threshold: {self.threshold}). "
                      f"New best loss: {self.best_loss:.6f}. Keeping LR: {self.current_lr:.1e}")
        else:
            old_lr = self.current_lr
            new_lr = old_lr * self.decay_factor

            if new_lr < self.min_lr:
                new_lr = self.min_lr
                if self.verbose > 0:
                    print(f"\n[Newbob] Plateau detected, but learning rate is locked at min_lr boundary: {self.min_lr:.1e}")
            else:
                self.current_lr = new_lr
                if hasattr(self.model.optimizer, "learning_rate"):
                    self.model.optimizer.learning_rate.assign(self.current_lr)
                
                if self.verbose > 0:
                    print(f"\n[Newbob Plateau] Loss improvement ({improvement:.6f}) failed to clear threshold ({self.threshold}). "
                          f"Decaying learning rate from {old_lr:.1e} to {self.current_lr:.1e}")
    

# --- Export NNUE Weights for Rust ---
def export_dense_nnue_for_rust(model, file_path="model.nnue"):
    with open(file_path, "wb") as f:
        print("--- Commencing Weight Quantization & Serialization for Rust ---")
        
        # 1. Accumulator Layer (22532 -> 256)
        # Input: Binary (0/1) | Weights: i16 | Bias/Output Accumulator: i32
        acc_layer = model.get_layer("accumulator_layer")
        embedding_weights = acc_layer.get_weights()[0]

        # Accumulator Weights (Updated dimension bound for 16-bucket layout)
        w1_real = embedding_weights[1:INPUT_FEATURES + 1, :]
        w1_quant = np.ascontiguousarray(np.round(w1_real * 128.0)).astype(np.int16)

        f.write(w1_quant.tobytes())
        print(f"-> Accumulator Layer serialized. Shape: {w1_real.shape} (Weights: i16 / Bias: NONE)")

        # 2. Hidden Layer 2 (256*2 -> 32)
        # Input: i16 (Clipped from Accumulator) | Weights: i8 | Bias/Output: i32
        # Shift Right by 7 (>> 7) before clipping to next input scale.
        layer2 = model.get_layer("hidden_layer_2") 
        w2, b2 = layer2.get_weights()
        w2_quant = np.ascontiguousarray(np.clip(np.round(w2.T * 32.0), -128, 127).astype(np.int8))
        b2_quant = np.round(b2 * 32.0).astype(np.int32) 
        f.write(w2_quant.tobytes())
        f.write(b2_quant.tobytes())
        print(f"-> Hidden Layer 2 serialized. Shape: {w2.shape} (Weights: i8 / Bias: i32) [Rust -> Shift >> 7]")

        # 3. Hidden Layer 3 (32 -> 32)
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

def centipawn_probability_mse_loss(y_true, y_pred):
    """
    Tuned for your 32-neuron bottleneck outputting CENTIPAWN units.
    Ensures gradients heavily prioritize precision between -200 and +200 centipawns.
    """
    SF_SCALE = 360.0 
    
    # Mathematical conversion factor: ln(10) / 360.0
    # 2.30258509299 / 360.0 ≈ 0.00639607
    scale_factor = 2.30258509299 / SF_SCALE
    
    y_true_float = keras.ops.cast(y_true, "float32")
    y_pred_float = keras.ops.cast(y_pred, "float32")
    
    y_true_prob = keras.ops.sigmoid(y_true_float * scale_factor)
    y_pred_prob = keras.ops.sigmoid(y_pred_float * scale_factor)
    
    # Calculate MSE on the probability landscape using stable continuous operations
    loss = keras.ops.square(y_true_prob - y_pred_prob)
    return keras.ops.mean(loss)

def close_position_error_all(y_true, y_pred):
    """
    Calculates Mean Absolute Error only for Non-Mate positions evaluating 
    within standard tactical play boundaries (up to 1000 centipawns / 10 pawns).
    """
    y_true_float = keras.ops.cast(y_true, "float32")
    y_pred_float = keras.ops.cast(y_pred, "float32")

    # 1. Compute raw absolute error per position
    raw_error = keras.ops.abs(y_true_float - y_pred_float)
    
    # 2. Create a mask for positions within the 0 to 1000 centipawn range (absolute value)
    is_close_position = keras.ops.abs(y_true_float) <= 1000.0
    mask = keras.ops.cast(is_close_position, "float32")
    
    # 3. Filter the error using the mask
    masked_error = raw_error * mask
    
    # 4. Compute the mean using only the valid filtered positions
    total_error = keras.ops.sum(masked_error)
    valid_count = keras.ops.sum(mask)
    
    # Avoid division by zero if a batch happens to contain zero close positions
    return total_error / (valid_count + 1e-8)

def close_position_error(y_true, y_pred):
    """
    Calculates MAE only for critical competitive positions with a true evaluation 
    between 0 and 250 centipawns (2.5 pawns), where engine choice matters most.
    """
    y_true_float = keras.ops.cast(y_true, "float32")
    y_pred_float = keras.ops.cast(y_pred, "float32")

    # 1. Compute raw absolute error per position
    raw_error = keras.ops.abs(y_true_float - y_pred_float)
    
    # 2. Create a mask for positions within the 0 to 250 centipawn range (absolute value)
    is_close_position = keras.ops.abs(y_true_float) <= 250.0
    mask = keras.ops.cast(is_close_position, "float32")
    
    # 3. Filter the error using the mask
    masked_error = raw_error * mask
    
    # 4. Compute the mean using only the valid filtered positions
    total_error = keras.ops.sum(masked_error)
    valid_count = keras.ops.sum(mask)
    
    # Avoid division by zero if a batch happens to contain zero close positions
    return total_error / (valid_count + 1e-8)

def train_nnue_on_fens():
    # Setup Model
    model = StockfishNNUE(input_features=INPUT_FEATURES, scale_max=SCALE_MAX)

    # 2. Bind directly to your engineered AdamW
    STEPS_PER_EPOCH, VALIDATION_PER_EPOCH = compute_dynamic_epoch_geometry(
        train_dir="./balanced_shards/training",
        val_dir="./balanced_shards/validation",
        batch_size=BATCH_SIZE
    )
    
    stockfish_optimizer = keras.optimizers.AdamW(
        learning_rate=1.0e-3,
        weight_decay=1e-5,
        epsilon=1e-5,
        global_clipnorm=1.0 
    )

    model.compile(
        optimizer=stockfish_optimizer,
        loss=centipawn_probability_mse_loss,
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

    slight_keep_prob = tf.Variable(1.0, dtype=tf.float32, trainable=False)   # 150 to 300 cp
    mid_keep_prob = tf.Variable(1.0, dtype=tf.float32, trainable=False)      # 300 to 500 cp
    extreme_keep_prob = tf.Variable(1.0, dtype=tf.float32, trainable=False)  # > 500 cp

    def tiered_dynamic_filter(x, y):
        abs_y = tf.abs(tf.squeeze(y))

        prob = tf.where(
            abs_y > 500.0,
            extreme_keep_prob,
            tf.where(
                abs_y > 300.0,
                mid_keep_prob,
                tf.where(abs_y > 150.0, slight_keep_prob, 1.0)
            )
        )
        return tf.random.uniform([]) < prob

    # 3. Build structural dataset graph definitions EXACTLY ONCE with uniform infinite loops
    def get_fresh_train_dataset():
        return (
            tf.data.Dataset.from_generator(
                generator=train_generator_wrapper,
                output_signature=(
                    {
                        "active_features": tf.TensorSpec(shape=(BATCH_SIZE, MAX_ACTIVE_FEATURES), dtype=tf.int16),
                        "passive_features": tf.TensorSpec(shape=(BATCH_SIZE, MAX_ACTIVE_FEATURES), dtype=tf.int16),
                    },
                    tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.float32)
                )
            )
            .unbatch()
            .repeat()
            .filter(tiered_dynamic_filter)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

    def get_fresh_val_dataset():
        return tf.data.Dataset.from_generator(
            generator=val_generator_wrapper,
            output_signature=(
                {
                    "active_features": tf.TensorSpec(shape=(BATCH_SIZE, MAX_ACTIVE_FEATURES), dtype=tf.int16),
                    "passive_features": tf.TensorSpec(shape=(BATCH_SIZE, MAX_ACTIVE_FEATURES), dtype=tf.int16),
                },
                tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.float32)
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

    newbob_callback = NewbobLearningRateScheduler(
        initial_lr=1.0e-3,
        decay_factor=0.5, 
        threshold=1.0e-4, 
        min_lr=1.0e-5,
        verbose=1
    )

    # Decays |eval| > 150 / 300 / 500 CP advantages to preserve later gradient
    tiered_decay_cb = TieredPositionDecayCallback(
        slight_prob_var=slight_keep_prob,
        mid_prob_var=mid_keep_prob,
        extreme_prob_var=extreme_keep_prob,
        warmup_epochs=3,
        slight_end_prob=0.70,
        mid_end_prob=0.40,
        extreme_end_prob=0.10,
        decay_epochs=15,
        verbose=1
    )

    # 6. Fit Execution: Runs perfectly synchronized across all 30 epochs
    model.fit(
        train_dataset,
        epochs=TOTAL_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_dataset,
        validation_steps=VALIDATION_PER_EPOCH,
        shuffle=False, 
        callbacks=[tiered_decay_cb, newbob_callback, checkpoint_cb, cleanup_cb]
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