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
TOTAL_EPOCHS = 3

# Mixed Data Sets
CLEAN_DATASET_DIR = "./balanced_shards" 
BIN_SAVE_PATH = "nnue_q_search_weights.bin"

NNUE_WEIGHT_PATH = "nnue_weights.bin"

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

class QuietGateNNUE(Model):
    def __init__(
        self,
        input_features,
        nnue_path="nnue_weights.bin",
        scale_max=1.0,
        alpha_leak=0.01,
        noisy_prior=0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_features = input_features
        self.nnue_path = nnue_path
        self.scale_max = scale_max
        self.alpha_leak = alpha_leak
        self.noisy_prior = noisy_prior
        self.max_active_features = MAX_ACTIVE_FEATURES

        # --- Load & Quantization-Align Frozen Accumulator from .nnue ---
        acc_elements = self.input_features * 256
        acc_bytes = acc_elements * 2  # i16 = 2 bytes

        with open(self.nnue_path, "rb") as f:
            raw_bytes = f.read(acc_bytes)

        if len(raw_bytes) != acc_bytes:
            raise ValueError(
                f"Expected {acc_bytes} bytes for accumulator in {self.nnue_path}, got {len(raw_bytes)}"
            )

        w1_quant = np.frombuffer(raw_bytes, dtype=np.int16).reshape((self.input_features, 256))
        w1_real = w1_quant.astype(np.float32) / 128.0

        full_embedding = np.zeros((self.input_features + 1, 256), dtype=np.float32)
        full_embedding[1 : self.input_features + 1, :] = w1_real

        # --- Architecture Layer Instantiations ---
        self.embedding_layer = layers.Embedding(
            input_dim=self.input_features + 1,
            output_dim=256,
            mask_zero=True,
            embeddings_initializer=keras.initializers.Constant(full_embedding),
            trainable=False,
            name="accumulator_layer",
        )

        self.concat = layers.Concatenate(name="perspective_multiplex")

        hidden_init = keras.initializers.HeNormal(seed=42)
        prior_logit = float(np.log(self.noisy_prior / (1.0 - self.noisy_prior)))
        prior_bias_init = keras.initializers.Constant(prior_logit)

        self.dense2 = layers.Dense(
            32,
            activation=None,
            use_bias=True,
            kernel_initializer=hidden_init,
            name="noisy_hidden_layer_2",
        )
        self.dense3 = layers.Dense(
            32,
            activation=None,
            use_bias=True,
            kernel_initializer=hidden_init,
            name="noisy_hidden_layer_3",
        )
        self.noisy_gate = layers.Dense(
            1,
            activation=None,
            use_bias=True,
            kernel_initializer=hidden_init,
            bias_initializer=prior_bias_init,
            name="noisy_gate",
        )

    def get_config(self):
        """
        Enables Keras to cleanly serialize and recreate this exact model configuration
        during ModelCheckpoint saving and loading operations.
        """
        config = super().get_config()
        config.update(
            {
                "input_features": self.input_features,
                "nnue_path": self.nnue_path,
                "scale_max": self.scale_max,
                "alpha_leak": self.alpha_leak,
                "noisy_prior": self.noisy_prior,
            }
        )
        return config

    def safe_leaky_clipped_relu(self, x):
        floor_handled = keras.ops.maximum(x, 0.0) + self.alpha_leak * keras.ops.minimum(x, 0.0)
        capped = keras.ops.minimum(floor_handled, self.scale_max)
        overshoot = keras.ops.maximum(floor_handled - self.scale_max, 0.0)
        return capped + (self.alpha_leak * overshoot)

    def call(self, inputs, training=False):
        active_input = inputs["active_features"]
        passive_input = inputs["passive_features"]

        # Create binary masks (1.0 for valid features, 0.0 for padding index 0)
        a_mask = keras.ops.expand_dims(keras.ops.cast(active_input != 0, "float32"), -1)
        p_mask = keras.ops.expand_dims(keras.ops.cast(passive_input != 0, "float32"), -1)

        # Look up frozen quantized embeddings and zero out padded entries before summing
        a_embed = self.embedding_layer(active_input) * a_mask
        p_embed = self.embedding_layer(passive_input) * p_mask

        a_acc_raw = keras.ops.sum(a_embed, axis=1)
        p_acc_raw = keras.ops.sum(p_embed, axis=1)

        # Activate each perspective separately before concatenating (matches StockfishNNUE)
        a_act = self.safe_leaky_clipped_relu(a_acc_raw)
        p_act = self.safe_leaky_clipped_relu(p_acc_raw)

        merged = self.concat([a_act, p_act])

        x2_raw = self.dense2(merged)
        x2_act = self.safe_leaky_clipped_relu(x2_raw)

        x3_raw = self.dense3(x2_act)
        x3_act = self.safe_leaky_clipped_relu(x3_raw)

        # Returns raw logits (compatible with binary_crossentropy(from_logits=True))
        noisy_logits = self.noisy_gate(x3_act)

        return noisy_logits

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)

        results = {m.name: m.result() for m in self.metrics if m.name != "loss"}
        results["loss"] = loss
        return results

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.compute_loss(x=x, y=y, y_pred=y_pred)

        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_pred)

        results = {m.name: m.result() for m in self.metrics if m.name != "loss"}
        results["loss"] = loss
        return results

# --- Export NNUE Weights for Rust ---
def export_dense_nnue_for_rust(model, file_path="model.nnue"):
    with open(file_path, "wb") as f:
        # 2. Hidden Layer 2 (256*2 -> 32)
        # Input: i16 (Clipped from Accumulator) | Weights: i8 | Bias/Output: i32
        # Shift Right by 7 (>> 7) before clipping to next input scale.
        layer2 = model.get_layer("noisy_hidden_layer_2") 
        w2, b2 = layer2.get_weights()
        w2_quant = np.ascontiguousarray(np.clip(np.round(w2.T * 32.0), -128, 127).astype(np.int8))
        b2_quant = np.round(b2 * 4096.0).astype(np.int32) 
        f.write(w2_quant.tobytes())
        f.write(b2_quant.tobytes())
        print(f"-> Hidden Layer 2 serialized. Shape: {w2.shape} (Weights: i8 / Bias: i32) [Rust -> Shift >> 7]")

        # 3. Hidden Layer 3 (32 -> 32)
        # Input: i16 | Weights: i8 | Bias/Output: i32 
        # Shift Right by 5 (>> 5) before clipping to next input scale.
        layer3 = model.get_layer("noisy_hidden_layer_3")
        w3, b3 = layer3.get_weights()
        w3_quant = np.ascontiguousarray(np.clip(np.round(w3.T * 32.0), -128, 127).astype(np.int8))
        b3_quant = np.round(b3 * 1024.0).astype(np.int32)
        f.write(w3_quant.tobytes())
        f.write(b3_quant.tobytes())
        print(f"-> Hidden Layer 3 serialized. Shape: {w3.shape} (Weights: i8 / Bias: i32) [Rust -> Shift >> 5]")

        # 4. Output Layer (32 -> 1)
        # Input: i16 | Weights: i8 | Bias/Output: i32 
        # Shift Right by 5 (>> 5) to get final scaled score.
        output_layer = model.get_layer("noisy_gate")
        w4, b4 = output_layer.get_weights()
        w4_quant = np.ascontiguousarray(np.clip(np.round(w4.T * 32.0), -128, 127).astype(np.int8))
        b4_quant = np.round(b4 * 1024.0).astype(np.int32)
        f.write(w4_quant.tobytes())
        f.write(b4_quant.tobytes())
        print(f"-> Output Layer serialized. Shape: {w4.shape} (Weights: i8 / Bias: i32) [Rust -> Shift >> 10]")

    print(f"\n[SUCCESS] Safe NNUE file successfully compiled and written to: {file_path}")

def quiet_loss(y_true, y_pred_logits):
    # y_true: 1.0 = noisy, 0.0 = quiet
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(y_true, tf.float32),
        logits=y_pred_logits,
    )
    return tf.reduce_mean(loss)

def q_search_prob_error(y_true, y_pred_logits):
    # Converts raw logits to [0, 1] probabilities before measuring error
    probs = keras.ops.sigmoid(y_pred_logits)
    avg_error = keras.ops.mean(keras.ops.abs(y_true - probs))
    return avg_error

def q_search_hard_error(y_true, y_pred_logits):
    # Ensure y_true is bounded to [0.0, 1.0]
    y_true_clipped = keras.ops.clip(keras.ops.cast(y_true, "float32"), 0.0, 1.0)
    # Exact 0 vs 1 prediction (logit > 0.0 -> 1.0, else 0.0)
    hard_preds = keras.ops.cast(y_pred_logits > 0.0, dtype="float32")
    error = keras.ops.mean(keras.ops.abs(y_true_clipped - hard_preds))
    return keras.ops.clip(error, 0.0, 1.0)

class LogitRecall(keras.metrics.Recall):
    def __init__(self, threshold=0.5, name="recall", **kwargs):
        super().__init__(thresholds=threshold, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.sigmoid(y_pred), sample_weight)

class LogitSkippedRate(keras.metrics.Metric):
    def __init__(self, threshold=0.10, name="skip_rate", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.skipped = self.add_weight(name="skipped", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        probs = tf.sigmoid(y_pred)
        is_skipped = tf.cast(probs <= self.threshold, tf.float32)
        self.skipped.assign_add(tf.reduce_sum(is_skipped))
        self.total.assign_add(tf.cast(tf.size(y_pred), tf.float32))

    def result(self):
        return self.skipped / self.total

def train_nnue_on_fens():
    # Setup Model
    model = QuietGateNNUE(
        input_features=INPUT_FEATURES,
        nnue_path="nnue_weights.bin",
        scale_max=SCALE_MAX,
    )

    # 2. Bind directly to your engineered AdamW
    STEPS_PER_EPOCH, VALIDATION_PER_EPOCH = compute_dynamic_epoch_geometry(
        train_dir="./balanced_shards/training",
        val_dir="./balanced_shards/validation",
        batch_size=BATCH_SIZE
    )
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1.0e-5,
        decay_steps=STEPS_PER_EPOCH * TOTAL_EPOCHS,
        alpha=1.0e-2,
        warmup_target=1.0e-3,
        warmup_steps=0.025 * STEPS_PER_EPOCH * TOTAL_EPOCHS,
    )

    stockfish_optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        beta_2=0.99,
        weight_decay=1.0e-6,
        epsilon=1e-5,
        global_clipnorm=1.0
    )
    stockfish_optimizer.exclude_from_weight_decay(var_names=["bias"])

    model.compile(
        optimizer=stockfish_optimizer,
        loss=quiet_loss,
        metrics=[
            keras.metrics.AUC(from_logits=True, name="pr_auc", curve="PR"),
        ],
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

    def get_fresh_train_dataset():
        return tf.data.Dataset.from_generator(
            generator=train_generator_wrapper,
            output_signature=(
                {
                    "active_features": tf.TensorSpec(shape=(BATCH_SIZE, MAX_ACTIVE_FEATURES), dtype=tf.int16),
                    "passive_features": tf.TensorSpec(shape=(BATCH_SIZE, MAX_ACTIVE_FEATURES), dtype=tf.int16),
                },
                tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.float32)
            )
        ).repeat().prefetch(tf.data.AUTOTUNE)
    
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
    checkpoint_path = "best_q_search_nnue.keras"
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