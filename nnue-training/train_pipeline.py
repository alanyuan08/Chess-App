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
INPUT_FEATURES = 64 * 64 * 12  # Dual-Perspective HalfKA Dimension (49152)
MAX_PIECES = 32                # Uniform layout array padding bound
SCALE_MAX = 1.0                # Bounded Clipped ReLU limit 

# --- Step-Based Architecture Configuration ---
# Auto-calculate exact step parameters from compiled files
BATCH_SIZE = 8192
TOTAL_EPOCHS = 30  

def compute_dynamic_epoch_geometry(train_dir="./balanced_shards/training", 
                                   val_dir="./balanced_shards/validation", 
                                   batch_size=8192):
    """
    Scans the compiled parquet sharding directories to instantly compute
    flawless, exact step parameters based on physical file constraints.
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
    
    # 3. Apply strict block geometry calculations
    steps_per_epoch = total_train_rows // batch_size
    validation_per_epoch = total_val_rows // batch_size
    
    print(f"   [TRAIN UNIVERSE] Physical Rows: {total_train_rows:,} -> Steps/Epoch: {steps_per_epoch}")
    print(f"   [VAL UNIVERSE]   Physical Rows: {total_val_rows:,} -> Steps/Epoch: {validation_per_epoch}")
    
    return steps_per_epoch, validation_per_epoch

STEPS_PER_EPOCH, VALIDATION_PER_EPOCH = compute_dynamic_epoch_geometry(
    train_dir="./balanced_shards/training",
    val_dir="./balanced_shards/validation",
    batch_size=BATCH_SIZE
)
WARMUP_STEPS = int(STEPS_PER_EPOCH * 0.25) 

# Mixed Data Sets
CLEAN_DATASET_DIR = "./balanced_shards" 
BIN_SAVE_PATH = "nnue_weights.bin"

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
    def __init__(self, warmup_steps, cosine_schedule, initial_lr=1e-6):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.cosine_schedule = cosine_schedule
        self.initial_lr = initial_lr
        
        # Dynamically extract peak LR from the underlying cosine schedule config
        if hasattr(cosine_schedule, 'initial_learning_rate'):
            self.peak_lr = cosine_schedule.initial_learning_rate
        else:
            # Fallback if config structure varies
            self.peak_lr = cosine_schedule.get_config().get('initial_learning_rate', 3e-4)

    def __call__(self, step):
        step_f = tf.cast(step, tf.float32)
        warmup_steps_f = tf.cast(self.warmup_steps, tf.float32)
        
        # Epoch 1: Linear upward ramp
        warmup_lr = self.initial_lr + (self.peak_lr - self.initial_lr) * (step_f / warmup_steps_f)
        
        # Epochs 2-30: Native Cosine Decay tracking steps past warmup
        decay_step = tf.maximum(step_f - warmup_steps_f, 0.0)
        cosine_lr = self.cosine_schedule(decay_step)
        
        # Graph-safe conditional selection
        return tf.where(step_f < warmup_steps_f, warmup_lr, cosine_lr)

    def get_config(self):
        # Must return primitive types or serializable Keras objects
        return {
            "warmup_steps": self.warmup_steps,
            "cosine_schedule": keras.optimizers.schedules.serialize(self.cosine_schedule),
            "initial_lr": self.initial_lr
        }

    @classmethod
    def from_config(cls, config):
        # Correctly deserializes the nested schedule upon loading
        config["cosine_schedule"] = keras.optimizers.schedules.deserialize(config["cosine_schedule"])
        return cls(**config)

@keras.utils.register_keras_serializable()
class SharedAccumulatorBias(layers.Layer):
    """
    Custom Keras layer hosting a single, shared trainable 
    256-dimensional accumulator bias vector (b1) applied to a single perspective.
    """
    def __init__(self, output_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.bias = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
            name="accumulator_bias_vector"
        )
        super().build(input_shape)

    def call(self, inputs):
        # Broadcasts the (256,) bias vector across the (Batch, 256) incoming tensor
        return inputs + self.bias

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        return config

def get_local_shard_directories():
    """
    Defines the storage locations for your exported clean Parquet files.
    Modify these strings to point directly to your dataset output directories.
    """
    train_dir = "./balanced_shards/training/"
    val_dir = "./balanced_shards/validation/"
    return train_dir, val_dir

# --- Export NNuE Weights for Rust ---
def export_dense_nnue_for_rust(model, file_path="model.nnue"):
    with open(file_path, "wb") as f:
        print("--- Commencing Weight Quantization & Serialization for Rust ---")
        
        # 1. Accumulator Layer (49152 -> 256)
        # Input: Binary (0/1) | Weights: i16 | Bias/Output Accumulator: i32
        acc_layer = model.get_layer("accumulator_layer")
        embedding_weights = acc_layer.get_weights()[0]

        # Accumlator Weights
        w1_real = embedding_weights[:49152, :]
        w1_quant = np.ascontiguousarray(np.round(w1_real * 128.0)).astype(np.int16)

        # Accumlator Bias
        bias_layer = model.get_layer("accumulator_bias")
        b1_real = bias_layer.get_weights()[0]
        b1_quant = np.round(b1_real * 128.0).astype(np.int32)
        # Write exactly the same bytes structure as before
        f.write(w1_quant.tobytes())
        f.write(b1_quant.tobytes())
        print(f"-> Accumulator Layer serialized. Shape: {w1_real.shape} (Weights: i16 / Synthetic Bias: i32)")

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

def train_nnue_on_fens():
    # 1. Inputs: Sequences of active feature index tokens matching your Parquet shapes
    active_input = keras.Input(shape=(MAX_PIECES,), dtype="int32", name="active_features")
    passive_input = keras.Input(shape=(MAX_PIECES,), dtype="int32", name="passive_features")

    # 2. Shared Accumulator Layer (Using Embedding Reduction to replace the old sparse matrix bottleneck)
    nnue_accumulator_init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.005)
    
    # We dimension the lookup array to INPUT_FEATURES + 1 to house the positive padding index row
    embedding_layer = layers.Embedding(
        input_dim=INPUT_FEATURES + 1,
        output_dim=256,
        embeddings_initializer=nnue_accumulator_init,
        mask_zero=True,
        name="accumulator_layer"
    )

    accumulator_bias_layer = SharedAccumulatorBias(
        name="accumulator_bias", 
        output_dim=256
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

    # INJECT THE ACCUMULATOR BIAS BEFORE CLIP/RELU
    a_acc_biased = accumulator_bias_layer(a_acc_raw)
    p_acc_biased = accumulator_bias_layer(p_acc_raw)

    # 5. Clipped ReLU Activation (ReLU1 / Bounded ReLU)
    a_act = keras.ops.clip(a_acc_biased, 0.0, SCALE_MAX)
    p_act = keras.ops.clip(p_acc_biased, 0.0, SCALE_MAX)
    
    # 6. Perspective Multiplexing Layer (Shape: Batch, 256*2)
    merged = layers.Concatenate(name="perspective_multiplex")([a_act, p_act]) 
    
    # 7. Hidden Layer 2 with ReLU1 activation
    x = layers.Dense(
        32, 
        activation=None, 
        use_bias=True, 
        kernel_initializer=keras.initializers.HeNormal(),
        name="hidden_layer_2"
    )(merged)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)

    # 8. Hidden Layer 3 with ReLU1 activation
    x = layers.Dense(
        32, 
        activation=None, 
        use_bias=True, 
        kernel_initializer=keras.initializers.HeNormal(),
        name="hidden_layer_3"
    )(x)
    x = keras.ops.clip(x, 0.0, SCALE_MAX)
    
    # 9. Output Layer (Linear, Pawn-Scale Evaluation Output)
    output = layers.Dense(
        1, 
        activation=None, 
        use_bias=True, 
        kernel_initializer=keras.initializers.HeNormal(),
        name="chess_eval"
    )(x)

    model = Model(
        inputs=[active_input, passive_input],
        outputs=output
    )
        
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

    # 2. Bind directly to your engineered AdamW
    base_cosine_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4,
        decay_steps=(TOTAL_EPOCHS - 1) * STEPS_PER_EPOCH,
        alpha=0.0167
    )
    
    lr_schedule = CustomWarmupCosineSchedule(
        warmup_steps=STEPS_PER_EPOCH, 
        cosine_schedule=base_cosine_schedule
    )

    stockfish_optimizer = keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01,
        epsilon=1e-8,
        global_clipnorm=1.0
    )
    # --- Compile the model ---
    model.compile(
        optimizer=stockfish_optimizer,
        loss=pawn_probability_mse_loss,
        metrics=[close_position_error_all, close_position_error_3]
    )

    # Track file storage roots for local Parquet files
    train_dir, val_dir = get_local_shard_directories()

    # Create the permanent managers ONCE. They spawn background processes that live forever.
    train_manager = EpochSynchronizedDatasetManager(
        shard_directory=train_dir, shard_pattern="*.parquet", num_workers=4, 
        queue_size=5000, batch_size=BATCH_SIZE
    )
    val_manager = EpochSynchronizedDatasetManager(
        shard_directory=val_dir, shard_pattern="*.parquet", num_workers=1,
        queue_size=5000, batch_size=BATCH_SIZE
    )

    # --- Train Dataset (Updated to accept dense list tokens signatures) ---
    train_dataset = tf.data.Dataset.from_generator(
        generator=lambda: train_manager.generator_fn(total_epoch_steps=STEPS_PER_EPOCH),
        output_signature=(
            {
                "active_features": tf.TensorSpec(shape=(BATCH_SIZE, MAX_PIECES), dtype=tf.int32),
                "passive_features": tf.TensorSpec(shape=(BATCH_SIZE, MAX_PIECES), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    # --- Validation Dataset ---
    val_dataset = tf.data.Dataset.from_generator(
        generator=lambda: val_manager.generator_fn(total_epoch_steps=VALIDATION_PER_EPOCH),
        output_signature=(
            {
                "active_features": tf.TensorSpec(shape=(BATCH_SIZE, MAX_PIECES), dtype=tf.int32),
                "passive_features": tf.TensorSpec(shape=(BATCH_SIZE, MAX_PIECES), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    print("\n--- Model compilation complete. Commencing Training Step ---")
    checkpoint_path = "best_chess_nnue.keras"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=False,
        verbose=1
    )

    cleanup_cb = AggressiveMemoryCleanup()

    # Train model execution call
    model.fit(
        train_dataset, 
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=TOTAL_EPOCHS, 
        validation_data=val_dataset,
        validation_steps=VALIDATION_PER_EPOCH,
        callbacks=[checkpoint_cb, cleanup_cb]
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