import os
import glob
import time
import queue
import numpy as np
import pandas as pd
import multiprocessing as mp

def _worker_loop(command_queue, data_queue, shutdown_event, padding_index_value, chunk_size, max_safe_buffer_size=32):
    """
    Persistent CPU worker routine that handles streaming chess data across multiple epochs.
    Blocks and waits for the master thread to drop fresh file pools into command_queue,
    with built-in mid-stream hot-swap checks to ensure seamless epoch boundaries.
    """
    # Unique high-entropy process-level randomization seeding
    np.random.seed(int(time.time() * 1000) % 2**32 ^ mp.current_process().pid)
    
    while not shutdown_event.is_set():
        try:
            # Block and wait for the master thread to drop this worker's file pool for the new epoch
            worker_files = command_queue.get(timeout=1.0)
            if worker_files is None:  # Sentinel shutdown signal
                return
        except queue.Empty:
            continue

        # Execute high-entropy shuffling logic over the assigned workload
        shuffled_files = list(worker_files)
        np.random.shuffle(shuffled_files)
        total_worker_files = len(shuffled_files)
        if total_worker_files == 0:
            continue
            
        # Dynamically compute buffer ceiling using your M4 Mac memory-tuning algorithm
        dynamic_mixing_buffer_ceiling = max_safe_buffer_size
        while dynamic_mixing_buffer_ceiling > 1:
            if total_worker_files % dynamic_mixing_buffer_ceiling == 0:
                break
            dynamic_mixing_buffer_ceiling -= 1
        if dynamic_mixing_buffer_ceiling == 1:
            dynamic_mixing_buffer_ceiling = min(max_safe_buffer_size, total_worker_files)
            
        mixed_active, mixed_passive, mixed_targets = [], [], []
        accumulated_files = 0
        hot_swap_triggered = False
        
        for file_path in shuffled_files:
            if shutdown_event.is_set():
                return
                
            # Intercept: Check if the master thread sent a hot-swap pool mid-epoch
            if not command_queue.empty():
                hot_swap_triggered = True
                break
                
            try:
                # Load columns explicitly to maximize decompression speed
                df = pd.read_parquet(file_path, columns=['active_indices', 'passive_indices', 'target'], engine='pyarrow')
                if len(df) == 0: 
                    continue
                
                a_matrix = np.stack(df['active_indices'].to_numpy()).astype(np.int16)
                p_matrix = np.stack(df['passive_indices'].to_numpy()).astype(np.int16)
                targets = np.stack(df['target'].to_numpy()).astype(np.float32).reshape(-1, 1)
                del df
                
                # Apply structural pad values
                a_matrix[a_matrix == -1] = np.int16(padding_index_value)
                p_matrix[p_matrix == -1] = np.int16(padding_index_value)
                
                mixed_active.append(a_matrix)
                mixed_passive.append(p_matrix)
                mixed_targets.append(targets)
                accumulated_files += 1
                
                if accumulated_files >= dynamic_mixing_buffer_ceiling:
                    flat_active = np.concatenate(mixed_active, axis=0)
                    flat_passive = np.concatenate(mixed_passive, axis=0)
                    flat_targets = np.concatenate(mixed_targets, axis=0)
                    
                    mixed_active, mixed_passive, mixed_targets = [], [], []
                    accumulated_files = 0
                    
                    perm = np.random.permutation(len(flat_targets))
                    flat_active = flat_active[perm]
                    flat_passive = flat_passive[perm]
                    flat_targets = flat_targets[perm]
                    
                    num_full_chunks = len(flat_targets) // chunk_size
                    total_usable_rows = num_full_chunks * chunk_size
                    
                    for i in range(0, total_usable_rows, chunk_size):
                        if shutdown_event.is_set(): 
                            return
                            
                        # Double-check command queue inside the chunk distribution loop
                        if not command_queue.empty():
                            hot_swap_triggered = True
                            break
                            
                        chunk_item = (
                            {'active_features': flat_active[i:i+chunk_size], 'passive_features': flat_passive[i:i+chunk_size]},
                            flat_targets[i:i+chunk_size]
                        )
                        
                        while not shutdown_event.is_set():
                            # Break processing immediately if a hot-swap command drops
                            if not command_queue.empty():
                                hot_swap_triggered = True
                                break
                            try:
                                data_queue.put(chunk_item, timeout=0.1)
                                break
                            except queue.Full: 
                                continue
                            except (ValueError, OSError): 
                                return
                                
                        if hot_swap_triggered:
                            break
                            
                    if hot_swap_triggered:
                        del flat_active, flat_passive, flat_targets
                        break
                        
                    if total_usable_rows < len(flat_targets):
                        mixed_active = [flat_active[total_usable_rows:]]
                        mixed_passive = [flat_passive[total_usable_rows:]]
                        mixed_targets = [flat_targets[total_usable_rows:]]
                        accumulated_files = 1
                    del flat_active, flat_passive, flat_targets
                    
            except Exception:
                continue
                
        # Skip remainder flush logic entirely if we are abandoning data for a hot-swap
        if hot_swap_triggered:
            del mixed_active, mixed_passive, mixed_targets
            continue
            
        # Remainder padding loop
        if accumulated_files > 0 and not shutdown_event.is_set():
            if not command_queue.empty():
                del mixed_active, mixed_passive, mixed_targets
                continue
                
            flat_active = np.concatenate(mixed_active, axis=0)
            flat_passive = np.concatenate(mixed_passive, axis=0)
            flat_targets = np.concatenate(mixed_targets, axis=0)
            mixed_active, mixed_passive, mixed_targets = [], [], []
            
            perm = np.random.permutation(len(flat_targets))
            flat_active = flat_active[perm]
            flat_passive = flat_passive[perm]
            flat_targets = flat_targets[perm]
            
            if len(flat_targets) < chunk_size:
                pad_len = chunk_size - len(flat_targets)
                flat_active = np.pad(flat_active, ((0, pad_len), (0, 0)), constant_values=padding_index_value)
                flat_passive = np.pad(flat_passive, ((0, pad_len), (0, 0)), constant_values=padding_index_value)
                flat_targets = np.pad(flat_targets, ((0, pad_len), (0, 0)), constant_values=0.0)
                
            for i in range(0, len(flat_targets) - chunk_size + 1, chunk_size):
                if shutdown_event.is_set(): 
                    return
                if not command_queue.empty():
                    break
                    
                chunk_item = (
                    {'active_features': flat_active[i:i+chunk_size], 'passive_features': flat_passive[i:i+chunk_size]},
                    flat_targets[i:i+chunk_size]
                )
                
                while not shutdown_event.is_set():
                    if not command_queue.empty():
                        break
                    try:
                        data_queue.put(chunk_item, timeout=0.1)
                        break
                    except queue.Full: 
                        continue
                    except (ValueError, OSError): 
                        return
            del flat_active, flat_passive, flat_targets

    return

class EpochSynchronizedDatasetManager:
    def __init__(self, shard_directory, shard_pattern="*.parquet", num_workers=4, queue_size=50, batch_size=8192, name="training"):
        self.shard_directory = shard_directory
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.name = name
        self.PADDING_INDEX_VALUE = 10 * 11 * 64 + 4
        
        self.all_files = glob.glob(os.path.join(shard_directory, shard_pattern))
        if not self.all_files:
            raise FileNotFoundError(f"No shards found in: {shard_directory}")
            
        print(f"[DatasetManager] Discovered {len(self.all_files)} total shards.")
        
        # Instantiate execution primitives EXACTLY ONCE at script initialization
        self.shutdown_event = mp.Event()
        
        # NEW: Thread-safe fence that halts the generator on epoch step ceilings 
        # to freeze the TensorFlow prefetch engine during background hot-swaps.
        self.epoch_ready_event = mp.Event()
        self.epoch_ready_event.set()  # Prime it so Epoch 1 starts streaming immediately
        
        self.queue = mp.Queue(maxsize=self.queue_size)
        self.queue.cancel_join_thread()
        
        # Create persistent command queues to feed the background processes
        self.worker_command_queues = [mp.Queue() for _ in range(num_workers)]
        self.workers = []
        
        # Spawn daemon worker forks immediately
        for i in range(self.num_workers):
            p = mp.Process(
                target=_worker_loop,
                args=(
                    self.worker_command_queues[i], 
                    self.queue, 
                    self.shutdown_event, 
                    self.PADDING_INDEX_VALUE, 
                    self.batch_size
                ),
                daemon=True
            )
            p.start()
            self.workers.append(p)

    def generator_fn(self, total_epoch_steps):
        """
        Continuous stream that satisfies Keras 3 across all epochs, using an
        Event lock to completely halt prefetching during file hot-swaps.
        """
        steps_yielded = 0
        
        while not self.shutdown_event.is_set():
            # CRITICAL SAFETY VALVE:
            # If we hit the exact epoch step boundary, lock the gate and wait for the callback.
            if steps_yielded % total_epoch_steps == 0 and steps_yielded > 0:
                self.epoch_ready_event.clear()  # Drop the gate to False (Red Light)
                print(f"\n[Generator] Step {steps_yielded:,} hit. Halting prefetcher for hot-swap...")
                
                # Block here. TensorFlow's prefetcher freezes right here and cannot bleed.
                # .wait() blocks at the OS level efficiently without spinning the CPU core.
                while not self.epoch_ready_event.is_set():
                    if self.shutdown_event.is_set():
                        return
                    self.epoch_ready_event.wait(timeout=0.1)
            
            try:
                # Pull raw data chunk (batch_size=8192) dropped by background worker processes
                chunk_item = self.queue.get(timeout=0.5)
                yield chunk_item
                steps_yielded += 1
            except queue.Empty:
                continue
            except (ValueError, OSError):
                # Data queue pipeline dropped or torn down by master process on exit
                return

    def start_epoch_stream(self, explicit_file_pool=None):
        # 1. Drain data queue completely to prevent epoch cross-contamination
        while not self.queue.empty():
            try: self.queue.get_nowait()
            except Exception: break
                
        # 2. Gather and slice the new shuffled file deck
        file_pool = explicit_file_pool if explicit_file_pool is not None else self.all_files
        worker_file_map = self.split_files_to_worker_chunks(file_pool) 
        
        # 3. Purge command queues and inject fresh files
        for i in range(self.num_workers):
            while not self.worker_command_queues[i].empty():
                try: self.worker_command_queues[i].get_nowait()
                except Exception: break
            self.worker_command_queues[i].put(list(worker_file_map[i]))
            
        # 4. RELEASE THE LOCK: Allow the generator to start building the next epoch
        print(f"[DatasetManager] {self.name} hot-swap complete. Releasing prefetch lock.")
        self.epoch_ready_event.set()
        print(f"[DatasetManager] {self.name} worker streams successfully hot-swapped for the new epoch loop.")

    def stop_epoch_stream(self):
        """ Clean termination of background daemons on script finalization. """
        self.shutdown_event.set()
        
        # NEW: Always unblock a waiting generator loop if it is currently locked 
        # at an epoch wall, allowing the master thread loop to unwind cleanly.
        try:
            self.epoch_ready_event.set()
        except Exception:
            pass

        for i in range(self.num_workers):
            try:
                self.worker_command_queues[i].put(None)
            except Exception: pass
            
        for p in self.workers:
            try:
                p.terminate()
            except Exception: pass

    def split_files_to_worker_chunks(self, file_pool=None):
        """
        Splits the provided globally shuffled file pool evenly across the active 
        worker count to ensure unique, non-overlapping workloads per process fork.
        """
        # 1. Fallback cleanly to master list if no dynamic pool is supplied
        files_to_slice = list(file_pool) if file_pool is not None else list(self.all_files)
        
        # Initialize a clean dictionary mapping worker index to its file list slice
        worker_file_map = {i: [] for i in range(self.num_workers)}
        
        # 2. Distribute files uniformly using a strided round-robin sequence
        # This acts as an extra layer of structural distribution across your 6 CPU processes
        for idx, file_path in enumerate(files_to_slice):
            worker_id = idx % self.num_workers
            worker_file_map[worker_id].append(file_path)
            
        # Optional validation telemetry log
        for i in range(self.num_workers):
            if len(worker_file_map[i]) == 0:
                print(f"[DatasetManager][WARNING] Worker {i} received 0 files to process!")
                
        return worker_file_map