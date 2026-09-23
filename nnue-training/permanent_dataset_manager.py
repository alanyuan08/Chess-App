import os
import glob
import math
import time
import queue
import numpy as np
import pandas as pd
import multiprocessing as mp

def _worker_loop(file_list, data_queue, shutdown_event, padding_index_value, chunk_size, files_per_mixing_buffer=4):
    np.random.seed(int(time.time() * 1000) % 2**32 ^ mp.current_process().pid)
    shuffled_files = list(file_list)
    
    # Shuffle once per epoch/worker lifespan
    np.random.shuffle(shuffled_files)
    
    mixed_active = []
    mixed_passive = []
    mixed_targets = []
    accumulated_files = 0
    
    for file_path in shuffled_files:
        if shutdown_event.is_set():
            return  # Exit immediately if main process requested shutdown
            
        try:
            df = pd.read_parquet(file_path, columns=['active_indices', 'passive_indices', 'target'])
            if len(df) == 0:
                continue
            
            a_matrix = np.stack(df['active_indices'].to_numpy()).astype(np.int32)
            p_matrix = np.stack(df['passive_indices'].to_numpy()).astype(np.int32)
            targets = np.stack(df['target'].to_numpy()).astype(np.float32).reshape(-1, 1)
            del df
            
            a_matrix[a_matrix == -1] = padding_index_value
            p_matrix[p_matrix == -1] = padding_index_value
            
            mixed_active.append(a_matrix)
            mixed_passive.append(p_matrix)
            mixed_targets.append(targets)
            accumulated_files += 1
            
            if accumulated_files >= files_per_mixing_buffer:
                flat_active = np.concatenate(mixed_active, axis=0)
                flat_passive = np.concatenate(mixed_passive, axis=0)
                flat_targets = np.concatenate(mixed_targets, axis=0)
                
                mixed_active, mixed_passive, mixed_targets = [], [], []
                
                perm = np.random.permutation(len(flat_targets))
                flat_active = flat_active[perm]
                flat_passive = flat_passive[perm]
                flat_targets = flat_targets[perm]
                
                for i in range(0, len(flat_targets), chunk_size):
                    if shutdown_event.is_set():
                        return
                    
                    chunk_item = (
                        {
                            'active_features': flat_active[i:i+chunk_size].copy(),
                            'passive_features': flat_passive[i:i+chunk_size].copy(),
                        },
                        flat_targets[i:i+chunk_size].copy()
                    )
                    
                    while not shutdown_event.is_set():
                        try:
                            data_queue.put(chunk_item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                
                del flat_active, flat_passive, flat_targets
                accumulated_files = 0
                
        except Exception as e:
            print(f"\n[Dataset Worker Error] Failed to process {file_path}: {e}")
            continue
            
    # Flush block: handles trailing files
    if accumulated_files > 0 and not shutdown_event.is_set():
        flat_active = np.concatenate(mixed_active, axis=0)
        flat_passive = np.concatenate(mixed_passive, axis=0)
        flat_targets = np.concatenate(mixed_targets, axis=0)
        
        mixed_active, mixed_passive, mixed_targets = [], [], []
        
        perm = np.random.permutation(len(flat_targets))
        flat_active = flat_active[perm]
        flat_passive = flat_passive[perm]
        flat_targets = flat_targets[perm]
        
        for i in range(0, len(flat_targets), chunk_size):
            if shutdown_event.is_set():
                return
            chunk_item = (
                {
                    'active_features': flat_active[i:i+chunk_size].copy(),
                    'passive_features': flat_passive[i:i+chunk_size].copy(),
                },
                flat_targets[i:i+chunk_size].copy()
            )
            while not shutdown_event.is_set():
                try:
                    data_queue.put(chunk_item, timeout=0.1)
                    break
                except queue.Full:
                    continue
        del flat_active, flat_passive, flat_targets

    # Single pass completed safely: push sentinel
    while not shutdown_event.is_set():
        try:
            data_queue.put(None, timeout=0.1)
            break
        except queue.Full:
            continue
            
    # Process naturally ends here, allowing the main script to join it cleanly.

class EpochSynchronizedDatasetManager:
    """
    High-performance multiprocessing data stream manager built specifically for Keras/TF.
    Spawns background processes dynamically at epoch boundaries to guarantee a fresh, 
    fully-randomized universe pass with zero residual cross-contamination.
    """
    def __init__(self, shard_directory, shard_pattern="*.parquet", num_workers=4, queue_size=100, batch_size=8192):
        self.shard_directory = shard_directory
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.batch_size = batch_size
        
        # Pull file paths directly matching your 99/1 split outputs
        self.all_files = glob.glob(os.path.join(shard_directory, shard_pattern))
        if not self.all_files:
            raise FileNotFoundError(f"No parquet shards discovered matching '{shard_pattern}' in: {shard_directory}")
            
        print(f"[DatasetManager] Initialized for {shard_directory} with {len(self.all_files)} shards.")
        
        # Dual-perspective HalfKA padding configuration constants (49,152 mapping index)
        self.PADDING_INDEX_VALUE = 64 * 64 * 12 
        
        # Core IPC state trackers
        self.shutdown_event = mp.Event()
        self.workers = []
        
    def _spawn_workers(self, epoch_queue, worker_file_slices):
        """
        Spawns the worker pool using the dynamic chunks calculated for the current epoch.
        """
        self.workers = []
        for i in range(self.num_workers):
            if not worker_file_slices[i]:
                continue
                
            p = mp.Process(
                target=_worker_loop,
                args=(
                    list(worker_file_slices[i]),
                    epoch_queue,
                    self.shutdown_event,
                    self.PADDING_INDEX_VALUE,
                    self.batch_size 
                ),
                daemon=True
            )
            p.start()
            self.workers.append(p)

    def shutdown(self):
        """Public interface to cleanly terminate background workers."""
        # Find the queue instance dynamically if it's attached to self, 
        # otherwise default to None or pass your queue instance variable.
        data_queue = getattr(self, 'data_queue', None)
        
        # If the queue isn't stored as self.data_queue, update the line above 
        # to match whatever your queue variable is named (e.g., self.queue)
        
        print("[DatasetManager] Initiating clean worker shutdown...")
        self._cleanup_workers(data_queue)

    def _cleanup_workers(self, data_queue):
        """Gracefully terminates and joins active worker processes at the epoch boundary."""
        # 1. Signal all workers to stop immediately
        self.shutdown_event.set()
        
        # 2. Continuous non-blocking drain loop to release workers blocked on queue puts
        # We try draining multiple times to catch delayed elements in the pipe
        for _ in range(5):
            while True:
                try:
                    data_queue.get_nowait()
                except queue.Empty:
                    break
                except:
                    break
            time.sleep(0.05) # Brief sleep to allow pipe serialization buffers to clear
                
        # 3. Cleanly join the processes
        for process in self.workers:
            if process.is_alive():
                process.join(timeout=0.5)
                if process.is_alive():
                    # If it refuses to die, kill it, but only after flushing resources
                    process.terminate()
                    process.join()
        
        self.workers = []

        # 4. Cleanly close the queue to prevent feeder thread deadlocks
        try:
            data_queue.close()
            data_queue.join_thread()
        except:
            pass

    def generator_fn(self, total_epoch_steps):
        """
        Keras-compliant Generator function.
        Guarantees that each epoch cycle triggers a fresh worker pool creation.
        """
        while True:
            epoch_queue = mp.Queue(maxsize=self.queue_size)
            
            # 1. FRESH GLOBAL FILE SHUFFLE BEFORE SPLITTING
            shuffled_all_files = list(self.all_files)
            np.random.shuffle(shuffled_all_files)
            
            # 2. CALCULATE AND GENERATE CHUNKS FOR THE SPAWNER
            files_per_worker = math.ceil(len(shuffled_all_files) / float(self.num_workers))
            worker_file_slices = []
            for i in range(self.num_workers):
                start_idx = i * files_per_worker
                end_idx = min(start_idx + files_per_worker, len(shuffled_all_files))
                worker_file_slices.append(shuffled_all_files[start_idx:end_idx])
                
            # 3. PASS CHUNKS TO MATCH THE 3-ARGUMENT CALL SIGNATURE PERFECTLY
            self.shutdown_event.clear()
            self._spawn_workers(epoch_queue, worker_file_slices)
            
            active_sentinels_expected = len(self.workers)
            steps_yielded = 0
            
            # Explicit sentinel tracking loop combined with step constraints
            while active_sentinels_expected > 0:
                try:
                    batch = epoch_queue.get(timeout=30.0)
                    
                    if batch is None:
                        active_sentinels_expected -= 1
                        continue
                        
                    # Yield data to Keras
                    yield batch
                    steps_yielded += 1
                    
                    # IMMEDIATE ESCAPE: Once Keras has all the steps it needs, break out!
                    if steps_yielded >= total_epoch_steps:
                        break
                        
                except queue.Empty:
                    print("\n[Pipeline Alert] Data queue starved! Forcing early epoch fallback transition.")
                    break
                
            # 4. ACTIVE DEADLOCK PREVENTION
            # Trigger the shutdown event so workers stop reading files and stop calling .put()
            self.shutdown_event.set()
            
            # Drain the queue completely. Workers blocked on .put() will be released 
            # so they can see the shutdown_event and exit.
            while not epoch_queue.empty():
                try:
                    epoch_queue.get_nowait()
                except queue.Empty:
                    break

            # Clean up processes right at the epoch boundary before looping
            self._cleanup_workers(epoch_queue)
            print("\n=== [Epoch Complete] Resetting infrastructure, re-shuffling universe shards... ===")
