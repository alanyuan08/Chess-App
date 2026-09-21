import os
import glob
import math
import time
import queue
import numpy as np
import pandas as pd
import multiprocessing as mp
import numpy as np
import pandas as pd
import time
import multiprocessing as mp
import queue

def _worker_loop(file_list, data_queue, shutdown_event, padding_index_value, chunk_size):
    # Unique stochastic seed generation per independent background process
    np.random.seed(int(time.time() * 1000) % 2**32 ^ mp.current_process().pid)
    
    shuffled_files = list(file_list)
    
    while not shutdown_event.is_set():
        # Scramble the macro file list path at the start of every iteration loop pass
        np.random.shuffle(shuffled_files)
        
        for file_path in shuffled_files:
            if shutdown_event.is_set():
                break
            try:
                # Explicitly load only what we need to minimize L3 cache pressure
                df = pd.read_parquet(file_path, columns=['active_indices', 'passive_indices', 'target'])
                total_rows = len(df)
                if total_rows < chunk_size:
                    continue
                
                # High-speed continuous block extraction 
                a_matrix = np.stack(df['active_indices'].to_numpy()).astype(np.int32)
                p_matrix = np.stack(df['passive_indices'].to_numpy()).astype(np.int32)
                targets = np.stack(df['target'].to_numpy()).astype(np.float32).reshape(-1, 1)
                
                # Clear raw dataframe from RAM instantly
                del df 
                
                # Apply padding arrays uniformly across un-set features
                a_matrix[a_matrix == -1] = padding_index_value
                p_matrix[p_matrix == -1] = padding_index_value
                
                # --- FIXED: HIGH-SPEED MICRO-BATCH ROW SCRAMBLER ---
                # Scramble the internal rows of this specific file chunk right inside the worker.
                # This breaks any lingering game-continuity or bucket-bias before queue submission.
                permutation = np.random.permutation(total_rows)
                a_matrix = a_matrix[permutation]
                p_matrix = p_matrix[permutation]
                targets = targets[permutation]
                
                # Calculate clean bounds matching our exact 8192 tensor size
                full_batch_cutoff = total_rows - (total_rows % chunk_size)
                
                for i in range(0, full_batch_cutoff, chunk_size):
                    if shutdown_event.is_set():
                        break
                    
                    chunk_item = (
                        {
                            'active_features': a_matrix[i:i+chunk_size],
                            'passive_features': p_matrix[i:i+chunk_size],
                        },
                        targets[i:i+chunk_size]
                    )
                    
                    # Yield perfectly scrambled block directly to the main consumer thread
                    while not shutdown_event.is_set():
                        try:
                            data_queue.put(chunk_item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                            
                del a_matrix, p_matrix, targets
                            
            except Exception as e:
                print(f"\n[Dataset Worker Error] Failed to process {file_path}: {e}")
                continue
                
        # End of file list pass: send sentinel if your consumer loop handles single-pass epochs,
        # or remove this block if your main training script explicitly manages epoch loops.
        while not shutdown_event.is_set():
            try:
                data_queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue

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
        self.manager = mp.Manager()
        self.shutdown_event = mp.Event()
        self.workers = []
        
    def _spawn_workers(self, data_queue):
        """Internal helper to split file matrices and start workers for the current epoch."""
        self.shutdown_event.clear()
        self.workers = []
        
        # Fresh global file shuffle before splitting work among background workers
        shuffled_all_files = list(self.all_files)
        np.random.shuffle(shuffled_all_files)
        
        files_per_worker = math.ceil(len(shuffled_all_files) / float(self.num_workers))
        
        for i in range(self.num_workers):
            start_idx = i * files_per_worker
            end_idx = min(start_idx + files_per_worker, len(shuffled_all_files))
            worker_files = shuffled_all_files[start_idx:end_idx]
            
            if not worker_files:
                continue
                
            process = mp.Process(
                target=_worker_loop,
                args=(worker_files, data_queue, self.shutdown_event, self.PADDING_INDEX_VALUE, self.batch_size),
                daemon=True 
            )
            self.workers.append(process)
            process.start()

    def _cleanup_workers(self, data_queue):
        """Gracefully terminates and joins active worker processes at the epoch boundary."""
        self.shutdown_event.set()
        
        # Fast queue flush to unblock workers stuck on data_queue.put()
        try:
            while not data_queue.empty():
                data_queue.get_nowait()
        except:
            Box = None
                
        for process in self.workers:
            if process.is_alive():
                # Give it a tiny moment to register the shutdown event
                process.join(timeout=0.2)
                # Forcefully terminate to ensure it drops the proxy locks immediately
                process.terminate()
                process.join()
        
        self.workers = []

    def generator_fn(self, total_epoch_steps):
        """
        Keras-compliant Generator function.
        Guarantees that each epoch cycle triggers a fresh worker pool creation,
        yielding clean, newly randomized data blocks matching your target steps perfectly.
        """
        while True:
            # Using standard mp.Queue instead of self.manager.Queue completely 
            # bypasses the SyncManager proxy dictionary KeyError trap!
            epoch_queue = mp.Queue(maxsize=self.queue_size)
            self._spawn_workers(epoch_queue)
            
            active_sentinels_expected = len(self.workers)
            steps_yielded = 0
            
            # Use an explicit sentinel tracking loop combined with step constraints
            while active_sentinels_expected > 0:
                try:
                    batch = epoch_queue.get(timeout=30.0)
                    
                    if batch is None:
                        active_sentinels_expected -= 1
                        continue
                        
                    # Only yield if we haven't exceeded Keras's requested steps
                    if steps_yielded < total_epoch_steps:
                        yield batch
                        steps_yielded += 1
                    else:
                        # Keras got all its data, but we keep looping quietly 
                        # to let workers deliver their sentinels naturally!
                        pass
                        
                except queue.Empty:
                    print("\n[Pipeline Alert] Data queue starved! Forcing early epoch fallback transition.")
                    break
                
            # Clean up processes right at the epoch boundary boundary before looping
            self._cleanup_workers(epoch_queue)
            print("\n=== [Epoch Complete] Resetting infrastructure, re-shuffling universe shards... ===")
