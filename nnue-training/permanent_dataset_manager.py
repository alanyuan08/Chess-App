# permanent_dataset_manager.py
import os
import glob
import time
import queue
import math
import multiprocessing as mp
import numpy as np
import pandas as pd

def _worker_loop(file_list, data_queue, shutdown_event, padding_index_value):
    """
    Isolated background process worker loop.
    Sequentially loads Parquet shards, shuffles rows locally, and feeds the queue.
    """
    # Seed the random number generator uniquely per background worker process
    np.random.seed(int(time.time() * 1000) % 2**32 ^ mp.current_process().pid)
    
    while not shutdown_event.is_set():
        # Local copy to shuffle shard file visitation orders each pass
        shuffled_files = list(file_list)
        np.random.shuffle(shuffled_files)
        
        for file_path in shuffled_files:
            if shutdown_event.is_set():
                break
                
            try:
                # Load the compact parquet file directly into RAM
                df = pd.read_parquet(file_path)
                
                # Perform an in-memory row shuffle within this shard to maximize randomness
                df = df.sample(frac=1.0).reset_index(drop=True)
                
                for _, row in df.iterrows():
                    if shutdown_event.is_set():
                        break
                        
                    w_idx = np.array(row['white_indices'], dtype=np.int32)
                    b_idx = np.array(row['black_indices'], dtype=np.int32)
                    
                    # Convert negative padding markers (-1) into your safe positive embedding token
                    w_idx[w_idx == -1] = padding_index_value
                    b_idx[b_idx == -1] = padding_index_value
                    
                    item = (
                        {
                            "white_features": w_idx,
                            "black_features": b_idx,
                            "side_to_move": [bool(row['is_black_turn'])]
                        },
                        [float(row['target'])]
                    )
                    
                    # Block if the queue is full; continuously poll shutdown event status
                    while not shutdown_event.is_set():
                        try:
                            data_queue.put(item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                            
            except Exception as e:
                print(f"\n[Dataset Worker Error] Failed to read shard {file_path}: {e}")
                continue
                
    print(f"[Dataset Worker Process {mp.current_process().pid}] Background thread exited cleanly.")

class PermanentDatasetManager:
    """
    Multiprocessing data stream manager that leverages background processes 
    to concurrently read, parse, and pre-buffer Parquet training shards.
    """
    def __init__(self, shard_directory, shard_pattern="production_wave_*.parquet", num_workers=4, queue_size=50000):
        self.shard_directory = shard_directory
        self.num_workers = num_workers
        self.queue_size = queue_size
        
        # Determine the total list of exported shards matching your file pattern
        self.all_files = glob.glob(os.path.join(shard_directory, shard_pattern))
        if not self.all_files:
            raise FileNotFoundError(f"No parquet shards discovered matching pattern '{shard_pattern}' inside directory: {shard_directory}")
            
        print(f"[PermanentDatasetManager] Initializing stream across {len(self.all_files)} Parquet shards with {num_workers} background workers.")
        
        # Constants matching your dual-perspective HalfKA input embedding specifications
        self.PADDING_INDEX_VALUE = 64 * 64 * 12 
        
        # Inter-process communication infrastructure setup
        self.manager = mp.Manager()
        self.data_queue = self.manager.Queue(maxsize=self.queue_size)
        self.shutdown_event = mp.Event()
        self.workers = []
        
        # Distribute shards as evenly as possible across your available background process pool
        files_per_worker = math.ceil(len(self.all_files) / float(self.num_workers))
        
        for i in range(self.num_workers):
            start_idx = i * files_per_worker
            end_idx = min(start_idx + files_per_worker, len(self.all_files))
            worker_files = self.all_files[start_idx:end_idx]
            
            # Skip initialization loops if there are more processes configured than matching files
            if not worker_files:
                continue
                
            process = mp.Process(
                target=_worker_loop,
                args=(worker_files, self.data_queue, self.shutdown_event, self.PADDING_INDEX_VALUE),
                daemon=True # Ensures tasks get forcefully culled if parent script terminates abruptly
            )
            self.workers.append(process)
            process.start()
            
        print(f"[PermanentDatasetManager] Spawning complete. Pre-buffering background queue pool...")

    def generator_fn(self):
        """
        Yields structured position dictionaries directly into your tf.data.Dataset pipeline.
        """
        while not self.shutdown_event.is_set():
            try:
                # Fetch parsed indices from queue; do not block indefinitely to allow graceful shutdown handling
                yield self.data_queue.get(timeout=1.0)
            except (queue.Empty, AssertionError, KeyboardInterrupt):
                if self.shutdown_event.is_set():
                    break
                continue

    def shutdown(self):
        """
        Safely signals background processes to stop and completely drains outstanding elements.
        """
        print("\n[PermanentDatasetManager] Initiating clean background worker pool shutdown...")
        self.shutdown_event.set()
        
        # Aggressively empty queue content to prevent background worker lockups during joins
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except Exception:
                break
                
        # Join worker instances back to parent thread loops safely
        for process in self.workers:
            if process.is_alive():
                process.join(timeout=2.0)
                if process.is_alive():
                    process.terminate() # Fallback kill instruction if a thread remains stubborn
                    
        print("[PermanentDatasetManager] All background dataset processes successfully terminated.")
