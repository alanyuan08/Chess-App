import os
import glob
import time
import queue
import numpy as np
import pandas as pd
import multiprocessing as mp

def _worker_loop(file_list, data_queue, shutdown_event, padding_index_value, chunk_size, files_per_mixing_buffer=8):
    # Dynamic re-seeding to prevent cloned worker outputs
    np.random.seed(int(time.time() * 1000) % 2**32 ^ mp.current_process().pid)
    shuffled_files = list(file_list)
    np.random.shuffle(shuffled_files)
    
    mixed_active, mixed_passive, mixed_targets = [], [], []
    accumulated_files = 0
    
    for file_path in shuffled_files:
        if shutdown_event.is_set():
            return
            
        try:
            df = pd.read_parquet(file_path, columns=['active_indices', 'passive_indices', 'target'])
            if len(df) == 0:
                continue
            
            a_matrix = np.stack(df['active_indices'].to_numpy()).astype(np.int16)
            p_matrix = np.stack(df['passive_indices'].to_numpy()).astype(np.int16)
            targets = np.stack(df['target'].to_numpy()).astype(np.float32).reshape(-1, 1)
            del df
            
            a_matrix[a_matrix == -1] = padding_index_value
            p_matrix[p_matrix == -1] = padding_index_value
            
            mixed_active.append(a_matrix)
            mixed_passive.append(p_matrix)
            mixed_targets.append(targets)
            accumulated_files += 1
            
            # Increased buffer size to ensure deeper variance mixing across position histories
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
            
    # Handle remainder elements cleanly
    if accumulated_files > 0 and not shutdown_event.is_set():
        flat_active = np.concatenate(mixed_active, axis=0)
        flat_passive = np.concatenate(mixed_passive, axis=0)
        flat_targets = np.concatenate(mixed_targets, axis=0)
        perm = np.random.permutation(len(flat_targets))
        for i in range(0, len(flat_targets), chunk_size):
            if shutdown_event.is_set(): return
            chunk_item = ({'active_features': flat_active[perm[i:i+chunk_size]], 'passive_features': flat_passive[perm[i:i+chunk_size]]}, flat_targets[perm[i:i+chunk_size]])
            while not shutdown_event.is_set():
                try:
                    data_queue.put(chunk_item, timeout=0.1)
                    break
                except queue.Full: continue

    # Inform the manager this worker finished its pass natively
    while not shutdown_event.is_set():
        try:
            data_queue.put(None, timeout=0.1)
            break
        except queue.Full:
            continue

class EpochSynchronizedDatasetManager:
    def __init__(self, shard_directory, shard_pattern="*.parquet", num_workers=4, queue_size=50, batch_size=8192):
        self.shard_directory = shard_directory
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.PADDING_INDEX_VALUE = 10 * 11 * 64
        
        self.all_files = glob.glob(os.path.join(shard_directory, shard_pattern))
        if not self.all_files:
            raise FileNotFoundError(f"No shards found in: {shard_directory}")
            
        print(f"[DatasetManager] Discovered {len(self.all_files)} total shards.")
        
        self.shutdown_event = None
        self.workers = []
        self.queue = None

    def start_epoch_stream(self):
        """Hard reset: Destroys previous structures to completely prevent memory leaks."""
        self.stop_epoch_stream()
        
        self.shutdown_event = mp.Event()
        self.queue = mp.Queue(maxsize=self.queue_size)
        
        # Globally shuffle your primary file list before mapping slices to background processes
        shuffled_master = list(self.all_files)
        np.random.shuffle(shuffled_master)
        worker_file_slices = np.array_split(shuffled_master, self.num_workers)
        
        self.workers = []
        for i in range(self.num_workers):
            if len(worker_file_slices[i]) == 0: continue
            p = mp.Process(
                target=_worker_loop,
                args=(list(worker_file_slices[i]), self.queue, self.shutdown_event, self.PADDING_INDEX_VALUE, self.batch_size),
                daemon=True
            )
            p.start()
            self.workers.append(p)
        print(f"[DatasetManager] Workers deployed fresh for upcoming epoch.")

    def generator_fn(self, total_epoch_steps):
        """Yields tensors safely. Cleans itself up natively if Keras exceptions arise."""
        try:
            for _ in range(total_epoch_steps):
                item = self.queue.get()
                if item is None:  # Worker reached sentinel
                    continue
                yield item
        finally:
            self.stop_epoch_stream()

    def stop_epoch_stream(self):
        """Force kills workers and clears the multiprocessing queue framework cleanly."""
        if self.shutdown_event:
            self.shutdown_event.set()
            
        # Hard-terminate processes to clear hidden pre-fetch queues instantly
        for p in self.workers:
            if p.is_alive():
                p.terminate()
                p.join()
        
        if self.queue:
            # Fully purge references to clear underlying handles
            self.queue.close()
            self.queue.join_thread()
            
        self.workers = []
        self.shutdown_event = None
        self.queue = None