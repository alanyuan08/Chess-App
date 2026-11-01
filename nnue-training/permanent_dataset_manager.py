import os
import glob
import time
import queue
import numpy as np
import pandas as pd
import multiprocessing as mp

def _worker_loop(file_list, data_queue, shutdown_event, padding_index_value, chunk_size, files_per_mixing_buffer=8):
    # Dynamic re-seeding to prevent cloned worker outputs across split fork processes
    np.random.seed(int(time.time() * 1000) % 2**32 ^ mp.current_process().pid)
    
    # Outer infinite loop ensures workers never run dry before Keras finishes the epoch
    while not shutdown_event.is_set():
        shuffled_files = list(file_list)
        np.random.shuffle(shuffled_files)
        
        mixed_active, mixed_passive, mixed_targets = [], [], []
        accumulated_files = 0
        
        for file_path in shuffled_files:
            if shutdown_event.is_set():
                return
                
            try:
                df = pd.read_parquet(
                    file_path, 
                    columns=['active_indices', 'passive_indices', 'target'], 
                    engine='pyarrow', 
                    use_threads=True
                )
                if len(df) == 0:
                    continue
                
                a_matrix = np.stack(df['active_indices'].to_numpy()).astype(np.int16)
                p_matrix = np.stack(df['passive_indices'].to_numpy()).astype(np.int16)
                targets = np.stack(df['target'].to_numpy()).astype(np.float32).reshape(-1, 1)
                del df
                
                a_matrix[a_matrix == -1] = np.int16(padding_index_value)
                p_matrix[p_matrix == -1] = np.int16(padding_index_value)
                
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
                        
                        # Graph-safe queue ingestion with responsive shutdown checks
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
                
        # Handle remainder elements cleanly before evaluating loop wrap
        if accumulated_files > 0 and not shutdown_event.is_set():
            flat_active = np.concatenate(mixed_active, axis=0)
            flat_passive = np.concatenate(mixed_passive, axis=0)
            flat_targets = np.concatenate(mixed_targets, axis=0)
            perm = np.random.permutation(len(flat_targets))
            for i in range(0, len(flat_targets), chunk_size):
                if shutdown_event.is_set(): 
                    return
                chunk_item = (
                    {
                        'active_features': flat_active[perm[i:i+chunk_size]].copy(), 
                        'passive_features': flat_passive[perm[i:i+chunk_size]].copy()
                    }, 
                    flat_targets[perm[i:i+chunk_size]].copy()
                )
                while not shutdown_event.is_set():
                    try:
                        data_queue.put(chunk_item, timeout=0.1)
                        break
                    except queue.Full: 
                        continue
            del flat_active, flat_passive, flat_targets

    # Graceful exit sequence: drop out cleanly if shutdown event is tripped
    return

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
        """Hard reset: Destroys previous structures cleanly to prevent deadlocks and leaks."""
    
        self.shutdown_event = mp.Event()
        self.queue = mp.Queue(maxsize=self.queue_size)
        
        # Prevent the queue background feeder threads from permanently locking up the master process
        self.queue.cancel_join_thread()
        
        shuffled_master = list(self.all_files)
        np.random.shuffle(shuffled_master)
        worker_file_slices = np.array_split(shuffled_master, self.num_workers)
        
        self.workers = []
        for i in range(self.num_workers):
            if len(worker_file_slices[i]) == 0: 
                continue
            p = mp.Process(
                target=_worker_loop,
                args=(list(worker_file_slices[i]), self.queue, self.shutdown_event, self.PADDING_INDEX_VALUE, self.batch_size),
                daemon=True
            )
            p.start()
            self.workers.append(p)
        print(f"[DatasetManager] Workers deployed fresh for upcoming epoch.")

    def generator_fn(self, total_epoch_steps):
        """Yields tensors safely, handling asynchronous queue drops gracefully."""
        try:
            steps_yielded = 0
            while steps_yielded < total_epoch_steps:
                if self.queue is None or (self.shutdown_event and self.shutdown_event.is_set()):
                    return
                
                try:
                    item = self.queue.get(timeout=1.0)
                except queue.Empty:
                    if self.shutdown_event and self.shutdown_event.is_set():
                        return
                    continue

                if item is None:
                    continue
                
                yield item
                steps_yielded += 1 
                
        except (AttributeError, TypeError):
            return
        finally:
            pass

    def stop_epoch_stream(self):
        """Force kills workers and clears the multiprocessing queue framework cleanly."""
        # 1. Trigger shutdown event immediately so workers drop out of their file-reading routines
        if self.shutdown_event:
            self.shutdown_event.set()
            
        # 2. Break down the queue buffers cleanly to drop memory references instantly
        if self.queue is not None:
            try:
                # Direct the underlying OS thread pool not to attempt flushing remaining bytes
                self.queue.cancel_join_thread()
                
                # Drain queue immediately on the main thread to unblock any workers stuck on .put()
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        break
                self.queue.close()
            except Exception as e:
                print(f"[DatasetManager Warning] Non-fatal queue disposal friction: {e}")
            self.queue = None

        # 3. Cleanly harvest background threads with fixed deadlines to avoid zombie states
        if self.workers:
            for p in self.workers:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=0.2)
            self.workers = []
            
        self.shutdown_event = None