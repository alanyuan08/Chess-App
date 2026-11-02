import os
import glob
import time
import queue
import numpy as np
import pandas as pd
import multiprocessing as mp

def _worker_loop(file_list, data_queue, shutdown_event, padding_index_value, chunk_size, files_per_mixing_buffer=20):
    """
    Parallel CPU worker routine that extracts Parquet shards, applies deep localized
    shuffling via expanded buffer pools, and pushes static chunks to the main queue.
    """
    # Dynamic salt re-seeding to guarantee unique permutations across process forks
    np.random.seed(int(time.time() * 1000) % 2**32 ^ mp.current_process().pid)
    
    while not shutdown_event.is_set():
        shuffled_files = list(file_list)
        np.random.shuffle(shuffled_files)
        
        mixed_active, mixed_passive, mixed_targets = [], [], []
        accumulated_files = 0
        
        for file_path in shuffled_files:
            if shutdown_event.is_set():
                return
                
            try:
                # Read specific columns via Arrow engine to protect unified memory bandwidth
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
                
                # Replace unallocated features with out-of-bounds padding token
                a_matrix[a_matrix == -1] = np.int16(padding_index_value)
                p_matrix[p_matrix == -1] = np.int16(padding_index_value)
                
                mixed_active.append(a_matrix)
                mixed_passive.append(p_matrix)
                mixed_targets.append(targets)
                accumulated_files += 1
                
                # Trigger the master block permutation when the 80-file limit is met
                if accumulated_files >= files_per_mixing_buffer:
                    flat_active = np.concatenate(mixed_active, axis=0)
                    flat_passive = np.concatenate(mixed_passive, axis=0)
                    flat_targets = np.concatenate(mixed_targets, axis=0)
                    
                    # Instantly release references to clear RAM overhead
                    mixed_active, mixed_passive, mixed_targets = [], [], []
                    accumulated_files = 0
                    
                    perm = np.random.permutation(len(flat_targets))
                    flat_active = flat_active[perm]
                    flat_passive = flat_passive[perm]
                    flat_targets = flat_targets[perm]
                    
                    for i in range(0, len(flat_targets), chunk_size):
                        if shutdown_event.is_set():
                            return
                        
                        chunk_item = (
                            {
                                'active_features': flat_active[i:i+chunk_size],
                                'passive_features': flat_passive[i:i+chunk_size],
                            },
                            flat_targets[i:i+chunk_size]
                        )
                        
                        while not shutdown_event.is_set():
                            try:
                                data_queue.put(chunk_item, timeout=0.1)
                                break
                            except queue.Full:
                                continue
                            except (ValueError, OSError):
                                # Immediately drop out if the queue architecture was dismantled by the main thread
                                return
                    
                    del flat_active, flat_passive, flat_targets
                    
            except Exception as e:
                print(f"\n[Dataset Worker Error] Skipping corrupted shard {file_path}: {e}")
                continue
                
        # Handle remainder elements cleanly when total file slice finishes
        if accumulated_files > 0 and not shutdown_event.is_set():
            flat_active = np.concatenate(mixed_active, axis=0)
            flat_passive = np.concatenate(mixed_passive, axis=0)
            flat_targets = np.concatenate(mixed_targets, axis=0)
            
            # Flush working lists immediately to prevent inter-epoch lingering memory bloat
            mixed_active, mixed_passive, mixed_targets = [], [], []
            accumulated_files = 0
            
            perm = np.random.permutation(len(flat_targets))
            for i in range(0, len(flat_targets), chunk_size):
                if shutdown_event.is_set(): 
                    return
                chunk_item = (
                    {
                        'active_features': flat_active[perm[i:i+chunk_size]], 
                        'passive_features': flat_passive[perm[i:i+chunk_size]]
                    }, 
                    flat_targets[perm[i:i+chunk_size]]
                )
                while not shutdown_event.is_set():
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
        
        self.shutdown_event = None
        self.workers = []
        self.queue = None

    def start_epoch_stream(self):
        """
        Hard reset: Dismantles previous frameworks and deploys strided workers using a
        hardware-regulated buffer cap designed to maximize entropy safely on a 24GB Apple M4 Mac.
        """
        self.shutdown_event = mp.Event()
        self.queue = mp.Queue(maxsize=self.queue_size)
        self.queue.cancel_join_thread()
        
        # Pull your newly isolated, strided file chunks
        worker_file_map = self.get_shuffled_worker_slices()
        
        self.workers = []
        for i in range(self.num_workers):
            worker_files = worker_file_map[i]
            total_worker_files = len(worker_files)
            if total_worker_files == 0: 
                continue
            
            max_safe_buffer_size = 32            
            dynamic_mixing_buffer_ceiling = max_safe_buffer_size
            while dynamic_mixing_buffer_ceiling > 1:
                if total_worker_files % dynamic_mixing_buffer_ceiling == 0:
                    break
                dynamic_mixing_buffer_ceiling -= 1
                
            # Safe structural fallback
            if dynamic_mixing_buffer_ceiling == 1:
                dynamic_mixing_buffer_ceiling = min(max_safe_buffer_size, total_worker_files)
                
            p = mp.Process(
                target=_worker_loop,
                args=(
                    list(worker_files), 
                    self.queue, 
                    self.shutdown_event, 
                    self.PADDING_INDEX_VALUE, 
                    self.batch_size,
                    dynamic_mixing_buffer_ceiling  # Cleanly segmented memory sweeps
                ),
                daemon=True
            )
            p.start()
            self.workers.append(p)
            
        print(f"[DatasetManager] {self.name} worker streams deployed. Hardware-regulated buffer locked at {dynamic_mixing_buffer_ceiling} files per sweep.")

    def get_shuffled_worker_slices(self):
        # 1. Clone and deeply randomize the absolute master list
        shuffled_master = list(self.all_files)
        np.random.shuffle(shuffled_master)
        
        # 2. Interleave the file paths across your worker threads using striding [i::num_workers]
        worker_assignments = {}
        for worker_idx in range(self.num_workers):
            assigned_files = shuffled_master[worker_idx::self.num_workers]
            worker_assignments[worker_idx] = assigned_files
            
        return worker_assignments

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
        print(f"[DatasetManager] {self.name} worker stopped")

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