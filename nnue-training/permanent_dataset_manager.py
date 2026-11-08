import os
import glob
import time
import queue
import numpy as np
import pandas as pd
import multiprocessing as mp

def _infinite_worker_loop(all_files, data_queue, shutdown_event, padding_index_value, chunk_size):
    """
    100% autonomous data pump. Shuffles the master file pool, parses and chunks 
    the parquets, and loops back to the beginning forever until shutdown.
    
    CRUCIAL: Features are shifted up by 1 (+1) to free up index 0 strictly for 
    Keras mask_zero=True structural padding.
    """
    np.random.seed(int(time.time() * 1000) % 2**32 ^ mp.current_process().pid)
    worker_files = list(all_files)
    
    while not shutdown_event.is_set():
        # Globally shuffle the local file pool at the start of every pass
        np.random.shuffle(worker_files)
        
        for file_path in worker_files:
            if shutdown_event.is_set():
                return
                
            try:
                # High-speed column decompression pass
                df = pd.read_parquet(file_path, columns=['active_indices', 'passive_indices', 'target'], engine='pyarrow')
                if len(df) == 0: 
                    continue
                
                # Transform arrays into aligned matrix layouts
                flat_active = np.stack(df['active_indices'].to_numpy()).astype(np.int16)
                flat_passive = np.stack(df['passive_indices'].to_numpy()).astype(np.int16)
                flat_targets = np.stack(df['target'].to_numpy()).astype(np.float32).reshape(-1, 1)
                del df
                
                # -------------------------------------------------------------
                # STRUCTURAL MASK_ZERO ALIGNMENT (BUGBUSTED FIXED)
                # -------------------------------------------------------------
                # Create a strict boolean mask of legitimate, unshifted features
                valid_active_mask = (flat_active >= 0) & (flat_active < padding_index_value)
                valid_passive_mask = (flat_passive >= 0) & (flat_passive < padding_index_value) 
                
                # Only shift verified, legitimate board/castling features up by 1
                flat_active[valid_active_mask] += 1
                flat_passive[valid_passive_mask] += 1
                
                # Explicitly catch EVERYTHING else (including old -1 markers, 
                # or rogue padding placeholders) and ground them to structural 0
                flat_active[~valid_active_mask] = np.int16(0)
                flat_passive[~valid_passive_mask] = np.int16(0)

                # Row-level shuffle within the file to maximize information entropy
                perm = np.random.permutation(len(flat_targets))
                flat_active = flat_active[perm]
                flat_passive = flat_passive[perm]
                flat_targets = flat_targets[perm]
                
                num_usable_rows = (len(flat_targets) // chunk_size) * chunk_size
                
                for i in range(0, num_usable_rows, chunk_size):
                    if shutdown_event.is_set():
                        return
                        
                    chunk_item = (
                        {
                            'active_features': flat_active[i:i+chunk_size], 
                            'passive_features': flat_passive[i:i+chunk_size]
                        },
                        flat_targets[i:i+chunk_size]
                    )
                    
                    # Pump natively into the shared conveyor belt
                    while not shutdown_event.is_set():
                        try:
                            data_queue.put(chunk_item, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                        except (ValueError, OSError):
                            return
                            
                del flat_active, flat_passive, flat_targets
                
            except Exception:
                continue

class InfiniteConveyorDatasetManager:
    def __init__(self, shard_directory, shard_pattern="*.parquet", num_workers=4, queue_size=50, batch_size=8192, name="training"):
        self.shard_directory = shard_directory
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.name = name
        self.PADDING_INDEX_VALUE =  32 * 11 * 64 + 4 
        
        self.all_files = glob.glob(os.path.join(shard_directory, shard_pattern))
        if not self.all_files:
            raise FileNotFoundError(f"No shards found in: {shard_directory}")
            
        print(f"[DatasetManager] {self.name} system online. Running with {len(self.all_files)} files.")
        
        self.shutdown_event = mp.Event()
        self.queue = mp.Queue(maxsize=self.queue_size)
        self.queue.cancel_join_thread()
        
        # Distribute the master file chunks round-robin once at initialization
        worker_file_map = {i: [] for i in range(self.num_workers)}
        for idx, file_path in enumerate(self.all_files):
            worker_id = idx % self.num_workers
            worker_file_map[worker_id].append(file_path)
            
        self.workers = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=_infinite_worker_loop,
                args=(
                    worker_file_map[i], 
                    self.queue, 
                    self.shutdown_event, 
                    self.PADDING_INDEX_VALUE, 
                    self.batch_size
                ),
                daemon=True
            )
            p.start()
            self.workers.append(p)

    def generator_fn(self):
        """ Flat, infinite stream generator. Never locks, never stops. """
        while not self.shutdown_event.is_set():
            try:
                yield self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except (ValueError, OSError):
                return

    def stop_epoch_stream(self):
        self.shutdown_event.set()
        for p in self.workers:
            try: p.terminate()
            except Exception: pass