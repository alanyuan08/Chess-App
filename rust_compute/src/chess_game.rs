use std::sync::atomic::{ AtomicBool };
use crossbeam_channel::{ unbounded, Sender, Receiver };
use pyo3::prelude::*;
use pyo3::types::PyString;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::bishop_mask::*;
use crate::rook_mask::*;
use crate::lmr_table::*;
use crate::transposition_table::*;
use crate::search_worker::*;
use crate::parser::*;
use crate::nnue_network::*;
use crate::search_command::*;
use crate::move_command::*;

pub const PV_DEPTH: i32 = 18;
pub const MAX_DEPTH: i32 = 22;

// When a thread finishes, if it exceeds the time, it will send the 
// termination signal to the other threads.

// The other threads the single every 2048 executions
pub const SEARCH_TIME_LIMIT: u64 = 10;

pub const INFINITY: i32 = 32000;
pub const MATE_VALUE: i32 = 30000;

pub const MATE_THRESHOLD: i32 = MATE_VALUE - (MAX_DEPTH * 2);

// This is tuned for the Mac M4 Pro Chip
// Thread Count is set to the number of Performance Cores to avoid 
// Heterogeneous Thread Migration between Performance and Efficiency
pub const NUM_THREADS: usize = 8;

// Condon-Thompson Bucket Transposition Table
pub const CACHE_SIZE: usize = 12 * 1024;

// Model path
pub const MODEL_PATH: &str = "nnue-training/nnue_weights.bin";

#[pyclass]
pub struct ChessGame {
    transposition_table: Arc<TranspositionTable>,
    nnue_network: &'static NnueNetwork, 

    worker_channels: Vec<Sender<SearchCommand>>, 
    search_result_channel: Receiver<WorkerSearchResult>,
    stop_signal: Arc<AtomicBool>, 
}

#[pymethods]
impl ChessGame {
    #[new]
    fn new() -> Self {
        // 1. Safely stream the 25MB packed matrix from disk straight onto the heap
        let network_box = NnueNetwork::load_from_file(MODEL_PATH)
            .expect("Catastrophic Initializer Failure: Could not load NNUE model file matrices");

        // 2. Leak the Box memory to acquire an immutable reference for all search threads
        let nnue_network: &'static NnueNetwork = Box::leak(network_box);    

        // 3. Transposition Tables
        let transposition_table = Arc::new(TranspositionTable::new(CACHE_SIZE));

        // 3. Create a Fleet of Workers to maintain in between
        let mut worker_channels = Vec::with_capacity(NUM_THREADS);
        let stop_signal = Arc::new(AtomicBool::new(false));

        // 4. Main Thread Receiver
        let (main_sender, main_reciever) = unbounded::<WorkerSearchResult>(); 

        // 5. Create Worker Threads
        for thread_id in 0..NUM_THREADS {
            // Create Worker Sender / Reciever
            let (worker_sender, worker_reciever) = unbounded::<SearchCommand>();
            worker_channels.push(worker_sender);

            // Clone Sender for Main Thread
            let main_sender_clone = main_sender.clone(); 

            // Clone Transposition table Arc
            let tt_arc = Arc::clone(&transposition_table);

            // Clone Stop Signal
            let stop_signal_clone = Arc::clone(&stop_signal);

            thread::spawn(move || {
                let mut search_worker = SearchWorker::new(tt_arc, nnue_network, thread_id);

                // Worker Thread Execution
                while let Ok(command) = worker_reciever.recv() {
                    match command {
                        SearchCommand::UpdateHistory { uci_move } => {
                            // Incrementally update the worker's internal state/NNUE accumulators
                            search_worker.process_move(uci_move);
                        }
                        SearchCommand::StartSearch { max_depth } => {
                            let (thread_best_move, nodes_processed) = 
                                search_worker.root_search(max_depth, &stop_signal_clone);

                            let best_move: ForwardMove = thread_best_move.unwrap();
                            let _ = main_sender_clone.send(
                                WorkerSearchResult { nodes_processed, best_move }
                            );
                        }
                    }
                }
            });
        }
        Self {
            transposition_table,
            nnue_network,

            worker_channels,
            search_result_channel: main_reciever,
            stop_signal,
        }
    }

    // Update Workers
    pub fn update_workers<'py>(
        &self,
        py: Python<'py>, 
        uci_move: String
    ) -> PyResult<Bound<'py, PyString>> { 
        for result_tx in &self.worker_channels {
            let _ = result_tx.send(SearchCommand::UpdateHistory { 
                uci_move: uci_move.clone() 
            });
        }

        Ok(PyString::new(py, ""))
    }

    // Prev Moves provided in UCI Format
    pub fn compute_next_move<'py>(
        &self,
        py: Python<'py>, 
    ) -> PyResult<Bound<'py, PyString>> {        
        // Propagate Move to Worker Channels
        let start_time = Instant::now();

        // Initiate Search
        let mut total_nodes_processed = 0;
        for result_tx in &self.worker_channels {
            let _ = result_tx.send(SearchCommand::StartSearch {
                max_depth: PV_DEPTH,
            });
        }

        // Return Result to Python
        match self.search_result_channel.recv() {
            Ok(result) => {
                let elapsed_time = start_time.elapsed();
                total_nodes_processed += result.nodes_processed;
                println!("{} Nodes Procesed in {} milliseconds", 
                    result.nodes_processed, elapsed_time.as_millis());
                        
                let ai_move = parse_uci(result.best_move);
                let py_str = PyString::new(py, &ai_move);

                let py_obj = py_str.into_pyobject(py)?.unbind();
                Ok(py_obj.into_bound(py))
            }
            Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err("Search threads crashed")),
        }
    }
}

#[pyfunction]
pub fn init_attack_tables() {
    let _ = *BISHOP_ATTACKS;
    let _ = *ROOK_ATTACKS;
    let _ = *LMR_TABLE;
}
