use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
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

pub const PV_DEPTH: i32 = 16;
pub const MAX_DEPTH: i32 = 18;

// When a thread finishes, if it exceeds the time, it will send the 
// termination signal to the other threads.

// The other threads the single every 2048 executions
pub const SEARCH_TIME_LIMIT: u64 = 20;

pub const INFINITY: i32 = 32000;
pub const MATE_VALUE: i32 = 30000;

// This is tuned for the Mac M4 Pro Chip
// Thread Count is set to the number of Performance Cores to avoid 
// Heterogeneous Thread Migration between Performance and Efficiency
pub const NUM_THREADS: i32 = 8;

// Condon-Thompson Bucket Transposition Table
pub const CACHE_SIZE: usize = 256;

// Model path
pub const MODEL_PATH: &str = "nnue-training/nnue_weights.bin";

#[pyclass]
pub struct ChessGame {
    nodes_processed: Arc<AtomicUsize>,
    transposition_table: Arc<TranspositionTable>,
    nnue_network: &'static NnueNetwork, 
}

#[pymethods]
impl ChessGame {
    #[new]
    fn new() -> Self {
        // 1. Safely stream the 25MB packed matrix from disk straight onto the heap
        let network_box = NnueNetwork::load_from_file(MODEL_PATH)
            .expect("Catastrophic Initializer Failure: Could not load NNUE model file matrices");

        // 2. Leak the Box memory to acquire an immutable reference for all search threads
        let network_static: &'static NnueNetwork = Box::leak(network_box);

        Self {
            nodes_processed: Arc::new(AtomicUsize::new(0)),
            transposition_table: Arc::new(TranspositionTable::new(CACHE_SIZE)),
            nnue_network: network_static,
        }
    }

    // Prev Moves provided in UCI Format
    pub fn compute_next_move<'py>(
        &self,
        py: Python<'py>, 
        prev_moves: Vec<String>
    ) -> PyResult<Bound<'py, PyString>> {
        let mut final_best_move = None;

        // Search Time
        let start_time = Instant::now();

        self.nodes_processed.store(0, Ordering::Relaxed);
        let network_ref = self.nnue_network;

        // Shared stop signal across all M4 Pro performance cores
        let stop_search = Arc::new(AtomicBool::new(false));

        let tt_ref = &*self.transposition_table; 
        let nodes_counter_ref = &*self.nodes_processed;
        
        // Clone Search Worker
        let mut clone_search_worker = SearchWorker::new(tt_ref, network_ref);
        clone_search_worker.process_moves(prev_moves);

        // Reset Count
        let worker_source_ptr: &SearchWorker<'_> = &clone_search_worker;

        thread::scope(|s| {
            let mut handlers = Vec::new();

            for thread_id in 0..NUM_THREADS {
                let worker_ref = worker_source_ptr; 

                let thread_stop_signal = Arc::clone(&stop_search);

                let handle = s.spawn(move || {
                    let mut search_worker = SearchWorker::from_game_state(
                        tt_ref, worker_ref, thread_id
                    );

                    let (thread_best_move, nodes_processed) = search_worker.root_search(
                        PV_DEPTH, 
                        &thread_stop_signal
                    );

                    nodes_counter_ref.fetch_add(nodes_processed, Ordering::Relaxed);

                    (thread_id, thread_best_move)
                });

                handlers.push(handle);
            }

            // Aggregate Responses (Fixed Scoped Handle Join API)
            for handle in handlers {
                let (thread_id, best_move) = handle.join().unwrap(); 
                
                if thread_id == 0 {
                    final_best_move = best_move;
                }
            }
        });

        let elapsed_time = start_time.elapsed();
        let node_procesed =  self.nodes_processed.load(Ordering::Relaxed);
        println!("{} Nodes Procesed in {} milliseconds", 
            node_procesed, elapsed_time.as_millis());
                
        if let Some(mv) = final_best_move {
            let uci = parse_uci(mv);

            // Step 1: Bound<PyString>
            let py_str = PyString::new(py, &uci);

            // Step 2: Bound<PyString> → Py<PyString>
            let py_obj = py_str.into_pyobject(py)?.unbind();

            // Step 3: Py<PyString> → Bound<PyAny>
            Ok(py_obj.into_bound(py))
        } else {
            Ok(PyString::new(py, ""))
        }
    }
}

#[pyfunction]
pub fn init_attack_tables() {
    let _ = *BISHOP_ATTACKS;
    let _ = *ROOK_ATTACKS;
    let _ = *LMR_TABLE;
}
