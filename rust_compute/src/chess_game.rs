use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use pyo3::prelude::*;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::bishop_mask::*;
use crate::rook_mask::*;
use crate::transposition_table::*;
use crate::search_worker::*;

pub const PV_DEPTH: i32 = 14;
pub const MAX_DEPTH: i32 = 20;

// When a thread finishes, if it exceeds the time, it will send the 
// termination signal to the other threads.

// The other threads the single every 2048 executions
pub const DEPTH_SEARCH_LIMIT: u64 = 15;

pub const INFINITY: i32 = 32000;
pub const MATE_VALUE: i32 = 30000;

// This is tuned for the Mac M4 Pro Chip
// Thread Count is set to the number of Performance Cores to avoid 
// Heterogeneous Thread Migration between Performance and Efficiency
pub const NUM_THREADS: i32 = 8;

// Condon-Thompson Bucket Transposition Table
pub const CACHE_SIZE: usize = 64;

#[pyclass]
pub struct ChessGame {
    nodes_processed: Arc<AtomicUsize>,
    transposition_table: Arc<TranspositionTable>,
}

#[pymethods]
impl ChessGame {
    #[new]
    fn new() -> Self {
        Self {
            nodes_processed: Arc::new(AtomicUsize::new(0)),
            transposition_table: Arc::new(TranspositionTable::new(CACHE_SIZE))
        }
    }

    pub fn compute_next_move<'py>(
        &self,
        py: Python<'py>, 
        prev_moves: Vec<String>
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut final_best_move = None;

        // Search Time
        let start_time = Instant::now();

        // Shared stop signal across all M4 Pro performance cores
        let stop_search = Arc::new(AtomicBool::new(false));

        let tt_ref = &*self.transposition_table; 
        let nodes_counter_ref = &*self.nodes_processed;
        
        // Clone Search Worker
        let mut clone_search_worker = SearchWorker::new(tt_ref);
        clone_search_worker.process_moves(prev_moves);

        let worker_source_ptr: &SearchWorker<'_> = &clone_search_worker;

        thread::scope(|s| {
            let mut handlers = Vec::new();

            for thread_id in 0..NUM_THREADS {
                let worker_ref = worker_source_ptr; 

                let thread_stop_signal = Arc::clone(&stop_search);

                let handle = s.spawn(move || {
                    let mut search_worker = 
                        SearchWorker::from_game_state(tt_ref, worker_ref, thread_id);

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
        
        // Python Import
        let module = py.import("modelComponent.moveCommand")?;
        let move_command_class = module.getattr("MoveCommand")?;

        let enum_mod = py.import("appEnums")?;
        let move_command_type_enum = enum_mod.getattr("MoveCommandType")?;

        if let Some(mv) = final_best_move {
            let move_type_value = mv.move_type as i32;
            let enum_instance = move_command_type_enum.call1((move_type_value,))?;

            let args = (
                (mv.start_sq / 8) as i32,
                (mv.start_sq % 8) as i32,
                (mv.end_sq / 8) as i32,
                (mv.end_sq % 8) as i32,
                enum_instance,
            ).into_pyobject(py)?;

            let move_command_instance = move_command_class.call1(args)?;
            Ok(move_command_instance)
        } else {
            Ok(py.None().into_bound(py))
        }
    }
}

#[pyfunction]
pub fn init_attack_tables() {
    let _ = *BISHOP_ATTACKS;
    let _ = *ROOK_ATTACKS;
}
