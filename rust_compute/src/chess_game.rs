use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use pyo3::prelude::*;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crate::bishop_mask::*;
use crate::rook_mask::*;
use crate::transposition_table::*;
use crate::search_worker::*;

pub const PV_DEPTH: i32 = 9;
pub const MATE_VALUE: i32 = 30000;
pub const MAX_DEPTH: i32 = 20;

pub const INFINITY: i32 = 32000;

// This is tuned for a Mac M4 Pro Chip
pub const NUM_THREADS: i32 = 10;

struct ChessGame {
    nodes_processed: Arc<AtomicI32>,
    transposition_table: TranspositionTable,
}

impl ChessGame {
    fn new() -> Self {
        Self {
            nodes_processed: Arc::new(AtomicI32::new(0)),
            transposition_table: TranspositionTable::new(1024 * 8)
        }
    }
}

#[pyfunction]
pub fn compute_next_move<'py>(py: Python<'py>, prev_moves: Vec<String>) -> PyResult<Bound<'py, PyAny>> {
    let chess_game = ChessGame::new();
    let mut final_best_move = None;

    // Search Time
    let start_time = Instant::now();

    // Shared stop signal across all M4 Pro performance cores
    let stop_search = Arc::new(AtomicBool::new(false));
    
    let tt_ref = &chess_game.transposition_table;
    let mut default_worker = SearchWorker::new(tt_ref);
    default_worker.process_moves(prev_moves);

    thread::scope(|s| {
        let mut handlers = Vec::new();

        for thread_id in 0..NUM_THREADS {
            let stop_signal = Arc::clone(&stop_search);
            let nodes_counter = Arc::clone(&chess_game.nodes_processed); 

            let mut thread_worker = default_worker.clone(); 

            let handle = s.spawn(move || {
                let (thread_best_move, nodes_processed) = thread_worker.root_search(
                    thread_id, 
                    PV_DEPTH, 
                    &stop_signal
                );

                nodes_counter.fetch_add(nodes_processed, Ordering::Relaxed);

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
    let node_procesed =  chess_game.nodes_processed.load(Ordering::Relaxed);
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

#[pyfunction]
pub fn init_attack_tables() {
    let _ = *BISHOP_ATTACKS;
    let _ = *ROOK_ATTACKS;
}
