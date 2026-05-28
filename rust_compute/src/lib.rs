use pyo3::prelude::*;

pub mod bishop_mask;
pub mod king_mask;
pub mod knight_mask;
pub mod pawn_mask;
pub mod rook_mask;
pub mod queen_mask;

pub mod move_command;
pub mod chess_board;
pub mod chess_game;
pub mod zobrist_hash;
pub mod transposition_table;

use crate::chess_game::{init_attack_tables};
use crate::chess_game::{compute_next_move};

#[pymodule]
fn rust_compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_attack_tables, m)?)?;
    m.add_function(wrap_pyfunction!(compute_next_move, m)?)?;

    Ok(())
}