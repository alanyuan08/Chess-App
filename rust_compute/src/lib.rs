use pyo3::prelude::*;

pub mod bishop_mask;
pub mod king_mask;
pub mod knight_mask;
pub mod pawn_mask;
pub mod rook_mask;

pub mod chess_board;

use crate::chess_board::{compute_next_move};

#[pymodule]
fn rust_compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_next_move, m)?)?;

    Ok(())
}