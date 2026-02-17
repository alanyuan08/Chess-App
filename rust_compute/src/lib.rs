use pyo3::prelude::*;

pub mod bishop_model;
pub mod chess_board;
pub mod chess_piece;
pub mod king_model;
pub mod knight_model;
pub mod move_command;
pub mod pawn_model;
pub mod queen_model;
pub mod rook_model;

use crate::chess_board::{new_chess_board};

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rust_compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(new_chess_board, m)?)?;

    Ok(())
}