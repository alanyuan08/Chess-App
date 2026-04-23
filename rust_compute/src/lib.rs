use pyo3::prelude::*;

pub mod bishop_mask;
pub mod chess_board;

use crate::chess_board::{new_chess_board};

#[pymodule]
fn rust_compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(new_chess_board, m)?)?;

    Ok(())
}