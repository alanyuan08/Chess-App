use pyo3::prelude::*;

pub mod bishop_mask;
pub mod chess_board;

use crate::chess_board::{new_chess_board};

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rust_compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(new_chess_board, m)?)?;

    Ok(())
}