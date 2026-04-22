use pyo3::prelude::*;

pub mod bishop_mask;
pub mod chess_board;

use crate::chess_board::{new_chess_board};

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rust_compute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let masks = bishop_mask::precompute_all();
    bishop_mask::BISHOP_MASKS.set(masks).unwrap();

    // 4. Also add it to the module so it's visible in Python as chess_lib.Bishop_Mask
    m.add("Bishop_Mask", masks)?;
        m.add_function(wrap_pyfunction!(new_chess_board, m)?)?;

    Ok(())
}