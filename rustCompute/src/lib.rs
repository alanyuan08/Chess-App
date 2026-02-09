use pyo3::prelude::*;

#[pyfunction]
fn test_print() {
    println!("Rust Integration!");
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn rustCompute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_print, m)?)?;

    Ok(())
}