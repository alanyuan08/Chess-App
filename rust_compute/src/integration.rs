use std::sync::{
    atomic::AtomicBool,
    Arc, OnceLock,
};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::{
    chess_game::init_attack_tables,
    nnue_network::NnueNetwork,
    search_worker::SearchWorker,
    transposition_table::TranspositionTable,
};

static EVAL_NNUE: OnceLock<&'static NnueNetwork> = OnceLock::new();

fn get_eval_nnue(model_path: &str) -> PyResult<&'static NnueNetwork> {
    if let Some(&nnue) = EVAL_NNUE.get() {
        return Ok(nnue);
    }

    let network_box = NnueNetwork::load_from_file(model_path).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "Failed to load NNUE model at '{}': {}",
            model_path, e
        ))
    })?;

    let leaked: &'static NnueNetwork = Box::leak(network_box);
    Ok(*EVAL_NNUE.get_or_init(|| leaked))
}

#[pyfunction]
pub fn is_unsafe_batch(
    py: Python<'_>,
    fens: Vec<String>,
    model_path: String,
) -> PyResult<Vec<f32>> {
    init_attack_tables();
    let nnue_ref = get_eval_nnue(&model_path)?;

    let results = py.detach(move || {
        fens.par_iter()
            .map_init(
                || {
                    let tt = Arc::new(TranspositionTable::new(4));
                    let stop = Arc::new(AtomicBool::new(false));
                    SearchWorker::new(tt, nnue_ref, stop, 0)
                },

                |worker, fen| worker.compute_volatility_target(fen),
            )
            .collect::<Vec<f32>>()
    });

    Ok(results)
}