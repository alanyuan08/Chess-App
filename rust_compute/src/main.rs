use std::sync::{Arc, Mutex};

mod uci_protocol;
mod search_worker;
mod chess_board;
mod move_command;
mod parser;

use uci_protocol::run_uci_server;
use search_worker::SearchWorker;

fn main() {
    // Initialize search worker
    let worker = Arc::new(Mutex::new(SearchWorker::new()));
    
    // Start UCI protocol loop
    run_uci_server(worker);
}
