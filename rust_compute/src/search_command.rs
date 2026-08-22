use crate::move_command::*;

pub enum SearchCommand {
    UpdateHistory {
        uci_move: String,
    },
    
    StartSearch {
        max_depth: i32,
    },
}

#[derive(Clone, Copy)]
pub struct SearchResult {
    pub score: i32,
    pub best_move: Option<ForwardMove>,
}

#[derive(Clone, Copy)]
pub struct WorkerSearchResult {
    pub nodes_processed: usize,
    pub best_move: ForwardMove,
}