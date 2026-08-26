use crate::move_command::*;

pub enum SearchCommand {
    UpdateHistory {
        uci_move: String,
    },
    
    StartSearch
}

#[derive(Clone, Copy)]
pub struct SearchResult {
    pub score: i32,
    pub best_move: ForwardMove,
    pub was_aborted: bool,
}

#[derive(Clone, Copy)]
pub struct WorkerSearchResult {
    pub nodes_processed: usize,
    pub thread_best_move: ForwardMove,
    pub thread_id: usize,
}

    // Process Negamax
    // Transposition Table Summary
    // Exact -> Preivous Function call found a value between Alpha and Beta
    // - Pass Value to parent

    // LOWER BOUND (Fail-High / Beta Cutoff): 
    // - Previous search found a value that exceeded or met Beta.
    // - The true value is AT LEAST this high. 
    // - Action: alpha = cmp::max(alpha, retrieved_score)

    // UPPER BOUND (Fail-Low): 
    // - Previous search couldn't find any move that beat Alpha.
    // - The true value is AT MOST this low. 
    // - Action: beta = cmp::min(beta, retrieved_score)
    
    // True Value -> If we ran Min-Max all the way down with zero pruning
    // Alpha - Best value a Maximizer can guarantee, hence true value is greater than or equal to this
    // Beta - Worst value a Minimizer can guarantee, hence true value is else than or equal to this