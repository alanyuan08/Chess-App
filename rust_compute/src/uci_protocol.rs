use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use crate::search_worker::SearchWorker;

/// UCI Protocol Handler for FastChess integration
pub struct UCIHandler {
    worker: Arc<Mutex<SearchWorker>>,
    engine_name: String,
    author: String,
}

impl UCIHandler {
    pub fn new(worker: Arc<Mutex<SearchWorker>>) -> Self {
        UCIHandler {
            worker,
            engine_name: "ChessAI".to_string(),
            author: "Alan Yuan".to_string(),
        }
    }

    /// Start UCI protocol loop
    pub fn start(&self) {
        let stdin = io::stdin();
        let reader = stdin.lock();

        for line in reader.lines() {
            if let Ok(input) = line {
                let trimmed = input.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let tokens: Vec<&str> = trimmed.split_whitespace().collect();
                if tokens.is_empty() {
                    continue;
                }

                match tokens[0] {
                    "uci" => self.handle_uci(),
                    "isready" => self.handle_isready(),
                    "setoption" => self.handle_setoption(&tokens),
                    "position" => self.handle_position(&tokens),
                    "go" => self.handle_go(&tokens),
                    "stop" => self.handle_stop(),
                    "quit" => {
                        println!("Exiting UCI protocol.");
                        break;
                    }
                    _ => {
                        // Ignore unknown commands silently
                    }
                }
            }
        }
    }

    /// Handle 'uci' command - identify engine
    fn handle_uci(&self) {
        println!("id name {}", self.engine_name);
        println!("id author {}", self.author);
        println!("option name Hash type spin default 256 min 1 max 33554432");
        println!("option name Threads type spin default 1 min 1 max 512");
        println!("uciok");
        io::stdout().flush().ok();
    }

    /// Handle 'isready' command - confirm engine is ready
    fn handle_isready(&self) {
        println!("readyok");
        io::stdout().flush().ok();
    }

    /// Handle 'setoption' command - set engine options
    fn handle_setoption(&self, tokens: &[&str]) {
        // Format: setoption name <name> value <value>
        // For now, we acknowledge but don't process specific options
        // Future: implement hash size and thread configuration
        io::stdout().flush().ok();
    }

    /// Handle 'position' command - set board position
    fn handle_position(&self, tokens: &[&str]) {
        if tokens.len() < 2 {
            return;
        }

        let mut worker = self.worker.lock().unwrap();

        match tokens[1] {
            "startpos" => {
                worker.reset_to_startpos();
                
                // Process moves if provided
                if tokens.len() > 2 && tokens[2] == "moves" {
                    let moves = tokens[3..].to_vec();
                    let moves_str: Vec<String> = moves.iter().map(|s| s.to_string()).collect();
                    worker.process_moves(moves_str);
                }
            }
            "fen" => {
                // Parse FEN string
                if tokens.len() >= 8 {
                    let fen_parts = &tokens[2..8];
                    let fen = fen_parts.join(" ");
                    worker.load_fen(&fen);

                    // Process moves if provided
                    if tokens.len() > 8 && tokens[8] == "moves" {
                        let moves = tokens[9..].to_vec();
                        let moves_str: Vec<String> = moves.iter().map(|s| s.to_string()).collect();
                        worker.process_moves(moves_str);
                    }
                }
            }
            _ => {}
        }
    }

    /// Handle 'go' command - start search
    fn handle_go(&self, tokens: &[&str]) {
        let mut depth = 20; // Default depth
        let mut movetime = None;
        let mut wtime = None;
        let mut btime = None;
        let mut winc = None;
        let mut binc = None;

        // Parse go parameters
        let mut i = 1;
        while i < tokens.len() {
            match tokens[i] {
                "depth" if i + 1 < tokens.len() => {
                    depth = tokens[i + 1].parse().unwrap_or(20);
                    i += 2;
                }
                "movetime" if i + 1 < tokens.len() => {
                    movetime = tokens[i + 1].parse().ok();
                    i += 2;
                }
                "wtime" if i + 1 < tokens.len() => {
                    wtime = tokens[i + 1].parse().ok();
                    i += 2;
                }
                "btime" if i + 1 < tokens.len() => {
                    btime = tokens[i + 1].parse().ok();
                    i += 2;
                }
                "winc" if i + 1 < tokens.len() => {
                    winc = tokens[i + 1].parse().ok();
                    i += 2;
                }
                "binc" if i + 1 < tokens.len() => {
                    binc = tokens[i + 1].parse().ok();
                    i += 2;
                }
                "infinite" => {
                    depth = 100; // Search until stopped
                    i += 1;
                }
                _ => i += 1,
            }
        }

        // Perform search
        let start_time = Instant::now();
        let mut worker = self.worker.lock().unwrap();
        
        // Execute iterative deepening search
        let best_move = worker.search(depth);

        // Output result in UCI format
        if let Some(mv) = best_move {
            let elapsed = start_time.elapsed().as_millis() as u64;
            let nodes = worker.get_nodes_searched();
            
            println!("bestmove {}", mv);
            io::stdout().flush().ok();
        }
    }

    /// Handle 'stop' command - halt search
    fn handle_stop(&self) {
        // Signal search to stop (implementation depends on search architecture)
        // For now, this is a placeholder
        io::stdout().flush().ok();
    }
}

/// Standalone UCI server for FastChess integration
pub fn run_uci_server(worker: Arc<Mutex<SearchWorker>>) {
    let handler = UCIHandler::new(worker);
    handler.start();
}
