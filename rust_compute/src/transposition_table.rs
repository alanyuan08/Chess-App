use crate::move_command::*;
use crate::chess_game::*;

// 1. TT Entry Flag definitions
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum HashFlag {
    EXACT = 0,
    LOWERBOUND = 1,
    UPPERBOUND = 2,
}

// 2. The streamlined entry structure
#[derive(Clone, Copy, Debug)]
pub struct TTEntry {
    pub key: u64,
    pub move_id: u16,
    pub score: i16,
    pub depth: i8,
    pub flag: HashFlag,
}

// 3. Simple Transposition Table mapping 1 hash key to 1 slot
pub struct TranspositionTable {
    entries: Vec<Option<TTEntry>>,
    mask: usize,
}

impl TranspositionTable {
    /// Creates a flat table matching the nearest power-of-two megabytes
    pub fn new(mb: usize) -> Self {
        let size_bytes = mb * 1024 * 1024;
        let count = size_bytes / std::mem::size_of::<Option<TTEntry>>();
        
        // Round down to power of two for fast bitwise indexing
        let power_of_two_count = count.next_power_of_two() >> 1;
        let final_count = std::cmp::max(1, power_of_two_count);

        Self {
            entries: vec![None; final_count],
            mask: final_count - 1,
        }
    }

    pub fn clear(&mut self) {
        self.entries.fill(None);
    }

    // Looks up a Zobrist hash key
    // Stored Value accounts for ply
    #[inline(always)]
    pub fn probe(&self, key: u64, ply: i32) -> Option<TTEntry> {
        let index = (key as usize) & self.mask;
        if let Some(mut entry) = self.entries[index] {
            if entry.key == key {
                // Adjust stored absolute mate score back to a search-relative score
                if entry.score > (MATE_VALUE - MAX_DEPTH) as i16 { 
                    entry.score -= ply as i16; 
                } else if entry.score < (-MATE_VALUE + MAX_DEPTH) as i16 { 
                    entry.score += ply as i16; 
                }
                return Some(entry);
            }
        }
        None
    }

    /// Simple Always-Replace Strategy: overwrite whatever is currently there
    #[inline(always)]
    pub fn store(&mut self, key: u64, score: i32, ply: i32,
        forward_move: Option<ForwardMove>, depth: i32, flag: HashFlag) 
    {
        let index = (key as usize) & self.mask;
        let move_id = forward_move.map_or(0, |mv| mv.pack()); 

        // Remove Ply for Store_store
        let mut store_score = score as i16;
        if store_score > (MATE_VALUE - MAX_DEPTH) as i16 { 
            store_score += ply as i16; 
        } else if store_score < (-MATE_VALUE + MAX_DEPTH) as i16 { 
            store_score -= ply as i16; 
        }
        self.entries[index] = Some(TTEntry {
            key,
            move_id,
            score: store_score,
            depth: depth as i8,
            flag,
        });
    }
}
