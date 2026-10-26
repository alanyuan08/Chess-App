#[derive(Debug, Clone, Copy)]
#[repr(C)] // 6 Bytes
pub struct ForwardMove {
    pub pv_score: i16,       // 2 Byte
    pub start_sq: u8,        // 1 Byte
    pub end_sq: u8,          // 1 Byte
    pub capture_sq: u8,       // 1 Byte
    pub move_type: MoveFlag, // 1 Byte
}

// Packs the move into a single u16 for the Transposition Table
impl ForwardMove {
    pub const NULL_MOVE: Self = ForwardMove {
        start_sq: 0,
        end_sq: 0,
        capture_sq: 0,
        move_type: MoveFlag::NULL,
        pv_score: 0,
    };
}

// Custom PartialEq to exclude pv_score from equality checks
impl PartialEq for ForwardMove {
    fn eq(&self, other: &Self) -> bool {
        self.start_sq == other.start_sq 
            && self.end_sq == other.end_sq 
            && self.move_type == other.move_type
            && self.capture_sq == other.capture_sq
    }
}

impl Eq for ForwardMove {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)] // 6 Bytes
pub struct UndoMove {
    pub start_sq: u8,
    pub end_sq: u8,

    pub capture_sq: u8,
    pub prev_castle_rights: u8,
    pub prev_en_passant: u8,
    
    pub move_type: MoveFlag, 
    pub captured_piece: BoardPiece,
}

impl UndoMove {
    pub const NULL_UNDO_MOVE: Self = UndoMove {
        start_sq: 0,
        end_sq: 0,
        capture_sq: 0,
        prev_castle_rights: 0,
        prev_en_passant: 64, 
        move_type: MoveFlag::NULL,
        captured_piece: BoardPiece::NONE,
    };
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum MoveFlag {
    MOVE = 0,
    PAWNOPENMOVE = 1,
    QUEENSIDECASTLE = 2,
    KINGSIDECASTLE = 3,
    ENPASSANT = 4,
    CAPTURE = 5,
    NULL = 6,

    PROMOTIONQUEEN = 7,
    PROMOTIONROOK = 8,
    PROMOTIONBISHOP = 9,
    PROMOTIONKNIGHT = 10,
}

pub fn promotion_bonus(forward_move: ForwardMove) -> i16 {
    match forward_move.move_type {
        MoveFlag::PROMOTIONQUEEN => 900,
        MoveFlag::PROMOTIONROOK => 500,
        MoveFlag::PROMOTIONBISHOP | MoveFlag::PROMOTIONKNIGHT => 300,
        _ => 0,
    }
}

pub fn white_promotion_piece(promotion_flag: MoveFlag) -> BoardPiece {
    match promotion_flag {
        MoveFlag::PROMOTIONQUEEN => {
            BoardPiece::WQUEEN
        },
        MoveFlag::PROMOTIONROOK => {
            BoardPiece::WROOK
        },
        MoveFlag::PROMOTIONBISHOP => {
            BoardPiece::WBISHOP
        },
        MoveFlag::PROMOTIONKNIGHT => {
            BoardPiece::WKNIGHT
        },
        _ => {
            BoardPiece::NONE
        },
    }
}

pub fn black_promotion_piece(promotion_flag: MoveFlag) -> BoardPiece {
    match promotion_flag {
        MoveFlag::PROMOTIONQUEEN => {
            BoardPiece::BQUEEN
        },
        MoveFlag::PROMOTIONROOK => {
            BoardPiece::BROOK
        },
        MoveFlag::PROMOTIONBISHOP => {
            BoardPiece::BBISHOP
        },
        MoveFlag::PROMOTIONKNIGHT => {
            BoardPiece::BKNIGHT
        },
        _ => {
            BoardPiece::NONE
        },
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Side {
    WHITE = 0,
    BLACK = 1,
    NONE = 2,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum BoardPiece {
    NONE = 0,

    WPAWN = 1,
    WBISHOP = 2,
    WKNIGHT = 3,
    WROOK = 4,
    WQUEEN = 5,
    WKING = 6,

    BPAWN = 7,
    BBISHOP = 8,
    BKNIGHT = 9,
    BROOK = 10,
    BQUEEN = 11,
    BKING = 12,
}

impl BoardPiece {
    /// Maps the BoardPiece enum to a 0..11 integer matching your Python layout.
    /// Panic-free optimization since we filter out NONE at call sites.
    #[inline(always)]
    pub fn to_nnue_type(self) -> u8 {
        debug_assert!(self != BoardPiece::NONE);
        (self as u8) - 1
    }
}

/// Flips a square vertically for Black's perspective (e.g., square 0/a1 becomes 56/a8)
#[inline(always)]
pub fn flip_square(sq: u8) -> u8 {
    sq ^ 63
}

// The 14-bucket coordinate mapping layout array with horizontal mirroring.
// Re-indexed to 14 buckets (0..13) instead of 16 to match the exact mathematical dimension.
pub const KING_BUCKET_MAP: [u8; 64] = [
    // Rank 1 (Each file A-D gets its own unique bucket)
    0,  1,  2,  3,    3,  2,  1,  0,
    // Rank 2 (Each file A-D gets its own unique bucket)
    4,  5,  6,  7,    7,  6,  5,  4,
    
    // Rank 3 & 4 (Combined 2x2 grids to combat mid-board data starvation)
    8,  8,  9,  9,    9,  9,  8,  8,
    8,  8,  9,  9,    9,  9,  8,  8,
    
    // Rank 5 & 6 (Combined 2x2 quadrant blocks to isolate edge vs center)
    10, 10, 11, 11,  11, 11, 10, 10,
    10, 10, 11, 11,  11, 11, 10, 10,
    
    // Rank 7 & 8 (Combined 2x2 quadrant blocks to isolate edge vs center)
    12, 12, 13, 13,  13, 13, 12, 12,
    12, 12, 13, 13,  13, 13, 12, 12
];

/// Computes the unique index [0..9856] for a piece from a specific player's perspective.
/// Returns `None` if the piece is a King, signaling that it should not be added to the features.
#[inline(always)]
pub fn get_feature_index(
    king_sq: u8, 
    piece: BoardPiece, 
    piece_sq: u8, 
    is_black_active: bool
) -> usize {
    let piece_type = piece.to_nnue_type(); // 0..11 based on your PIECE_MAP

    // 1. Establish perspective baseline coordinates and handle color inversions
    let (k_sq, p_sq, b_orig_type) = if is_black_active {
        // Rotate 180 degrees vertically and invert side types for Black's perspective
        let inverted_type = (piece_type + 6) % 12;
        (king_sq ^ 63, piece_sq ^ 63, inverted_type)
    } else {
        (king_sq, piece_sq, piece_type)
    };

    // 3. Compress piece IDs into a contiguous 0..10 space
    let final_p_type = if b_orig_type < 5 {
        b_orig_type
    } else {
        b_orig_type - 1
    };

    // 4. Resolve horizontal mirroring conditions based on active King's file position
    let king_file = k_sq % 8;
    let is_mirrored = king_file >= 4;

    // Determine target lookup square for the bucket layout and piece features using branchless select
    let lookup_king_sq = if is_mirrored { k_sq ^ 7 } else { k_sq };
    let final_p_sq = if is_mirrored { p_sq ^ 7 } else { p_sq };

    // Resolve the bucket ID (0..13)
    let king_bucket = KING_BUCKET_MAP[lookup_king_sq as usize] as usize;

    // 5. Index Formula: (KingBucket * 704) + (PieceType * 64) + PieceSquare
    (king_bucket * 704) + (final_p_type as usize * 64) + (final_p_sq as usize)
}

pub fn is_pawn(piece: BoardPiece) -> bool {
    matches!(piece, BoardPiece::WPAWN | BoardPiece::BPAWN)
}

pub fn is_king(piece: BoardPiece) -> bool {
    matches!(piece, BoardPiece::WKING | BoardPiece::BKING)
}

pub fn is_none(piece: BoardPiece) -> bool {
    matches!(piece, BoardPiece::NONE)
}

pub fn piece_value(piece_type: BoardPiece) -> i16 {
    match piece_type {
        BoardPiece::WPAWN | BoardPiece::BPAWN => {
            100
        },
        BoardPiece::WBISHOP | BoardPiece::BBISHOP |
        BoardPiece::WKNIGHT | BoardPiece::BKNIGHT => {
            300
        },
        BoardPiece::WROOK | BoardPiece::BROOK => {
            500
        },
        BoardPiece::WQUEEN | BoardPiece::BQUEEN => {
            900
        },
        BoardPiece::WKING | BoardPiece::BKING => {
            20000
        },
        BoardPiece::NONE => {
            0
        },
    }
}

pub fn piece_player(piece_type: BoardPiece) -> Side {
    match piece_type {
        BoardPiece::WPAWN | BoardPiece::WBISHOP |
        BoardPiece::WKNIGHT | BoardPiece::WROOK |
        BoardPiece::WQUEEN | BoardPiece::WKING  => {
            Side::WHITE
        },
        BoardPiece::BPAWN | BoardPiece::BBISHOP |
        BoardPiece::BKNIGHT | BoardPiece::BROOK |
        BoardPiece::BQUEEN | BoardPiece::BKING  => {
            Side::BLACK
        },
        BoardPiece::NONE => {
            Side::NONE
        },
    }
}
