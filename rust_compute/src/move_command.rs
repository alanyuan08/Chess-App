#[derive(Debug, Clone, Copy)]
#[repr(C)] // 5 Bytes
pub struct ForwardMove {
    pub pv_score: i16, // 2 Byte
    pub start_sq: u8, // 1 Byte
    pub end_sq: u8, // 1 Byte
    pub move_type: MoveFlag, // 1 Byte
}

// Packs the move into a single u16 for the Transposition Table
impl ForwardMove {
    pub const NULL_MOVE: Self = ForwardMove {
        start_sq: 0,
        end_sq: 0,
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
    }
}

impl Eq for ForwardMove {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)] // 6 Bytes
pub struct UndoMove {
    pub start_sq: u8, // 1 Byte
    pub end_sq: u8, // 1 Byte
    pub prev_castle_rights: u8, // 1 Byte
    pub prev_en_passant: u8, // 1 Byte
    
    pub move_type: MoveFlag, // 1 Byte
    pub captured_piece: BoardPiece, // 1 Byte
}

impl UndoMove {
    pub const NULL_UNDO_MOVE: Self = UndoMove {
        start_sq: 0,
        end_sq: 0,
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
            panic!("Invalid Promotion Flag");
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
            panic!("Invalid Promotion Flag");
        },
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum Side {
    WHITE = 0,
    BLACK = 1,
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

/// Computes the unique index [0..49151] for a piece from a specific player's perspective
#[inline(always)]
pub fn get_feature_index(king_sq: u8, piece: BoardPiece, 
    piece_sq: u8, is_black_active: bool) -> usize {
    let piece_type = piece.to_nnue_type();

    let (k_sq, p_sq, p_type) = if is_black_active {
        // From Black's perspective, flip the board vertically and invert piece colors
        // In Python layout: White pieces are 0..5, Black pieces are 6..11
        let inverted_type = (piece_type + 6) % 12;
        (flip_square(king_sq), flip_square(piece_sq), inverted_type)
    } else {
        (king_sq, piece_sq, piece_type)
    };

    // Index Formula: (KingSquare * 768) + (PieceType * 64) + PieceSquare
    (k_sq as usize) * 768 + (p_type as usize) * 64 + (p_sq as usize)
}

pub fn is_pawn(piece: BoardPiece) -> bool {
    matches!(piece, BoardPiece::WPAWN | BoardPiece::BPAWN)
}

pub fn is_king(piece: BoardPiece) -> bool {
    matches!(piece, BoardPiece::WKING | BoardPiece::BKING)
}

pub fn is_some(piece: BoardPiece) -> bool {
    !matches!(piece, BoardPiece::NONE)
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
            panic!("Passed None");
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
            panic!("Passed None");
        },
    }
}
