#[derive(Debug, Clone, Copy)]
#[repr(C)] // Ensures a predictable layout in memory
pub struct ForwardMove {
    pub pv_score: i32,       // 32 bits | Move ordering score
    pub start_sq: u8,        //  8 bits | 0 to 63
    pub end_sq: u8,          //  8 bits | 0 to 63
    pub move_type: MoveFlag, //  8 bits | Enum marked with #[repr(u8)]
}

// Packs the move into a single u16 for the Transposition Table
impl ForwardMove {
    // Packs the move into a single u16 for the Transposition Table
    pub fn pack(&self) -> u16 {
        (self.start_sq as u16) | ((self.end_sq as u16) << 6) | ((self.move_type as u16) << 12)
    }

    // Unpacks the u16 back into our 64-bit struct safely with zero overhead
    pub fn unpack(packed: u16) -> Self {
        // Extract the 4-bit move type flag (bits 12 to 15)
        let raw_flag = ((packed >> 12) & 0x0F) as u8;
        
        // Map the raw integer back to your exact enum variants safely
        let move_type = MoveFlag::try_from(raw_flag).unwrap_or(MoveFlag::NULL);

        Self {
            start_sq: (packed & 0x3F) as u8,
            end_sq: ((packed >> 6) & 0x3F) as u8,
            move_type,
            pv_score: 0,
        }
    }

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
#[repr(C, align(16))]
pub struct UndoMove {
    pub start_sq: u8,               // 8 bits | 0 to 63
    pub end_sq: u8,                 // 8 bits | 0 to 63
    pub prev_castle_rights: u8,     // 8 bits | Packed bitmask
    pub prev_en_passant: u8,     // 8 bits | Square index (0-63), or 64 for None
    
    pub move_type: MoveFlag,        // 8 bits | (Requires #[repr(u8)] on MoveFlag)
    pub captured_piece: BoardPiece, // 8 bits | (Requires #[repr(u8)] on BoardPiece)
}

impl UndoMove {
    pub const NULL_UNDO_MOVE: Self = UndoMove {
        start_sq: 0,
        end_sq: 0,
        prev_castle_rights: 0,
        prev_en_passant: 0, 
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

impl TryFrom<u8> for MoveFlag {
    type Error = ();

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(MoveFlag::MOVE),
            1 => Ok(MoveFlag::PAWNOPENMOVE),
            2 => Ok(MoveFlag::QUEENSIDECASTLE),
            3 => Ok(MoveFlag::KINGSIDECASTLE),
            4 => Ok(MoveFlag::ENPASSANT),
            5 => Ok(MoveFlag::CAPTURE),
            6 => Ok(MoveFlag::NULL),

            7 => Ok(MoveFlag::PROMOTIONQUEEN),
            8 => Ok(MoveFlag::PROMOTIONROOK),
            9 => Ok(MoveFlag::PROMOTIONBISHOP),
            10 => Ok(MoveFlag::PROMOTIONKNIGHT),
            _ => Err(()),
        }
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

    // Fen Notation used is training assumes incorrect order
    let king_sq_flip = king_sq ^ 56;
    let piece_sq_flip = piece_sq ^ 56;

    let (k_sq, p_sq, p_type) = if is_black_active {
        // From Black's perspective, flip the board vertically and invert piece colors
        // In Python layout: White pieces are 0..5, Black pieces are 6..11
        let inverted_type = (piece_type + 6) % 12;
        (flip_square(king_sq_flip), flip_square(piece_sq_flip), inverted_type)
    } else {
        (king_sq_flip, piece_sq_flip, piece_type)
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

pub fn piece_value(piece_type: BoardPiece) -> i32 {
    match piece_type {
        BoardPiece::WPAWN | BoardPiece::BPAWN => {
            1
        },
        BoardPiece::WBISHOP | BoardPiece::BBISHOP |
        BoardPiece::WKNIGHT | BoardPiece::BKNIGHT => {
            2
        },
        BoardPiece::WROOK | BoardPiece::BROOK => {
            3
        },
        BoardPiece::WQUEEN | BoardPiece::BQUEEN => {
            4
        },
        BoardPiece::WKING | BoardPiece::BKING => {
            5
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
