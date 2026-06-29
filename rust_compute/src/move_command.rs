#[derive(Debug, Clone, Copy)]
pub struct ForwardMove {
    pub start_sq: usize,
    pub end_sq: usize,
    pub move_type: MoveFlag,
    pub pv_score: i32,
}

// Packs the move into a single u16 for the Transposition Table
impl ForwardMove {
    pub fn pack(&self) -> u16 {
        (self.start_sq as u16) | ((self.end_sq as u16) << 6) | ((self.move_type as u16) << 12)
    }

    pub fn unpack(packed: u16) -> Self {
        Self {
            start_sq: (packed & 0x3F) as usize,
            end_sq: ((packed >> 6) & 0x3F) as usize,
            move_type: unsafe { std::mem::transmute::<u32, MoveFlag>((packed >> 12) as u32) }, 
            pv_score: 0,
        }
    }
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
pub struct UndoMove {
    pub start_sq: usize,
    pub end_sq: usize,
    pub move_type: MoveFlag,
    pub captured_piece: Option<BoardPiece>,
    pub prev_castle_rights: u8,
    pub prev_en_passant: u64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
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

#[derive(Clone, Copy)]
pub struct SearchResult {
    pub score: i32,
    pub best_move: Option<ForwardMove>,
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

impl TryFrom<u32> for MoveFlag {
    type Error = ();

    fn try_from(v: u32) -> Result<Self, Self::Error> {
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
