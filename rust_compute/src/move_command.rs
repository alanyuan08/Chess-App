#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardMove {
    pub start_sq: usize,
    pub end_sq: usize,
    pub move_type: MoveFlag,
    pub pv_score: i32,
}

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

pub const PROMOTION_FLAGS: [MoveFlag; 4] = [
    MoveFlag::PROMOTIONQUEEN,
    MoveFlag::PROMOTIONROOK,
    MoveFlag::PROMOTIONBISHOP,
    MoveFlag::PROMOTIONKNIGHT,
];

#[derive(Clone, Copy)]
pub struct SearchResult {
    pub score: i32,
    pub best_move: Option<ForwardMove>,
}

pub fn white_promotion_piece(promotion_flag: MoveFlag) -> BoardPiece {
    match promotion_flag {
        MoveFlag::PROMOTIONQUEEN => {
            return BoardPiece::WQUEEN;
        },
        MoveFlag::PROMOTIONROOK => {
            return BoardPiece::WROOK;
        },
        MoveFlag::PROMOTIONBISHOP => {
            return BoardPiece::WBISHOP;
        },
        MoveFlag::PROMOTIONKNIGHT => {
            return BoardPiece::WKNIGHT;
        },
        _ => {
            panic!("Invalid Promotion Flag");
        },
    }
}

pub fn black_promotion_piece(promotion_flag: MoveFlag) -> BoardPiece {
    match promotion_flag {
        MoveFlag::PROMOTIONQUEEN => {
            return BoardPiece::BQUEEN;
        },
        MoveFlag::PROMOTIONROOK => {
            return BoardPiece::BROOK;
        },
        MoveFlag::PROMOTIONBISHOP => {
            return BoardPiece::BBISHOP;
        },
        MoveFlag::PROMOTIONKNIGHT => {
            return BoardPiece::BKNIGHT;
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

pub fn piece_value(piece_type: BoardPiece) -> i32 {
    match piece_type {
        BoardPiece::WPAWN | BoardPiece::BPAWN => {
            return 1;
        },
        BoardPiece::WBISHOP | BoardPiece::BBISHOP |
        BoardPiece::WKNIGHT | BoardPiece::BKNIGHT => {
            return 2;
        },
        BoardPiece::WROOK | BoardPiece::BROOK => {
            return 3;
        },
        BoardPiece::WQUEEN | BoardPiece::BQUEEN => {
            return 4;
        },
        BoardPiece::WKING | BoardPiece::BKING => {
            return 5;
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
            return Side::WHITE;
        },
        BoardPiece::BPAWN | BoardPiece::BBISHOP |
        BoardPiece::BKNIGHT | BoardPiece::BROOK |
        BoardPiece::BQUEEN | BoardPiece::BKING  => {
            return Side::BLACK;
        },
        BoardPiece::NONE => {
            panic!("Passed None");
        },
    }
}
