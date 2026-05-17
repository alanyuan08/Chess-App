#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForwardMove {
    pub startSq: usize,
    pub endSq: usize,
    pub moveType: MoveFlag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UndoMove {
    pub startSq: usize,
    pub endSq: usize,
    pub moveType: MoveFlag,
    pub capturedPiece: Option<BoardPiece>,
    pub prevCastleRights: u8,
    pub prevEnPassant: u64,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum MoveFlag {
    MOVE = 0,
    KINGSIDECASTLE = 1,
    QUEENSIDECASTLE = 2,
    PROMOTION = 3,
    ENPASSANT = 4,
    CAPTURE = 5,
}

impl TryFrom<u32> for MoveFlag {
    type Error = ();

    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(MoveFlag::MOVE),
            1 => Ok(MoveFlag::KINGSIDECASTLE),
            2 => Ok(MoveFlag::QUEENSIDECASTLE),
            3 => Ok(MoveFlag::PROMOTION),
            4 => Ok(MoveFlag::ENPASSANT),
            5 => Ok(MoveFlag::CAPTURE),
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