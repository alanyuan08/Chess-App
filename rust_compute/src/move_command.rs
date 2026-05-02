#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Move {
    startSq: usize,
    endSq: usize,
    moveType: MoveFlag,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct UndoMove {
    startSq: usize,
    endSq: usize,
    moveType: MoveFlag,
    capturedPiece: Option<Piece>,
    prevCastleRights: u8,
    prevEnPassant: u64,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum MoveFlag {
    MOVE = 0,
    KINGSIDECASTLE = 1,
    QUEENSIDECASTLE = 2,
    PROMOTION = 3,
    ENPASSANT = 4,
    CAPTURE = 5,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Side {
    WHITE = 0,
    BLACK = 1,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Piece {
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