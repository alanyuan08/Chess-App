use pyo3::prelude::*;

pub struct MoveCommand {
    startRow: u32,
    startCol: u32,
    endRow: u32,
    endCol: u32,
    moveType: MoveCommandType
}

#[derive(Debug, PartialEq)]
enum MoveCommandType {
    Move,
    QueenSideCastle,
    KingSideCastle,
    Capture,
    Enpassant,
    PawnOpenMove,
    Promote
}