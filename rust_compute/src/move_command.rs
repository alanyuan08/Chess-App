pub struct MoveCommand {
    start_row: u32,
    start_col: u32,
    end_row: u32,
    end_col: u32,
    move_type: MoveCommandType
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