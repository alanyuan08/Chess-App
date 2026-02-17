use crate::chess_board::ChessBoard;
use crate::move_command::MoveCommand;

#[derive(Debug, PartialEq)]
enum Color {
    White,
    Black,
}

pub trait ChessPiece {
    fn phase_weight(&self) -> u32;

    fn piece_value(&self) -> u32;

    // fn computed_value(&self, chess_board: &ChessBoard, phase_weight: u32) -> u32;

    // fn possible_moves(&self, chess_board: &ChessBoard) -> Vec<MoveCommand>;

    // fn capture_targets(&self, chess_board: &ChessBoard) -> Vec<(u32, u32)>;
}