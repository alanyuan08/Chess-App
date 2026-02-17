use pyo3::prelude::*;

mod chess_board;
use chess_board::ChessBoard;

mod move_command;
use move_command::MoveCommand;

#[derive(Debug, PartialEq)]
enum Color {
    White,
    Black,
}

struct ChessPiece {
    row: u32,
    col: u32,
    color: Color,
    moves: u32,
}

trait chess_piece {
    fn phase_weight(&self) -> u32;

    fn piece_value(&self) -> u32;

    fn computed_value(&self, chess_board: &ChessBoard, phase_weight: u32) -> u32: 

    fn possible_moves(&self, chess_board: &ChessBoard) -> Vec<MoveCommand>:

    fn capture_targets(&self, chess_board: &ChessBoard) -> Vec<(u32, u32)):
}