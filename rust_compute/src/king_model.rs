use crate::chess_piece::ChessPiece;
use crate::chess_piece::Color;

pub struct KingModel {
    player: Color,
}

impl ChessPiece for KingModel {
    fn phase_weight(&self) -> u32 {
        0
    }

    fn piece_value(&self) -> u32 {
    	10000
    }
}