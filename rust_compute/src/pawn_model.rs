use crate::chess_piece::ChessPiece;

pub struct PawnModel;

impl ChessPiece for PawnModel {
    fn phase_weight(&self) -> u32 {
        0
    }

    fn piece_value(&self) -> u32 {
    	100
    }
}