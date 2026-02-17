use crate::chess_piece::ChessPiece;

pub struct KnightModel;

impl ChessPiece for KnightModel {
    fn phase_weight(&self) -> u32 {
        1
    }

    fn piece_value(&self) -> u32 {
    	300
    }
}