use crate::chess_piece::ChessPiece;

pub struct KingModel;

impl ChessPiece for KingModel {
    fn phase_weight(&self) -> u32 {
        0
    }

    fn piece_value(&self) -> u32 {
    	10000
    }
}