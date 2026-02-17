use crate::chess_piece::ChessPiece;

pub struct RookModel;

impl ChessPiece for RookModel {
    fn phase_weight(&self) -> u32 {
        2
    }

    fn piece_value(&self) -> u32 {
    	500
    }
}