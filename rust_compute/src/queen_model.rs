use crate::chess_piece::ChessPiece;

pub struct QueenModel;

impl ChessPiece for QueenModel {
    fn phase_weight(&self) -> u32 {
        4
    }

    fn piece_value(&self) -> u32 {
    	900
    }
}