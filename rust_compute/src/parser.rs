use crate::move_command::*;
use std::collections::HashMap;

pub fn parse_uci(forward_move: ForwardMove) -> String {
    let map = HashMap::from([
        (0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'),
        (4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'),
    ]);

    let start_file = *map.get(&(forward_move.start_sq % 8)).unwrap_or(&'a'); 
    let start_rank = (forward_move.start_sq / 8) + 1;
    let end_file = *map.get(&(forward_move.end_sq % 8)).unwrap_or(&'a'); 
    let end_rank = (forward_move.end_sq / 8) + 1;

    let promo = match forward_move.move_type {
        MoveFlag::PROMOTIONQUEEN => "q",
        MoveFlag::PROMOTIONROOK => "r",
        MoveFlag::PROMOTIONBISHOP => "b",
        MoveFlag::PROMOTIONKNIGHT => "n",
        _ => "",
    };
    format!("{}{}{}{}{}", start_file, start_rank, end_file, end_rank, promo)
}

pub fn parse_forward_move(raw_move: &str) -> ForwardMove {    
    let result: Vec<u32> = raw_move.chars().map(|c: char| {
        c.to_digit(10).unwrap_or(0)
    }).collect();

    ForwardMove { 
        start_sq: (result[1] * 8 + result[0]) as usize, 
        end_sq: (result[3] * 8 + result[2]) as usize, 
        move_type: MoveFlag::try_from(result[4]).expect("Corrupted move data"),
        pv_score: 0,
    }
}