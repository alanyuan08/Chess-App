use crate::bishop_mask::*;
use crate::king_mask::*;
use crate::knight_mask::*;
use crate::pawn_mask::*;
use crate::rook_mask::*;
use crate::move_command::*;
use crate::zobrist_hash::*;

// 0 -> White / 1 -> Black
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ChessBoard {
    pawns: [u64; 2],
    knights: [u64; 2],
    bishops: [u64; 2],
    rooks: [u64; 2],
    queens: [u64; 2],
    kings: [u64; 2],
    
    all_pieces: [u64; 2],
    occupied: u64,

    castling_rights: u8,
    en_passant: u64,
    active_player: Side,
    total_moves: i32,

    mailbox: [Piece; 64],

    // Zobrist Hash
    zobrist_hash: u64,
}

pub const WHITE_KINGSIDE: u8 = 0b0001; // 1
pub const WHITE_QUEENSIDE: u8 = 0b0010; // 2
pub const BLACK_KINGSIDE: u8 = 0b0100; // 4
pub const BLACK_QUEENSIDE: u8 = 0b1000; // 8

impl ChessBoard {
    // A constructor-like associated function
    // [0] - White / [1] - Black
    pub fn new() -> Self {
        Self {
            pawns: [0, 0],
            knights: [0, 0],
            bishops: [0, 0],
            rooks: [0, 0],
            queens: [0, 0],
            kings: [0, 0],
            
            all_pieces: [0, 0],
            occupied: 0,

            castling_rights: 0b1111,
            en_passant: 0,
            active_player: Side::WHITE,
            total_moves: 0,

            mailbox: [Piece::NONE; 64],
            zobrist_hash: 0,
        }
    }

    pub fn init_board(&mut self) {
        for color in 0..2 {
            // Ranks: White = 0 & 1, Black = 6 & 7
            let piece_rank_offset = if color == 0 { 0 } else { 56 };
            let pawn_rank_offset  = if color == 0 { 8 } else { 48 };

            // 1. Initialize Pawns
            for i in 0..8 {
                let sq = pawn_rank_offset + i;
                self.pawns[color] |= 1u64 << sq;
                self.mailbox[sq] = if color == 0 { Piece::WPAWN } else { Piece::BPAWN };
            }

            // 2. Initialize Major Pieces (Bitboards)
            self.rooks[color]   |= (1u64 << (piece_rank_offset + 0)) | (1u64 << (piece_rank_offset + 7));
            self.knights[color] |= (1u64 << (piece_rank_offset + 1)) | (1u64 << (piece_rank_offset + 6));
            self.bishops[color] |= (1u64 << (piece_rank_offset + 2)) | (1u64 << (piece_rank_offset + 5));
            self.queens[color]  |= 1u64 << (piece_rank_offset + 3);
            self.kings[color]   |= 1u64 << (piece_rank_offset + 4);

            // 3. Initialize Major Pieces (Mailbox)
            let pieces = if color == 0 {
                [Piece::WROOK, Piece::WKNIGHT, Piece::WBISHOP, Piece::WQUEEN, Piece::WKING, Piece::WBISHOP, Piece::WKNIGHT, Piece::WROOK]
            } else {
                [Piece::BROOK, Piece::BKNIGHT, Piece::BBISHOP, Piece::BQUEEN, Piece::BKING, Piece::BBISHOP, Piece::BKNIGHT, Piece::BROOK]
            };

            for i in 0..8 {
                self.mailbox[piece_rank_offset + i] = pieces[i];
            }

            // 4. Composite Bitboards
            self.all_pieces[color] = self.pawns[color] | self.rooks[color] | 
                                    self.knights[color] | self.bishops[color] | 
                                    self.queens[color] | self.kings[color];
            self.occupied |= self.all_pieces[color];

            // 5. Compute Zobrist
            self.zobrist_hash = self.compute_init_Zobrist();
        }
    }

    // Compute Init Zobritist
    pub fn compute_init_Zobrist(self) -> u64 { 
        let mut hash = 0u64;

        // If piece_idx is 'Empty', this XORs with 0, which does nothing.
        for (index, piece) in self.mailbox.iter().enumerate() {
            let piece_idx = piece_type_zobrist(*piece);
            hash ^= ZOBRIST_TABLE_MAP[piece_idx][index];
        }

        // Side to Move
        hash ^= ZOBRIST_SIDE_TO_MOVE[active_player_zobrist(self.active_player)];

        // Castling
        hash ^= ZOBRIST_CASTLING[self.castling_rights as usize];

        // En Passant
        hash ^= ZOBRIST_EN_PASSANT[en_passant_zobrist(self.en_passant)];

        hash
    }

    // Return Castle Rights
    pub fn castle_rights(self) -> u8 {   
        let castle_rights = self.castling_rights.clone();
        castle_rights
    }

    pub fn en_passant(self) -> u64 {   
        let en_passant = self.en_passant.clone();
        en_passant
    }

    // Used to Calculate Castling / King Safety
    fn opponent_attack_targets(&mut self) -> u64 {
        let mut attacks = 0u64;
        let opp = if self.active_player == Side::WHITE { 1 } else { 0 };
        let occ = self.occupied;

        // 1. Pawns - Print Opponent Attack Pawns
        let pawns = self.pawns[opp];
        if self.active_player == Side::WHITE {  
            attacks |= black_pawn_attacks(pawns);
        } else {
            attacks |= white_pawn_attacks(pawns);
        }

        // 2. Knights
        let mut knights = self.knights[opp];
        while knights != 0 {
            attacks |= KNIGHT_ATTACKS[knights.trailing_zeros() as usize];
            knights &= knights - 1;
        }

        // 3. Kings
        let mut kings = self.kings[opp];
        while kings != 0 {
            attacks |= KING_ATTACKS[kings.trailing_zeros() as usize];
            kings &= kings - 1;
        }

        // 4. Sliders (Bishops, Rooks, Queens)
        let mut bishops = self.bishops[opp] | self.queens[opp];
        while bishops != 0 {
            attacks |= bishop_attack_paths(bishops.trailing_zeros() as usize, occ);
            bishops &= bishops - 1;
        }

        let mut rooks = self.rooks[opp] | self.queens[opp];
        while rooks != 0 {
            attacks |= rook_attack_paths(rooks.trailing_zeros() as usize, occ);
            rooks &= rooks - 1;
        }

        attacks
    }

    // Generate Pseudo-Moves - Only Validate King Safety for Castle / King Movement
    pub fn generate_moves(&mut self) -> Vec<Move> {
        let mut gen_moves: Vec<Move> = Vec::with_capacity(64);
        let player_index = if self.active_player == Side::WHITE { 0 } else { 1 }; 
        let opp_index = if self.active_player == Side::WHITE { 1 } else { 0 }; 

        let _opponent_attack_targets = self.opponent_attack_targets();

        let king_positon = get_lsb_indices(self.kings[player_index]);
        king_moves(king_positon, self.occupied, _opponent_attack_targets,
        self.active_player, self.castling_rights, self.all_pieces[opp_index], &mut gen_moves);

        let knight_positon = get_lsb_indices(self.knights[player_index]);
        knight_moves(knight_positon, self.occupied, self.all_pieces[opp_index], &mut gen_moves);

        let rook_position = get_lsb_indices(self.rooks[player_index]);
        rook_moves(rook_position, self.occupied, self.all_pieces[opp_index], &mut gen_moves);

        let bishop_position = get_lsb_indices(self.bishops[player_index]);
        bishop_moves(bishop_position, self.occupied, self.all_pieces[opp_index], &mut gen_moves);

        let queen_position = get_lsb_indices(self.queens[player_index]);
        bishop_moves(queen_position, self.occupied, self.all_pieces[opp_index], &mut gen_moves);

        match self.active_player {
            Side::WHITE => {
                white_pawn_moves(self.pawns[player_index], self.occupied, 
                    self.all_pieces[opp_index], self.en_passant, &mut gen_moves);
            },
            Side::BLACK => {
                black_pawn_moves(self.pawns[player_index], self.occupied, 
                    self.all_pieces[opp_index], self.en_passant, &mut gen_moves);
            }
        }

        gen_moves
    }

    // helper method for move piece
    fn _move_piece(&mut self, move_command: Move) {
        let move_piece = self.mailbox[move_command.startSq];

        let player = piece_player(move_piece);
        let player_index = if player == Side::WHITE { 0 } else { 1 }; 

        match move_piece {
            Piece::WPAWN => {
                self.pawns[player_index] ^= 1u64 << move_command.startSq;
                self.pawns[player_index] ^= 1u64 << move_command.endSq;
            },
            Piece::BPAWN => {
                self.pawns[player_index] ^= 1u64 << move_command.startSq;
                self.pawns[player_index] ^= 1u64 << move_command.endSq;
            },
            Piece::WBISHOP | Piece::BBISHOP => {
                self.bishops[player_index] ^= 1u64 << move_command.startSq;
                self.bishops[player_index] ^= 1u64 << move_command.endSq;
            },
            Piece::WKNIGHT | Piece::BKNIGHT => {
                self.knights[player_index] ^= 1u64 << move_command.startSq;
                self.knights[player_index] ^= 1u64 << move_command.endSq;
            },
            Piece::WROOK | Piece::BROOK => {
                self.rooks[player_index] ^= 1u64 << move_command.startSq;
                self.rooks[player_index] ^= 1u64 << move_command.endSq;
            },
            Piece::WQUEEN | Piece::BQUEEN => {
                self.queens[player_index] ^= 1u64 << move_command.startSq;
                self.queens[player_index] ^= 1u64 << move_command.endSq;
            },
            Piece::WKING | Piece::BKING=> {
                self.kings[player_index] ^= 1u64 << move_command.startSq;
                self.kings[player_index] ^= 1u64 << move_command.endSq;
            },
            Piece::NONE => {},
        }

        // Remove Start Piece / Add End Piece
        let start_piece_type = piece_type_zobrist(self.mailbox[move_command.startSq]);
        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[start_piece_type][move_command.startSq];

        let end_piece_type = piece_type_zobrist(self.mailbox[move_command.endSq]);
        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[end_piece_type][move_command.endSq];

        self.mailbox[move_command.startSq] = Piece::NONE;
        self.mailbox[move_command.endSq] = move_piece;

        self.all_pieces[player_index] &= !(1u64 << move_command.startSq);
        self.all_pieces[player_index] |= 1u64 << move_command.endSq;

        self.occupied &= !(1u64 << move_command.startSq);
        self.occupied |= 1u64 << move_command.endSq;
    }

    // helper method for remove piece
    fn _remove_piece(&mut self, remove_sq: usize) {
        let remove_piece = self.mailbox[remove_sq];

        let player = piece_player(remove_piece);
        let player_index = if player == Side::WHITE { 0 } else { 1 }; 

        match remove_piece {
            Piece::WPAWN | Piece::BPAWN => {
                self.pawns[player_index] ^= 1u64 << remove_sq;
            },
            Piece::WBISHOP | Piece::BBISHOP => {
                self.bishops[player_index] ^= 1u64 << remove_sq;
            },
            Piece::WKNIGHT | Piece::BKNIGHT => {
                self.knights[player_index] ^= 1u64 << remove_sq;
            },
            Piece::WROOK | Piece::BROOK => {
                self.rooks[player_index] ^= 1u64 << remove_sq;
            },
            Piece::WQUEEN | Piece::BQUEEN => {
                self.queens[player_index] ^= 1u64 << remove_sq;
            },
            Piece::WKING | Piece::BKING=> {
                self.kings[player_index] ^= 1u64 << remove_sq;
            },
            Piece::NONE => {},
        }
        // Update Zobrist
        let remove_piece_type = piece_type_zobrist(self.mailbox[remove_sq]);
        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[remove_piece_type][remove_sq];

        self.mailbox[remove_sq] = Piece::NONE;
        self.all_pieces[player_index] ^= 1u64 << remove_sq;
        self.occupied ^= 1u64 << remove_sq;
    }

    fn _place_piece(&mut self, place_sq: usize, piece_type: Piece) {        
        let player = piece_player(piece_type);
        let player_index = if player == Side::WHITE { 0 } else { 1 }; 

        match piece_type {
            Piece::WPAWN => {
                self.pawns[player_index] ^= 1u64 << place_sq;
            },
            Piece::BPAWN => {
                self.pawns[player_index] ^= 1u64 << place_sq;
            },
            Piece::WBISHOP | Piece::BBISHOP => {
                self.bishops[player_index] ^= 1u64 << place_sq;
            },
            Piece::WKNIGHT | Piece::BKNIGHT => {
                self.knights[player_index] ^= 1u64 << place_sq;
            },
            Piece::WROOK | Piece::BROOK => {
                self.rooks[player_index] ^= 1u64 << place_sq;
            },
            Piece::WQUEEN | Piece::BQUEEN => {
                self.queens[player_index] ^= 1u64 << place_sq;
            },
            Piece::WKING | Piece::BKING=> {
                self.kings[player_index] ^= 1u64 << place_sq;
            },
            Piece::NONE => {},
        }

        // Update Zobrist
        let add_piece_type = piece_type_zobrist(piece_type);
        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[add_piece_type][place_sq];

        self.mailbox[place_sq] = piece_type;
        self.all_pieces[player_index] ^= 1u64 << place_sq;
        self.occupied ^= 1u64 << place_sq;
    }

    // Used Prior / After Execute Move/ Undo
    fn zobrist_xor(&mut self) {
        self.zobrist_hash ^= ZOBRIST_SIDE_TO_MOVE[active_player_zobrist(self.active_player)];
        self.zobrist_hash ^= ZOBRIST_EN_PASSANT[en_passant_zobrist(self.en_passant)];
        self.zobrist_hash ^= ZOBRIST_CASTLING[self.castling_rights as usize];
    }

    pub fn execute_move(&mut self, move_command: Move) -> Option<Piece> {
        // XOR the current State for Castle, En Passant and Side to Move
        self.zobrist_xor();

        let mut remove_piece = None;
        match move_command.moveType { 
            MoveFlag::CAPTURE | MoveFlag::PROMOTION => {
                remove_piece = Some(self.mailbox[move_command.endSq]);
            },
            MoveFlag::ENPASSANT => {
                match self.active_player {
                    Side::WHITE => {
                        remove_piece = Some(self.mailbox[move_command.endSq - 8]);
                    },
                    Side::BLACK => {
                        remove_piece = Some(self.mailbox[move_command.endSq + 8]);
                    },
                }
            },
            _ => {},
        }

        // Clear the Previous En Passant
        self.en_passant = 0;
        self.total_moves += 1;

        match move_command.moveType {
            MoveFlag::MOVE => {
                // Update En Passant
                let piece = self.mailbox[move_command.startSq];
                if (piece == Piece::WPAWN || piece == Piece::BPAWN) && 
                (move_command.startSq as i8 - move_command.endSq as i8).abs() == 16 {
                    match self.active_player {
                        Side::WHITE => {
                            self.en_passant = 1u64 << (move_command.startSq + 8);
                        },
                        Side::BLACK => {
                            self.en_passant = 1u64 << (move_command.startSq - 8);
                        },
                    }
                }

                // Update Castle
                let target_piece = self.mailbox[move_command.startSq];
                match target_piece {
                    Piece::WKING => {
                        self.castling_rights &= !(WHITE_KINGSIDE | WHITE_QUEENSIDE);
                    },
                    Piece::BKING => {
                        self.castling_rights &= !(BLACK_KINGSIDE | BLACK_QUEENSIDE);
                    },
                    Piece::WROOK => {
                        if move_command.startSq == 7 { self.castling_rights &= !WHITE_KINGSIDE; }
                        if move_command.startSq == 0 { self.castling_rights &= !WHITE_QUEENSIDE; }
                    },
                    Piece::BROOK => {
                        if move_command.startSq == 63 { self.castling_rights &= !BLACK_KINGSIDE; }
                        if move_command.startSq == 56 { self.castling_rights &= !BLACK_QUEENSIDE; }
                    },
                    _ => {}
                }
                
                self._move_piece(move_command);
          
            },
            MoveFlag::KINGSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        self.castling_rights &= !(WHITE_KINGSIDE | WHITE_QUEENSIDE);

                        let king_move_cmd = Move { startSq: 4, endSq: 6, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = Move { startSq: 7, endSq: 5, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        self.castling_rights &= !(BLACK_KINGSIDE | BLACK_QUEENSIDE);

                        let king_move_cmd = Move { startSq: 60, endSq: 62, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = Move { startSq: 63, endSq: 61, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::QUEENSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        self.castling_rights &= !(WHITE_KINGSIDE | WHITE_QUEENSIDE);

                        let king_move_cmd = Move { startSq: 4, endSq: 2, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = Move { startSq: 0, endSq: 3, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        self.castling_rights &= !(BLACK_KINGSIDE | BLACK_QUEENSIDE);

                        let king_move_cmd = Move { startSq: 60, endSq: 58, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = Move { startSq: 56, endSq: 59, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::PROMOTION => {
                self._remove_piece(move_command.startSq);

                if self.mailbox[move_command.endSq] != Piece::NONE {
                    self._remove_piece(move_command.endSq);
                }

                match move_command.endSq {
                    0  => self.castling_rights &= !WHITE_QUEENSIDE,
                    7  => self.castling_rights &= !WHITE_KINGSIDE,
                    56 => self.castling_rights &= !BLACK_QUEENSIDE,
                    63 => self.castling_rights &= !BLACK_KINGSIDE,
                    _ => {}
                }

                match self.active_player {
                    Side::WHITE => {
                        self._place_piece(move_command.endSq, Piece::WQUEEN);
                    },
                    Side::BLACK => {
                        self._place_piece(move_command.endSq, Piece::BQUEEN);
                    },
                }
            },
            MoveFlag::ENPASSANT => {
                self._move_piece(move_command);

                match self.active_player {
                    Side::WHITE => {
                        self._remove_piece(move_command.endSq - 8);
                    },
                    Side::BLACK => {
                        self._remove_piece(move_command.endSq + 8);
                    },
                }
            },
            MoveFlag::CAPTURE => {
                match move_command.endSq {
                    0  => self.castling_rights &= !WHITE_QUEENSIDE,
                    7  => self.castling_rights &= !WHITE_KINGSIDE,
                    56 => self.castling_rights &= !BLACK_QUEENSIDE,
                    63 => self.castling_rights &= !BLACK_KINGSIDE,
                    _ => {}
                }

                self._remove_piece(move_command.endSq);
                self._move_piece(move_command);
            },
        }

        self.active_player = match self.active_player {
            Side::WHITE => Side::BLACK,
            Side::BLACK => Side::WHITE,
        };

        // XOR in current state for Castle, En Passant and Side to Move
        self.zobrist_xor();

        remove_piece
    }

    // Undo Move
    pub fn unexecute_move(&mut self, undo_move_cmd: UndoMove) {
        // XOR the current State for Castle, En Passant and Side to Move
        self.zobrist_xor();

        // Swap Active
        self.active_player = match self.active_player {
            Side::WHITE => Side::BLACK,
            Side::BLACK => Side::WHITE,
        };
        
        // Undo Move
        match undo_move_cmd.moveType {
            MoveFlag::MOVE | MoveFlag::CAPTURE => {
                let undo_command = Move { 
                    startSq: undo_move_cmd.endSq, 
                    endSq: undo_move_cmd.startSq, 
                    moveType: MoveFlag::MOVE 
                };
                self._move_piece(undo_command);
          
            },
            MoveFlag::KINGSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = Move { startSq: 6, endSq: 4, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = Move { startSq: 5, endSq: 7, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = Move { startSq: 62, endSq: 60, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = Move { startSq: 61, endSq: 63, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::QUEENSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = Move { startSq: 2, endSq: 4, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = Move { startSq: 3, endSq: 0, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = Move { startSq: 58, endSq: 60, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = Move { startSq: 59, endSq: 56, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::PROMOTION => {
                self._remove_piece(undo_move_cmd.endSq);

                match self.active_player {
                    Side::WHITE => {
                        self._place_piece(undo_move_cmd.startSq, Piece::WPAWN);
                    },
                    Side::BLACK => {
                        self._place_piece(undo_move_cmd.startSq, Piece::BPAWN);
                    },
                }
            },
            MoveFlag::ENPASSANT => {
                let undo_command = Move { 
                    startSq: undo_move_cmd.endSq, 
                    endSq: undo_move_cmd.startSq, 
                    moveType: MoveFlag::MOVE 
                };
                self._move_piece(undo_command);

                match self.active_player {
                    Side::WHITE => {
                        self._remove_piece(undo_move_cmd.endSq - 8);
                    },
                    Side::BLACK => {
                        self._remove_piece(undo_move_cmd.endSq + 8);
                    },
                }
            },
        }

        // Restore Piece only if one was actually captured
        if let Some(piece) = undo_move_cmd.capturedPiece {
            match undo_move_cmd.moveType { 
                MoveFlag::CAPTURE | MoveFlag::PROMOTION => {
                    self.mailbox[undo_move_cmd.endSq] = piece;
                },
                MoveFlag::ENPASSANT => {
                    let offset = if self.active_player == Side::WHITE { -8 } else { 8 };
                    self.mailbox[(undo_move_cmd.endSq as i32 + offset) as usize] = piece;
                },
                _ => {},
            }
        }

        // Restore En Passant
        self.en_passant = undo_move_cmd.prevEnPassant;
        self.castling_rights = undo_move_cmd.prevCastleRights;
        self.total_moves -= 1;

        // XOR in current state for Castle, En Passant and Side to Move
        self.zobrist_xor();
    }
}

fn piece_player(piece_type: Piece) -> Side {
    match piece_type {
        Piece::WPAWN | Piece::WBISHOP | Piece::WKNIGHT |
        Piece::WROOK | Piece::WQUEEN | Piece::WKING => {
            return Side::WHITE;
        },
        Piece::BPAWN | Piece::BBISHOP | Piece::BKNIGHT |
        Piece::BROOK | Piece::BQUEEN | Piece::BKING => {
            return Side::BLACK;
        },
        Piece::NONE => {
            panic!("Passed None");
        },
    }
}

fn get_lsb_indices(board: u64) -> Vec<usize> {
    let mut bitboard = board;
    let mut indices = Vec::new();
    
    while bitboard != 0 {
        let lsb_index = bitboard.trailing_zeros();
        indices.push(lsb_index as usize);
        
        bitboard &= bitboard - 1;
    }
    
    indices
}

// DEBUG
pub fn print_board(board: u64, debug_string: &str) {
    println!("{}", debug_string);
    for r in (0..8).rev() {
        for c in 0..8 {
            print!("{}", (board >> (r * 8 + c)) & 1);
        }
        println!("");
    }
}