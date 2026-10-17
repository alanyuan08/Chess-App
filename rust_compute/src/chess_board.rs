use crate::bishop_mask::*;
use crate::king_mask::*;
use crate::knight_mask::*;
use crate::pawn_mask::*;
use crate::rook_mask::*;
use crate::queen_mask::*;
use crate::move_command::*;
use crate::zobrist_hash::*;
use crate::nnue_network::*;
use crate::board_accumlator::*;
use arrayvec::ArrayVec;

// 0 -> White / 1 -> Black
#[derive(Debug, Clone)] 
pub struct ChessBoard {
    pub pawns: [u64; 2],
    pub knights: [u64; 2],
    pub bishops: [u64; 2],
    pub rooks: [u64; 2],
    pub queens: [u64; 2],
    pub kings: [u64; 2],
    
    pub all_pieces: [u64; 2],
    pub occupied: u64,
    pub mailbox: [BoardPiece; 64],

    castling_rights: u8,
    en_passant: u8,
    active_player: Side,

    zobrist_hash: u64,
    ply: usize,
    nnue_network: &'static NnueNetwork,
}

pub const WHITE_KINGSIDE: u8 = 0b0001; // 1
pub const WHITE_QUEENSIDE: u8 = 0b0010; // 2
pub const BLACK_KINGSIDE: u8 = 0b0100; // 4
pub const BLACK_QUEENSIDE: u8 = 0b1000; // 8

impl ChessBoard {
    // A constructor-like associated function
    // [0] - White / [1] - Black
    pub fn new(
        nnue_network: &'static NnueNetwork
    ) -> Self {
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
            en_passant: 64,
            active_player: Side::WHITE,

            mailbox: [BoardPiece::NONE; 64],
            zobrist_hash: 0,

            ply: 0,
            nnue_network,
        }
    }

    pub fn init_board(&mut self, accumulators: &mut AccumulatorHistory) {
        for color in 0..2 {
            // Ranks: White = 0 & 1, Black = 6 & 7
            let piece_rank_offset = if color == 0 { 0 } else { 56 };
            let pawn_rank_offset  = if color == 0 { 8 } else { 48 };

            // 1. Initialize Pawns
            for i in 0..8 {
                let sq = pawn_rank_offset + i;
                self.pawns[color] |= 1u64 << sq;
                self.mailbox[sq] = if color == 0 { BoardPiece::WPAWN } else { BoardPiece::BPAWN };
            }

            // 2. Initialize Major Pieces (Bitboards)
            self.rooks[color]   |= (1u64 << piece_rank_offset) | (1u64 << (piece_rank_offset + 7));
            self.knights[color] |= (1u64 << (piece_rank_offset + 1)) | (1u64 << (piece_rank_offset + 6));
            self.bishops[color] |= (1u64 << (piece_rank_offset + 2)) | (1u64 << (piece_rank_offset + 5));
            self.queens[color]  |= 1u64 << (piece_rank_offset + 3);
            self.kings[color]   |= 1u64 << (piece_rank_offset + 4);

            // 3. Initialize Major Pieces (Mailbox)
            let pieces = if color == 0 {
                [BoardPiece::WROOK, BoardPiece::WKNIGHT, BoardPiece::WBISHOP, BoardPiece::WQUEEN, 
                BoardPiece::WKING, BoardPiece::WBISHOP, BoardPiece::WKNIGHT, BoardPiece::WROOK]
            } else {
                [BoardPiece::BROOK, BoardPiece::BKNIGHT, BoardPiece::BBISHOP, BoardPiece::BQUEEN, 
                BoardPiece::BKING, BoardPiece::BBISHOP, BoardPiece::BKNIGHT, BoardPiece::BROOK]
            };

            self.mailbox[piece_rank_offset..8 + piece_rank_offset].copy_from_slice(&pieces);

            // 4. Composite Bitboards
            self.all_pieces[color] = self.pawns[color] | self.rooks[color] | 
                                    self.knights[color] | self.bishops[color] | 
                                    self.queens[color] | self.kings[color];
            self.occupied |= self.all_pieces[color];

            // 5. Compute Zobrist
            self.zobrist_hash = self.compute_init_zobrist();
        }

        // Init Accumulator
        self.create_accumulator_from_scratch(accumulators);    
    }

    // Null Move Pruning Zugzwang
    #[inline]
    pub fn has_major_pieces(&self) -> bool {
        let side_idx = self.active_player as usize;

        // Check only the active player's pieces
        (self.knights[side_idx] 
            | self.bishops[side_idx] 
            | self.rooks[side_idx] 
            | self.queens[side_idx]) != 0
    }

    // Compute Init Zobritist
    pub fn compute_init_zobrist(&self) -> u64 { 
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

    // Return Opponent 
    #[inline]
    pub fn opponent_player(&self) -> Side {
        match self.active_player {
            Side::WHITE => {
                Side::BLACK
            },
            Side::BLACK => {
                Side::WHITE
            }
        }
    }

    // Return Index
    #[inline]
    pub fn player_index(&self, player_side: Side) -> usize {
        match player_side {
            Side::WHITE => {
                0
            },
            Side::BLACK => {
                1
            }
        }
    }

    // Return Castle Rights
    #[inline]
    pub fn castle_rights(&self) -> u8 {   
        self.castling_rights
    }

    // Return En Passant
    #[inline]
    pub fn en_passant(&self) -> u8 {   
        self.en_passant
    }

    // Return Zobrist Hash
    #[inline]
    pub fn zobrist_hash(&self) -> u64 {   
        self.zobrist_hash
    }

    // Return Active Player
    #[inline]
    pub fn active_player(&self) -> Side {   
        self.active_player
    }

    // Return Mailbox Piece
    #[inline]
    pub fn mailbox_piece(&self, target: u8) -> BoardPiece {   
        self.mailbox[target as usize]
    }

    // Used to Calculate Castling / King Safety
    pub fn compute_attack_targets(&self, attacking_side: Side) -> u64 {
        let mut attacks = 0u64;
        let index = self.player_index(attacking_side);
        
        let occ = self.occupied;

        // 1. Pawns
        let pawns = self.pawns[index];
        if attacking_side == Side::BLACK {  
            attacks |= black_pawn_attacks(pawns);
        } else {
            attacks |= white_pawn_attacks(pawns);
        }

        // 2. Knights
        let mut knights = self.knights[index];
        while knights != 0 {
            attacks |= KNIGHT_ATTACKS[knights.trailing_zeros() as usize];
            knights &= knights - 1;
        }

        // 3. Kings
        let mut kings = self.kings[index];
        while kings != 0 {
            attacks |= KING_ATTACKS[kings.trailing_zeros() as usize];
            kings &= kings - 1;
        }

        // 4. Sliders (Bishops, Rooks, Queens)
        let mut bishops = self.bishops[index] | self.queens[index];
        while bishops != 0 {
            attacks |= bishop_attack_paths(bishops.trailing_zeros() as u8, occ);
            bishops &= bishops - 1;
        }

        let mut rooks = self.rooks[index] | self.queens[index];
        while rooks != 0 {
            attacks |= rook_attack_paths(rooks.trailing_zeros() as u8, occ);
            rooks &= rooks - 1;
        }

        attacks
    }

    // Check if current player can capture opponent King
    pub fn is_previous_player_king_in_check(&mut self) -> bool {
        let _curr_attack_targets = self.compute_attack_targets(self.active_player);

        let opponent_index = self.player_index(self.opponent_player());
        (self.kings[opponent_index] & _curr_attack_targets) != 0
    }

    // Check if current board is in check
    pub fn is_in_check(&mut self) -> bool {
        let _curr_opp_attack_targets = self.compute_attack_targets(self.opponent_player());

        let current_player_index = self.player_index(self.active_player);
        (self.kings[current_player_index] & _curr_opp_attack_targets) != 0
    }

    // Generate Pseudo-Moves - Only Validate King Safety for Castle / King Movement
    pub fn generate_moves(&mut self, 
        gen_moves: &mut ArrayVec::<ForwardMove, 256>, 
        pv_move_hint: ForwardMove,
        ply_idx: i16,
        killer_move_table: &[[ForwardMove; 2]; 128],
        only_tactical: bool, 
    ) {        
        let player_index = self.player_index(self.active_player);
        let opp_index = self.player_index(self.opponent_player());
        let _opponent_attack_targets = self.compute_attack_targets(self.opponent_player());

        // Generate Moves
        king_moves(self, player_index, opp_index, _opponent_attack_targets, gen_moves, only_tactical);

        knight_moves(self, player_index, opp_index, gen_moves, only_tactical);

        rook_moves(self, player_index, opp_index, gen_moves, only_tactical);

        bishop_moves(self, player_index, opp_index, gen_moves, only_tactical);

        queen_moves(self, player_index, opp_index, gen_moves, only_tactical);

        match self.active_player {
            Side::WHITE => {
                white_pawn_moves(self, player_index, opp_index, gen_moves, only_tactical);
            },
            Side::BLACK => {
                black_pawn_moves(self, player_index, opp_index, gen_moves, only_tactical);
            }
        }
        
        if ply_idx < 128 {
            let has_valid_pv = pv_move_hint.move_type != MoveFlag::NULL;
            let k0 = killer_move_table[ply_idx as usize][0];
            let k1 = killer_move_table[ply_idx as usize][1];
            
            let has_k0 = k0.move_type != MoveFlag::NULL && (!has_valid_pv || k0 != pv_move_hint);
            let has_k1 = k1.move_type != MoveFlag::NULL && (!has_valid_pv || k1 != pv_move_hint);

            for m in gen_moves.iter_mut() {
                if has_valid_pv && *m == pv_move_hint {
                    m.pv_score = 32_000;
                } else if has_k0 && *m == k0 {
                    m.pv_score = 500;
                } else if has_k1 && *m == k1 {
                    m.pv_score = 400;
                }
            }
        }

        gen_moves.sort_unstable_by_key(|cmd| -cmd.pv_score);
    }

    // helper method for move piece
    fn _move_piece(&mut self, move_command: ForwardMove) {
        // Remove Start Piece / Add End Piece
        let piece_type = piece_type_zobrist(self.mailbox[move_command.start_sq as usize]);

        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[piece_type][move_command.start_sq as usize];
        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[piece_type][move_command.end_sq as usize];

        let move_piece = self.mailbox[move_command.start_sq as usize];
        let player_index = self.player_index(piece_player(move_piece));

        match move_piece {
            BoardPiece::WPAWN => {
                self.pawns[player_index] ^= 1u64 << move_command.start_sq;
                self.pawns[player_index] ^= 1u64 << move_command.end_sq;
            },
            BoardPiece::BPAWN => {
                self.pawns[player_index] ^= 1u64 << move_command.start_sq;
                self.pawns[player_index] ^= 1u64 << move_command.end_sq;
            },
            BoardPiece::WBISHOP | BoardPiece::BBISHOP => {
                self.bishops[player_index] ^= 1u64 << move_command.start_sq;
                self.bishops[player_index] ^= 1u64 << move_command.end_sq;
            },
            BoardPiece::WKNIGHT | BoardPiece::BKNIGHT => {
                self.knights[player_index] ^= 1u64 << move_command.start_sq;
                self.knights[player_index] ^= 1u64 << move_command.end_sq;
            },
            BoardPiece::WROOK | BoardPiece::BROOK => {
                self.rooks[player_index] ^= 1u64 << move_command.start_sq;
                self.rooks[player_index] ^= 1u64 << move_command.end_sq;
            },
            BoardPiece::WQUEEN | BoardPiece::BQUEEN => {
                self.queens[player_index] ^= 1u64 << move_command.start_sq;
                self.queens[player_index] ^= 1u64 << move_command.end_sq;
            },
            BoardPiece::WKING | BoardPiece::BKING=> {
                self.kings[player_index] ^= 1u64 << move_command.start_sq;
                self.kings[player_index] ^= 1u64 << move_command.end_sq;
            },
            BoardPiece::NONE => {
                println!("Tried to move empty");
            },
        }

        self.mailbox[move_command.start_sq as usize] = BoardPiece::NONE;
        self.mailbox[move_command.end_sq as usize] = move_piece;

        self.all_pieces[player_index] &= !(1u64 << move_command.start_sq);
        self.all_pieces[player_index] |= 1u64 << move_command.end_sq;

        self.occupied &= !(1u64 << move_command.start_sq);
        self.occupied |= 1u64 << move_command.end_sq;
    }

    // helper method for remove piece
    fn _remove_piece(&mut self, remove_sq: usize) {
        // Update Zobrist
        let remove_piece_type = piece_type_zobrist(self.mailbox[remove_sq]);
        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[remove_piece_type][remove_sq];

        let remove_piece = self.mailbox[remove_sq];
        let player_index = self.player_index(piece_player(remove_piece));

        match remove_piece {
            BoardPiece::WPAWN | BoardPiece::BPAWN => {
                self.pawns[player_index] ^= 1u64 << remove_sq;
            },
            BoardPiece::WBISHOP | BoardPiece::BBISHOP => {
                self.bishops[player_index] ^= 1u64 << remove_sq;
            },
            BoardPiece::WKNIGHT | BoardPiece::BKNIGHT => {
                self.knights[player_index] ^= 1u64 << remove_sq;
            },
            BoardPiece::WROOK | BoardPiece::BROOK => {
                self.rooks[player_index] ^= 1u64 << remove_sq;
            },
            BoardPiece::WQUEEN | BoardPiece::BQUEEN => {
                self.queens[player_index] ^= 1u64 << remove_sq;
            },
            BoardPiece::WKING | BoardPiece::BKING=> {
                self.kings[player_index] ^= 1u64 << remove_sq;
            },
            BoardPiece::NONE => {
                println!("Tried to remove empty");
            },
        }

        self.mailbox[remove_sq] = BoardPiece::NONE;
        self.all_pieces[player_index] ^= 1u64 << remove_sq;
        self.occupied ^= 1u64 << remove_sq;
    }

    fn _place_piece(&mut self, place_sq: usize, piece_type: BoardPiece) {   
        // Update Zobrist
        let add_piece_type = piece_type_zobrist(piece_type);
        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[add_piece_type][place_sq];

        let player_index = self.player_index(piece_player(piece_type));

        match piece_type {
            BoardPiece::WPAWN => {
                self.pawns[player_index] ^= 1u64 << place_sq;
            },
            BoardPiece::BPAWN => {
                self.pawns[player_index] ^= 1u64 << place_sq;
            },
            BoardPiece::WBISHOP | BoardPiece::BBISHOP => {
                self.bishops[player_index] ^= 1u64 << place_sq;
            },
            BoardPiece::WKNIGHT | BoardPiece::BKNIGHT => {
                self.knights[player_index] ^= 1u64 << place_sq;
            },
            BoardPiece::WROOK | BoardPiece::BROOK => {
                self.rooks[player_index] ^= 1u64 << place_sq;
            },
            BoardPiece::WQUEEN | BoardPiece::BQUEEN => {
                self.queens[player_index] ^= 1u64 << place_sq;
            },
            BoardPiece::WKING | BoardPiece::BKING=> {
                self.kings[player_index] ^= 1u64 << place_sq;
            },
            BoardPiece::NONE => {},
        }

        self.mailbox[place_sq] = piece_type;
        self.all_pieces[player_index] ^= 1u64 << place_sq;
        self.occupied ^= 1u64 << place_sq;
    }

    // Used Prior / After Execute Move/ Undo
    #[inline(always)]
    fn zobrist_xor(&mut self) {
        self.zobrist_hash ^= ZOBRIST_SIDE_TO_MOVE[active_player_zobrist(self.active_player)];
        self.zobrist_hash ^= ZOBRIST_EN_PASSANT[en_passant_zobrist(self.en_passant)];
        self.zobrist_hash ^= ZOBRIST_CASTLING[self.castling_rights as usize];
    }

    pub fn execute_move(&mut self, move_command: ForwardMove) -> BoardPiece {
        // XOR the current State for Castle, En Passant and Side to Move
        self.zobrist_xor();

        let mut remove_piece = BoardPiece::NONE;
        // Store Removed Piece / No bitboard Operations
        match move_command.move_type { 
            MoveFlag::CAPTURE | MoveFlag::PROMOTIONQUEEN |
            MoveFlag::PROMOTIONROOK | MoveFlag::PROMOTIONBISHOP | 
            MoveFlag::PROMOTIONKNIGHT => {
                if self.mailbox[move_command.end_sq as usize] != BoardPiece::NONE {
                    remove_piece = self.mailbox[move_command.end_sq as usize];
                }
            },
            MoveFlag::ENPASSANT => {
                match self.active_player {
                    Side::WHITE => {
                        remove_piece = self.mailbox[(move_command.end_sq - 8) as usize];
                    },
                    Side::BLACK => {
                        remove_piece = self.mailbox[(move_command.end_sq + 8) as usize];
                    },
                }
            },
            _ => {},
        }

        // Clear the Previous En Passant
        self.en_passant = 64;

        // Update Castling
        match move_command.start_sq {
            4  => self.castling_rights &= !(WHITE_KINGSIDE | WHITE_QUEENSIDE),
            60 => self.castling_rights &= !(BLACK_KINGSIDE | BLACK_QUEENSIDE),
            7  => self.castling_rights &= !WHITE_KINGSIDE,
            0  => self.castling_rights &= !WHITE_QUEENSIDE,
            63 => self.castling_rights &= !BLACK_KINGSIDE,
            56 => self.castling_rights &= !BLACK_QUEENSIDE,
            _ => {}
        }
        match move_command.end_sq {
            7  => self.castling_rights &= !WHITE_KINGSIDE,
            0  => self.castling_rights &= !WHITE_QUEENSIDE,
            63 => self.castling_rights &= !BLACK_KINGSIDE,
            56 => self.castling_rights &= !BLACK_QUEENSIDE,
            _ => {}
        }

        match move_command.move_type {
            MoveFlag::MOVE => {
                self._move_piece(move_command);
            },
            MoveFlag::PAWNOPENMOVE => {
                // Update En Passant
                let piece = self.mailbox[move_command.start_sq as usize];
                if (piece == BoardPiece::WPAWN || piece == BoardPiece::BPAWN) && 
                (move_command.start_sq as i8 - move_command.end_sq as i8).abs() == 16 {
                    match self.active_player {
                        Side::WHITE => {
                            self.en_passant = move_command.start_sq + 8;
                        },
                        Side::BLACK => {
                            self.en_passant = move_command.start_sq - 8;
                        },
                    }
                }
                
                self._move_piece(move_command);
            },
            MoveFlag::KINGSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 4, end_sq: 6, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 7, end_sq: 5, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 60, end_sq: 62, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 63, end_sq: 61, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::QUEENSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 4, end_sq: 2, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 0, end_sq: 3, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 60, end_sq: 58, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 56, end_sq: 59, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::PROMOTIONQUEEN | MoveFlag::PROMOTIONROOK |
            MoveFlag::PROMOTIONBISHOP | MoveFlag::PROMOTIONKNIGHT => {
                self._remove_piece(move_command.start_sq as usize);

                if remove_piece != BoardPiece::NONE {
                    self._remove_piece(move_command.end_sq as usize);
                }

                match self.active_player {
                    Side::WHITE => {
                        self._place_piece(move_command.end_sq as usize, 
                            white_promotion_piece(move_command.move_type)
                        );
                    },
                    Side::BLACK => {
                        self._place_piece(move_command.end_sq as usize, 
                            black_promotion_piece(move_command.move_type)
                        );
                    },
                }
            },
            MoveFlag::ENPASSANT => {
                self._move_piece(move_command);
                match self.active_player {
                    Side::WHITE => {
                        self._remove_piece((move_command.end_sq - 8) as usize);
                    },
                    Side::BLACK => {
                        self._remove_piece((move_command.end_sq + 8) as usize);
                    },
                }
            },
            MoveFlag::CAPTURE => {
                self._remove_piece(move_command.end_sq as usize);
                self._move_piece(move_command);
            },
            MoveFlag::NULL => {},
        }

        self.active_player = self.opponent_player();

        // XOR in current state for Castle, En Passant and Side to Move
        self.zobrist_xor();

        remove_piece
    }

    // Undo Move
    pub fn unexecute_move(&mut self, undo_move_cmd: UndoMove) {
        // XOR the current State for Castle, En Passant and Side to Move
        self.zobrist_xor();

        // Swap Active
        self.active_player = self.opponent_player();
        
        // Undo Move
        match undo_move_cmd.move_type {
            MoveFlag::MOVE | MoveFlag::CAPTURE | MoveFlag::PAWNOPENMOVE | MoveFlag::ENPASSANT=> {
                let undo_command = ForwardMove { 
                    start_sq: undo_move_cmd.end_sq, 
                    end_sq: undo_move_cmd.start_sq, 
                    move_type: MoveFlag::MOVE, 
                    pv_score: 0,
                };
                self._move_piece(undo_command);
          
            },
            MoveFlag::KINGSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 6, end_sq: 4, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 5, end_sq: 7, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 62, end_sq: 60, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 61, end_sq: 63, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::QUEENSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 2, end_sq: 4, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 3, end_sq: 0, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 58, end_sq: 60, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 59, end_sq: 56, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::PROMOTIONQUEEN | MoveFlag::PROMOTIONROOK | 
            MoveFlag::PROMOTIONBISHOP | MoveFlag::PROMOTIONKNIGHT => {
                self._remove_piece(undo_move_cmd.end_sq as usize);

                match self.active_player {
                    Side::WHITE => {
                        self._place_piece(undo_move_cmd.start_sq as usize, BoardPiece::WPAWN);
                    },
                    Side::BLACK => {
                        self._place_piece(undo_move_cmd.start_sq as usize, BoardPiece::BPAWN);
                    },
                }
            },
            MoveFlag::NULL => {},
        }

        // Restore Piece only if one was actually captured
        if undo_move_cmd.captured_piece != BoardPiece::NONE {
            match undo_move_cmd.move_type { 
                MoveFlag::CAPTURE | MoveFlag::PROMOTIONQUEEN | 
                MoveFlag::PROMOTIONROOK | MoveFlag::PROMOTIONBISHOP | 
                MoveFlag::PROMOTIONKNIGHT => {
                    self._place_piece(undo_move_cmd.end_sq as usize, undo_move_cmd.captured_piece);
                },
                MoveFlag::ENPASSANT => {
                    let ep_square = if self.active_player == Side::WHITE { 
                        undo_move_cmd.end_sq - 8 } else { undo_move_cmd.end_sq + 8 };
                    self._place_piece(ep_square as usize, undo_move_cmd.captured_piece);
                },
                _ => {},
            }
        }

        // Restore En Passant
        self.en_passant = undo_move_cmd.prev_en_passant;
        self.castling_rights = undo_move_cmd.prev_castle_rights;

        // XOR in current state for Castle, En Passant and Side to Move
        self.zobrist_xor();
    }

    #[inline(always)]
    pub fn null_move_accumulator(&mut self, accumulators: &mut AccumulatorHistory) {
        let curr_ply = self.ply as usize;
        let prev_ply = curr_ply - 1; 

        let (left, right) = accumulators.moves.split_at_mut(curr_ply);
        let prev_frame = &left[prev_ply];
        let curr_frame = &mut right[0];

        curr_frame.white.copy_from_slice(&prev_frame.white);
        curr_frame.black.copy_from_slice(&prev_frame.black);
    }

    // Create from Scratch
    pub fn create_accumulator_from_scratch(&mut self, accumulators: &mut AccumulatorHistory) {
        // Retrieve King Squares
        let w_king_sq = self.kings[Side::WHITE as usize].trailing_zeros() as u8;
        let b_king_sq = self.kings[Side::BLACK as usize].trailing_zeros() as u8;
        
        // Grab current accumulator frame using explicit slice indexing on the Boxed array
        let current_frame = &mut accumulators.moves[self.ply];
        let target_white: &mut [i16; 256] = &mut current_frame.white;
        let target_black: &mut [i16; 256] = &mut current_frame.black;

        // Initialize both sides with biases first (casting i32 to i16)
        let biases = &self.nnue_network.l1_biases;

        // Using zip allows the compiler to optimize the loop boundaries and apply SIMD
        for (i, &bias) in biases.iter().enumerate().take(256) {
            let casted_bias = bias as i16;
            target_white[i] = casted_bias;
            target_black[i] = casted_bias;
        }

        // Combine ALL piece bitboards
        let mut pieces_bitboard = (self.pawns[0] | self.pawns[1])

            | (self.knights[0] | self.knights[1])
            | (self.bishops[0] | self.bishops[1])
            | (self.rooks[0] | self.rooks[1])
            | (self.queens[0] | self.queens[1])
            | (self.kings[0] | self.kings[1]);

        let mut white_indices = [0usize; 32];
        let mut black_indices = [0usize; 32];
        let mut active_piece_count = 0;

        // --- SINGLE PASS: GATHER ALL FEATURE INDICES ---
        while pieces_bitboard != 0 {
            let sq = pieces_bitboard.trailing_zeros() as u8;
            let piece = self.mailbox[sq as usize];

            white_indices[active_piece_count] = get_feature_index(w_king_sq, piece, sq, false);
            black_indices[active_piece_count] = get_feature_index(b_king_sq, piece, sq, true);
            
            active_piece_count += 1;
            pieces_bitboard &= pieces_bitboard - 1;
        }

        // --- LOOP 1: PROCESS WHITE PERSPECTIVE COMPLETELY ---
        // Exact [i16; 256] array sizing unlocks aggressive SIMD/AVX2 loop unrolling
        for &w_idx in white_indices.iter().take(active_piece_count) {
            let w_row: &[i16; 256] = &self.nnue_network.l1_weights[w_idx];
            
            for i in 0..256 {
                target_white[i] = target_white[i].wrapping_add(w_row[i]);
            }
        }

        // --- LOOP 2: PROCESS BLACK PERSPECTIVE COMPLETELY ---
        for &b_idx in black_indices.iter().take(active_piece_count) {
            let b_row: &[i16; 256] = &self.nnue_network.l1_weights[b_idx];
            
            for i in 0..256 {
                target_black[i] = target_black[i].wrapping_add(b_row[i]);
            }
        }
    }

    pub fn evaluate(&mut self, accumulators: &AccumulatorHistory, buffer: &mut NnueInferenceBuffer) -> i16 {
        // --- PERSPECTIVE ROUTING ---
        // Side to move (US) always fills the first 256 inputs.
        // Opponent (THEM) always fills the second 256 inputs.
        let (active_acc, opp_acc) = match self.active_player {
            Side::WHITE => (
                &accumulators.moves[self.ply as usize].white, 
                &accumulators.moves[self.ply as usize].black
            ),
            Side::BLACK => (
                &accumulators.moves[self.ply as usize].black, 
                &accumulators.moves[self.ply as usize].white
            ),
        };

        // --- STEP 0: ACCUMULATOR (Input -> Accumulator) ---
        // Coerce destination into exact fixed-size slices to completely eliminate bounds checks
        let (l2_left, l2_right) = buffer.l2_inputs.split_at_mut(256);
        let l2_left: &mut [i8; 256] = l2_left.try_into().unwrap();
        let l2_right: &mut [i8; 256] = l2_right.try_into().unwrap();

        // --- STEP 1: CONCATENATION & ACTIVATION (L1 -> L2 / Clipped ReLU) ---
        for i in 0..256 {
            l2_left[i] = active_acc[i].clamp(0, 127) as i8;
        }
        for i in 0..256 {
            l2_right[i] = opp_acc[i].clamp(0, 127) as i8;
        }

        // --- STEP 2: HIDDEN LAYER 2 (512 -> 32) ---
        let l2_biases = &self.nnue_network.l2_biases;
        let l2_inputs: &[i8; 512] = (&buffer.l2_inputs).try_into().unwrap();
        let l3_inputs: &mut [i8; 32] = (&mut buffer.l3_inputs).try_into().unwrap();

        for neuron in 0..32 {
            let mut sum: i32 = l2_biases[neuron];
            let row: &[i8; 512] = &self.nnue_network.l2_weights[neuron];

            // 512 iterations with exact-size arrays: compiler unrolls this beautifully via SIMD/AVX2
            for i in 0..512 {
                sum += (l2_inputs[i] as i32) * (row[i] as i32);
            }

            let activated = sum >> 7;
            l3_inputs[neuron] = activated.clamp(0, 32) as i8;
        }

        // --- STEP 3: HIDDEN LAYER 3 (32 -> 32) ---
        let l3_biases = &self.nnue_network.l3_biases;
        let l4_inputs: &mut [i8; 32] = (&mut buffer.l4_inputs).try_into().unwrap();

        for neuron in 0..32 {
            let mut sum: i32 = l3_biases[neuron];
            let row: &[i8; 32] = &self.nnue_network.l3_weights[neuron];

            for i in 0..32 {
                sum += (l3_inputs[i] as i32) * (row[i] as i32);
            }

            let activated = sum >> 5;
            l4_inputs[neuron] = activated.clamp(0, 32) as i8;
        }

        // --- STEP 4: OUTPUT LAYER (32 -> 1) ---
        let mut final_sum: i32 = self.nnue_network.output_bias[0];
        let row: &[i8; 32] = &self.nnue_network.output_weights[0];

        for i in 0..32 {
            final_sum += (l4_inputs[i] as i32) * (row[i] as i32);
        }

        let internal_pawns_scaled = final_sum >> 5;

        // Shift by >> 7 to remove remaining scale context factors
        ((internal_pawns_scaled * 100) >> 7) as i16
    }

    /// Progresses the network forward incrementally during move making
    pub fn make_move(&mut self, mv: ForwardMove, accumulators: &mut AccumulatorHistory) {
        // Retrieve King Squares
        let w_king_sq = self.kings[Side::WHITE as usize].trailing_zeros() as u8;
        let b_king_sq = self.kings[Side::BLACK as usize].trailing_zeros() as u8;

        let move_piece: BoardPiece = self.mailbox[mv.start_sq as usize];

        // --- 1. Identify Target Added Piece (Handles Promotions) ---
        let mut added_piece = move_piece;
        match mv.move_type {
            MoveFlag::PROMOTIONQUEEN => {
                added_piece = if move_piece == BoardPiece::WPAWN { BoardPiece::WQUEEN } else { BoardPiece::BQUEEN };
            }
            MoveFlag::PROMOTIONROOK => {
                added_piece = if move_piece == BoardPiece::WPAWN { BoardPiece::WROOK } else { BoardPiece::BROOK };
            }
            MoveFlag::PROMOTIONBISHOP => {
                added_piece = if move_piece == BoardPiece::WPAWN { BoardPiece::WBISHOP } else { BoardPiece::BBISHOP };
            }
            MoveFlag::PROMOTIONKNIGHT => {
                added_piece = if move_piece == BoardPiece::WPAWN { BoardPiece::WKNIGHT } else { BoardPiece::BKNIGHT };
            }
            _ => {}
        }

         // --- 2. Identify Captured Piece & Coordinate (Handles En Passant) ---
        let (captured_sq, captured_piece) = if mv.move_type == MoveFlag::ENPASSANT {
            let sq = if move_piece == BoardPiece::WPAWN { mv.end_sq - 8 } else { mv.end_sq + 8 };
            (sq, self.mailbox[sq as usize])
        } else {
            (mv.end_sq, self.mailbox[mv.end_sq as usize])
        };

        // --- 3. Compute Sparse Feature Indices for the Moving Piece ---
        let w_remove = get_feature_index(w_king_sq, move_piece, mv.start_sq, false);
        let b_remove = get_feature_index(b_king_sq, move_piece, mv.start_sq, true);

        let w_add = get_feature_index(w_king_sq, added_piece, mv.end_sq, false);
        let b_add = get_feature_index(b_king_sq, added_piece, mv.end_sq, true);

        let w_rem_row: &[i16; 256] = &self.nnue_network.l1_weights[w_remove];
        let b_rem_row: &[i16; 256] = &self.nnue_network.l1_weights[b_remove];
        let w_add_row: &[i16; 256] = &self.nnue_network.l1_weights[w_add];
        let b_add_row: &[i16; 256] = &self.nnue_network.l1_weights[b_add];

        // --- 4. Setup Target and Source Frame Boundaries Safely ---
        let curr_ply = self.ply;
        let prev_ply = curr_ply - 1;

        // Split the mutable array slice to satisfy the borrow checker cleanly
        let (left, right) = accumulators.moves.split_at_mut(curr_ply);
        let prev_acc = &left[prev_ply];
        let curr_acc = &mut right[0];

        let curr_white: &mut [i16; 256] = &mut curr_acc.white;
        let curr_black: &mut [i16; 256] = &mut curr_acc.black;
        
        let prev_white: &[i16; 256] = &prev_acc.white;
        let prev_black: &[i16; 256] = &prev_acc.black;

        // --- 5. High-Density Auto-Vectorized Accumulation Pass ---
        if captured_piece != BoardPiece::NONE {
            let w_cap = get_feature_index(w_king_sq, captured_piece, captured_sq, false);
            let b_cap = get_feature_index(b_king_sq, captured_piece, captured_sq, true);
            
            let w_cap_row: &[i16; 256] = &self.nnue_network.l1_weights[w_cap];
            let b_cap_row: &[i16; 256] = &self.nnue_network.l1_weights[b_cap];

            // Single loop updates both perspectives at once, minimizing cache misses
            for i in 0..256 {
                curr_white[i] = prev_white[i]
                    .wrapping_add(w_add_row[i])
                    .wrapping_sub(w_rem_row[i])
                    .wrapping_sub(w_cap_row[i]);
                
                curr_black[i] = prev_black[i]
                    .wrapping_add(b_add_row[i])
                    .wrapping_sub(b_rem_row[i])
                    .wrapping_sub(b_cap_row[i]);
            }
        } else {
            // Standard quiet move pattern: 1 addition, 1 subtraction
            for i in 0..256 {
                curr_white[i] = prev_white[i]
                    .wrapping_add(w_add_row[i])
                    .wrapping_sub(w_rem_row[i]);
                
                curr_black[i] = prev_black[i]
                    .wrapping_add(b_add_row[i])
                    .wrapping_sub(b_rem_row[i]);
            }
        }
    }

    #[inline]
    pub fn unmake_move(&mut self) {
        // 🛡️ Safe development check: verifies we never unroll past the root.
        // It disappears completely in --release mode, leaving zero runtime overhead.
        debug_assert!(self.ply > 0, "Catastrophic: Attempted to unmake_move at root ply 0!");

        self.ply -= 1;
    }

    #[inline]
    pub fn increment_ply(&mut self) {
        self.ply += 1;
    }
}

