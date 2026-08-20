use crate::bishop_mask::*;
use crate::king_mask::*;
use crate::knight_mask::*;
use crate::pawn_mask::*;
use crate::rook_mask::*;
use crate::queen_mask::*;
use crate::move_command::*;
use crate::zobrist_hash::*;
use crate::chess_game::*;
use crate::board_accumlator::*;
use crate::nnue_network::*;
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
    en_passant: u64,
    active_player: Side,

    // Zobrist Hash
    zobrist_hash: u64,

    // --- INTEGRATED NNUE ACCUMULATOR BASE ---
    accumulators: Box<[BoardAccumulators; 256]>, 
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
            en_passant: 0,
            active_player: Side::WHITE,

            mailbox: [BoardPiece::NONE; 64],
            zobrist_hash: 0,

            accumulators: Box::new([BoardAccumulators::default(); 256]),
            ply: 0,

            nnue_network,
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
        self.create_accumlator_from_scratch();    
    }

    // Null Move Pruning Zugzwang
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
    pub fn castle_rights(&self) -> u8 {   
        self.castling_rights
    }

    // Return En Passant
    pub fn en_passant(&self) -> u64 {   
        self.en_passant
    }

    // Return Zobrist Hash
    pub fn zobrist_hash(&self) -> u64 {   
        self.zobrist_hash
    }

    // Return Active Player
    pub fn active_player(&self) -> Side {   
        self.active_player
    }

    // Return Mailbox Piece
    pub fn mailbox_piece(&self, target: usize) -> BoardPiece {   
        self.mailbox[target]
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
            attacks |= bishop_attack_paths(bishops.trailing_zeros() as usize, occ);
            bishops &= bishops - 1;
        }

        let mut rooks = self.rooks[index] | self.queens[index];
        while rooks != 0 {
            attacks |= rook_attack_paths(rooks.trailing_zeros() as usize, occ);
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
        pv_move_hint: Option<ForwardMove>,
        depth: i32,
        killer_move_table: &[[Option<ForwardMove>; MAX_DEPTH as usize]; 2]
    ) {        
        let player_index = self.player_index(self.active_player);
        let opp_index = self.player_index(self.opponent_player());
        let _opponent_attack_targets = self.compute_attack_targets(self.opponent_player());

        // Generate Moves
        king_moves(self, player_index, opp_index, _opponent_attack_targets, gen_moves);

        knight_moves(self, player_index, opp_index, gen_moves);

        rook_moves(self, player_index, opp_index, gen_moves);

        bishop_moves(self, player_index, opp_index, gen_moves);

        queen_moves(self, player_index, opp_index, gen_moves);

        match self.active_player {
            Side::WHITE => {
                white_pawn_moves(self, player_index, opp_index, gen_moves);
            },
            Side::BLACK => {
                black_pawn_moves(self, player_index, opp_index, gen_moves);
            }
        }
        
        if depth >= 0 {
            // 1. Allocate PV Move in Front (Highest Priority)
            if let Some(hint) = pv_move_hint {
                // Rust automatically uses your custom PartialEq logic via ==
                if let Some(cmd) = gen_moves.iter_mut().find(|m| **m == hint) {
                    cmd.pv_score = -2_000_000; 
                }
            }
            // 2. Allocate Killer Moves (High Priority, but below PV move)

            // Primary Killer
            let killer_0 = killer_move_table[0][depth as usize];
            if killer_0 != pv_move_hint {
                if let Some(cmd) = gen_moves.iter_mut().find(|m| Some(**m) == killer_0) {
                    cmd.pv_score = 200;
                }
            }

            // Secondary Killer
            let killer_1 = killer_move_table[1][depth as usize];
            if killer_1 != pv_move_hint {
                if let Some(cmd) = gen_moves.iter_mut().find(|m| Some(**m) == killer_1) {
                    cmd.pv_score = 210;
                }
            }
        }

        gen_moves.sort_unstable_by_key(|cmd| cmd.pv_score);
    }

    // helper method for move piece
    fn _move_piece(&mut self, move_command: ForwardMove) {
        // Remove Start Piece / Add End Piece
        let piece_type = piece_type_zobrist(self.mailbox[move_command.start_sq]);

        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[piece_type][move_command.start_sq];
        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[piece_type][move_command.end_sq];

        let move_piece = self.mailbox[move_command.start_sq];
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

        self.mailbox[move_command.start_sq] = BoardPiece::NONE;
        self.mailbox[move_command.end_sq] = move_piece;

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
    fn zobrist_xor(&mut self) {
        self.zobrist_hash ^= ZOBRIST_SIDE_TO_MOVE[active_player_zobrist(self.active_player)];
        self.zobrist_hash ^= ZOBRIST_EN_PASSANT[en_passant_zobrist(self.en_passant)];
        self.zobrist_hash ^= ZOBRIST_CASTLING[self.castling_rights as usize];
    }

    pub fn execute_move(&mut self, move_command: ForwardMove) -> Option<BoardPiece> {
        // XOR the current State for Castle, En Passant and Side to Move
        self.zobrist_xor();

        let mut remove_piece = None;
        // Store Removed Piece / No bitboard Operations
        match move_command.move_type { 
            MoveFlag::CAPTURE | MoveFlag::PROMOTIONQUEEN |
            MoveFlag::PROMOTIONROOK | MoveFlag::PROMOTIONBISHOP | 
            MoveFlag::PROMOTIONKNIGHT => {
                if self.mailbox[move_command.end_sq] != BoardPiece::NONE {
                    remove_piece = Some(self.mailbox[move_command.end_sq]);
                }
            },
            MoveFlag::ENPASSANT => {
                match self.active_player {
                    Side::WHITE => {
                        remove_piece = Some(self.mailbox[move_command.end_sq - 8]);
                    },
                    Side::BLACK => {
                        remove_piece = Some(self.mailbox[move_command.end_sq + 8]);
                    },
                }
            },
            _ => {},
        }

        // Clear the Previous En Passant
        self.en_passant = 0;

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
                let piece = self.mailbox[move_command.start_sq];
                if (piece == BoardPiece::WPAWN || piece == BoardPiece::BPAWN) && 
                (move_command.start_sq as i8 - move_command.end_sq as i8).abs() == 16 {
                    match self.active_player {
                        Side::WHITE => {
                            self.en_passant = 1u64 << (move_command.start_sq + 8);
                        },
                        Side::BLACK => {
                            self.en_passant = 1u64 << (move_command.start_sq - 8);
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
                self._remove_piece(move_command.start_sq);

                if remove_piece.is_some() {
                    self._remove_piece(move_command.end_sq);
                }

                match self.active_player {
                    Side::WHITE => {
                        self._place_piece(move_command.end_sq, 
                            white_promotion_piece(move_command.move_type)
                        );
                    },
                    Side::BLACK => {
                        self._place_piece(move_command.end_sq, 
                            black_promotion_piece(move_command.move_type)
                        );
                    },
                }
            },
            MoveFlag::ENPASSANT => {
                self._move_piece(move_command);
                match self.active_player {
                    Side::WHITE => {
                        self._remove_piece(move_command.end_sq - 8);
                    },
                    Side::BLACK => {
                        self._remove_piece(move_command.end_sq + 8);
                    },
                }
            },
            MoveFlag::CAPTURE => {
                self._remove_piece(move_command.end_sq);
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
                self._remove_piece(undo_move_cmd.end_sq);

                match self.active_player {
                    Side::WHITE => {
                        self._place_piece(undo_move_cmd.start_sq, BoardPiece::WPAWN);
                    },
                    Side::BLACK => {
                        self._place_piece(undo_move_cmd.start_sq, BoardPiece::BPAWN);
                    },
                }
            },
            MoveFlag::NULL => {},
        }

        // Restore Piece only if one was actually captured
        if let Some(piece) = undo_move_cmd.captured_piece {
            match undo_move_cmd.move_type { 
                MoveFlag::CAPTURE | MoveFlag::PROMOTIONQUEEN | 
                MoveFlag::PROMOTIONROOK | MoveFlag::PROMOTIONBISHOP | 
                MoveFlag::PROMOTIONKNIGHT => {
                    self._place_piece(undo_move_cmd.end_sq, piece);
                },
                MoveFlag::ENPASSANT => {
                    let ep_square = if self.active_player == Side::WHITE { 
                        undo_move_cmd.end_sq - 8 } else { undo_move_cmd.end_sq + 8 };
                    self._place_piece(ep_square, piece);
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

    // Create from Scratch
    pub fn create_accumlator_from_scratch(&mut self) {
        // Retrieve King Squares
        let w_king_sq = self.kings[Side::WHITE as usize].trailing_zeros() as usize;
        let b_king_sq = self.kings[Side::BLACK as usize].trailing_zeros() as usize;

        let target_white = &mut self.accumulators[self.ply].white.vals[..256];
        let target_black = &mut self.accumulators[self.ply].black.vals[..256];
        let biases = &self.nnue_network.l1_biases[..256];

        // --- 1. PROCESS WHITE ACCUMULATOR COMPLETELY ---
        // Initialize White with a clean copy (compiler can optimize this easily)
        for i in 0..256 {
            target_white[i] = biases[i] as i16;
        }
        
        // Accumulate all active pieces for White in a single contiguous memory stream
        for (sq, &piece) in self.mailbox.iter().enumerate().take(64) {
            if piece != BoardPiece::NONE {
                let w_idx = get_feature_index(w_king_sq, piece, sq, false);
                let w_row = &self.nnue_network.l1_weights[w_idx][..256];

                for i in 0..256 {
                    target_white[i] = target_white[i].wrapping_add(w_row[i]);
                }
            }
        }

        // --- 2. PROCESS BLACK ACCUMULATOR COMPLETELY ---
        // Initialize Black cleanly
        for i in 0..256 {
            target_black[i] = biases[i] as i16;
        }

        // Accumulate all active pieces for Black in a single contiguous memory stream
        for (sq, &piece) in self.mailbox.iter().enumerate().take(64) {
            if piece != BoardPiece::NONE {
                let b_idx = get_feature_index(b_king_sq, piece, sq, true);
                let b_row = &self.nnue_network.l1_weights[b_idx][..256];

                for i in 0..256 {
                    target_black[i] = target_black[i].wrapping_add(b_row[i]); 
                }
            }
        }
    }

    pub fn evaluate(&mut self, buffer: &mut NnueInferenceBuffer) -> i32 {
        // --- PERSPECTIVE ROUTING ---
        // Side to move (US) always fills the first 256 inputs.
        // Opponent (THEM) always fills the second 256 inputs.
        let (active_acc, opp_acc) = match self.active_player {
            Side::WHITE => (
                &self.accumulators[self.ply].white, &self.accumulators[self.ply].black
            ),
            Side::BLACK => (
                &self.accumulators[self.ply].black, &self.accumulators[self.ply].white
            ),
        };

        // --- STEP 0: ACCUMLATOR (Input -> Accumlator) ---
        // The accumlator is maintained by the init / move functions

        // --- STEP 1: CONCATENATION & ACTIVATION (L1 -> L2) ---
        for (i, &val) in active_acc.vals.iter().enumerate().take(256) {
            buffer.l2_inputs[i] = val.clamp(0, 127) as i8;
        }

        for (i, &val) in opp_acc.vals.iter().enumerate().take(256) {
            buffer.l2_inputs[i + 256] = val.clamp(0, 127) as i8;
        }

        // --- STEP 2: HIDDEN LAYER 2 (512 -> 64) ---
        // Input Scale (128) * Weight Scale (32) = Sum Scale (4096).
        // Shift Down by >> 7 to Scale (32)
        // Clamp at 32 to match Python's ReLU1 (1.0).
        let l2_layer = self.nnue_network.l2_weights.iter().zip(self.nnue_network.l2_biases.iter());
        for (neuron, (row, &bias)) in l2_layer.enumerate().take(64) {
            let mut sum: i32 = bias;

            // Process chunks of 16 elements to enable aggressive SIMD auto-vectorization
            let inputs = &buffer.l2_inputs[..512];
            for (chunk_weights, chunk_inputs) in row.chunks_exact(16).zip(inputs.chunks_exact(16)) {
                for (&w, &inp) in chunk_weights.iter().zip(chunk_inputs.iter()) {
                    sum += (inp as i32) * (w as i32);
                }
            }

            let activated = sum >> 7;
            buffer.l3_inputs[neuron] = activated.clamp(0, 32) as i8;
        }

        // --- STEP 3: HIDDEN LAYER 3 (64 -> 32) ---
        // Input Scale (32) * Weight Scale (32) = Sum Scale (1024).
        // Shift Down by 5 to Scale (32)
        // Clamp at 32 to match Python's ReLU1 (1.0).
        let l3_layer = self.nnue_network.l3_weights.iter().zip(self.nnue_network.l3_biases.iter());
        for (neuron, (row, &bias)) in l3_layer.enumerate().take(32) {
            let mut sum: i32 = bias;

            let inputs = &buffer.l3_inputs[..64];
            for (chunk_weights, chunk_inputs) in row.chunks_exact(16).zip(inputs.chunks_exact(16)) {
                for (&w, &inp) in chunk_weights.iter().zip(chunk_inputs.iter()) {
                    sum += (inp as i32) * (w as i32);
                }
            }

            let activated = sum >> 5;
            buffer.l4_inputs[neuron] = activated.clamp(0, 32) as i8;
        }


        // --- STEP 4: OUTPUT LAYER (32 -> 1) ---
        // Input Scale (32) * Weight Scale (128) = Sum Scale (4096).
        // Shift Down by 5 to Scale (128)
        let mut final_sum: i32 = self.nnue_network.output_bias[0];
        let row = &self.nnue_network.output_weights[0];

        let inputs = &buffer.l4_inputs[..32];
        for (chunk_weights, chunk_inputs) in row.chunks_exact(16).zip(inputs.chunks(16)) {
            for (&w, &inp) in chunk_weights.iter().zip(chunk_inputs.iter()) {
                final_sum += (inp as i32) * (w as i32);
            }
        }
        let internal_pawns_scaled = final_sum >> 5;

        // Shift by >> 7 remove remaining scale
        (internal_pawns_scaled * 100) / 128
    }

    /// Progresses the network forward incrementally during move making
    #[inline(always)]
    pub fn make_move(
        &mut self, 
        mv: ForwardMove,
    ) {
        self.increment_ply();

        // Retrieve King Squares
        let w_king_sq = self.kings[Side::WHITE as usize].trailing_zeros() as usize;
        let b_king_sq = self.kings[Side::BLACK as usize].trailing_zeros() as usize;

        let move_piece: BoardPiece = self.mailbox[mv.start_sq];

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
            let sq = if move_piece == BoardPiece::WPAWN {
                mv.end_sq - 8 
            } else {
                mv.end_sq + 8 
            };
            (sq, self.mailbox[sq])
        } else {
            (mv.end_sq, self.mailbox[mv.end_sq])
        };

        // --- 3. Compute Sparse Feature Indices ---
        let w_remove = get_feature_index(w_king_sq, move_piece, mv.start_sq, false);
        let b_remove = get_feature_index(b_king_sq, move_piece, mv.start_sq, true);

        let w_add = get_feature_index(w_king_sq, added_piece, mv.end_sq, false);
        let b_add = get_feature_index(b_king_sq, added_piece, mv.end_sq, true);

        // Get basic rows
        let w_rem_row = &self.nnue_network.l1_weights[w_remove][..256];
        let b_rem_row = &self.nnue_network.l1_weights[b_remove][..256];
        let w_add_row = &self.nnue_network.l1_weights[w_add][..256];
        let b_add_row = &self.nnue_network.l1_weights[b_add][..256];

        // --- 4. High-Density Auto-Vectorized Parallel Loop Block ---
        let prev_ply = self.ply - 1;
        let (left, right) = self.accumulators.split_at_mut(self.ply);
        
        let prev_acc = &left[prev_ply];
        let curr_acc = &mut right[0];

        // Slice both targets to exactly 256 to remove runtime bounds checking
        let curr_white = &mut curr_acc.white.vals[..256];
        let curr_black = &mut curr_acc.black.vals[..256];
        
        let prev_white = &prev_acc.white.vals[..256];
        let prev_black = &prev_acc.black.vals[..256];

        if captured_piece != BoardPiece::NONE {
            let w_cap = get_feature_index(w_king_sq, captured_piece, captured_sq, false);
            let b_cap = get_feature_index(b_king_sq, captured_piece, captured_sq, true);
            
            let w_cap_row = &self.nnue_network.l1_weights[w_cap][..256];
            let b_cap_row = &self.nnue_network.l1_weights[b_cap][..256];

            // 1. Process White entirely in a clean, isolated memory pipeline
            for i in 0..256 {
                curr_white[i] = prev_white[i]
                    .wrapping_add(w_add_row[i])
                    .wrapping_sub(w_rem_row[i])
                    .wrapping_sub(w_cap_row[i]);
            }

            // 2. Process Black entirely in a clean, isolated memory pipeline
            for i in 0..256 {
                curr_black[i] = prev_black[i]
                    .wrapping_add(b_add_row[i])
                    .wrapping_sub(b_rem_row[i])
                    .wrapping_sub(b_cap_row[i]);
            }
        } else {
            // 1. Process White entirely
            for i in 0..256 {
                curr_white[i] = prev_white[i]
                    .wrapping_add(w_add_row[i])
                    .wrapping_sub(w_rem_row[i]);
            }

            // 2. Process Black entirely
            for i in 0..256 {
                curr_black[i] = prev_black[i]
                    .wrapping_add(b_add_row[i])
                    .wrapping_sub(b_rem_row[i]);
            }
        }
    }

    pub fn unmake_move(&mut self) {
        if self.ply > 0 {
            self.ply -= 1; 
        } else {
            eprintln!("Warning: Attempted to unmake_move at ply 0!");
        }
    }

    pub fn increment_ply(&mut self) {
        self.ply += 1;
    }
}

