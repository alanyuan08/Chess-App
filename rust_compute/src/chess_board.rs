use crate::bishop_mask::*;
use crate::king_mask::*;
use crate::knight_mask::*;
use crate::pawn_mask::*;
use crate::rook_mask::*;
use crate::queen_mask::*;
use crate::move_command::*;
use crate::zobrist_hash::*;
use timecat::prelude::*;

// 0 -> White / 1 -> Black
#[derive(Debug, Clone)] 
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

    mailbox: [BoardPiece; 64],

    // Zobrist Hash
    zobrist_hash: u64,

    // Time Cat board
    timecat_board: Board,
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

            mailbox: [BoardPiece::NONE; 64],
            zobrist_hash: 0,
            timecat_board: Board::default(),
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
            self.rooks[color]   |= (1u64 << (piece_rank_offset + 0)) | (1u64 << (piece_rank_offset + 7));
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

            for i in 0..8 {
                self.mailbox[piece_rank_offset + i] = pieces[i];
            }

            // 4. Composite Bitboards
            self.all_pieces[color] = self.pawns[color] | self.rooks[color] | 
                                    self.knights[color] | self.bishops[color] | 
                                    self.queens[color] | self.kings[color];
            self.occupied |= self.all_pieces[color];

            // 5. Compute Zobrist
            self.zobrist_hash = self.compute_init_zobrist();
        }
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
                return Side::BLACK;
            },
            Side::BLACK => {
                return Side::WHITE;
            }
        }
    }

    // Return Index
    pub fn player_index(&self, player_side: Side) -> usize {
        match player_side {
            Side::WHITE => {
                return 0;
            },
            Side::BLACK => {
                return 1;
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

    // Return Time Cat Score
    pub fn eval(&mut self) -> i32 {   
        self.timecat_board.evaluate() as i32
    }

    // Forward Time Cat
    pub fn timecat_push_move(&mut self, uci_input: String) {
        if self.timecat_board.push_move(&uci_input).expect("ValidOrNullMove").is_none() {
            panic!("Invalid UCI move or illegal move: {}", uci_input);
        }
    }

    // Timecat FEN 
    pub fn timecat_print_fen(&mut self) {
        println!("{}", self.timecat_board);
    }

    // Return Piece Value
    pub fn index_piece_value(&self, index: usize) -> i32 {   
        piece_value(self.mailbox[index])
    }

    // Undo Time Cat Move
    pub fn timecat_pop_move(&mut self) {
        self.timecat_board.pop();

        return;
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
    pub fn generate_moves(&mut self) -> Vec<ForwardMove> {
        let mut gen_moves: Vec<ForwardMove> = Vec::with_capacity(218);
        let player_index = self.player_index(self.active_player);

        let opp_index = self.player_index(self.opponent_player());
        let _opponent_attack_targets = self.compute_attack_targets(self.opponent_player());

        // Generate Moves
        king_moves(self.kings[player_index], self.occupied, _opponent_attack_targets,
            self.active_player, self.castling_rights, self.all_pieces[opp_index], &mut gen_moves);

        knight_moves(self.knights[player_index], self.occupied, self.all_pieces[opp_index], &mut gen_moves);

        rook_moves(self.rooks[player_index], self.occupied, self.all_pieces[opp_index], &mut gen_moves);

        bishop_moves(self.bishops[player_index], self.occupied, self.all_pieces[opp_index], &mut gen_moves);

        queen_moves(self.queens[player_index], self.occupied, self.all_pieces[opp_index], &mut gen_moves);

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
    fn _move_piece(&mut self, move_command: ForwardMove) {
        // Remove Start Piece / Add End Piece
        let piece_type = piece_type_zobrist(self.mailbox[move_command.startSq]);

        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[piece_type][move_command.startSq];
        self.zobrist_hash ^= ZOBRIST_TABLE_MAP[piece_type][move_command.endSq];

        let move_piece = self.mailbox[move_command.startSq];
        let player_index = self.player_index(piece_player(move_piece));

        match move_piece {
            BoardPiece::WPAWN => {
                self.pawns[player_index] ^= 1u64 << move_command.startSq;
                self.pawns[player_index] ^= 1u64 << move_command.endSq;
            },
            BoardPiece::BPAWN => {
                self.pawns[player_index] ^= 1u64 << move_command.startSq;
                self.pawns[player_index] ^= 1u64 << move_command.endSq;
            },
            BoardPiece::WBISHOP | BoardPiece::BBISHOP => {
                self.bishops[player_index] ^= 1u64 << move_command.startSq;
                self.bishops[player_index] ^= 1u64 << move_command.endSq;
            },
            BoardPiece::WKNIGHT | BoardPiece::BKNIGHT => {
                self.knights[player_index] ^= 1u64 << move_command.startSq;
                self.knights[player_index] ^= 1u64 << move_command.endSq;
            },
            BoardPiece::WROOK | BoardPiece::BROOK => {
                self.rooks[player_index] ^= 1u64 << move_command.startSq;
                self.rooks[player_index] ^= 1u64 << move_command.endSq;
            },
            BoardPiece::WQUEEN | BoardPiece::BQUEEN => {
                self.queens[player_index] ^= 1u64 << move_command.startSq;
                self.queens[player_index] ^= 1u64 << move_command.endSq;
            },
            BoardPiece::WKING | BoardPiece::BKING=> {
                self.kings[player_index] ^= 1u64 << move_command.startSq;
                self.kings[player_index] ^= 1u64 << move_command.endSq;
            },
            BoardPiece::NONE => {
                println!("Tried to move empty");
            },
        }

        self.mailbox[move_command.startSq] = BoardPiece::NONE;
        self.mailbox[move_command.endSq] = move_piece;

        self.all_pieces[player_index] &= !(1u64 << move_command.startSq);
        self.all_pieces[player_index] |= 1u64 << move_command.endSq;

        self.occupied &= !(1u64 << move_command.startSq);
        self.occupied |= 1u64 << move_command.endSq;
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
        match move_command.moveType { 
            MoveFlag::CAPTURE | MoveFlag::PROMOTIONQUEEN |
            MoveFlag::PROMOTIONROOK | MoveFlag::PROMOTIONBISHOP | 
            MoveFlag::PROMOTIONKNIGHT => {
                if self.mailbox[move_command.endSq] != BoardPiece::NONE {
                    remove_piece = Some(self.mailbox[move_command.endSq]);
                }
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

        // Update Castling
        match move_command.startSq {
            4  => self.castling_rights &= !(WHITE_KINGSIDE | WHITE_QUEENSIDE),
            60 => self.castling_rights &= !(BLACK_KINGSIDE | BLACK_QUEENSIDE),
            7  => self.castling_rights &= !WHITE_KINGSIDE,
            0  => self.castling_rights &= !WHITE_QUEENSIDE,
            63 => self.castling_rights &= !BLACK_KINGSIDE,
            56 => self.castling_rights &= !BLACK_QUEENSIDE,
            _ => {}
        }
        match move_command.endSq {
            7  => self.castling_rights &= !WHITE_KINGSIDE,
            0  => self.castling_rights &= !WHITE_QUEENSIDE,
            63 => self.castling_rights &= !BLACK_KINGSIDE,
            56 => self.castling_rights &= !BLACK_QUEENSIDE,
            _ => {}
        }

        match move_command.moveType {
            MoveFlag::MOVE => {
                self._move_piece(move_command);
            },
            MoveFlag::PAWNOPENMOVE => {
                // Update En Passant
                let piece = self.mailbox[move_command.startSq];
                if (piece == BoardPiece::WPAWN || piece == BoardPiece::BPAWN) && 
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
                
                self._move_piece(move_command);
            },
            MoveFlag::KINGSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { startSq: 4, endSq: 6, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { startSq: 7, endSq: 5, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { startSq: 60, endSq: 62, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { startSq: 63, endSq: 61, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::QUEENSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { startSq: 4, endSq: 2, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { startSq: 0, endSq: 3, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { startSq: 60, endSq: 58, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { startSq: 56, endSq: 59, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::PROMOTIONQUEEN | MoveFlag::PROMOTIONROOK |
            MoveFlag::PROMOTIONBISHOP | MoveFlag::PROMOTIONKNIGHT => {
                self._remove_piece(move_command.startSq);

                if remove_piece.is_some() {
                    self._remove_piece(move_command.endSq);
                }

                match self.active_player {
                    Side::WHITE => {
                        self._place_piece(move_command.endSq, 
                            white_promotion_piece(move_command.moveType)
                        );
                    },
                    Side::BLACK => {
                        self._place_piece(move_command.endSq, 
                            black_promotion_piece(move_command.moveType)
                        );
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
                self._remove_piece(move_command.endSq);
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
        match undo_move_cmd.moveType {
            MoveFlag::MOVE | MoveFlag::CAPTURE | MoveFlag::PAWNOPENMOVE | MoveFlag::ENPASSANT=> {
                let undo_command = ForwardMove { 
                    startSq: undo_move_cmd.endSq, 
                    endSq: undo_move_cmd.startSq, 
                    moveType: MoveFlag::MOVE 
                };
                self._move_piece(undo_command);
          
            },
            MoveFlag::KINGSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { startSq: 6, endSq: 4, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { startSq: 5, endSq: 7, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { startSq: 62, endSq: 60, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { startSq: 61, endSq: 63, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::QUEENSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { startSq: 2, endSq: 4, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { startSq: 3, endSq: 0, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { startSq: 58, endSq: 60, moveType: MoveFlag::MOVE };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { startSq: 59, endSq: 56, moveType: MoveFlag::MOVE };
                        self._move_piece(rook_move_cmd);
                    },
                }
            },
            MoveFlag::PROMOTIONQUEEN | MoveFlag::PROMOTIONROOK | 
            MoveFlag::PROMOTIONBISHOP | MoveFlag::PROMOTIONKNIGHT => {
                self._remove_piece(undo_move_cmd.endSq);

                match self.active_player {
                    Side::WHITE => {
                        self._place_piece(undo_move_cmd.startSq, BoardPiece::WPAWN);
                    },
                    Side::BLACK => {
                        self._place_piece(undo_move_cmd.startSq, BoardPiece::BPAWN);
                    },
                }
            },
            MoveFlag::NULL => {},
        }

        // Restore Piece only if one was actually captured
        if let Some(piece) = undo_move_cmd.capturedPiece {
            match undo_move_cmd.moveType { 
                MoveFlag::CAPTURE | MoveFlag::PROMOTIONQUEEN | 
                MoveFlag::PROMOTIONROOK | MoveFlag::PROMOTIONBISHOP | 
                MoveFlag::PROMOTIONKNIGHT => {
                    self._place_piece(undo_move_cmd.endSq, piece);
                },
                MoveFlag::ENPASSANT => {
                    let ep_square = if self.active_player == Side::WHITE { 
                        undo_move_cmd.endSq - 8 } else { undo_move_cmd.endSq + 8 };
                    self._place_piece(ep_square, piece);
                },
                _ => {},
            }
        }

        // Restore En Passant
        self.en_passant = undo_move_cmd.prevEnPassant;
        self.castling_rights = undo_move_cmd.prevCastleRights;

        // XOR in current state for Castle, En Passant and Side to Move
        self.zobrist_xor();
    }
}
