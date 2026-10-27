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
            },
            Side::NONE => {
                Side::NONE
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
            },
            Side::NONE => {
                2
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

    // Return En Passant as a board
    #[inline]
    pub fn en_passant_board(&self) -> u64 {   
         1u64 <<  self.en_passant
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
        let defender_index = 1 - index;
        
        let defender_king_bb = self.kings[defender_index];
        let occ = self.occupied ^ defender_king_bb;

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
            },
            Side::NONE => {},
        }
        
        if ply_idx < 128 {
            let has_valid_pv = (pv_move_hint.move_type != MoveFlag::NULL) as i16;
            let k0 = killer_move_table[ply_idx as usize][0];
            let k1 = killer_move_table[ply_idx as usize][1];
            
            let has_k0 = (k0.move_type != MoveFlag::NULL && (has_valid_pv == 0 || k0 != pv_move_hint)) as i16;
            let has_k1 = (k1.move_type != MoveFlag::NULL && (has_valid_pv == 0 || k1 != pv_move_hint)) as i16;

            for m in gen_moves.iter_mut() {
                // 1. Evaluate matching properties using strict i16 bitwise mapping
                let is_pv = has_valid_pv & (*m == pv_move_hint) as i16;
                let is_k0 = (1 - is_pv) * (has_k0 & (*m == k0) as i16);
                let is_k1 = (1 - is_pv) * (1 - is_k0) * (has_k1 & (*m == k1) as i16);

                // 2. Accumulate values over a strict linear i16 equation
                let added_score = (is_pv * 32_000) + (is_k0 * 500) + (is_k1 * 400);

                // 3. Scale baseline scores safely without control flow jumps
                let remains_unchanged = 1 - is_pv - is_k0 - is_k1;
                m.pv_score = (m.pv_score * remains_unchanged) + added_score;
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
            BoardPiece::WPAWN | BoardPiece::BPAWN=> {
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
                return;
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
                return;
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
            BoardPiece::WPAWN | BoardPiece::BPAWN => {
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
            BoardPiece::NONE => {
                return;
            },
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

    #[inline(always)]
    fn get_normalized_king_state(king_sq: u8, is_black: bool) -> (u8, bool) {
        // 1. If it's a black king, swap it to a white "side to move" king via vertical flip
        let k_sq = if is_black { king_sq ^ 63 } else { king_sq };
        
        // 2. Resolve horizontal wing mirroring (Files E-H map to A-D)
        let is_mirrored = (k_sq % 8) >= 4;
        let lookup_sq = if is_mirrored { k_sq ^ 7 } else { k_sq };
        
        (KING_BUCKET_MAP[lookup_sq as usize], is_mirrored)
    }

    pub fn does_move_cross_king_bucket(&self, forward_move: ForwardMove) -> bool {
        let move_piece = self.mailbox_piece(forward_move.start_sq);
        
        // 1. Extract baseline king squares from bitboard registers
        let w_king_before = self.kings[0].trailing_zeros() as u8;
        let b_king_before = self.kings[1].trailing_zeros() as u8;

        // 2. Project destinations assuming start/end squares are correct for king tracking
        let w_king_after = if move_piece == BoardPiece::WKING { forward_move.end_sq } else { w_king_before };
        let b_king_after = if move_piece == BoardPiece::BKING { forward_move.end_sq } else { b_king_before };

        // 3. Normalize both sides into their own side-to-move frames
        let (w_bucket_before, w_mirror_before) =  Self::get_normalized_king_state(w_king_before, false);
        let (w_bucket_after, w_mirror_after) =  Self::get_normalized_king_state(w_king_after, false);

        let (b_bucket_before, b_mirror_before) =  Self::get_normalized_king_state(b_king_before, true);
        let (b_bucket_after, b_mirror_after) =  Self::get_normalized_king_state(b_king_after, true);

        // 4. Return true if a bucket boundary OR a horizontal reflection line is crossed
        (w_bucket_before != w_bucket_after) || (w_mirror_before != w_mirror_after) ||
        (b_bucket_before != b_bucket_after) || (b_mirror_before != b_mirror_after)
    }

    pub fn execute_move(&mut self, move_command: ForwardMove) -> BoardPiece {
        // XOR the current State for Castle, En Passant and Side to Move
        self.zobrist_xor();

        // Remove the Capture Square
        let remove_piece = self.mailbox[move_command.capture_sq as usize];

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
                match self.active_player {
                    Side::WHITE => {
                        self.en_passant = move_command.start_sq + 8;
                    },
                    Side::BLACK => {
                        self.en_passant = move_command.start_sq - 8;
                    },
                    Side::NONE => {},
                }
            
                self._move_piece(move_command);
            },
            MoveFlag::KINGSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 4, end_sq: 6, capture_sq: 6, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 7, end_sq: 5, capture_sq: 5, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 60, end_sq: 62, capture_sq: 62, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 63, end_sq: 61, capture_sq: 61, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::NONE => {},
                }
            },
            MoveFlag::QUEENSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 4, end_sq: 2, capture_sq: 2, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 0, end_sq: 3, capture_sq: 3, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 60, end_sq: 58, capture_sq: 58, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 56, end_sq: 59, capture_sq: 59, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::NONE => {},
                }
            },
            MoveFlag::PROMOTIONQUEEN | MoveFlag::PROMOTIONROOK |
            MoveFlag::PROMOTIONBISHOP | MoveFlag::PROMOTIONKNIGHT => {
                self._remove_piece(move_command.start_sq as usize);
                self._remove_piece(move_command.capture_sq as usize);

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
                    Side::NONE => {},
                }
            },
            MoveFlag::CAPTURE | MoveFlag::ENPASSANT => {
                self._remove_piece(move_command.capture_sq as usize);
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
            MoveFlag::MOVE | MoveFlag::CAPTURE | MoveFlag::PAWNOPENMOVE | MoveFlag::ENPASSANT => {
                let undo_command = ForwardMove { 
                    start_sq: undo_move_cmd.end_sq, 
                    end_sq: undo_move_cmd.start_sq, 
                    capture_sq: undo_move_cmd.start_sq,
                    move_type: MoveFlag::MOVE, 
                    pv_score: 0,
                };
                self._move_piece(undo_command);
          
            },
            MoveFlag::KINGSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 6, end_sq: 4, capture_sq: 4, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 5, end_sq: 7, capture_sq: 7, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 62, end_sq: 60, capture_sq: 60, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 61, end_sq: 63, capture_sq: 63, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::NONE => {},
                }
            },
            MoveFlag::QUEENSIDECASTLE => {
                match self.active_player {
                    Side::WHITE => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 2, end_sq: 4, capture_sq: 4, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 3, end_sq: 0, capture_sq: 0, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::BLACK => {
                        let king_move_cmd = ForwardMove { 
                            start_sq: 58, end_sq: 60, capture_sq: 60, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(king_move_cmd);

                        let rook_move_cmd = ForwardMove { 
                            start_sq: 59, end_sq: 56, capture_sq: 56, move_type: MoveFlag::MOVE, pv_score: 0
                        };
                        self._move_piece(rook_move_cmd);
                    },
                    Side::NONE => {},
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
                    Side::NONE => {},
                }
            },
            MoveFlag::NULL => {},
        }

        // Restore Captured Piece
        self._place_piece(undo_move_cmd.capture_sq as usize, undo_move_cmd.captured_piece);

        // Restore En Passant
        self.en_passant = undo_move_cmd.prev_en_passant;
        self.castling_rights = undo_move_cmd.prev_castle_rights;

        // XOR in current state for Castle, En Passant and Side to Move
        self.zobrist_xor();
    }

    #[inline(always)]
    pub fn null_move_accumulator(&mut self, accumulators: &mut AccumulatorHistory) {
        let curr_ply = self.ply;
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
        let (active_acc, opp_acc) = match self.active_player {
            Side::WHITE => (
                &accumulators.moves[self.ply].white, 
                &accumulators.moves[self.ply].black
            ),
            Side::BLACK => (
                &accumulators.moves[self.ply].black, 
                &accumulators.moves[self.ply].white
            ),
            Side::NONE => panic!("Invalid Game State: active_player cannot be Side::NONE during NNUE accumulator fetching!"),
        };

        // --- STEP 0: ACCUMULATOR ---
        let l2_left: &mut [i8; 256] = 
            unsafe { &mut *buffer.l2_inputs.as_mut_ptr().cast::<[i8; 256]>() };
        let l2_right: &mut [i8; 256] = 
            unsafe { &mut *buffer.l2_inputs.as_mut_ptr().add(256).cast::<[i8; 256]>() };

        // --- STEP 1: Clipped ReLU ---
        for i in 0..256 {
            l2_left[i] = active_acc[i].clamp(0, 127) as i8;
        }
        for i in 0..256 {
            l2_right[i] = opp_acc[i].clamp(0, 127) as i8;
        }

        // --- STEP 2: HIDDEN LAYER 2 (512 -> 32) ---
        let l2_biases = &self.nnue_network.l2_biases;
        let l2_inputs: &[i8; 512] = 
            unsafe { &*buffer.l2_inputs.as_ptr().cast::<[i8; 512]>() };
        let l3_inputs: &mut [i8; 32] = 
            unsafe { &mut *buffer.l3_inputs.as_mut_ptr().cast::<[i8; 32]>() };

        for neuron in 0..32 {
            let mut sum: i32 = l2_biases[neuron];
            let row: &[i8; 512] = &self.nnue_network.l2_weights[neuron];

            for i in 0..512 {
                sum += (l2_inputs[i] as i32) * (row[i] as i32);
            }

            let activated = sum >> 6;
            l3_inputs[neuron] = activated.clamp(0, 127) as i8;
        }

        // --- STEP 3: HIDDEN LAYER 3 (32 -> 32) ---
        let l3_biases = &self.nnue_network.l3_biases;
        let l4_inputs: &mut [i8; 32] = 
            unsafe { &mut *buffer.l4_inputs.as_mut_ptr().cast::<[i8; 32]>() };

        for neuron in 0..32 {
            let mut sum: i32 = l3_biases[neuron];
            let row: &[i8; 32] = &self.nnue_network.l3_weights[neuron];

            for i in 0..32 {
                sum += (l3_inputs[i] as i32) * (row[i] as i32);
            }

            let activated = sum >> 6;
            l4_inputs[neuron] = activated.clamp(0, 127) as i8;
        }

        // --- STEP 4: OUTPUT LAYER (32 -> 1) ---
        let mut final_sum: i32 = self.nnue_network.output_bias[0];
        let row: &[i8; 32] = &self.nnue_network.output_weights[0];

        for i in 0..32 {
            final_sum += (l4_inputs[i] as i32) * (row[i] as i32);
        }

        let score_scaled = (final_sum * 100) >> 12;

        score_scaled as i16
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
        let (captured_sq, captured_piece) = (mv.capture_sq, self.mailbox[mv.capture_sq as usize]);

        // --- 3. Setup Target and Source Frame Boundaries Safely ---
        let curr_ply = self.ply;
        let prev_ply = curr_ply - 1;

        let (left, right) = accumulators.moves.split_at_mut(curr_ply);
        let prev_acc = &left[prev_ply];
        let curr_acc = &mut right[0];

        let curr_white: &mut [i16; 256] = &mut curr_acc.white;
        let curr_black: &mut [i16; 256] = &mut curr_acc.black;
        
        // Base initialization: copy the previous accumulator state forward
        curr_white.copy_from_slice(&prev_acc.white);
        curr_black.copy_from_slice(&prev_acc.black);

        // --- 4. Process Moving / Added Piece ---
        // White Perspective: Tracks everything except its own White King
        if move_piece != BoardPiece::WKING {
            let w_remove = get_feature_index(w_king_sq, move_piece, mv.start_sq, false);
            let w_add = get_feature_index(w_king_sq, added_piece, mv.end_sq, false);
            
            let w_rem_row = &self.nnue_network.l1_weights[w_remove];
            let w_add_row = &self.nnue_network.l1_weights[w_add];
            
            for i in 0..256 {
                curr_white[i] = curr_white[i].wrapping_add(w_add_row[i]).wrapping_sub(w_rem_row[i]);
            }
        }

        // Black Perspective: Tracks everything except its own Black King
        if move_piece != BoardPiece::BKING {
            let b_remove = get_feature_index(b_king_sq, move_piece, mv.start_sq, true);
            let b_add = get_feature_index(b_king_sq, added_piece, mv.end_sq, true);
            
            let b_rem_row = &self.nnue_network.l1_weights[b_remove];
            let b_add_row = &self.nnue_network.l1_weights[b_add];
            
            for i in 0..256 {
                curr_black[i] = curr_black[i].wrapping_add(b_add_row[i]).wrapping_sub(b_rem_row[i]);
            }
        }

        // --- 5. Process Captured Piece ---
        if captured_piece != BoardPiece::NONE {
            let w_cap = get_feature_index(w_king_sq, captured_piece, captured_sq, false);
            let b_cap = get_feature_index(b_king_sq, captured_piece, captured_sq, true);
            
            let w_cap_row = &self.nnue_network.l1_weights[w_cap];
            let b_cap_row = &self.nnue_network.l1_weights[b_cap];

            for i in 0..256 {
                curr_white[i] = curr_white[i].wrapping_sub(w_cap_row[i]);
                curr_black[i] = curr_black[i].wrapping_sub(b_cap_row[i]);
            }
        }

        // --- 6. Special Castling Logic (Incremental Rook Updating) ---
        // If the King castles but stays within its bucket boundary, we must manually update the Rook features.
        if mv.move_type == MoveFlag::KINGSIDECASTLE || mv.move_type == MoveFlag::QUEENSIDECASTLE {
            let moving_side = piece_player(move_piece);
            
            let (r_start, r_end, rook_piece) = match (mv.move_type, moving_side) {
                (MoveFlag::KINGSIDECASTLE, Side::WHITE)  => (7, 5, BoardPiece::WROOK),   // H1 -> F1
                (MoveFlag::KINGSIDECASTLE, Side::BLACK)  => (63, 61, BoardPiece::BROOK), // H8 -> F8
                (MoveFlag::QUEENSIDECASTLE, Side::WHITE) => (0, 3, BoardPiece::WROOK),   // A1 -> D1
                (MoveFlag::QUEENSIDECASTLE, Side::BLACK) => (56, 59, BoardPiece::BROOK), // A8 -> D8
                _ => (0, 0, BoardPiece::NONE),
            };

            let w_r_rem = get_feature_index(w_king_sq, rook_piece, r_start, false);
            let b_r_rem = get_feature_index(b_king_sq, rook_piece, r_start, true);
            
            let w_r_add = get_feature_index(w_king_sq, rook_piece, r_end, false);
            let b_r_add = get_feature_index(b_king_sq, rook_piece, r_end, true);

            let w_r_rem_row = &self.nnue_network.l1_weights[w_r_rem];
            let b_r_rem_row = &self.nnue_network.l1_weights[b_r_rem];
            let w_r_add_row = &self.nnue_network.l1_weights[w_r_add];
            let b_r_add_row = &self.nnue_network.l1_weights[b_r_add];

            for i in 0..256 {
                curr_white[i] = curr_white[i].wrapping_add(w_r_add_row[i]).wrapping_sub(w_r_rem_row[i]);
                curr_black[i] = curr_black[i].wrapping_add(b_r_add_row[i]).wrapping_sub(b_r_rem_row[i]);
            }
        }
    }

    #[inline]
    pub fn unmake_move(&mut self) {
        // Safe development check: verifies we never unroll past the root.
        // It disappears completely in --release mode, leaving zero runtime overhead.
        debug_assert!(self.ply > 0, "Catastrophic: Attempted to unmake_move at root ply 0!");

        self.ply -= 1;
    }

    #[inline(always)]
    pub fn total_piece_count(&self) -> u8 {
        let all_pieces_mask = self.all_pieces[0] | self.all_pieces[1];
        all_pieces_mask.count_ones() as u8
    }

    #[inline]
    pub fn increment_ply(&mut self) {
        self.ply += 1;
    }

    pub fn compute_pinned_mask_with_custom_state(&self, custom_occupancy: u64, friendly_pieces: u64, stm: usize) -> u64 {
        let opponent_side = 1 - stm;
        let king_sq = self.kings[stm].trailing_zeros() as u8;
        if king_sq >= 64 { return 0; }

        // 1. Gather all sliding pinners using X-ray lookups matching the custom sandbox layout
        let friendly_blockers_ortho = rook_attack_paths(king_sq, custom_occupancy) & friendly_pieces;
        let orthogonal_pinners = rook_attack_paths(king_sq, custom_occupancy ^ friendly_blockers_ortho) 
            & (self.rooks[opponent_side] | self.queens[opponent_side]);

        let friendly_blockers_diag = bishop_attack_paths(king_sq, custom_occupancy) & friendly_pieces;
        let diagonal_pinners = bishop_attack_paths(king_sq, custom_occupancy ^ friendly_blockers_diag) 
            & (self.bishops[opponent_side] | self.queens[opponent_side]);

        let active_pinners = orthogonal_pinners | diagonal_pinners;
        let mut pinned_mask = 0u64;

        // 2. Identify which internal friendly blockers sit perfectly online with active threats
        let mut potential_pieces = friendly_blockers_ortho | friendly_blockers_diag;
        while potential_pieces != 0 {
            let p_sq = potential_pieces.trailing_zeros() as u8;
            let single_friendly_piece = 1u64 << p_sq;

            let mut pinners = active_pinners;
            while pinners != 0 {
                let pinner_sq = pinners.trailing_zeros() as u8;
                if Self::aligned(p_sq, pinner_sq, king_sq) {
                    pinned_mask |= single_friendly_piece;
                    break;
                }
                pinners &= pinners - 1;
            }
            potential_pieces &= potential_pieces - 1;
        }

        pinned_mask
    }

    #[inline(always)]
    pub fn aligned(p_sq: u8, target_sq: u8, king_sq: u8) -> bool {
        let p_rank = (p_sq >> 3) as i8;
        let p_file = (p_sq & 7) as i8;
        
        let t_rank = (target_sq >> 3) as i8;
        let t_file = (target_sq & 7) as i8;
        
        let k_rank = (king_sq >> 3) as i8;
        let k_file = (king_sq & 7) as i8;

        // Isolate vectors from the King to the Pinned piece and Target square
        let dr_p = p_rank - k_rank;
        let df_p = p_file - k_file;
        
        let dr_t = t_rank - k_rank;
        let df_t = t_file - k_file;

        // 1. Check if all three squares are colinear
        let colinear = (dr_p * df_t) == (dr_t * df_p);

        // 2. Check if the line matches a valid orthogonal or perfect 45-degree diagonal ray
        let valid_ray = (dr_p == 0) | (df_p == 0) | ((dr_p * dr_p) == (df_p * df_p));

        // 3. Directional Guard via Dot Product
        let same_direction = (dr_p * dr_t + df_p * df_t) >= 0;

        colinear & valid_ray & same_direction
    }

    /// Generates sliding attacks on the fly using your engine's ray paths.
    fn get_slider_attackers(&self, target_sq: u8, occupied: u64) -> u64 {
        let mut attackers = 0u64;

        // Bishop-like attacks (Bishops and Queens from both sides)
        let bishop_rays = bishop_attack_paths(target_sq, occupied);
        let diagonal_pieces = self.bishops[0] | self.bishops[1] | self.queens[0] | self.queens[1];
        attackers |= bishop_rays & diagonal_pieces;

        // Rook-like attacks (Rooks and Queens from both sides)
        let rook_rays = rook_attack_paths(target_sq, occupied);
        let orthogonal_pieces = self.rooks[0] | self.rooks[1] | self.queens[0] | self.queens[1];
        attackers |= rook_rays & orthogonal_pieces;

        attackers
    }

    /// Pulls the least valuable piece out of the attacker pool for the active side to move (stm).
    pub fn get_least_valuable_attacker(
        &self, 
        attackers_pool: u64, 
        pinned_mask: u64, 
        king_sq: u8, 
        target_sq: u8,
        stm: usize,
        occupancy: u64 
    ) -> (u8, BoardPiece) {
        let mut pool = attackers_pool;

        // --- PIN FILTRATION ---
        let pinned_attackers = pool & pinned_mask;
        if pinned_attackers != 0 {
            let mut temp = pinned_attackers;
            while temp != 0 {
                let p_sq = temp.trailing_zeros() as u8;
                
                // If the pinned piece attempts to move off its pin line, eliminate it from this turn's pool
                if !Self::aligned(p_sq, target_sq, king_sq) {
                    pool &= !(1u64 << p_sq);
                }
                temp &= temp - 1;
            }
        }

        // --- LEAST VALUABLE PIECE CHECKS ---
        let pawns = pool & self.pawns[stm];
        if pawns != 0 {
            return (pawns.trailing_zeros() as u8, if stm == 0 { BoardPiece::WPAWN } else { BoardPiece::BPAWN });
        }

        let knights = pool & self.knights[stm];
        if knights != 0 {
            return (knights.trailing_zeros() as u8, if stm == 0 { BoardPiece::WKNIGHT } else { BoardPiece::BKNIGHT });
        }

        let bishops = pool & self.bishops[stm];
        if bishops != 0 {
            return (bishops.trailing_zeros() as u8, if stm == 0 { BoardPiece::WBISHOP } else { BoardPiece::BBISHOP });
        }

        let rooks = pool & self.rooks[stm];
        if rooks != 0 {
            return (rooks.trailing_zeros() as u8, if stm == 0 { BoardPiece::WROOK } else { BoardPiece::BROOK });
        }

        let queens = pool & self.queens[stm];
        if queens != 0 {
            return (queens.trailing_zeros() as u8, if stm == 0 { BoardPiece::WQUEEN } else { BoardPiece::BQUEEN });
        }

        // --- KING RECAPTURE PROTECTION ---
        let kings = pool & self.kings[stm];
        if kings != 0 {
            let opponent_side = 1 - stm;

            // 1. FIXED: Isolate direct opponent defenders currently in the pool.
            // Since pool is already pre-filtered in the SEE loop by the active live side masks,
            // we simply subtract the active stm's known pieces. Everything left belongs to alive opponents!
            let opponent_attackers_in_pool = pool & !self.all_pieces[stm];
            
            // If the opponent has ANY direct leaper or remaining attacker currently targeting this square,
            // the king cannot move into check. Break the chain immediately.
            let direct_opponent_leapers = opponent_attackers_in_pool & (
                self.pawns[opponent_side] | self.knights[opponent_side] | self.kings[opponent_side]
            );
            if direct_opponent_leapers != 0 {
                return (0, BoardPiece::NONE);
            }

            // 2. Check live sliding attacks targeting the square if the king lands on it.
            let king_bit = 1u64 << kings.trailing_zeros();
            let safe_occ = occupancy ^ king_bit;

            // Generate fresh diagonal paths from target_sq and ensure no ghost sliders are registered
            let diagonal_threats = bishop_attack_paths(target_sq, safe_occ) 
                & opponent_attackers_in_pool
                & (self.bishops[opponent_side] | self.queens[opponent_side]);
            if diagonal_threats != 0 {
                return (0, BoardPiece::NONE);
            }

            // Generate fresh orthogonal paths from target_sq and ensure no ghost sliders are registered
            let orthogonal_threats = rook_attack_paths(target_sq, safe_occ) 
                & opponent_attackers_in_pool
                & (self.rooks[opponent_side] | self.queens[opponent_side]);
            if orthogonal_threats != 0 {
                return (0, BoardPiece::NONE);
            }

            return (kings.trailing_zeros() as u8, if stm == 0 { BoardPiece::WKING } else { BoardPiece::BKING });
        }

        (0, BoardPiece::NONE)
    }

    /// Aggregates non-sliding attackers targeting a square using your native function hooks.
    pub fn get_leaper_attacks_to(&self, target_sq: u8) -> u64 {
        let mut leapers = 0u64;
        let target_bb = 1u64 << target_sq;

        // 1. Knights & Kings (Standard Lookups)
        leapers |= KNIGHT_ATTACKS[target_sq as usize] & (self.knights[0] | self.knights[1]);
        leapers |= KING_ATTACKS[target_sq as usize] & (self.kings[0] | self.kings[1]);

         // 2. Pawns (Branchless - STM Perspective)
    
        // --- White Pawns ---
        // White Pawn attacks left
        leapers |= (((self.pawns[0] & NOT_A_FILE) << 9) & target_bb) >> 9;
        // White Pawn attacks right
        leapers |= (((self.pawns[0] & NOT_H_FILE) << 7) & target_bb) >> 7;

        // --- Black Pawns ---
        // Black Pawn attacks left
        leapers |= (((self.pawns[1] & NOT_H_FILE) >> 7) & target_bb) << 7;
        // Black Pawn attacks right
        leapers |= (((self.pawns[1] & NOT_A_FILE) >> 9) & target_bb) << 9;

        leapers
    }

    // Static Exchange Evaluation (SEE)
    pub fn static_exchange_evaluation(&self, forward_move: ForwardMove) -> i16 {
        let from_sq = forward_move.start_sq;
        let target_sq = forward_move.end_sq;
        let capture_sq = forward_move.capture_sq;

        let stm = self.player_index(self.active_player);
        
        let mut capture_gain = [0i16; 32];
        let mut defender = 0;

        let piece_moving = self.mailbox[from_sq as usize];
        let piece_captured = self.mailbox[capture_sq as usize];
        
        let base_captured_value = piece_value(piece_captured);
        let mut current_piece_value = piece_value(piece_moving);
        
        let promotion_bonus = promotion_bonus(forward_move);
        capture_gain[defender] = base_captured_value + promotion_bonus;
        current_piece_value += promotion_bonus;

        // 1. Initial State Sandbox
        let mut occupancy = self.occupied;
        occupancy &= !(1u64 << from_sq);
        occupancy &= !(1u64 << capture_sq);

        if capture_sq == target_sq {
            occupancy |= 1u64 << target_sq;
        }

        let mut side_pieces = [self.all_pieces[0], self.all_pieces[1]];
        side_pieces[stm] &= !(1u64 << from_sq);
        side_pieces[1 - stm] &= !(1u64 << capture_sq);
        
        if capture_sq == target_sq {
            occupancy |= 1u64 << target_sq;
        }

        let initial_pinned_white = self.compute_pinned_mask_with_custom_state(self.occupied, self.all_pieces[0], 0);
        let initial_pinned_black = self.compute_pinned_mask_with_custom_state(self.occupied, self.all_pieces[1], 1);
        let pinned_masks = [initial_pinned_white, initial_pinned_black];

        // 2. Initial Attack Pool Construction
        let mut attackers_pool = self.get_leaper_attacks_to(target_sq) 
            | self.get_slider_attackers(target_sq, occupancy);
        attackers_pool &= side_pieces[0] | side_pieces[1];

        let mut loop_stm = 1 - stm;

        // 3. Recapture Simulation Loop
        while attackers_pool != 0 {
            let defender_king_sq = self.kings[loop_stm].trailing_zeros() as u8;
            
            // Fast, cheap retrieval using our single-pass pre-calculated pin-mask
            let current_pinned_mask = pinned_masks[loop_stm];

            let (att_sq, att_piece) = self.get_least_valuable_attacker(
                attackers_pool, 
                current_pinned_mask, 
                defender_king_sq, 
                target_sq,
                loop_stm,
                occupancy
            );

            if att_piece == BoardPiece::NONE {
                break; 
            }

            defender += 1;
            // Standard Negamax exchange calculation step
            capture_gain[defender] = current_piece_value - capture_gain[defender - 1];
            current_piece_value = piece_value(att_piece);

            // Update sandbox tracking arrays
            let clear_mask = !(1u64 << att_sq);
            occupancy &= clear_mask;
            side_pieces[loop_stm] &= clear_mask;

            // Re-calculate attacks looking through cleared slider lines
            attackers_pool = self.get_leaper_attacks_to(target_sq) 
                | self.get_slider_attackers(target_sq, occupancy);
            attackers_pool &= side_pieces[0] | side_pieces[1];

            loop_stm = 1 - loop_stm;
        }

        // 4. Corrected Minimax Rollback (Min Bound Truncation)
        while defender > 0 {
            capture_gain[defender - 1] = (-capture_gain[defender]).min(capture_gain[defender - 1]);
            defender -= 1;
        }

        capture_gain[0]
    }

}
