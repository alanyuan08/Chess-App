# Import Model
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.moveCommand import MoveCommand

# Enum 
from appEnums import PieceType, Player, MoveCommandType

# Used to test next move
import copy

# Controller 
class ChessBoardModel():
	def __init__(self, humanPlayers: list[Player]):
		# Create the Chess Board
		self.board = ChessPieceModel.createInitChessBoard()

		# Init Player as White
		self.playerTurn = Player.WHITE

		# Set Human Player
		self.humanPlayer = humanPlayers

		# En Passant Column - Set after pawn move, then cleared 
		self.enPassantColumn = None

		# Used to check if player can Castle
		self.blackKingMoved = False
		self.blackKingSideRookMoved = False
		self.blackQueenSideRookMoved = False

		self.whiteKingMoved = False
		self.whiteKingSideRookMoved = False
		self.whiteQueenSideRookMoved = False 

		# Used to Check for King Safety
		self.whiteKingSquare = (0, 4)
		self.blackKingSquare = (7, 4)

	# Validate the Move
	def validateMove(self, initRow: int, initCol: int, targetRow: int, targetCol: int, player: Player):
		# It's not your turn to move
		if player != self.playerTurn:
			return None

		# Validate the Move Command is a Possible Move
		possibleMoves = self._possibleMoveFromPosition(initRow, initCol, player)
		for cmd in possibleMoves:
			if cmd.endRow == targetRow and cmd.endCol == targetCol:
				return cmd

		return None

	# Moves the Piece
	def movePiece(self, moveCommand: MoveCommand):
		# Set enPassant to Null - Reset this if the opponent does a double pawn move
		self.enPassantColumn = None

		match moveCommand.moveType:
			# Queen Side Castle
			case MoveCommandType.QUEENSIDECASTLE:
				if moveCommand.player == Player.BLACK:
					self.blackKingMoved = True
					self.blackQueenSideRookMoved = True

					# Move the King 2 steps to the left
					self.board[7][2], self.board[7][4] = self.board[7][4], self.board[7][2]

					# Move the Rook to the right of the king
					self.board[7][3] = self.board[7][0]
					self.board[7][0] = None

					# Set the Black King Square
					self.blackKingSquare = (7, 2)
					self.blackKingMoved = True

				elif moveCommand.player == Player.WHITE:
					self.whiteKingMoved = True
					self.whiteQueenSideRookMoved = True

					# Move the King 2 steps to the left
					self.board[0][2], self.board[0][4] = self.board[0][4], self.board[0][2]

					# Move the Rook to the right of the king
					self.board[0][3] = self.board[0][0]
					self.board[0][0] = None

					# Set the White King Square
					self.whiteKingSquare = (0, 2)
					self.whiteKingMoved = True

			# King Side Castle
			case MoveCommandType.KINGSIDECASTLE:
				if moveCommand.player == Player.BLACK:
					self.blackKingMoved = True
					self.blackQueenSideRookMoved = True

					# Move the King 2 steps to the right
					self.board[7][6], self.board[7][4] = self.board[7][4], self.board[7][6]

					# Move the Rook to the right of the king
					self.board[7][5] = self.board[7][7]
					self.board[7][7] = None

					# Set the Black King Square
					self.blackKingSquare = (7, 6)
					self.blackKingMoved = True

				elif moveCommand.player == Player.WHITE:
					self.whiteKingMoved = True
					self.whiteQueenSideRookMoved = True

					# Move the King 2 steps to the right
					self.board[0][6], self.board[0][4] = self.board[0][4], self.board[0][6]

					# Move the Rook to the right of the king
					self.board[0][5] = self.board[0][7]
					self.board[0][7] = None

					# Set the White King Square
					self.whiteKingSquare = (0, 6)
					self.whiteKingMoved = True

			# Move Piece
			case MoveCommandType.MOVE | MoveCommandType.CAPTURE:
				# Move the piece from start to end
				startingPiece = self.board[moveCommand.startRow][moveCommand.startCol]
				self.board[moveCommand.endRow][moveCommand.endCol] = startingPiece
				self.board[moveCommand.startRow][moveCommand.startCol] = None

				# Update the King Square
				if startingPiece.type == PieceType.KING:
					if moveCommand.player == Player.BLACK:
						self.blackKingSquare = (moveCommand.endRow, moveCommand.endCol)
						self.blackKingMoved = True
					elif moveCommand.player == Player.WHITE:
						self.whiteKingSquare = (moveCommand.endRow, moveCommand.endCol)
						self.whiteKingMoved = True

			# Double Pawn Move
			case MoveCommandType.PAWNOPENMOVE:
				# Move the piece from start to end
				startingPiece = self.board[moveCommand.startRow][moveCommand.startCol]
				self.board[moveCommand.endRow][moveCommand.endCol] = startingPiece
				self.board[moveCommand.startRow][moveCommand.startCol] = None

				self.enPassantColumn = moveCommand.endCol

			# Promote Pawn
			case MoveCommandType.PROMOTE:
				# Promote the Pawn to a Queen
				self.board[moveCommand.startRow][moveCommand.startCol] = None
				self.board[moveCommand.endRow][moveCommand.endCol] = ChessPieceModel(moveCommand.player, PieceType.QUEEN)

			# En Passant
			case MoveCommandType.ENPASSANT:
				# Move the piece from start to end
				startingPiece = self.board[moveCommand.startRow][moveCommand.startCol]

				self.board[moveCommand.endRow][moveCommand.endCol] = startingPiece
				self.board[moveCommand.startRow][moveCommand.startCol] = None

				# Remove En Passant Pawn
				self.board[moveCommand.startRow][moveCommand.endCol] = None

		# Swap the Player Turn
		self.playerTurn = ChessBoardModel.opponent(self.playerTurn)
		return

	# Compute Board Value -> [White, Black]
	def computeBoardValue(self):
		whiteValue = 0
		blackValue = 0

		# Compute Base Value
		for row in range(0, 8):
			for col in range(0, 8):
				if self.board[row][col] != None:
					if self.board[row][col].player == Player.WHITE:
						whiteValue += self.board[row][col].pieceValue()
					else:
						blackValue += self.board[row][col].pieceValue()
					
		return [whiteValue, valueValue]

	@staticmethod
	def opponent(player: Player):
		if player == Player.WHITE:
			return Player.BLACK
		else:
			return Player.WHITE

	# Return a list of move commands for a player
	def listPossibleMovesForPlayer(self, player: Player):
		totalMove = []

		for row in range(0, 8):
			for col in range(0, 8):
				if self.board[row][col] != None and self.board[row][col].player == player:
					totalMove += self._possibleMoveFromPosition(row, col, player)

		return totalMove

	# This returns all possibleMoves as a move object
	def _possibleMoveFromPosition(self, row: int, col: int, player: Player):
		returnMoves = []

		if self.board[row][col] == None:
			return returnMoves
		else:
			match self.board[row][col].type:
				case PieceType.KING:
					for possibleMoves in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
						newRow = row + possibleMoves[0]
						newCol = col + possibleMoves[1]
						if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
							if self.board[newRow][newCol] == None:
								returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.MOVE, player))
							elif self.board[newRow][newCol].player != self.board[row][col].player:
								returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.CAPTURE, player))

					# King + Move to Castle are not in check
					opponent = ChessBoardModel.opponent(self.board[row][col].player)
					opponentAttackTargets = self.allOpponentCaptureTargets(opponent)

					# Black Queen Side Castle
					if not self.blackKingMoved and not self.blackQueenSideRookMoved:
						nullBlock = 0
						for i in [1, 2, 3]:
							if self.board[7][i] == None and (7, i) not in opponentAttackTargets:
								nullBlock += 1
							
						if nullBlock == 3 and (row, col) not in opponentAttackTargets:
							returnMoves.append(MoveCommand(row, col, row, col-2, MoveCommandType.QUEENSIDECASTLE, player))

					# Black King Side Castle
					if not self.blackKingMoved and not self.blackKingSideRookMoved:
						nullBlock = 0
						for i in [5, 6]:
							if self.board[7][i] == None and (7, i) not in opponentAttackTargets:
								nullBlock += 1

						if nullBlock == 2 and (row, col) not in opponentAttackTargets:
							returnMoves.append(MoveCommand(row, col, row, col+2, MoveCommandType.KINGSIDECASTLE, player))

					# White Queen Side Castle
					if not self.whiteKingMoved and not self.whiteQueenSideRookMoved:
						nullBlock = 0
						for i in [1, 2, 3]:
							if self.board[0][i] == None and (0, i) not in opponentAttackTargets:
								nullBlock += 1

						if nullBlock == 3 and (row, col) not in opponentAttackTargets:
							returnMoves.append(MoveCommand(row, col, row, col-2, MoveCommandType.QUEENSIDECASTLE, player))

					# White King Side Castle
					if not self.whiteKingMoved and not self.whiteKingSideRookMoved:
						nullBlock = 0
						for i in [5, 6]:
							if self.board[0][i] == None and (0, i) not in opponentAttackTargets:
								nullBlock += 1

						if nullBlock == 2 and (row, col) not in opponentAttackTargets:
							returnMoves.append(MoveCommand(row, col, row, col+2, MoveCommandType.KINGSIDECASTLE, player))

				case PieceType.QUEEN:
					for direction in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
						i = 1
						while(True):
							newRow = row + direction[0] * i
							newCol = col + direction[1] * i
							i += 1

							if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
								if self.board[newRow][newCol] == None:
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.MOVE, player))
								elif self.board[newRow][newCol].player != self.board[row][col].player:
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.CAPTURE, player))
									break
								else:
									break
							else:
								break

				case PieceType.KNIGHT:
					for possibleMoves in [(2, 1), (1, 2), (-2, -1), (-1, -2), (-2, 1), (-1, 2), (2, -1), (1, -2)]:
						newRow = row + possibleMoves[0]
						newCol = col + possibleMoves[1]
						if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
							if self.board[newRow][newCol] == None:
								returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.MOVE, player))
							elif self.board[newRow][newCol].player != self.board[row][col].player:
								returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.CAPTURE, player))

				case PieceType.ROOK:
					for direction in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
						i = 1
						while(True):
							newRow = row + direction[0] * i
							newCol = col + direction[1] * i
							i += 1

							if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
								if self.board[newRow][newCol] == None:
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.MOVE, player))
								elif self.board[newRow][newCol].player != self.board[row][col].player:
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.CAPTURE, player))
									break
								else:
									break
							else:
								break

				case PieceType.PAWN:
					# Black Player - Pawn Moves Down
					if self.board[row][col].player == Player.BLACK:
						if row == 1:
							# Promote
							if col > 0:
								if self.board[0][col-1] != None and self.board[0][col-1].player != player:
									returnMoves.append(MoveCommand(1, col, 0, col-1, MoveCommandType.PROMOTE, player))

							if col < 7:
								if self.board[0][col+1] != None and self.board[0][col+1].player != player:
									returnMoves.append(MoveCommand(1, col, 0, col+1, MoveCommandType.PROMOTE, player))

							if self.board[0][col] == None:
								returnMoves.append(MoveCommand(1, col, 0, col, MoveCommandType.PROMOTE, player))

						else:
							# Pawn Capture
							if col > 0 and self.board[row-1][col-1] != None and self.board[row-1][col-1].player != self.board[row][col].player:
								returnMoves.append(MoveCommand(row, col, row-1, col-1, MoveCommandType.CAPTURE, player))

							if col < 7 and self.board[row-1][col+1] != None and self.board[row-1][col+1].player != self.board[row][col].player:
								returnMoves.append(MoveCommand(row, col, row-1, col+1, MoveCommandType.CAPTURE, player))

							# Pawn Move
							if self.board[row-1][col] == None:
								returnMoves.append(MoveCommand(row, col, row-1, col, MoveCommandType.MOVE, player))

							if row == 6 and self.board[row-1][col] == None and self.board[row-2][col] == None and row == 6:
								returnMoves.append(MoveCommand(row, col, row-2, col, MoveCommandType.PAWNOPENMOVE, player))

							# En passant
							if row == 3 and (self.enPassantColumn == col-1 or self.enPassantColumn == col+1):
								returnMoves.append(MoveCommand(row, col, row-1, self.enPassantColumn , MoveCommandType.ENPASSANT, player))

					# White Player - Pawn Moves Up
					elif self.board[row][col].player == Player.WHITE:
						if row == 6:
							# Promote
							if col > 0:
								if self.board[7][col-1] != None and self.board[7][col-1].player != player:
									returnMoves.append(MoveCommand(6, col, 7, col-1, MoveCommandType.PROMOTE, player))

							if col < 7:
								if self.board[7][col+1] != None and self.board[7][col+1].player != player:
									returnMoves.append(MoveCommand(6, col, 7, col+1, MoveCommandType.PROMOTE, player))

							if self.board[7][col] == None:
								returnMoves.append(MoveCommand(6, col, 7, col, MoveCommandType.PROMOTE, player))

						else:
							# Pawn Capture
							if col > 0 and self.board[row+1][col-1] != None and self.board[row+1][col-1].player != self.board[row][col].player:
								returnMoves.append(MoveCommand(row, col, row+1, col-1, MoveCommandType.CAPTURE, player))

							if col < 7 and self.board[row+1][col+1] != None and self.board[row+1][col+1].player != self.board[row][col].player:
								returnMoves.append(MoveCommand(row, col, row+1, col+1, MoveCommandType.CAPTURE, player))

							# Pawn Move
							if self.board[row+1][col] == None:
								returnMoves.append(MoveCommand(row, col, row+1, col, MoveCommandType.MOVE, player))

							if row == 1 and self.board[row+1][col] == None and self.board[row+2][col] == None and row == 1:
								returnMoves.append(MoveCommand(row, col, row+2, col, MoveCommandType.PAWNOPENMOVE, player))

							# En passant
							if row == 4 and (self.enPassantColumn == col-1 or self.enPassantColumn == col+1):
								returnMoves.append(MoveCommand(row, col, row+1, self.enPassantColumn , MoveCommandType.ENPASSANT, player))

				case PieceType.BISHOP:
					for direction in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
						i = 1
						while(True):
							newRow = row + direction[0] * i
							newCol = col + direction[1] * i
							i += 1

							if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
								if self.board[newRow][newCol] == None:
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.MOVE, player))
								elif self.board[newRow][newCol].player != self.board[row][col].player:
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.CAPTURE, player))
									break
								else:
									break
							else:
								break

			# Validate For King Safety
			kingSafeArray = [] 
			for command in returnMoves:
				if self.kingSafety(command, player):
					kingSafeArray.append(command)

			return kingSafeArray

	# Validate King Safety
	def kingSafety(self, moveCommand: MoveCommand, player: Player):
		# Validate King Safety
		testBoard = copy.deepcopy(self)
		testBoard.movePiece(moveCommand)

		opponent = ChessBoardModel.opponent(player)
		if player == Player.WHITE:
			if testBoard.whiteKingSquare in testBoard.allOpponentCaptureTargets(opponent):
				return False
			else:
				return True
		elif player == Player.BLACK:
			if testBoard.blackKingSquare in testBoard.allOpponentCaptureTargets(opponent):
				return False
			else:
				return True

	# This is used to determine king safety and castle
	def allOpponentCaptureTargets(self, player: Player):
		attackSquare = {}
		for row in range(0, 8):
			for col in range(0, 8):
				if self.board[row][col] != None and self.board[row][col].player == player:
					possibleMoves = self._opponentCaptureTargetsFromLocation(row, col)
					for move in possibleMoves:
						attackSquare[move] = True
		
		return attackSquare

	# Return the list of targets the opponent could attack as [row, col]
	def _opponentCaptureTargetsFromLocation(self, row: int, col: int):
		if self.board[row][col] == None:
			return []
		else:
			match self.board[row][col].type:
				case PieceType.KING:
					returnMoves = []
					for possibleMoves in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
						newRow = row + possibleMoves[0]
						newCol = col + possibleMoves[1]
						if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
							if self.board[newRow][newCol] == None or self.board[newRow][newCol].player != self.board[row][col].player:
								returnMoves.append((newRow, newCol))

					return returnMoves

				case PieceType.QUEEN:
					returnMoves = []
					for direction in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
						i = 1
						while(True):
							newRow = row + direction[0] * i
							newCol = col + direction[1] * i
							i += 1

							if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
								if self.board[newRow][newCol] == None:
									returnMoves.append((newRow, newCol))
								elif self.board[newRow][newCol].player != self.board[row][col].player:
									returnMoves.append((newRow, newCol))
									break
								else:
									break
							else:
								break

					return returnMoves

				case PieceType.KNIGHT:
					returnMoves = []
					for possibleMoves in [(2, 1), (1, 2), (-2, -1), (-1, -2), (-2, 1), (-1, 2), (2, -1), (1, -2)]:
						newRow = row + possibleMoves[0]
						newCol = col + possibleMoves[1]
						if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
							if self.board[newRow][newCol] == None or self.board[newRow][newCol].player != self.board[row][col].player:
								returnMoves.append((newRow, newCol))

					return returnMoves

				case PieceType.ROOK:
					returnMoves = []
					for direction in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
						i = 1
						while(True):
							newRow = row + direction[0] * i
							newCol = col + direction[1] * i
							i += 1

							if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
								if self.board[newRow][newCol] == None:
									returnMoves.append((newRow, newCol))
								elif self.board[newRow][newCol].player != self.board[row][col].player:
									returnMoves.append((newRow, newCol))
									break
								else:
									break
							else:
								break

					return returnMoves

				case PieceType.PAWN:
					returnMoves = []
					if self.board[row][col].player == Player.BLACK:
						if row > 0:
							# Capture Point
							if col > 0:
								returnMoves.append((row-1, col-1))

							if col < 7:
								returnMoves.append((row-1, col+1))

					elif self.board[row][col].player == Player.WHITE:
						if row < 7:
							# Capture Point
							if col > 0:
								returnMoves.append((row+1, col-1))

							if col < 7:
								returnMoves.append((row+1, col+1))

					return returnMoves

				case PieceType.BISHOP:
					returnMoves = []
					for direction in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
						i = 1
						while(True):
							newRow = row + direction[0] * i
							newCol = col + direction[1] * i
							i += 1

							if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
								if self.board[newRow][newCol] == None:
									returnMoves.append((newRow, newCol))
								elif self.board[newRow][newCol].player != self.board[row][col].player:
									returnMoves.append((newRow, newCol))
									break
								else:
									break
							else:
								break

					return returnMoves

