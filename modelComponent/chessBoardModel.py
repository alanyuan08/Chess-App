# Import Model
from modelComponent.chessPieceModel import ChessPieceModel
from modelComponent.moveCommand import MoveCommand

# Enum 
from appEnums import PieceType, Player, MoveCommandType

# Used to check King Safety s
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

		# Used to check En passant
		self.enPassantColumn = None

		# Used to check if player can Castle
		self.blackKingMoved = False
		self.blackKingSideRookMoved = False
		self.blackQueenSideRookMoved = False

		self.whiteKingMoved = False
		self.whiteKingSideRookMoved = False
		self.whiteQueenSideRookMoved = False 

	# Validate the command and check if it satisfys king safety 
	def validateAndReturnCommand(self, initRow: int, initCol: int, targetRow: int, targetCol: int, player: Player):
		# It's not your turn to move
		if player != self.playerTurn:
			return None

		# Validate the Move Command is a Possible Move
		moveCommand = None
		possibleMoves = self._possibleMoves(initRow, initCol, player)
		for cmd in possibleMoves:
			if cmd.endRow == targetRow and cmd.endCol == targetCol:
				moveCommand = cmd

		# No Valid Moves
		if moveCommand == None:
			return None
		# Validate King Safety
		elif self.kingSafety(moveCommand, player):
			return moveCommand
		else:
			return None

	# Validate King Safety
	def kingSafety(self, moveCommand: MoveCommand, player: Player):
		# Validate King Safety
		testBoard = copy.deepcopy(self)
		testBoard.movePiece(moveCommand)

		opponent = ChessBoardModel.returnOpponent(player)
		opponentAttackTargets = testBoard._totalAttackTargets(opponent)

		for row in range(0, 8):
			for col in range(0, 8):
				if testBoard.board[row][col] != None and testBoard.board[row][col].player == player:
					if testBoard.board[row][col].type == PieceType.KING and (row, col) in opponentAttackTargets:
						return False

		return True

	# Moves the Piece if its valid
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

				else:
					self.whiteKingMoved = True
					self.whiteQueenSideRookMoved = True

					# Move the King 2 steps to the left
					self.board[0][2], self.board[0][4] = self.board[0][4], self.board[0][2]

					# Move the Rook to the right of the king
					self.board[0][3] = self.board[0][0]
					self.board[0][0] = None

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

				else:
					self.whiteKingMoved = True
					self.whiteQueenSideRookMoved = True

					# Move the King 2 steps to the right
					self.board[0][6], self.board[0][4] = self.board[0][4], self.board[0][6]

					# Move the Rook to the right of the king
					self.board[0][5] = self.board[0][7]
					self.board[0][7] = None

			# Move Piece
			case MoveCommandType.MOVE | MoveCommandType.CAPTURE:
				# Move the piece from start to end
				startingPiece = self.board[moveCommand.startRow][moveCommand.startCol]
				self.board[moveCommand.endRow][moveCommand.endCol] = startingPiece
				self.board[moveCommand.startRow][moveCommand.startCol] = None

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
		self.playerTurn = ChessBoardModel.returnOpponent(self.playerTurn)

		return

	# Opponent Takes Local Min Move
	def takeBestMove(self, player: Player):
		returnCommand = None
		cmdBoardValue = (-1) * 10**10

		for cmd in self._listPossibleMovesForPlayer(player):
			testBoard = copy.deepcopy(self)
			testBoard.movePiece(cmd)

			computedPair = testBoard.computeBoardValue()

			diffValue = 0
			if Player == Player.WHITE:
				diffValue = computedPair[1] - computedPair[0]
			else:
				diffValue = computedPair[0] - computedPair[1]

			if diffValue > cmdBoardValue:
				cmdBoardValue = diffValue
				returnCommand = cmd

		self.movePiece(returnCommand)
		return

	# Compute Best Move for Player
	def computeBestValue(self, player: Player):
		returnCommand = None
		cmdBoardValue = (-1) * 10**10

		for cmd in self._listPossibleMovesForPlayer(player):
			testBoard = copy.deepcopy(self)
			testBoard.movePiece(cmd)

			opponentPlayer = ChessBoardModel.returnOpponent(player)
			testBoard.takeBestMove(opponentPlayer)
			computedPair = testBoard.computeBoardValue()

			diffValue = 0
			if Player == Player.WHITE:
				diffValue = computedPair[1] - computedPair[0]
			else:
				diffValue = computedPair[0] - computedPair[1]

			if diffValue > cmdBoardValue:
				cmdBoardValue = diffValue
				returnCommand = cmd

		return returnCommand


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
					
		# Compute Based on Possible Value
		blackValue += self._computeAttackDefendValue(Player.BLACK)
		whiteValue += self._computeAttackDefendValue(Player.WHITE)

		return [blackValue, whiteValue]

	def _computeAttackDefendValue(self, player: Player):
		totalValue = 0

		# Move Value
		for cmd in self._listPossibleMovesForPlayer(Player):
			match cmd.moveType:
				# Queen Side Castle
				case MoveCommandType.QUEENSIDECASTLE:
					totalValue += 10
				case MoveCommandType.KINGSIDECASTLE:
					totalValue += 10
				case MoveCommandType.ENPASSANT:
					totalValue += 50
				case MoveCommandType.CAPTURE:
					initPieceValue = self.board[cmd.startRow][cmd.startCol].pieceValue()
					targetValue = self.board[cmd.endRow][cmd.endCol].pieceValue()
					if targetValue <= initPieceValue:
						totalvalue += targetValue * 0.2
					else:
						totalvalue += targetValue * 0.3
				case MoveCommandType.PROMOTE:
					totalValue += 100

		# Capture Value
		opponentPlayer = ChessBoardModel.returnOpponent(player)
		for cmd in self._listPossibleMovesForPlayer(opponentPlayer):
			match cmd.moveType:
				case MoveCommandType.CAPTURE:
					initPieceValue = self.board[cmd.startRow][cmd.startCol].pieceValue()
					targetValue = self.board[cmd.endRow][cmd.endCol].pieceValue()

					if initPieceValue <= targetValue:
						totalValue -= targetValue
					else:
						totalValue -= targetValue * 0.5

		return totalValue

	# Return a list of move commands for a player
	def _listPossibleMovesForPlayer(self, player: Player):
		totalMove = []

		for row in range(0, 8):
			for col in range(0, 8):
				if self.board[row][col] != None and self.board[row][col].player == player:
					for command in self._possibleMoves(row, col, player):
						if self.kingSafety(command, player):
							totalMove.append(command)

		return totalMove

	@staticmethod
	def returnOpponent(player: Player):
		if player == Player.WHITE:
			return Player.BLACK
		else:
			return Player.WHITE

	# This returns all possibleMoves as a move object
	def _possibleMoves(self, row: int, col: int, player: Player):
		if self.board[row][col] == None:
			return []
		else:
			match self.board[row][col].type:
				case PieceType.KING:
					returnMoves = []
					opponent = ChessBoardModel.returnOpponent(self.board[row][col].player)
					opponentAttackTargets = self._totalAttackTargets(opponent)
					
					for possibleMoves in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
						newRow = row + possibleMoves[0]
						newCol = col + possibleMoves[1]
						if newRow >= 0 and newRow < 8 and newCol >= 0 and newCol < 8:
							if self.board[newRow][newCol] == None:
								if (newRow, newCol) not in opponentAttackTargets:
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.MOVE, player))
							elif self.board[newRow][newCol].player != self.board[row][col].player:
								returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.CAPTURE, player))

					# Black Queen Side Castle
					if not self.blackKingMoved and not self.blackQueenSideRookMoved:
						nullBlock = 0
						for i in [1, 2, 3]:
							if self.board[7][i] == None and (7, i) not in opponentAttackTargets:
								nullBlock += 1

						kingSafety = True
						if (row, col) in opponentAttackTargets:
							kingSafety = False
							
						if nullBlock == 3 and kingSafety:
							returnMoves.append(MoveCommand(row, col, row, col-2, MoveCommandType.QUEENSIDECASTLE, player))

					# Black King Side Castle
					if not self.blackKingMoved and not self.blackKingSideRookMoved:
						nullBlock = 0
						for i in [5, 6]:
							if self.board[7][i] == None and (7, i) not in opponentAttackTargets:
								nullBlock += 1

						kingSafety = True
						if (row, col) in opponentAttackTargets:
							kingSafety = False

						if nullBlock == 2 and kingSafety:
							returnMoves.append(MoveCommand(row, col, row, col+2, MoveCommandType.KINGSIDECASTLE, player))

					# White Queen Side Castle
					if not self.whiteKingMoved and not self.whiteQueenSideRookMoved:
						nullBlock = 0
						for i in [1, 2, 3]:
							if self.board[0][i] == None and (0, i) not in opponentAttackTargets:
								nullBlock += 1

						kingSafety = True
						if (row, col) in opponentAttackTargets:
							kingSafety = False

						if nullBlock == 3 and kingSafety:
							returnMoves.append(MoveCommand(row, col, row, col-2, MoveCommandType.QUEENSIDECASTLE, player))

					# White King Side Castle
					if not self.whiteKingMoved and not self.whiteKingSideRookMoved:
						nullBlock = 0
						for i in [5, 6]:
							if self.board[0][i] == None and (0, i) not in opponentAttackTargets:
								nullBlock += 1

						kingSafety = True
						if (row, col) in opponentAttackTargets:
							kingSafety = False

						if nullBlock == 2 and kingSafety:
							returnMoves.append(MoveCommand(row, col, row, col+2, MoveCommandType.KINGSIDECASTLE, player))

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
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.MOVE, player))
								elif self.board[newRow][newCol].player != self.board[row][col].player:
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.CAPTURE, player))
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
							if self.board[newRow][newCol] == None:
								returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.MOVE, player))
							elif self.board[newRow][newCol].player != self.board[row][col].player:
								returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.CAPTURE, player))

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
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.MOVE, player))
								elif self.board[newRow][newCol].player != self.board[row][col].player:
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.CAPTURE, player))
									break
								else:
									break
							else:
								break

					return returnMoves

				case PieceType.PAWN:
					returnMoves = []
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
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.MOVE, player))
								elif self.board[newRow][newCol].player != self.board[row][col].player:
									returnMoves.append(MoveCommand(row, col, newRow, newCol, MoveCommandType.CAPTURE, player))
									break
								else:
									break
							else:
								break

					return returnMoves

	# This returns all attack targets for a player
	def _totalAttackTargets(self, player: Player):
		attackSquare = {}
		for row in range(0, 8):
			for col in range(0, 8):
				if self.board[row][col] != None and self.board[row][col].player == player:
					possibleMoves = self._attackTargets(row, col)

					for move in possibleMoves:
						moveTarget = move[0]
						if moveTarget not in attackSquare:
							attackSquare[move] = True
		
		return attackSquare

	# Return the list of targets the opponent could attack as [row, col]
	def _attackTargets(self, row: int, col: int):
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

