# Enum 
from appEnums import PieceType, Player, MoveCommandType

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Model
from modelComponent.moveCommand import MoveCommand

# Controller 
class ChessBoardModel():
    def __init__(self):
        # Create the Chess Board
        self.board = []

        # Init Player as White
        self.playerTurn = Player.WHITE

        # Set Human Player
        self.humanPlayers = []

        # En Passant Column - Set after pawn move, then cleared 
        self.enPassantColumn = None

        # Used to check if player can Castle
        self.blackKingSideCanCastle = True
        self.blackQueenSideCanCastle = True

        self.whiteKingSideCanCastle = True
        self.whiteQueenSideCanCastle = True

        # Used to Keep Score for Castle
        self.whiteCastled = False
        self.blackCastled = False

        # Used to Check for King Safety
        self.whiteKingSquareRow = 0
        self.whiteKingSquareCol = 4

        self.blackKingSquareRow = 7
        self.blackKingSquareCol = 4
        
    # Danger Moves
    quiescenceMoveCmd = [MoveCommandType.PROMOTE, MoveCommandType.CAPTURE, MoveCommandType.ENPASSANT]

    # Store NonBoard States -> Used for Undo
    def nonBoardState(self):
        nonBoardStates = {}
        nonBoardStates['enPassantColumn'] = self.enPassantColumn 
        
        nonBoardStates['blackKingSideCanCastle'] = self.blackKingSideCanCastle
        nonBoardStates['blackQueenSideCanCastle'] = self.blackQueenSideCanCastle

        nonBoardStates['whiteKingSideCanCastle'] = self.whiteKingSideCanCastle
        nonBoardStates['whiteQueenSideCanCastle'] = self.whiteQueenSideCanCastle

        nonBoardStates['whiteCastled'] = False
        nonBoardStates['blackCastled'] = False
        
        nonBoardStates['whiteKingSquareRow'] = self.whiteKingSquareRow
        nonBoardStates['whiteKingSquareCol'] = self.whiteKingSquareCol

        nonBoardStates['blackKingSquareRow'] = self.blackKingSquareRow
        nonBoardStates['blackKingSquareCol'] = self.blackKingSquareCol

        return nonBoardStates

    # Reset Board State
    def resetBoardState(self, nonBoardStates):
        self.enPassantColumn = nonBoardStates['enPassantColumn']
        
        self.blackKingSideCanCastle = nonBoardStates['blackKingSideCanCastle'] 
        self.blackQueenSideCanCastle = nonBoardStates['blackQueenSideCanCastle']

        self.whiteKingSideCanCastle = nonBoardStates['whiteKingSideCanCastle']
        self.whiteQueenSideCanCastle = nonBoardStates['whiteQueenSideCanCastle']

        self.whiteCastled = nonBoardStates['whiteCastled']
        self.blackCastled = nonBoardStates['blackCastled']
        
        self.whiteKingSquareRow = nonBoardStates['whiteKingSquareRow'] 
        self.whiteKingSquareCol = nonBoardStates['whiteKingSquareCol']

        self.blackKingSquareRow = nonBoardStates['blackKingSquareRow']
        self.blackKingSquareCol = nonBoardStates['blackKingSquareCol']

        return

    @staticmethod
    def opponent(player: Player):
        if player == Player.WHITE:
            return Player.BLACK
        else:
            return Player.WHITE

    # Validate the Move
    def validateMove(self, initRow: int, initCol: int, targetRow: int, targetCol: int, player: Player):
        # It's not your turn to move
        if player != self.playerTurn:
            return None

        # Validate the Move Command is a Possible Move
        targetPiece = self.board[initRow][initCol]
        if targetPiece == None:
            return None

        if targetPiece.player == player:
            possibleMoves = targetPiece.possibleMoves(self)
            for cmd in possibleMoves:
                if cmd.endRow == targetRow and cmd.endCol == targetCol:
                    return cmd

        return None

    # Validate King Safety for player after making move
    def validateKingSafety(self, cmd: MoveCommand):
        # This is explictly defined here to avoid confusion after the move
        currentPlayer = self.playerTurn

        nonBoardState = self.nonBoardState()
        removedPiece = self.movePiece(cmd)

        returnValue = True

        if currentPlayer == Player.BLACK:
            kingPiece = self.board[self.blackKingSquareRow][self.blackKingSquareCol]
            returnValue = kingPiece.evaluateKingSafety(self)

        elif currentPlayer == Player.WHITE:
            kingPiece = self.board[self.whiteKingSquareRow][self.whiteKingSquareCol]
            returnValue = kingPiece.evaluateKingSafety(self)

        self.undoMove(cmd, removedPiece)
        self.resetBoardState(nonBoardState)
        return returnValue

    # Return all Valid moves for currentPlayer
    def allValidMoves(self):
        validMoves = []

        for row in range(0, 8):
            for col in range(0, 8):
                if self.board[row][col] != None:
                    if self.board[row][col].player == self.playerTurn:
                        for cmd in self.board[row][col].possibleMoves(self):
                            validMoves.append(cmd)

        return validMoves

    # Compute Board Value - Assume White is the Protagonist
    def computeBoardValue(self):
        returnValue = 0

        for row in range(0, 8):
            for col in range(0, 8):
                if self.board[row][col] != None:
                    if self.board[row][col].player == Player.WHITE:
                        returnValue += self.board[row][col].pieceValue(self)
                    else:
                        returnValue -= self.board[row][col].pieceValue(self)

        return returnValue

    # Return all Capture Moves
    def allQuiesceneMoves(self):
        validMoves = []

        for cmd in self.allValidMoves():
            if cmd.moveType in self.quiescenceMoveCmd:
                validMoves.append(cmd)

        return validMoves

    # MinMaxSearch -> General
    def quiesceneSearch(self, alpha, beta):

        staticEval = self.computeBoardValue()
        if staticEval >= beta:
            return beta

        if staticEval > alpha:
            alpha = staticEval

        for cmd in self.allQuiesceneMoves():
            nonBoardState = self.nonBoardState()
            removedPiece = self.movePiece(cmd)

            score = (-1) * self.quiesceneSearch((-1) * beta, (-1) * alpha)

            self.undoMove(cmd, removedPiece)
            self.resetBoardState(nonBoardState)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # MinMaxSearch -> General
    def negamax(self, depth, alpha, beta):
        # Termination Condition
        if depth == 0:
            return self.quiesceneSearch(float('-inf'), float('inf'))
        else:
            maxEval = float('inf')

            for cmd in self.allValidMoves():
                nonBoardState = self.nonBoardState()
                removedPiece = self.movePiece(cmd)

                score = (-1) * self.negamax(depth - 1, (-1) * beta, (-1) * alpha)
                
                self.undoMove(cmd, removedPiece)
                self.resetBoardState(nonBoardState)

                maxEval = max(maxEval, score)
                alpha = max(alpha, score)
                
                if alpha >= beta:
                    break
                    
            return alpha

    # Take Opponent Turn
    def computeBestMove(self):
        returnCmd = None
        bestScore = float('-inf')

        for cmd in self.allValidMoves():
            nonBoardState = self.nonBoardState()
            removedPiece = self.movePiece(cmd)

            print(cmd)
            returnValue = (-1) * self.negamax(2, float('-inf'), float('inf'))

            if returnValue > bestScore:
                bestScore = returnValue
                returnCmd = cmd
                                
            self.undoMove(cmd, removedPiece)
            self.resetBoardState(nonBoardState)

        return returnCmd

    # This is used to determine king safety and castle
    def _allPlayerCaptureTargets(self, player):
        attackSquare = {}

        for row in range(0, 8):
            for col in range(0, 8):
                targetPiece = self.board[row][col]
                if targetPiece != None and targetPiece.player == player:

                    possibleMoves = targetPiece.captureTargets(self)
                    for move in possibleMoves:
                        attackSquare[move] = True
        
        return attackSquare

    # Move Piece on ChessBoard
    def _movePieceOnBoard(self, startRow: int, startCol: int, endRow: int, endCol: int):
        self.board[endRow][endCol] = self.board[startRow][startCol]
        self.board[endRow][endCol].row = endRow
        self.board[endRow][endCol].col = endCol

        self.board[startRow][startCol] = None

        # Update the King Square
        if self.board[endRow][endCol].type == PieceType.KING:
            if self.board[endRow][endCol].player == Player.BLACK:
                self.blackKingSquareRow = endRow
                self.blackKingSquareCol = endCol

                self.blackKingSideCanCastle = False
                self.blackQueenSideCanCastle = False

            elif self.board[endRow][endCol].player == Player.WHITE:
                self.whiteKingSquareRow = endRow
                self.whiteKingSquareCol = endCol

                self.whiteKingSideCanCastle = False
                self.whiteQueenSideCanCastle = False

        # Update Rook 
        if self.board[endRow][endCol].type == PieceType.ROOK:
            if self.board[endRow][endCol].player == Player.BLACK:
                if startCol == 0:
                    self.blackQueenSideCanCastle = False
                elif startCol == 7:
                    self.blackKingSideCanCastle = False

            elif self.board[endRow][endCol].player == Player.WHITE:
                if startCol == 0:
                    self.whiteQueenSideCanCastle = False
                elif startCol == 7:
                    self.whiteKingSideCanCastle = False  

    def _undoMoveOnBoard(self, originalRow: int, originalCol: int, currentRow: int, currentCol: int):        
        self.board[originalRow][originalCol] = self.board[currentRow][currentCol] 
        self.board[originalRow][originalCol].row = originalRow
        self.board[originalRow][originalCol].col = originalCol

        self.board[currentRow][currentCol] = None

    def undoMove(self, cmd: MoveCommand, restorePiece):
        # Swap the Player Turn
        self.playerTurn = ChessBoardModel.opponent(self.playerTurn)

        match cmd.moveType:
            # Queen Side Castle (Undo)
            case MoveCommandType.QUEENSIDECASTLE:
                if self.playerTurn == Player.BLACK:
                    # Move the King 2 steps to the left
                    self._undoMoveOnBoard(7, 4, 7, 2)

                    # Move the Rook to the right of the king
                    self._undoMoveOnBoard(7, 0, 7, 3)

                    # Update Castle
                    self.blackCastled = False

                elif self.playerTurn == Player.WHITE:
                    # Move the King 2 steps to the left
                    self._undoMoveOnBoard(0, 4, 0, 2)

                    # Move the Rook to the right of the king
                    self._undoMoveOnBoard(0, 0, 0, 3)

                    # Update Castle
                    self.whiteCastled = False

            # King Side Castle
            case MoveCommandType.KINGSIDECASTLE:
                if self.playerTurn == Player.BLACK:
                    # Move the King 2 steps to the right
                    self._undoMoveOnBoard(7, 4, 7, 6)

                    # Move the Rook to the right of the king
                    self._undoMoveOnBoard(7, 7, 7, 5)

                    # Update Castle
                    self.blackCastled = False

                elif self.playerTurn == Player.WHITE:
                    # Move the King 2 steps to the right
                    self._undoMoveOnBoard(0, 4, 0, 6)

                    # Move the Rook to the right of the king
                    self._undoMoveOnBoard(0, 7, 0, 5)

                    # Update Castle
                    self.whiteCastled = False

            # Move Piece
            case MoveCommandType.MOVE:
                # Move Starting Piece to Move Point
                self._undoMoveOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            case MoveCommandType.CAPTURE:
                # Move Starting Piece to Capture Point
                self._undoMoveOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                # Restore Piece
                self.board[cmd.endRow][cmd.endCol] = restorePiece

            # Double Pawn Move
            case MoveCommandType.PAWNOPENMOVE:
                # Move the piece from start to end
                self._undoMoveOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            # Promote Pawn
            case MoveCommandType.PROMOTE:
                # Promote the Pawn to a Queen
                self.board[cmd.endRow][cmd.endCol] = restorePiece

                # Store Removed Piece
                self.board[cmd.startRow][cmd.startCol] = ChessPieceFactory.createChessPiece(
                    PieceType.PAWN, self.playerTurn, cmd.startRow, cmd.startCol)

            # En Passant
            case MoveCommandType.ENPASSANT:
                # Move the Pawn to the target
                self._undoMoveOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                # Store Removed Piece
                self.board[cmd.startRow][cmd.endCol] = restorePiece

        return 

    def movePiece(self, cmd: MoveCommand):
        # Set enPassant to Null - Reset this if the opponent does a double pawn move
        self.enPassantColumn = None

        # Captured Piece 
        removedPiece = None

        match cmd.moveType:
            # Queen Side Castle
            case MoveCommandType.QUEENSIDECASTLE:
                if self.playerTurn == Player.BLACK:
                    # Move the King 2 steps to the left
                    self._movePieceOnBoard(7, 4, 7, 2)

                    # Move the Rook to the right of the king
                    self._movePieceOnBoard(7, 0, 7, 3)

                    # Update Castle
                    self.blackCastled = True

                elif self.playerTurn == Player.WHITE:
                    # Move the King 2 steps to the left
                    self._movePieceOnBoard(0, 4, 0, 2)

                    # Move the Rook to the right of the king
                    self._movePieceOnBoard(0, 0, 0, 3)

                    # Update Castle
                    self.whiteCastled = True

            # King Side Castle
            case MoveCommandType.KINGSIDECASTLE:
                if self.playerTurn == Player.BLACK:
                    # Move the King 2 steps to the right
                    self._movePieceOnBoard(7, 4, 7, 6)

                    # Move the Rook to the right of the king
                    self._movePieceOnBoard(7, 7, 7, 5)

                    # Update Castle
                    self.blackCastled = True

                elif self.playerTurn == Player.WHITE:
                    # Move the King 2 steps to the right
                    self._movePieceOnBoard(0, 4, 0, 6)

                    # Move the Rook to the right of the king
                    self._movePieceOnBoard(0, 7, 0, 5)

                    # Update Castle
                    self.whiteCastled = True

            # Move Piece
            case MoveCommandType.MOVE:
                # Move Starting Piece to Capture Point
                self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            case MoveCommandType.CAPTURE:
                # Store the Removed Piece
                removedPiece = self.board[cmd.endRow][cmd.endCol]
                self.board[cmd.endRow][cmd.endCol] = None

                # Move Starting Piece to Capture Point
                self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

            # Double Pawn Move
            case MoveCommandType.PAWNOPENMOVE:
                # Move the piece from start to end
                self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                self.enPassantColumn = cmd.endCol

            # Promote Pawn
            case MoveCommandType.PROMOTE:
                # Promote MAY be a capture or a move
                removedPiece = self.board[cmd.endRow][cmd.endCol]

                self.board[cmd.endRow][cmd.endCol] = ChessPieceFactory.createChessPiece(
                    PieceType.QUEEN, self.playerTurn, cmd.endRow, cmd.endCol)

            # En Passant
            case MoveCommandType.ENPASSANT:
                # Store Removed Piece
                removedPiece = self.board[cmd.startRow][cmd.endCol]

                # Move the Pawn to the target
                self._movePieceOnBoard(cmd.startRow, cmd.startCol, cmd.endRow, cmd.endCol)

                # Remove En Passant Pawn
                self.board[cmd.startRow][cmd.endCol] = None

        # Swap the Player Turn
        self.playerTurn = ChessBoardModel.opponent(self.playerTurn)

        # Create a new copy of the removed Piece
        return removedPiece