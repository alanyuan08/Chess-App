# Import Factory
from modelFactory.chessBoardFactory import ChessBoardFactory

# Model
from modelComponent.chessBoardModel import ChessBoardModel
from modelComponent.moveCommand import MoveCommand
from modelComponent.ttEntry import TTEntry

# Factory
from modelFactory.chessPieceFactory import ChessPieceFactory

# Enum
from appEnums import PieceType, Player, MoveCommandType, TTFlag

# Normal
import math
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Controller 
class ChessGameModel():
    def __init__(self, playersArray):
        self.chessBoard = ChessBoardFactory.createChessBoard(playersArray)

        # Stores the Transposition Table -> (value, depth, flag, cmd
        self.transpositionTable = {}

        # Set Human Player
        self.humanPlayers = playersArray

        # Set Game Turn 
        self.gamePlayerTurn = Player.WHITE

    # Move Piece
    def movePiece(self, cmd: MoveCommand):
        self.chessBoard.movePiece(cmd)

        self.gamePlayerTurn = ChessBoardModel.opponent(self.gamePlayerTurn)

    # Validate Move
    def validateMove(self, initRow: int, initCol: int, targetRow: int, 
        targetCol: int, player: Player) -> MoveCommand:
        # It's not your turn to move
        if player != self.gamePlayerTurn:
            return None

        return self.chessBoard.validateMove(initRow, initCol, targetRow, targetCol, player)

    def checkTable(self, entrykey, searchDepth, alpha, beta):
        if entrykey in self.transpositionTable:
            entry = self.transpositionTable[entrykey]

            # 3. Check if the stored information is deep enough to be useful (pruning check)
            if entry.depth >= searchDepth:
                # Check the flag to determine if we can cut off the search
                if entry.flag == TTFlag.EXACT:
                    return entry.value
                if entry.flag == TTFlag.BETA and entry.value >= beta:
                    # We found a move that is too good; we can prune this branch
                    return entry.value
                if entry.flag == TTFlag.ALPHA and entry.value <= alpha:
                    # All moves here were too bad; we can prune this branch
                    return entry.value
        
        # 5. Hash miss, no useful data found for this search
        return None 

    def storeHash(self, key: str, maxEval: int, depth: int, flag: TTFlag):
        self.transpositionTable[key] = TTEntry(key, maxEval, depth, flag)

    def computeBestMove(self) -> MoveCommand:
        commandList = self.chessBoard.allValidMoves()
        commandList.sort(key=lambda move: self._getMovePriority(self.chessBoard, move), reverse=True)

        bestScore = float('-inf')
        bestMove = None

        # Use ThreadPoolExecutor for concurrent search
        with ThreadPoolExecutor() as executor:
            # Create a board clone for each thread to avoid state conflicts
            future_to_cmd = {
                executor.submit(self._search_task, copy.deepcopy(self.chessBoard), cmd): cmd 
                for cmd in commandList
            }

            for future in as_completed(future_to_cmd):
                cmd = future_to_cmd[future]
                try:
                    score = future.result()
                    print(f"Move: {cmd} | Score: {score}")

                    if score > bestScore:
                        bestScore = score
                        bestMove = cmd
                except Exception as exc:
                    print(f"Move {cmd} generated an exception: {exc}")

        return bestMove

    def _search_task(self, testBoard: ChessBoardModel, cmd: MoveCommand):
        """Worker function for individual threads."""
        removedPiece, prevEnPassant, prevCastleIndex = testBoard.movePiece(cmd)
        # Perform negamax on the thread-local board copy
        # Threads automatically share self.tt_table
        score = (-1) * self._negamax(testBoard, 5, float('-inf'), float('inf'))
        return score

    # Compute Board Value - White is Positive/ Black is Negative
    def _computeBoardValue(self, testBoard: ChessBoardModel) -> int:
        returnValue = 0

        phaseWeight = self._calculateGamePhase(testBoard)
        board = testBoard.board
        # Compute for Piece
        for row in range(0, 8):
            for col in range(0, 8):
                gamePiece = board[row][col]
                if gamePiece != None:
                    if gamePiece.player == Player.WHITE:
                        returnValue += gamePiece.computedValue(
                            self.chessBoard, phaseWeight)
                    else:
                        returnValue -= gamePiece.computedValue(
                            self.chessBoard, phaseWeight)

        # Compute for Double/ Isolated Pawns
        returnValue -= self._pawnPenalizer(testBoard, Player.WHITE, phaseWeight)
        returnValue += self._pawnPenalizer(testBoard, Player.BLACK, phaseWeight)

        if testBoard.playerTurn == Player.WHITE:
            return returnValue
        else:
            return (-1) * returnValue

    # MinMaxSearch -> General
    def _negamax(self, testBoard: ChessBoardModel, depth: int, alpha: int, beta: int) -> int:
        # TT Table 
        tt_result = self.checkTable(testBoard.zobristHash, depth, alpha, beta)
        if isinstance(tt_result, int):
            return tt_result

        validMoves = testBoard.allValidMoves()
        validMoves.sort(key=lambda move: self._getMovePriority(testBoard, move), reverse=True)
        
        # No Valid Moves = Lose
        if len(validMoves) == 0:
            return self.resolveEndGame(testBoard)

        alphaOriginal = alpha  # Save the original alpha value

        # Termination Condition
        if depth == 0:
            return self._quiesceneSearch(testBoard, alpha, beta)
        else:
            maxEval = float('-inf')

            for cmd in validMoves:
                removedPiece, prevEnPassant, prevCastleIndex = testBoard.movePiece(cmd)
                score = (-1) * self._negamax(testBoard, depth - 1, (-1) * beta, (-1) * alpha)
                testBoard.undoMove(cmd, removedPiece, prevEnPassant, prevCastleIndex)

                maxEval = max(maxEval, score)
                alpha = max(alpha, score)
                
                if alpha >= beta:
                    break
            
            # At the end, determine the flag
            if maxEval <= alphaOriginal:
                flag = TTFlag.ALPHA 
            elif maxEval >= beta:
                flag = TTFlag.BETA
            else:
                flag = TTFlag.EXACT

            self.storeHash(testBoard.zobristHash, maxEval, depth, flag)
            return maxEval

    # MinMaxSearch -> General
    def _quiesceneSearch(self, testBoard: ChessBoardModel, alpha, beta, depth = 0) -> int:
        staticEval = self._computeBoardValue(testBoard)

        if depth >= 10:
            return staticEval

        if staticEval >= beta:
            return beta

        if staticEval > alpha:
            alpha = staticEval

        validMoves = testBoard.allValidMoves()
        validMoves.sort(key=lambda move: self._getMovePriority(testBoard, move), reverse=True)

        # No Valid Moves = Lose
        if len(validMoves) == 0:
            return self.resolveEndGame(testBoard)

        for cmd in self._allQuiesceneMoves(validMoves):
            removedPiece, prevEnPassant, prevCastleIndex = testBoard.movePiece(cmd)
            score = (-1) * self._quiesceneSearch(testBoard, (-1) * beta, (-1) * alpha, depth + 1)
            testBoard.undoMove(cmd, removedPiece, prevEnPassant, prevCastleIndex)

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # Resolve End Game -> Called when No Valid Moves
    def resolveEndGame(self, testBoard: ChessBoardModel) -> int:
        opponentAttackTargets = testBoard.allOpponentCaptureTargets()

        kingTuple = testBoard.kingTuple(testBoard.playerTurn)
        if kingTuple in opponentAttackTargets:
            if testBoard.playerTurn == Player.WHITE:
                return float('-inf')
            else:
                return float('inf')
        else:
            return 0

    # Return all Capture Moves
    def _allQuiesceneMoves(self, validMoves) -> list[MoveCommand]:
        # QuiescenceMoves
        quiescenceMoveCmd = [MoveCommandType.PROMOTE, MoveCommandType.CAPTURE, \
            MoveCommandType.ENPASSANT]

        return list(filter(
            lambda cmd: cmd.moveType in quiescenceMoveCmd, validMoves
        ))

    # Compute Pawn Penalizer for Isolated / Double Pawn
    def _pawnPenalizer(self, testBoard: ChessBoardModel, player, phaseWeight) -> int:
        filePawnCount = [0 for _ in range(8)]

        board = testBoard.board
        for row in range(0, 8):
            for col in range(0, 8):
                piece = board[row][col]
                if piece != None and piece.type == PieceType.PAWN and piece.player == player:
                    filePawnCount[col] += 1

        penalizer = 0
        # Pawn Penalizer
        for file in range(0, 8):
            pawnCount = filePawnCount[file]

            if pawnCount > 0:
                # Penalizer for Double / Triple/ Quad Pawns
                if pawnCount > 1:
                    penalizer += (pawnCount - 1) * 15

                # Compute for Isolated Pawns
                isIsolated = True
                if file > 0 and filePawnCount[file-1] > 0:
                    isIsolated = False
                if file < 7 and filePawnCount[file+1] > 0:
                    isIsolated = False
                    
                if isIsolated:
                    earlyGame = 15
                    endGame = 25
                    computedPhase = earlyGame * phaseWeight + endGame * (24 - phaseWeight) 
                
                    penalizer += math.ceil(computedPhase / 24)

        return penalizer

    # Maximum Value is 24, used for early/ mid board evaluation
    def _calculateGamePhase(self, testBoard: ChessBoardModel) -> int:
        totalPhaseWeight = 0

        board = testBoard.board
        for row in range(0, 8):
            for col in range(0, 8):
                if board[row][col] != None:
                    totalPhaseWeight += board[row][col].phaseWeight()

        return totalPhaseWeight


    # Compute Move Priority
    def _getMovePriority(self, testBoard: ChessBoardModel, cmd: MoveCommand) -> int:
        board = testBoard.board

        # 1. Promotions (High Priority)
        if cmd.moveType == MoveCommandType.PROMOTE:
            return 90000 # Treat as Queen value
        
        # 2. Captures (MVV-LVA)
        if cmd.moveType in [MoveCommandType.CAPTURE, MoveCommandType.ENPASSANT]:
            capturedPieceVal = 0
            if cmd.moveType == MoveCommandType.ENPASSANT:
                capturedPieceVal = board[cmd.startRow][cmd.endCol].pieceValue()
            else:
                capturedPieceVal = board[cmd.endRow][cmd.endCol].pieceValue()

            startingPieceVal = board[cmd.startRow][cmd.startCol].pieceValue()  
            
            return (capturedPieceVal * 10) - startingPieceVal

        # 3. Castling (Mid Priority)
        if cmd.moveType in [MoveCommandType.KINGSIDECASTLE, MoveCommandType.QUEENSIDECASTLE]:
            return 50

        # 4. Standard Moves (Low Priority)
        return 0

