from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QApplication
import sys

# Import Enums
from appEnums import Player

# Import View
from viewComponent.chessBoardView import ChessBoardView

# Import ViewModel
from viewModelComponent.chessBoardViewModel import ChessBoardViewModel

# Import Rust Test
import rust_compute		

if __name__ == '__main__':
	# Import Model
	from modelFactory.chessGameFactory import ChessGameFactory

	app = QApplication(sys.argv)

	humanPlayer = [Player.BLACK, Player.WHITE]

	# Test Rust Integration
	# print(rust_compute.new_chess_board(15, 10))

	# Set Human Player
	if len(sys.argv) > 1:
	    match(sys.argv[1]):
	    	case "white":
	    		humanPlayer = [Player.WHITE]
	    	case "black":
	    		humanPlayer = [Player.BLACK]

	# ChessBoard View
	scene = QGraphicsScene(0, 0, 720, 840)
	view = QGraphicsView(scene)
	view.setFixedSize(720, 840)

	# Create Chess Game Moded / ChessBoard/ Chess Piece Model Components
	chessGameModel = ChessGameFactory.createChessGame(humanPlayer)

	# Create View Components
	chessBoardView = ChessBoardView(scene, chessGameModel)

	# Create ChessBoard ViewModel
	chessBoardController = ChessBoardViewModel(chessBoardView, chessGameModel)

	view.show()
	app.exec()
