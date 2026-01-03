from typing import Protocol

# Enum 
from appEnums import PieceType, Player, MoveCommandType

class ChessBoardProtocal(Protocol):
    board: list[list[any]]

    whiteEnPassantColumn: int
    blackEnPassantColumn: int

    whiteCastled: bool
    blackCastled: bool

    def allOpponentCaptureTargets(self, 
        player: Player) -> set[tuple[str, int]]:
        ... 

    def validateKingSafety(self, cmd: MoveCommandType) -> bool:
        ...