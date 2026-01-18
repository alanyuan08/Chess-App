# Enum
from dataclasses import dataclass
from appEnums import PieceType

@dataclass
class MoveCommand:
    startRow: int
    startCol: int
    endRow: int
    endCol: int
    moveType: PieceType
