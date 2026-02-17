# Enum
from dataclasses import dataclass
from appEnums import MoveCommandType

@dataclass
class MoveCommand:
    startRow: int
    startCol: int
    endRow: int
    endCol: int
    moveType: MoveCommandType
