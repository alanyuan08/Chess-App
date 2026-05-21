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

    @classmethod
    def from_dict(cls, data: dict) -> "MoveCommand":
        return cls(
            data["startRow"],
            data["startCol"],
            data["endRow"],
            data["endCol"],
            MoveCommandType(data["moveType"])
        )