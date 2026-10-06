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

def move_command_to_uci(cmd: MoveCommand) -> str:
    # Convert 0-based row/col → UCI file/rank
    def to_sq(row: int, col: int) -> str:
        file = chr(ord('a') + col)
        rank = str(row + 1)
        return file + rank

    start = to_sq(cmd.startRow, cmd.startCol)
    end   = to_sq(cmd.endRow, cmd.endCol)

    uci = start + end

    # Promotion
    promo_map = {
        MoveCommandType.PROMOTION_QUEEN: "q",
        MoveCommandType.PROMOTION_ROOK: "r",
        MoveCommandType.PROMOTION_BISHOP: "b",
        MoveCommandType.PROMOTION_KNIGHT: "n",
    }

    if cmd.moveType in promo_map:
        uci += promo_map[cmd.moveType]

    return uci