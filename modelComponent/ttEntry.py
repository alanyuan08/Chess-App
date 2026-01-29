# Enum
from appEnums import TTBoundType

# Enum
from dataclasses import dataclass
from appEnums import PieceType

@dataclass
class TTEntry:
    key: int
    score: int
    depth: int
    flag: TTBoundType
