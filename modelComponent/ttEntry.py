from appEnums import PieceType

# Model
from modelComponent.ttEntryCType import TTEntryCType

@dataclass
class TTEntry:
    key: int
    score: int
    depth: int
    flag: TTBoundType

    @staticmethod
    def mapToTTEntryCType(key: int, score: int, depth: int,
        flag: TTBoundType) -> TTEntryCType:
        flagMap = 0

        if flag == TTBoundType.EXACT:
            flagMap = 1
        elif flag == TTBoundType.UPPERBOUND:
            flagMap = 2
        elif flag == TTBoundType.LOWERBOUND:
            flagMap = 3

        return TTEntryCType(key, score, depth, flagMap)
  
    @staticmethod
    def mapFromTTEntryCType(ttEntryCType: TTEntryCType) -> TTEntry:
        flag = None

        if ttEntryCType == 1:
            flag = TTBoundType.EXACT
        elif ttEntryCType == 2:
            flag = TTBoundType.UPPERBOUND
        elif ttEntryCType == 3:
            flag = TTBoundType.LOWERBOUND

        return TTEntry(ttEntryCType.key, ttEntryCType.score,
            ttEntryCType.depth, flag)