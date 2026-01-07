# Enum
from appEnums import TTFlag

class TTEntry:
    def __init__(self, key: str, value: int, 
        depth: int, flag: TTFlag):
        self.key = key
        self.value = value
        self.depth = depth
        self.flag = flag
