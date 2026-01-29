# Enum
from appEnums import TTBoundType

# Model
from modelComponent.ttEntry import TTEntry

class TranspositionTable():
    def __init__(self):
        self.size = 2 ** 26
        self.mask = self.size - 1
        self.table = [None] * self.size

    def store(self, key: int, score: int, depth: int, flag: TTBoundType):
        index = key & self.mask
        existing = self.table[index]
        
        # Replacement Strategy: Depth-Preferred
        # Keep the search that went deeper, as it is more valuable
        if existing is None or depth >= existing.depth:
            self.table[index] = TTEntry(key, score, depth, flag)

    def probe(self, key: int):
        index = key & self.mask
        entry = self.table[index]
        # Verification: The full 64-bit key MUST match
        if entry and entry.key == key:
            return entry
        return None
