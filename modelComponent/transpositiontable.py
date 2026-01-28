
class TranspositionTable():
    def __init__(self):
        self.size = 2 ** 30
        self.mask = self.size - 1
        self.table = [None] * self.size

    def store(self, key, score, depth, flag):
        index = key & self.mask
        existing = self.table[index]
        
        # Replacement Strategy: Depth-Preferred
        # Keep the search that went deeper, as it is more valuable
        if existing is None or depth >= existing[2]:
            self.table[index] = [key, score, depth, flag]

    def probe(self, key):
        index = key & self.mask
        entry = self.table[index]
        # Verification: The full 64-bit key MUST match
        if entry and entry[0] == key:
            return entry
        return None
