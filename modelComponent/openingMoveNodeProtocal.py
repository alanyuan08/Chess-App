from typing import Optional

# Model
from modelComponent.moveCommand import MoveCommand

# Assumed interface based on your codebase structure
class OpeningMoveNodeProtocal:
    def stepForward(self, cmd: MoveCommand) -> Optional['OpeningMoveNodeProtocal']:
        raise NotImplementedError
        
    def hasSubsequentCmd(self) -> bool:
        raise NotImplementedError
        
    def randomSubsequentCmd(self) -> MoveCommand:
        raise NotImplementedError