from typing import Protocol

# Enum 
from appEnums import PieceType, Player, MoveCommandType

class OpeningMoveNodeProtocal(Protocol):

    cmd: Optional[MoveCommand]
    subsequentNodes: List[MoveCommand]

    def addSequence(self, cmdSequence: List[MoveCommand]):
        ... 

    def findTraverseNode(self, cmd: MoveCommand) -> Optional[OpeningMoveNodeProtocal]:
        ...

    def randomSubsequentCmd(self) -> MoveCommand:
        ...

    def stepForward(self, cmd: MoveCommand) -> Optional[OpeningMoveNodeProtocal]:
        ...

    def hasSubsequentCmd(self) -> bool:
        ...