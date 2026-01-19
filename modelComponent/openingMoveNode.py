import random

# Model 
from modelComponent.moveCommand import MoveCommand

# Enum
from appEnums import MoveCommandType

# Controller 
class OpeningMoveNode():
	def __init__(self, cmd: Optional[MoveCommand] = None):
		# Root -> None Cmd
		self.cmd = cmd

		# Subsequent Cmd
		self.subsequentNodes = []

	# Add Subsequent Cmd
	def addSequence(self, cmdSequence: List[MoveCommand]):
		traverseNode = self

		for cmd in cmdSequence:
			# Traverse Cmd in SubsequentNodes
			target = traverseNode.findTraverseNode(cmd)

			# Create Target if not in TraverseNode
			if not target:
				target = OpeningMoveNode(cmd)
				traverseNode.subsequentNodes.append(target)
			
			traverseNode = target

	# Find TraverseNode
	def findTraverseNode(self, cmd: MoveCommand):
		for cmdNode in self.subsequentNodes:
			if cmd == cmdNode.cmd:
				return cmdNode

		return None

	# Random Subsequent Cmd
	def randomSubsequentCmd(self):
		subsequentCmd = random.choice(self.subsequentNodes)

		return subsequentCmd.cmd

	# Step Forward
	def stepForward(self, cmd: MoveCommand):
		for cmdNode in self.subsequentNodes:
			if cmd == cmdNode.cmd:
				return cmdNode

		return None

	# Random Subsequent Cmd
	def hasSubsequentCmd(self):
		if len(self.subsequentNodes) > 0:
			return True
		else:
			return False

# Root
OpeningMoveCmd = OpeningMoveNode()

# -- E4 Pawn Moves

# Sicilian Defense
OpeningMoveCmd.addSequence([
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(6, 2, 4, 2, MoveCommandType.PAWNOPENMOVE)
])

# French Defense
OpeningMoveCmd.addSequence([
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(6, 4, 5, 4, MoveCommandType.MOVE)
])

# Ruy Lopez
OpeningMoveCmd.addSequence([
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(6, 4, 4, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(0, 6, 2, 5, MoveCommandType.MOVE),
	MoveCommand(7, 1, 5, 2, MoveCommandType.MOVE),
	MoveCommand(0, 5, 4, 1, MoveCommandType.MOVE)
])

# Caro-Kann Defense
OpeningMoveCmd.addSequence([
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(6, 2, 5, 2, MoveCommandType.MOVE)
])

# Italian Game
OpeningMoveCmd.addSequence([
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(6, 4, 4, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(0, 6, 2, 5, MoveCommandType.MOVE),
	MoveCommand(7, 1, 5, 2, MoveCommandType.MOVE),
	MoveCommand(0, 5, 3, 2, MoveCommandType.MOVE)
])

# Scandinavian Defense
OpeningMoveCmd.addSequence([
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(6, 3, 4, 3, MoveCommandType.PAWNOPENMOVE)
])

# Pirc Defense
OpeningMoveCmd.addSequence([
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(6, 3, 5, 3, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(1, 3, 3, 3, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(7, 6, 5, 5, MoveCommandType.PAWNOPENMOVE)
])

# Alekhine's Defense
OpeningMoveCmd.addSequence([
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(7, 6, 5, 5, MoveCommandType.PAWNOPENMOVE)
])

# King's Gambit
OpeningMoveCmd.addSequence([
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(6, 4, 4, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(1, 5, 3, 5, MoveCommandType.PAWNOPENMOVE)
])

# Scotch Game
OpeningMoveCmd.addSequence([
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(6, 4, 4, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(0, 6, 2, 5, MoveCommandType.MOVE),
	MoveCommand(7, 1, 5, 2, MoveCommandType.MOVE),
	MoveCommand(1, 3, 3, 3, MoveCommandType.PAWNOPENMOVE)
])

#Vienna Game
OpeningMoveCmd.addSequence([
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(6, 4, 4, 4, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(0, 1, 2, 2, MoveCommandType.MOVE)
])

# -- D4 Pawn Moves

# Sicilian Defense
OpeningMoveCmd.addSequence([
	MoveCommand(1, 3, 3, 3, MoveCommandType.PAWNOPENMOVE),
	MoveCommand(6, 2, 4, 2, MoveCommandType.PAWNOPENMOVE)
])

