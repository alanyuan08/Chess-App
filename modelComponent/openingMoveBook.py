import random

# Model 
from modelComponent.moveCommand import MoveCommand

# Enum
from appEnums import MoveCommandType

# Controller 
class OpeningMovebook():
	def __init__(self, cmd: MoveCommand = None):
		# Curr Node
		self.cmd = cmd

		# Paths to Subsequent Nodes
		self.subsequentCmd = [] 

	# Add Subsequent Cmd
	def addSubsequentCmd(self, subsequentMove: OpeningMovebook):
		self.subsequentCmd.append(subsequentMove)
		
	# HandBook has Subsequent Move
	def hasSubsequentCmd(self) -> bool:
		if len(self.subsequentCmd) > 0:
			return True
		else:
			return False

	# Random Subsequent Cmd
	def randomSubsequentCmd(self):
		subsequentCmd = random.choice(self.subsequentCmd)

		print(subsequentCmd)
		return subsequentCmd.cmd

	# Step Forward
	def stepForward(self, cmd: MoveCommand):
		for subsequentCmd in self.subsequentCmd:
			print(cmd, subsequentCmd)
			if subsequentCmd.cmd == cmd:
				return subsequentCmd

# E4 Pawn
e4Pawn = OpeningMovebook(
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE)
)
# Sicilian Defense
e4Pawn.addSubsequentCmd(
	OpeningMovebook(MoveCommand(6, 2, 4, 2, MoveCommandType.PAWNOPENMOVE))
)

# French Defense
e4Pawn.addSubsequentCmd(
	OpeningMovebook(MoveCommand(6, 4, 5, 4, MoveCommandType.MOVE))
)

# Ruy Lopez
b4Bishop = OpeningMovebook(MoveCommand(0, 5, 4, 1, MoveCommandType.MOVE))

c6Knight = OpeningMovebook(MoveCommand(7, 1, 5, 2, MoveCommandType.MOVE))
c6Knight.addSubsequentCmd(b4Bishop)

f3Knight = OpeningMovebook(MoveCommand(0, 6, 2, 5, MoveCommandType.MOVE))
f3Knight.addSubsequentCmd(c6Knight)

e5Pawn = OpeningMovebook(MoveCommand(6, 4, 4, 4, MoveCommandType.PAWNOPENMOVE))
e5Pawn.addSubsequentCmd(f3Knight)

e4Pawn.addSubsequentCmd(e5Pawn)

# Caro-Kann Defense
e4Pawn.addSubsequentCmd(
	OpeningMovebook(MoveCommand(6, 2, 5, 2, MoveCommandType.MOVE))
)

# Caro-Kann Defense
e4Pawn.addSubsequentCmd(
	OpeningMovebook(MoveCommand(6, 2, 5, 2, MoveCommandType.MOVE))
)


# Root
rootCmd = OpeningMovebook()	
rootCmd.addSubsequentCmd(e4Pawn)


