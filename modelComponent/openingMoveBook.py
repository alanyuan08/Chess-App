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
	def addSubsequentCmd(self, cmd: MoveCommand):
		self.subsequentCmd.append(
			OpeningMovebook(cmd)
		)
		
	# HandBook has Subsequent Move
	def hasSubsequentCmd(self) -> bool:
		if len(self.subsequentCmd) > 0:
			return True
		else:
			return False

	# Random Subsequent Cmd
	def randomSubsequentCmd(self):
		subsequentCmd = random.choice(self.subsequentCmd)

		return subsequentCmd.cmd

	# Step Forward
	def stepForward(self, cmd: MoveCommand):
		for subsequentCmd in self.subsequentCmd:
			if subsequentCmd.cmd == cmd:
				return subsequentCmd

# Root
rootCmd = OpeningMovebook()	
rootCmd.addSubsequentCmd(
	MoveCommand(1, 4, 3, 4, MoveCommandType.PAWNOPENMOVE)
)
rootCmd.addSubsequentCmd(
	MoveCommand(1, 3, 3, 3, MoveCommandType.PAWNOPENMOVE)
)
rootCmd.addSubsequentCmd(
	MoveCommand(0, 1, 2, 2, MoveCommandType.MOVE)
)
rootCmd.addSubsequentCmd(
	MoveCommand(0, 6, 2, 5, MoveCommandType.MOVE)
)