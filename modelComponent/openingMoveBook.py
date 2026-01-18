import random

# Model 
from modelComponent.moveCommand import MoveCommand

# Controller 
class OpeningMovebook():
	def __init__(self, cmd: MoveCommand = None):
		# Curr Node
		self.cmd = cmd

		# Paths to Subsequent Nodes
		self.subsequentCmd = [] 

	# Add Subsequent Cmd
	def addSubsequentCmd(self, cmd: MoveCommand):
		self.subsequentCmd.append(cmd)
		
	# Random Subsequent Cmd
	def randomSubsequentCmd(self, cmd: MoveCommand):
		if len(self.subsequentCmd) > 0:
			newNode = random.choice(self.subsequentCmd)

		else:
			return None

	# Step Forward
	def stepForward(self, cmd: MoveCommand):
		for subsequentCmd in self.subsequentCmd:
			if subsequentCmd.cmd == cmd:
				return subsequentCmd

		return None

# Root
rootCmd = OpeningMovebook()