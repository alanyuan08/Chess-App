# Chess App

This is a standard Chess-App built using PySide6 Library for the Front End and standard MinMax search with Alpha-Beta Pruning Optimizations.

# Search Algorithm 

The Chess App uses the standard MinMax search with Alpha-Beta Pruning for search optimizations; The MinMax algorithm recursives explores all possible moves 4-pry deep for both player and computes the best move, assuming both players play optimially.

The Alpha-Beta Pruning allows the MinMax algorithm to stop pruning a branch of the search tree if it determined that the current position is already worse than a position it has found elsewhere.

Furthermore, the algorithm uses Quiescence Search at the nodes of the search depths to ensure the algorithm isn't suspectiable to the "Horizon Effect".

Finally, the algorithm  uses the Younger Brother Parallel Search such that it first searches the most likely promising move to produce a suitable alpha cutoff that is passed to the subsequent processors to be searched in parallel.

The Board Evaluation is handcrafted based on heuristics provided online.

# Running the App

Playing as [black|white]
- python chessApp.py [black|white]

# Improvements

- The AI currently does not have a opening hand book and uses a stop-gap function to play one of four moves
- The AI does not have an endgame table and is suspectible to draws if there isn't sufficient material 
- The UI is restrictve and doesn't common features such as timers, player selection, promotion selection etc. 


# Playing Level

The Chess AI has been tested and defeated bots on Chess.com with ELO 2000+

- [Win - ELO 1800 Bot](https://www.chess.com/game/computer/519702629)
- [Win - ELO 2000 Bot](https://www.chess.com/analysis/game/computer/522586927/analysis).
- [Lose - ELO 2200 Bot](https://www.chess.com/game/computer/522001439)