# Chess App

This is a standard Chess-App built using PySide6 Library for the Front End and standard MinMax search with Alpha-Beta Pruning Optimizations.

## 1. Python Presentation & Validation Layer
    
- PySide6 User Interface: Renders the 2D chessboard, handles player drag-and-drop input. 

## 2. Rust Engine Core

- High-Performance Generation: Computes legal and pseudo-legal move paths across millions of positions per second using native multi-threading. MinMax with Alpha-Beta Pruning: Cuts off unpromising branches early to reduce the total search tree size exponentially.

## 3. Neural Network Evaluation

- NNUE Integration: Uses a custom, pre-trained neural network that updates incrementally as pieces move, bypassing slow, manual heuristic math.

# Search Algorithm 

The Chess App uses the standard MinMax search with Alpha-Beta Pruning for search optimizations; The MinMax algorithm recursives explores all possible moves 4-pry deep for both player and computes the best move, assuming both players play optimially.

The Alpha-Beta Pruning allows the MinMax algorithm to stop pruning a branch of the search tree if it determined that the current position is already worse than a position it has found elsewhere.

Furthermore, the algorithm uses Quiescence Search at the nodes of the search depths to ensure the algorithm isn't suspectiable to the "Horizon Effect".

Finally, the algorithm  uses the Younger Brother Parallel Search such that it first searches the most likely promising move to produce a suitable alpha cutoff that is passed to the subsequent processors to be searched in parallel.

The Board Evaluation is handcrafted based on heuristics provided online.

# Running the App

Playing as [black|white]
- /run.sh [black|white]

# Playing Level

The Chess AI has been tested and defeated bots on Chess.com with ELO 2000+

- [Win - ELO 2300 Bot](https://www.chess.com/analysis/game/computer/1397423175/review)
- [Win - ELO 2300 Bot](https://www.chess.com/analysis/game/computer/1393583077/analysis)

## Contact

Alan Yuan

| Platform | Link | Intent |
| :--- | :--- | :--- |
| **Email** | [alan0408yuan@gmail.com](mailto:alan0408yuan@gmail.com) | Direct inquiries & scheduling |
| **LinkedIn** | [linkedin.com](https://www.linkedin.com/in/alan-yuan-62301272/) | Professional networking & background |

*Response time: Typically within 24 hours.*