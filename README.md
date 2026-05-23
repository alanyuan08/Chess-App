# Chess App

This is a standard Chess-App built using PySide6 Library (Python) for the FrontEnd and Rust Multi-Threading for the computation heavy steps of move generation, board evaluation and search. 

## 1. Python Presentation & Validation Layer
    
- PySide6 User Interface: Renders the 2D chessboard, handles player drag-and-drop input in addition to Move Validation. It also handles the Three-Fold Repetition using a Zobrist Hash and maintains an opening handbook of the most commonly played moves. 

## 2. Rust Engine Core

- High-Performance Generation: Computes all pseudo-legal move paths across millions of positions per second using Bitboards for Move Generation. It uses Min-Max combined with Alpha-Beta Pruning for pruning unpromising branches early and Quiescence for extending unstable searches beyond the search horiozn. 

It uses Iterative Deepening combined with Principal Variations to search 10+ ply deep. In addition, the engine uses Transposition Table to store previously evaluated board positions and for coordinating results from Lock-Free Concurrent Tree Search (Lazy SMP).

## 3. Neural Network Evaluation

- NNUE Integration: Uses a custom, pre-trained neural network that updates incrementally using Chess UCI. This will be replaced by a self-trained NNUE as the timecat NNUE does not evaluate pseudo positions requred for Null-Move Pruning. 

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