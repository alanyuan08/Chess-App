# Chess AI

<img src="img/saved_game/saved_game.png" width="50%">

A hybrid desktop chess application pairing a responsive PySide6 user interface with a high-performance, multithreaded Rust engine core capable of evaluating millions of positions per second.

The engine has been unofficially benchmarked and validated against 3000 Elo bots on Chess.com.

## 1. Python Presentation & Validation Layer
    
- **PySide6 UI:** Renders a fluid 2D chessboard and manages real-time player drag-and-drop interactions

- **Move Validation:** Enforces legal moves and coordinates state synchronization with the engine core

- **Opening Handbook:** Integrates a built-in opening book containing standard opening lines

## 2. Rust Engine Core

- **Bitboard Move Generation:** Maximizes throughput by computing all pseudo-legal move paths across millions of positions per second

- **Adversarial Search:** Implements Minimax search enhanced by Alpha-Beta pruning and a Quiescence search to eliminate horizon-effect instability.

- **Advanced Pruning:** Uses Killer Move Heuristics and Late Move Reduction to improve the alpha / beta cutoff. The algorithm does not utilize Null-Move Pruning as it is currently using the Timecat NNUE for board evaluation and it is unable to process psuedo-moves

- **Deep Evaluation:** Combines Iterative Deepening with Principal Variation Search (PVS) to regularly achieve search depths of 12+ plies. (Average Move is approximately 20 seconds to 1 minutes)

- **Transposition Tables:** Caches previously evaluated board states to accelerate search paths and share data across threads. The tables uses the Condon-Thompson Replacement method to increase efficiency of L1 / L2 / L3 caches. 

- **Parallel Processing:** Scales performance across CPU threads using a lock-free concurrent tree search architecture (Lazy SMP)

- **Performance Benchmark:** Processes approximately 20 million nodes per second (NPS) on an Apple M4 Pro chip. (8 Performance Core Only - 4.5 GHz + On-Chip Cache Memory - 39.5 MB)

## 3. Neural Network Evaluation

- **NNUE Integration:** Features an incrementally updated Efficiently Updatable Neural Network (NNUE) paired with the Universal Chess Interface (UCI) protocol.

> **Current Limitations:**: The engine currently utilizes the Timecat NNUE backend; This dependency will need to be removed prior to Computer Chess Rating Listing (CCRL) submission.

> **Future Roadmap:**: This dependency will be replaced with a custom, self-trained NNUE framework designed to handle perspective shifts during abstract pruning phases.

# Running the App

## Prerequisites

The user nedes Python 3, PySide6, and Cargo (Rust) installed on their machine.

Playing as [black|white]
- /run.sh [black|white]

# Playing Level

The Chess AI has been tested and consistently drew against ELO 3000 chess.com bots.

- [DRAW - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1562860054/analysis)
- [DRAW - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1574164820/analysis)

- [WIN - ELO 2900 Bot](https://www.chess.com/analysis/game/computer/1550643276/analysis)
- [WIN - ELO 2900 Bot](https://www.chess.com/analysis/game/computer/1554932080/analysis)

## Contact

Alan Yuan

| Platform | Link | Intent |
| :--- | :--- | :--- |
| **Email** | [alan0408yuan@gmail.com](mailto:alan0408yuan@gmail.com) | Direct inquiries |
| **LinkedIn** | [linkedin.com](https://www.linkedin.com/in/alan-yuan-62301272/) | Professional networking |

*Response time: Typically within 24 hours.*
