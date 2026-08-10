# AlanBot Chess AI

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

- **Deep Evaluation:** Combines Iterative Deepening with Principal Variation Search (PVS) to regularly achieve search depths of 14+ plies. (Average Move is approximately 20 seconds to 1 minutes)

- **Transposition Tables:** Caches previously evaluated board states to accelerate search paths and share data across threads. The tables uses the Condon-Thompson Replacement method to increase efficiency of L1 / L2 / L3 caches. 

- **Parallel Processing:** Scales performance across CPU threads using a lock-free concurrent tree search architecture (Lazy SMP)

- **Performance Benchmark:** Processes approximately 10 million nodes per second (NPS) on an Apple M4 Pro chip. (8 Performance Core Only - 4.5 GHz + On-Chip Cache Memory - 39.5 MB)

## 3. Neural Network Evaluation

- **NNUE Architecture:** The engine features a customized **Dual-Perspective HalfKA** perspective neural network utilizing a hybrid quantization layout. The architectural data pathways progress as follows:
  
  $$\text{Inputs (49,152)} \rightarrow \text{Accumulator (256)} \rightarrow \text{Multiplexed Perspective (512)} \rightarrow \text{Hidden 2 (64)} \rightarrow \text{Hidden 3 (32)} \rightarrow \text{Output (1)}$$

  - **Input Layer:** $12 \times 64 \times 64 = 49,152$ sparse features mapping active piece-square configurations relative to your own active King's position.
  - **Accumulator Layer:** Shapes into $(49152, 256)$ weights and $(256,)$ biases quantized to signed 16-bit integers (`i16`). Uses branchless tensor multiplexing to concatenate White/Black points of view into a unified $512$-dimensional vector.
    - *Activation:* Clipped/Bounded Linear ReLU ($\text{ReLU1}$) bounded strictly between `0.0` and `1.0`.
  - **Hidden Layer 2:** Matrix transformation mapping $(512, 64)$ quantized to signed 8-bit weights (`i8`) and 32-bit biases (`i32`).
    - *Activation:* Clipped/Bounded Linear ReLU ($\text{ReLU1}$) bounded strictly between `0.0` and `1.0`.
  - **Hidden Layer 3:** Matrix transformation mapping $(64, 32)$ quantized to signed 8-bit weights (`i8`) and 32-bit biases (`i32`).
    - *Activation:* Clipped/Bounded Linear ReLU ($\text{ReLU1}$) bounded strictly between `0.0` and `1.0`.
  - **Output Layer:** Combines $(32, 1)$ outputs down to a single evaluation scalar using 8-bit weights (`i8`) and 32-bit biases (`i32`). Scaled dynamically by a target factor of $600.0$ to map output values straight to standard whole integer centipawns for the Alpha-Beta search tree.

- **NNUE Training Data:** The evaluation network is trained exclusively on normalized Stockfish evaluations mapped from standard Forsyth-Edwards Notation (FEN) profiles spanning varied positional lines and forced checkmate sequences.

- **Dataset Source:** [Lichess Chess Position Evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) The dataset is filtered for quiet positions to train the NNUE

# Running the App

Playing as [black|white]
- /run.sh [black|white]

# Playing Level

The Chess AI has been tested against ELO 3000+ chess.com bots. There is controversy that the chess.com bot score is likely inflated 100-150 ELO+.

- [WIN - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1617707258/analysis)
- [DRAW - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1562860054/analysis)
- [DRAW - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1574164820/analysis)

## Contact

Alan Yuan

| Platform | Link | Intent |
| :--- | :--- | :--- |
| **Email** | [alan0408yuan@gmail.com](mailto:alan0408yuan@gmail.com) | Direct inquiries |
| **LinkedIn** | [linkedin.com](https://www.linkedin.com/in/alan-yuan-62301272/) | Professional networking |

*Response time: Typically within 24 hours.*
