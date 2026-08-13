# AlanBot Chess AI

<img src="img/saved_game/saved_game.png" width="50%">

- **Presentation Layer:** Python PySide6 for drag and drop interface
  
- **Compute Engine:** Rust Maturin for Adversarial Search - Negamax with Quiescence Search with advanced pruning techniques, Killer Move Heuristics, Late Move Reduction, Principal Variation Search, and Null-Move Pruning.

  The engine processes 10 million nodes per second on Apple M4 Pro (8 Performance Threads) and averages 12+ depth on 20 second search. 
  
- **Evaluation:** Self-trained NNUE (Dual-Perspective HalfKA) using Lichess FEN -> Score Positions
  
## 1. Python Presentation & Validation Layer

- **PySide6 UI:** Renders a fluid 2D chessboard and manages real-time player drag-and-drop interactions

- **Move Validation:** Enforces legal moves and coordinates state synchronization with the engine core

- **Opening Handbook:** Integrates a built-in opening book containing standard opening lines

## 2. Rust Compute Engine

- **Bitboard Move Generation:** Maximizes throughput by computing all pseudo-legal move paths across millions of positions per second

- **Adversarial Search:** Implements Minimax (Negemax) search enhanced by Alpha-Beta pruning and a Quiescence search to eliminate horizon-effect instability.

- **Advanced Pruning:** Uses Killer Move Heuristics, Late Move Reduction and Null-Move Pruning to improve the alpha / beta cutoff. 

- **Deep Evaluation:** Combines Iterative Deepening with Principal Variation Search (PVS) to regularly achieve search depths of 14+ plies. (Average Move is approximately 20+ seconds)

- **Transposition Tables:** Caches previously evaluated board states to accelerate search paths and share data across threads. The tables uses the Condon-Thompson Replacement method to increase efficiency of L1 / L2 / L3 caches. 

- **Parallel Processing:** Scales performance across CPU threads using a lock-free concurrent tree search architecture (Lazy SMP)

- **Performance Benchmark:** Processes approximately 10 million nodes per second (NPS) on an Apple M4 Pro chip. (8 Performance Core Only - 4.5 GHz + On-Chip Cache Memory - 39.5 MB)

## 3. Neural Network Evaluation

- **NNUE Architecture:** The engine features a customized **Dual-Perspective HalfKA** perspective neural network utilizing a hybrid quantization layout. The architectural data pathways progress as follows:

  - **Input Preprocessing:** 
  1. Convert to CentiPawn: ($\text{Centipawn / 100.0}$) -> Pawn
  2. Smooth & Bound (Soft-Capping): ($\text{10.0 * tf.math.tanh(Pawn / 10.0)}$) -> Cap
  3. Compute Win Probability: ($\text{1.0 / (1.0 + tf.math.exp(-0.41 * Cap))}$) -> score

  $$\text{Inputs (49,152)} \rightarrow \text{Accumulator (256)} \rightarrow \text{Multiplexed Perspective (512)} \rightarrow \text{Hidden 2 (64)} \rightarrow \text{Hidden 3 (32)} \rightarrow \text{Output (1)}$$

  - **Input Layer:** $12 \times 64 \times 64 = 49,152$ sparse features mapping active piece-square configurations relative to your own active King's position.
  - **Accumulator Layer:** Shapes into $(49152, 256)$ weights and $(256,)$ biases quantized to signed 16-bit integers (`i16`). Uses branchless tensor multiplexing to concatenate White/Black points of view into a unified $512$-dimensional vector.
    - *Activation:* Clipped/Bounded Linear ReLU ($\text{ReLU1}$) bounded strictly between `0.0` and `1.0`.
  - **Hidden Layer 2:** Matrix transformation mapping $(512, 64)$ quantized to signed 8-bit weights (`i8`) and 32-bit biases (`i32`).
    - *Activation:* Clipped/Bounded Linear ReLU ($\text{ReLU1}$) bounded strictly between `0.0` and `1.0`.
  - **Hidden Layer 3:** Matrix transformation mapping $(64, 32)$ quantized to signed 8-bit weights (`i8`) and 32-bit biases (`i32`).
    - *Activation:* Clipped/Bounded Linear ReLU ($\text{ReLU1}$) bounded strictly between `0.0` and `1.0`.
  - **Output Layer:** Combines $(32, 1)$ outputs down to a single evaluation scalar using 8-bit weights (`i8`) and 32-bit biases (`i32`).
    - *Activation:* ($\text{activation=Tanh_Smooth}$). Output a value bounded strictly between `-10.0` and `10.0`. 

- **NNUE Training Data:** The evaluation network is trained exclusively on normalized Stockfish evaluations mapped from standard Forsyth-Edwards Notation (FEN) profiles spanning varied positional lines and forced checkmate sequences.

- **Dataset Source:** [Lichess Chess Position Evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations) The dataset is filtered for quiet positions (Not in Check, No Captures)

## 4. Playing Level

The Chess AI has been tested against ELO 3200+ Chess.com bots. There is a concensus that Chess.com bots are likely overrated by 200 ELO points. 

- [WIN - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1617707258/analysis)
- [DRAW - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1562860054/analysis)
- [DRAW - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1574164820/analysis)

- **Future Roadmap:** This engine has not been officially ratified by Computer Chess Rating Lists

## 5. Running the App

Playing as [black|white]
- /run.sh [black|white]

## 6. Contact

Alan Yuan

| Platform | Link | Intent |
| :--- | :--- | :--- |
| **Email** | [alan0408yuan@gmail.com](mailto:alan0408yuan@gmail.com) | Direct inquiries |
| **LinkedIn** | [linkedin.com](https://www.linkedin.com/in/alan-yuan-62301272/) | Professional networking |

*Response time: Typically within 24 hours.*
