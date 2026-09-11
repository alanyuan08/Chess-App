# AlanBot Chess AI

<img src="img/saved_game/saved_game.png" width="50%">

- **Presentation Layer:** Python PySide6 for drag and drop interface
  
- **Compute Engine:** Rust Maturin for Adversarial Search - Negamax with Quiescence Search with advanced pruning techniques such as Killer Move Heuristics, Late Move Reduction, Principal Variation Search, and Null-Move Pruning.

  The engine processes 17+ million nodes per second on Apple M4 Pro (8 Performance Threads) and averages 16+ depth on 15 second search. 

- **Move Generation:** BitBoard for board representation and BitBoard Magic Number to calculate moves for sliding pieces. 

- **Evaluation:** Self-trained NNUE (Dual-Perspective HalfKA) using Lichess FEN -> Score Positions.

- **NNuE Training:** Trained on [Lichess Chess Position Evaluations](https://huggingface.co/datasets/Lichess/chess-position-evaluations).

  The positions are filtered for Quiet Positions and the score is converted to a win percentage between [0 to 1] using a Sigmoid Function.
  
## 1. Python Presentation & Validation Layer

- **PySide6 UI:** Renders a fluid 2D chessboard and manages real-time player drag-and-drop interactions

- **Move Validation:** Enforces legal moves and coordinates state synchronization with the rust compute engine

- **Opening Handbook:** Integrates a built-in opening book containing standard opening lines

## 2. Rust Compute Engine

- **Bitboard Move Generation:** Uses 64-bit integers with fast AND / XOR logic to compute board occupancy. It also uses BitBoard Magic Number to instant compute sliding pieces moves and attacks.

- **Adversarial Search:** Implements Minimax (Negemax) adversarial search and uses Quiescence Search to extend the search for non-quiet positions to mitigate the horizon effect.

- **Advanced Pruning:** Uses Killer Move Heuristics, Late Move Reduction and Null-Move Pruning. to improve the Alpha / Beta cutoff. 

- **Deep Evaluation:** Combines Iterative Deepening with Principal Variation Search (PVS) to regularly achieve search depths of 16+ plies. (Average Move is approximately 14+ seconds).

- **Transposition Tables:** Caches previously evaluated board states to accelerate search paths in a lockless transposition Table. The tables uses the Condon-Thompson Replacement method to increase efficiency of L1 / L2 / L3 caches by prioritizng positions that are frequently traversed positions and evaluations with strong depth. 

- **Zobrist Hash:** Uses a unique 64-bit Zobrist Hashing for every board position and is incrementally updated using XOR operations; This is used for detecting three-move repetition and in the Transposition Table

- **Parallel Processing:** The search uses Lazy SMP (Symmetric Multiprocessing) which uses multiple search algorithms to independently process the same search evaluation agorithm and share position evaluations and cut-offs using a shared Lockless Transposition table.

- **Performance Benchmark:** Processes approximately 17+ million nodes per second (NPS) on an Apple M4 Pro chip. (8 Performance Core Only - 4.5 GHz + On-Chip Cache Memory - 39.5 MB)

## 3. Neural Network Evaluation

- **NNuE Architecture:** The engine features a customized **Dual-Perspective HalfKA** perspective neural network utilizing a hybrid quantization layout. The architectural data pathways progress as follows:

  - **Input Preprocessing:** 
    - Convert to CentiPawn: ($\text{Centipawn / 100.0}$) -> pawn_units

  $$\text{Inputs (49,152)} \rightarrow \text{Accumulator (256)} \rightarrow \text{Multiplexed Perspective (512)} \rightarrow \text{Hidden 2 (64)} \rightarrow \text{Hidden 3 (32)} \rightarrow \text{Output (1)}$$

  - **Input Layer:** $12 \times 64 \times 64 = 49,152$ sparse features mapping active piece-square configurations relative to your own active King's position.
  - **Accumulator Layer:** Shapes into $(49152, 256)$ weights and $(256,)$ biases quantized to signed 16-bit integers (`i16`). Uses branchless tensor multiplexing to concatenate White/Black points of view into a unified $512$-dimensional vector.
    - *Activation:* Clipped/Bounded Linear ReLU ($\text{ReLU1}$) bounded strictly between `0.0` and `1.0`.
  - **Hidden Layer 2:** Matrix transformation mapping $(512, 32)$ quantized to signed 8-bit weights (`i8`) and 32-bit biases (`i32`).
    - *Activation:* Clipped/Bounded Linear ReLU ($\text{ReLU1}$) bounded strictly between `0.0` and `1.0`.
  - **Hidden Layer 3:** Matrix transformation mapping $(32, 32)$ quantized to signed 8-bit weights (`i8`) and 32-bit biases (`i32`).
    - *Activation:* Clipped/Bounded Linear ReLU ($\text{ReLU1}$) bounded strictly between `0.0` and `1.0`.
  - **Output Layer:** Combines $(32, 1)$ outputs down to a single evaluation scalar using 8-bit weights (`i8`) and 32-bit biases (`i32`).
    - *Activation:* None
  - **Loss Function:** Custom Mean Squared Error Function where the model output is converted from pawnUnits to Loss/ Win [0, 1] using the function ($\text{1.0 / (1.0 + tf.math.exp(STOCKFISH-CONSTANT * pawnUnits))}$)

  STOCKFISH-CONSTANT = 0.244

## 4. NNUE Training

- **Training Data:** The model is trained on **394,669,566 chess positions** sourced from the *Lichess Chess Position Evaluations* dataset, which features evaluations calculated by Stockfish at various depths. To ensure an even and unbiased distribution, the training and validation sets utilize separate shards, and the training data is thoroughly shuffled.

- **Preprocessing & Data Preparation:**
  - **Quiet Position Filtering:** The dataset is filtered to include only "quiet" positions. Board states are excluded if the king is in check, an immediate tactical win is available via captures, or a forced checkmate sequence exists.
  - **Data Augmentation:** The filtered positions are augmented by rotating each board state 180 degrees. This process yields a total of **394,357,440 unique positions**.
  - **Deduplication & Balancing:** After deduplication, the data is randomized within the training sets to preserve its natural valuation distribution:
    - 0 to 150 Centipawns: 67.62%
    - 150 to 400 Centipawns: 19.55%
    - 400 to 800 Centipawns: 12.07%
    - 800 to 1000 Centipawns: 0.76%

- **Training Configuration:**
  - **Target Optimization:** The model applies a Sigmoid transformation to convert raw evaluation scores into a win probability scale where 1.0 represents a win, 0.5 a draw, and 0.0 a loss. This bounds the output and forces the model to focus on highly competitive positions rather than overwhelming outliers.
  - **Architecture & Schedule:** Training runs for **2,000 epochs**, featuring **4,000 steps per epoch** with a batch size of **8,192 positions per step**.
  - **Loss Function:** Performance is calculated using Mean Squared Error (MSE) between the predicted and expected win probabilities: 
    $$\text{MSE} = (Y_{\text{pred}} - Y_{\text{expected}})^2$$
  - **Learning Rate Schedule:** The optimization uses a stepped learning rate decay to fine-tune weights over time:
    - Epochs 0 to 139: 0.001
    - Epochs 140 to 175: 0.0001
    - Epochs 176 to 2000: 0.00001

- cd nnue-training
- /train_pipeline.sh

## 5. Playing Level

The Chess AI has been tested against ELO 3200+ Chess.com bots.

- [WIN - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1617707258/analysis)
- [WIN - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1978599994/analysis)

- **Future Roadmap:** This engine has not been officially ratified by Computer Chess Rating Lists

## 6. Running the App

Playing as [black|white]
- /run.sh [black|white]

## 7. Contact

Alan Yuan

| Platform | Link | Intent |
| :--- | :--- | :--- |
| **Email** | [alan0408yuan@gmail.com](mailto:alan0408yuan@gmail.com) | Direct inquiries |
| **LinkedIn** | [linkedin.com](https://www.linkedin.com/in/alan-yuan-62301272/) | Professional networking |

*Response time: Typically within 24 hours.*
