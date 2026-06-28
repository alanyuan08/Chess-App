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

- **Advanced Pruning:** Uses Killer Move Heuristics and Late Move Reduction to improve the alpha / beta cutoff. The algorithm does not utilize Null-Move Pruning as it is currently using the Timeca[...]

- **Deep Evaluation:** Combines Iterative Deepening with Principal Variation Search (PVS) to regularly achieve search depths of 14+ plies. (Average Move is approximately 20 seconds to 1 minutes)

- **Transposition Tables:** Caches previously evaluated board states to accelerate search paths and share data across threads. The tables uses the Condon-Thompson Replacement method to increase ef[...]

- **Parallel Processing:** Scales performance across CPU threads using a lock-free concurrent tree search architecture (Lazy SMP)

- **Performance Benchmark:** Processes approximately 10 million nodes per second (NPS) on an Apple M4 Pro chip. (8 Performance Core Only - 4.5 GHz + On-Chip Cache Memory - 39.5 MB)

## 3. Neural Network Evaluation

- **NNUE Integration:** Features an incrementally updated Efficiently Updatable Neural Network (NNUE) paired with the Universal Chess Interface (UCI) protocol.

- **UCI Protocol Support:** Standalone UCI server for FastChess integration and CCRL rating list submission.

> **Current Status:** The engine now supports full UCI protocol compliance, enabling compatibility with FastChess, CuteChess, and other UCI interfaces. See [FastChess Integration Guide](fastchess_integration.md) for details.

> **Roadmap:** Custom NNUE training framework to replace Timecat dependency, enabling CCRL submission without external dependencies.

# Running the App

## Prerequisites

The user needs Python 3, PySide6, and Cargo (Rust) installed on their machine.

### Desktop GUI Mode
Playing as [black|white]
```bash
./run.sh [black|white]
```

### UCI Server Mode (FastChess Integration)

```bash
cd rust_compute
cargo build --release
./target/release/chess_app
```

The UCI server will listen for FastChess commands on standard input/output.

# Playing Level

The Chess AI has been tested against ELO 3000+ chess.com bots. There is controversy that the chess.com bot score is likely inflated 100-150 ELO+.

- [WIN - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1617707258/analysis)
- [DRAW - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1562860054/analysis)
- [DRAW - ELO 3200 Bot](https://www.chess.com/analysis/game/computer/1574164820/analysis)

## FastChess Integration

For automated testing and CCRL rating submission:

```bash
# Build UCI server
cd rust_compute && cargo build --release

# Run with FastChess
fastchess -engine1 "./target/release/chess_app" \
  -engine2 "stockfish" \
  -games 100 \
  -concurrency 4
```

See [FastChess Integration Guide](fastchess_integration.md) for complete instructions.

## Contact

Alan Yuan

| Platform | Link | Intent |
| :--- | :--- | :--- |
| **Email** | [alan0408yuan@gmail.com](mailto:alan0408yuan@gmail.com) | Direct inquiries |
| **LinkedIn** | [linkedin.com](https://www.linkedin.com/in/alan-yuan-62301272/) | Professional networking |

*Response time: Typically within 24 hours.*
