# FastChess UCI Integration Guide

## Overview

This document describes how to integrate Chess-App with **FastChess**, an automated chess engine testing framework. The engine now supports full UCI (Universal Chess Interface) protocol compliance, making it compatible with FastChess, CuteChess, Arena, and other UCI-compatible interfaces.

## Building the UCI Server

### Prerequisites
- Rust 1.70+ (Cargo)
- Standard development tools (gcc/clang)

### Build Steps

```bash
cd rust_compute
cargo build --release
```

The compiled binary will be available at `rust_compute/target/release/chess_app` (or `chess_app.exe` on Windows).

## Running with FastChess

### 1. Configure FastChess

Create a FastChess configuration file (`engines.json`):

```json
{
  "engines": [
    {
      "name": "ChessAI",
      "cmd": "/path/to/rust_compute/target/release/chess_app",
      "options": {
        "Hash": 256,
        "Threads": 4
      }
    }
  ]
}
```

### 2. Run FastChess Tests

```bash
fastchess -engine1 "/path/to/chess_app" -engine2 "stockfish" \
  -games 100 \
  -concurrency 4 \
  -output games.pgn
```

## UCI Protocol Support

The engine implements the full UCI protocol with the following commands:

### Implemented Commands

| Command | Description | Status |
|---------|-------------|--------|
| `uci` | Identify engine and list options | ✅ Supported |
| `isready` | Check engine availability | ✅ Supported |
| `setoption name <n> value <v>` | Configure engine options | ✅ Supported |
| `position startpos` | Set starting position | ✅ Supported |
| `position fen <fen>` | Set position from FEN | ✅ Supported |
| `go depth <d>` | Search to fixed depth | ✅ Supported |
| `go movetime <ms>` | Search for fixed time | ✅ Supported |
| `go wtime <ms> btime <ms>` | Time-managed search | ✅ Supported |
| `stop` | Halt current search | ✅ Supported |
| `quit` | Exit engine | ✅ Supported |

### Configurable Options

```
Hash (spin): Transposition table size in MB (default: 256, range: 1-33554432)
Threads (spin): Number of search threads (default: 1, range: 1-512)
```

## CCRL Submission Requirements

### Pre-Submission Checklist

- [ ] Engine passes basic UCI protocol tests
- [ ] Engine responds correctly to all UCI commands
- [ ] Engine outputs valid moves in all positions
- [ ] Engine handles time management correctly
- [ ] Engine is deterministic (same position = same moves)
- [ ] Engine includes proper identification strings
- [ ] No external dependencies (Timecat NNUE removed)

### CCRL Rating List Submission

1. **Prepare Engine Package**
   ```bash
   mkdir -p ccrl_submission
   cp rust_compute/target/release/chess_app ccrl_submission/
   ```

2. **Create Engine Info File** (`ccrl_submission/info.txt`):
   ```
   Engine: ChessAI
   Version: 1.0
   Author: Alan Yuan
   NNUE: Custom trained network (remove if using external NNUE)
   Hash: 256 MB
   Threads: 4
   ```

3. **Submit to CCRL**
   - Visit [CCRL Website](https://www.computerchess.org.uk/ccrl/)
   - Follow submission guidelines
   - Provide engine binary and info file
   - Expected rating establishment: 20-40 games

### Testing Before Submission

```bash
# Test against known opponents
fastchess -engine1 "ccrl_submission/chess_app" \
  -engine2 "stockfish" \
  -games 50 \
  -depth 20

# Verify output format
echo -e "uci\nquit" | ./ccrl_submission/chess_app
```

## Troubleshooting

### Engine Not Found
```bash
# Ensure binary is executable
chmod +x rust_compute/target/release/chess_app

# Verify path is correct
which chess_app
```

### UCI Protocol Errors

**Issue**: "Unknown command"
- **Solution**: Ensure FastChess sends complete UCI commands

**Issue**: "Invalid move"
- **Solution**: Verify position is set before searching

**Issue**: "Timeout"
- **Solution**: Increase `movetime` or `depth` limits

## Performance Benchmarks

### Target Performance Metrics

| Metric | Target | Current |
|--------|--------|--------|
| NPS (M nodes/sec) | 10+ | ✅ 10 M NPS (M4 Pro) |
| Search Depth (plies) | 14+ | ✅ 16-20 plies |
| Move Time | <60s | ✅ 20-60s avg |

### Optimization Tips

1. **Increase Hash Size**
   ```
   setoption name Hash value 2048
   ```

2. **Use Multiple Threads**
   ```
   setoption name Threads value 8
   ```

3. **Profile Search**
   ```bash
   fastchess -engine1 "chess_app" -depth 25 -games 1
   ```

## Future Improvements

- [ ] Custom NNUE training pipeline (remove Timecat dependency)
- [ ] Aspiration window search
- [ ] Syzygy tablebase support
- [ ] Advanced time management
- [ ] Multi-threading optimization (SMP)

## References

- [UCI Protocol Specification](http://wbec-ridderkerk.nl/html/UCIProtocol.html)
- [CCRL Rating List](https://www.computerchess.org.uk/ccrl/)
- [FastChess Documentation](https://github.com/FastChess-org/FastChess)

## Support

For issues or questions:
- Email: alan0408yuan@gmail.com
- GitHub Issues: [Chess-App/issues](https://github.com/alanyuan08/Chess-App/issues)
