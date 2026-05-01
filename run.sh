#!/bin/bash

# 1. Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# 2. Activate and Install
source .venv/bin/activate
pip install .

# Build Rust
# maturin develop 

# 3. Run your main script
python chessApp.py black