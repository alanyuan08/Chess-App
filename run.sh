#!/bin/bash

# 1. Create venv if it doesn't exist
brew install python
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# 2. Activate and Install
source .venv/bin/activate
pip install --upgrade pip
pip install .

# Build Rust
cargo clean
RUSTFLAGS="-C target-cpu=native" maturin develop --release

python --version

# 3. Retrieve Weights from Hugging Face
TARGET_FILE="nnue-training/nnue_weights.bin"
DOWNLOAD_URL="https://huggingface.co/AlanYuan0408/nnue_weights.bin/resolve/main/nnue_weights.bin?download=true"

if [ ! -f "$TARGET_FILE" ]; then
    echo "'$TARGET_FILE' not found locally."
    
    # -L follows CDN redirects, -o writes the output to your target file path
    curl -L -o "$TARGET_FILE" "$DOWNLOAD_URL"
    
    # Verify the download succeeded
    if [ $? -eq 0 ]; then
        echo "Download complete! File saved as $TARGET_FILE"
    else
        echo "Error: Download failed."
        exit 1
    fi
else
    echo "'$TARGET_FILE' already exists. Skipping download step."
fi

# 4. Run your main script
python chessApp.py $1

# Rust Flame Graph
# samply record python chessApp.py $1