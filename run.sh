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

# == STEP 4: Retrieve Tighter Tactic-Focused Performance Opening Book ==
BOOK_FILE="performance.bin"
ZIP_FILE="Performance.zip"
BOOK_URL="https://github.com/ChrisWhittington/polyglot-books/releases/latest/download/Performance.zip"

if [ ! -f "$BOOK_FILE" ]; then
    echo "'$BOOK_FILE' handbook database missing. Fetching performance-constrained release..."
    
    # Download the Performance release archive
    curl -L -o "$ZIP_FILE" "$BOOK_URL"
    
    if [ $? -eq 0 ] && [ -f "$ZIP_FILE" ]; then
        echo "Archive received. Unpacking focused performance binary files..."
        
        # Unzip the content quietly
        unzip -q "$ZIP_FILE"
        
        # Dynamically locate the unpacked .bin file and normalize name to performance.bin
        EXTRACTED_BIN=$(ls *.bin 2>/dev/null | head -n 1)
        if [ -n "$EXTRACTED_BIN" ]; then
            mv "$EXTRACTED_BIN" "$BOOK_FILE"
            echo "Success! Specialized opening handbook active as: $BOOK_FILE"
        else
            echo "Error: Zip archive did not contain an expected .bin database file."
        fi
        
        # Cleanup the download wrapper zip
        rm -f "$ZIP_FILE"
    else
        echo "Warning: Download failed. Defaulting to engine computation framework."
    fi
else
    echo "'$BOOK_FILE' specialized opening dataset already active. Skipping setup."
fi


# 4. Run your main script
python chessApp.py $1

# Rust Flame Graph
# samply record python chessApp.py $1