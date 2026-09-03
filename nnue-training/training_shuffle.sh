#!/bin/bash

# Exit the shell script immediately if any individual command fails
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE}" )" && pwd )"
PRODUCTION_DIR="$SCRIPT_DIR/production_shards"

TRAIN_DIR="$PRODUCTION_DIR/training"
VAL_DIR="$PRODUCTION_DIR/validation"

# 2. Generate target subdirectories if they are missing
mkdir -p "$TRAIN_DIR"
mkdir -p "$VAL_DIR"

for i in {1..30}; do
    FILENAME="production_wave_${i}.parquet"
    SOURCE_PATH="$PRODUCTION_DIR/$FILENAME"
    
    # Verify the file actually exists before attempting to move it
    if [ ! -f "$SOURCE_PATH" ]; then
        echo "[WARNING] Target file not found, skipping: $FILENAME"
        continue
    fi
    
    # 4. Enforce your exact validation split rule: 
    # Waves 1 to 5 go to validation, the remaining go to training
    if [ "$i" -le 5 ]; then
        echo "Allocating to VALIDATION ──> $FILENAME"
        mv "$SOURCE_PATH" "$VAL_DIR/"
    else
        echo "Allocating to TRAINING   ──> $FILENAME"
        mv "$SOURCE_PATH" "$TRAIN_DIR/"
    fi
done
