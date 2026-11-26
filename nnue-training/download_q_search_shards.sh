#!/bin/bash
# Exit immediately if any command fails
set -e

# Require a 4-digit year as the first argument
YEAR="${1:?Usage: $0 <year> (e.g., $0 2022)}"

if ! [[ "$YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "Error: Year must be a 4-digit number (got '$YEAR')." >&2
    exit 1
fi

START_MONTH=1
END_MONTH=12

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"

BASE_URL="https://huggingface.co/datasets/Lichess/fishnet-evals/resolve/main"

echo "=== Downloading Lichess fishnet-evals shards for $YEAR (months $START_MONTH..$END_MONTH) ==="

for month in $(seq -f "%02g" "$START_MONTH" "$END_MONTH"); do
    FILENAME="standard_rated_${YEAR}_${month}.parquet"
    URL="$BASE_URL/$FILENAME"
    DEST="$DATA_DIR/$FILENAME"
    PART_DEST="${DEST}.part"

    if [ -f "$DEST" ]; then
        echo "[SKIP] $FILENAME already exists locally in data/."
        continue
    fi

    echo "Streaming $FILENAME down from Hugging Face..."
    # Download to a .part file first so interrupted downloads are never mistaken for complete files
    if curl -L -C - --fail "$URL" -o "$PART_DEST"; then
        mv "$PART_DEST" "$DEST"
        echo "[SUCCESS] Saved $FILENAME cleanly."
    else
        echo "[MISS] $FILENAME not found on remote (or failed), skipping."
        rm -f "$PART_DEST"
    fi
done

echo "Download complete for $YEAR! Files saved in: $DATA_DIR"