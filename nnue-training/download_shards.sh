#!/bin/bash
# Exit immediately if any command fails
set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$SCRIPT_DIR/data"
mkdir -p "$DATA_DIR"

# Official Hugging Face Parquet base directory link URL
BASE_URL="https://huggingface.co/datasets/Lichess/chess-position-evaluations/resolve/main/data"

# Adjust these bounds to download more data shards!
# {0..19} downloads 20 shards total (data_0000 to data_0019)
START_SHARD=0
END_SHARD=19

echo "====================================================================="
echo "       STARTING MASSIVE MULTI-SHARD PARQUET DOWNLOAD PASSTHROUGH     "
echo "====================================================================="

# Loop through the numbers, padding them with zeros to 4 digits (e.g., 0000, 0001)
for i in $(seq -f "%04g" $START_SHARD $END_SHARD); do
    FILENAME="data_$i.parquet"
    URL="$BASE_URL/$FILENAME"
    DEST="$DATA_DIR/$FILENAME"
    
    if [ -f "$DEST" ]; then
        echo "[SKIP] $FILENAME already exists locally in data/ folder."
    else
        echo "Streaming $FILENAME down from Hugging Face storage node..."
        # -L handles redirects, -C - automatically resumes if the internet drops
        curl -L -C - --fail "$URL" -o "$DEST"
        echo "[SUCCESS] Saved $FILENAME cleanly."
    fi
done

echo -e "\nAll requested Parquet shards successfully staged inside your local data/ folder!"
echo "====================================================================="
