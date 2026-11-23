#!/bin/bash

# Exit the shell script immediately if any individual command fails
set -e

# Keep-alive loop: Refreshes your sudo credential token every 60 seconds 
# in the background. It automatically terminates when this script ends.
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

# 1. Check if the environment already exists to prevent overwriting
brew install python@3.11
if [ ! -d "../.tf_venv" ]; then
    python3.11 -m venv ../.tf_venv
fi

# 2. Activate and Install
source ../.tf_venv/bin/activate
pip install --upgrade pip
pip install "../[training]" 

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE}" )" && pwd )"
cd "$SCRIPT_DIR"

mkdir -p ./data ./processed_shards

for YEAR in $(seq 2020 2021); do
    echo "=========================================================="
    echo ">>> STARTING PIPELINE FOR YEAR: $YEAR"
    echo "=========================================================="

    # 3. Download the files for $YEAR and $HALF -> Output ./data
    ./download_q_search_shards.sh "$YEAR"

    # Skip processing if no files were downloaded for this batch
    shopt -s nullglob
    parquet_files=(./data/*.parquet)
    shopt -u nullglob

    if [ ${#parquet_files[@]} -eq 0 ]; then
        echo "[INFO] No parquet files found in ./data for $YEAR Skipping exporter."
        continue
    fi

    # 4. Process the Positions -> Input ./data -> Output ./processed_shards
    python dataset_nnue_q_search_exporter.py sync

    # Clean up raw shards to reclaim disk space before next batch
    rm -f ./data/*.parquet ./data/*.part

    # Clean up any leftover background Python processes
    pkill -u "$(whoami)" -f "dataset_nnue_q_search_exporter.py" || true

    echo ">>> COMPLETED YEAR: $YEAR"
done

echo "All years (2013 2025) processed!"

# 5. Data Parse + Mirror -> Input /processed_shards -> Output /data_dedup
python global_dedup.py
sync
rm -f ./processed_shards/*

# 6. Count Distrubution
python count_targets.py

# 7. Shard Balancer -> Input /data_dedup -> Output /balanced_shards
python shard_balancer.py
sync
rm -f ./data_dedup/*

# 9. Run your main training script -> Input /balanced_shards
python train_q_search_pipeline.py

# 10. Upload Weights to Hugging Face
hf auth login
hf upload AlanYuan0408/nnue_q_search_weights.bin nnue_q_search_weights.bin
