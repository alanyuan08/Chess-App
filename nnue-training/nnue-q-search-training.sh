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

for YEAR in $(seq 2013 2 2021); do
    NEXT_YEAR=$((YEAR + 1))

    echo "=========================================================="
    if [ "$NEXT_YEAR" -le 2021 ]; then
        echo ">>> STARTING PIPELINE FOR YEARS: $YEAR and $NEXT_YEAR"
    else
        echo ">>> STARTING PIPELINE FOR YEAR: $YEAR"
    fi
    echo "=========================================================="

    # 3. Download the files for both years -> Output ./data
    ./download_q_search_shards.sh $YEAR
    if [ "$NEXT_YEAR" -le 2021 ]; then
        ./download_q_search_shards.sh $NEXT_YEAR
    fi

    # 4. Process the Positions -> Input ./data -> Output ./processed_shards
    python dataset_exporter.py --cut

    # Clean up raw shards to reclaim disk space before next batch
    rm -f ./data/*.parquet ./data/*.part

    # Clean up any leftover background Python processes
    pkill -u $(whoami) -f dataset_nnue_q_search_exporter.py || true

    echo ">>> COMPLETED BATCH STARTING: $YEAR"
done

echo "All years (2013 2021) processed!"

# 5. Data Parse + Mirror -> Input /processed_shards -> Output /data_dedup
python global_dedup.py
sync
rm -f ./processed_shards/*

# 7. Shard Balancer -> Input /data_dedup -> Output /balanced_shards
python shard_balancer.py
sync
rm -f ./data_dedup/*

# 9. Run your main training script -> Input /balanced_shards
python train_q_search_pipeline.py

# 10. Upload Weights to Hugging Face
hf upload AlanYuan0408/nnue_q_search_weights.bin nnue_q_search_weights.bin
