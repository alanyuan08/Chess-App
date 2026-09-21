#!/bin/bash

# Exit the shell script immediately if any individual command fails
set -e

# 1. Check if the environment already exists to prevent overwriting
brew install python@3.11
if [ ! -d "../.tf_venv" ]; then
    python3.11 -m venv ../.tf_venv
fi

# 2. Activate and Install
source ../.tf_venv/bin/activate
pip install --upgrade pip
pip install "../[training]" 

# 3. Download the files -> Output /data
# ./download_shards.sh

# 4. Process the Positions -> Input /data -> Output /processed_shards
# python dataset_exporter.py
# rm -f ./data/*

# 5. Data Parse + Mirror -> Input /processed_shards -> Output /data_dedup
# python global_dedup.py
# rm -f ./processed_shards/*

# 6. Shard Balancer -> Input /data_dedup -> Output /balanced_shards
python shard_balancer.py
# rm -f ./data_dedup/*

# 9. Run your main training script -> Input /balanced_shards
python train_pipeline.py

# 10. Upload Weights to Hugging Face
hf auth login
# hf upload AlanYuan0408/nnue_weights.bin nnue_weights.bin