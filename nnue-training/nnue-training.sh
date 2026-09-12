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

# 4. Deduplicate -> Input /data -> Output /data_dedup
# Prunes from 1 Billion Positons to 400 Million
# python global_dedup.py
# rm -f ./data/*

# 5. Data Parse + Mirror -> Input /data_dedup -> Output /data
# python dataset_exporter.py
# rm -f ./data_dedup/*

# 6. Deduplicate -> Input /data -> Output /data_dedup
# (Removes Duplicates created during Mirror)
# python global_dedup.py
# rm -f ./data/*

# 7. Data Mixer -> Input /data_dedup -> Output /production_shards
# python global_mixer.py
# rm -f ./data_dedup/*

# 8. Shard Balancer -> Input /production_shards -> Output /balanced_shards
python shard_balancer.py
# rm -f ./production_shards/*

# 9. Training / Validation Split -> Input /balanced_shards
./training_shuffle.sh

# 10. Run your main training script -> Input /balanced_shards
python train_pipeline.py

# 11. Upload Weights to Hugging Face
hf auth login
# hf upload AlanYuan0408/nnue_weights.bin nnue_weights.bin