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

# 4. Deduplicate and Mix -> Input /data -> Output /data_dedup
python global_dedup.py
rm -f ./data/*

# 5. Data Mixer -> Input /data_dedup -> Output /data_dedup_mixed
python global_mixer.py
rm -f ./temp_mixer_shards/*
rm -f ./data_dedup/*

# 6. Data Exporter -> Input /data_dedup_mixed -> Output /production_shards
python dataset_exporter.py
rm -f ./data_dedup_mixed/*

# 6. Training / Validation Split -> Input /production_shards
./training_shuffle.sh

# 7. Run your main training script -> Input /production_shards
python train_pipeline.py

# 8. Upload Weights to Hugging Face
hf auth login
hf upload AlanYuan0408/nnue_weights.bin nnue_weights.bin