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
./download_shards.sh

# 4. Deduplicate and Mix -> Input /data -> Output /data_dedup
python global_dedup.py
rm -f ./data/*

# 5. Data Parse + Mirror -> Input /data_dedup -> Output /data_mirrored
python dataset_exporter.py
rm -f ./data_dedup/*

# 6. Data Mixer -> Input /data_mirrored -> Output /production_shards
python global_mixer.py
rm -f ./data_mirrored/*

# 7. Training / Validation Split -> Input /production_shards
./training_shuffle.sh

# 8. Run your main training script -> Input /production_shards
python train_pipeline.py

# 9. Upload Weights to Hugging Face
hf auth login
# hf upload AlanYuan0408/nnue_weights.bin nnue_weights.bin