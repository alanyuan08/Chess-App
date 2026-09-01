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

# 3. Download the files
./download_shards.sh

# 4. Data Exporter
python dataset_exporter.py

# 5. Global Mixer 
python global_mixer.py

# 6. Training / Validation Split
./training_shuffle.sh

# 7. Run your main training script
python train_pipeline.py

# 8. Upload Weights to Hugging Face
hf auth login
hf upload AlanYuan0408/nnue_weights.bin nnue_weights.bin