#!/bin/bash

# 1. Check if the environment already exists to prevent overwriting
brew install python@3.11
if [ ! -d "../.tf_venv" ]; then
    python3.11 -m venv ../.tf_venv
fi

# 2. Activate and Install
source ../.tf_venv/bin/activate
pip install --upgrade pip

# FIX: Install the current directory package plus the training extras
pip install "../[training]" 

# 3. Run your main training script
python train_halfkpr_from_fen.py

# 4. Upload Weights to Hugging Face
brew install hf
hf auth login
hf upload AlanYuan0408/nnue_weights.bin nnue_weights.bin