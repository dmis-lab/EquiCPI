#!/bin/bash

# Prompt user for GPU selection
read -p "Enter the GPU ID to use: " GPU
echo "Selected GPU: $GPU"

# Define training parameters
MODEL_NAME="model9"
BATCH_SIZE=75
EPOCHS=30
TASK="novel_comp"
DROPOUT=0.9
RESULT_NAME="${MODEL_NAME}_classification_mean_aggdrop"

# Run training
echo "Starting training with model: $MODEL_NAME"
CUDA_VISIBLE_DEVICES="$GPU" python train.py \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --n_epochs "$EPOCHS" \
  --task "$TASK" \
  --dropout "$DROPOUT" \
  --result_name "$RESULT_NAME"
