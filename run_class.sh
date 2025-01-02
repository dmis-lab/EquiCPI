#!/bin/bash 
echo "Please enter the GPU address"
read GPU
echo "Selected GPU: $GPU"

#train
CUDA_VISIBLE_DEVICES=$GPU python train.py --model_name model9 --batch_size 75 --n_epochs 30 --task novel_comp --dropout 0.9 --result_name model9_classifcation_mean_aggdrop
