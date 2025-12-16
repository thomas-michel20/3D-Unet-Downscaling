#!/bin/bash
#SBATCH --job-name=unet3d_eval      # Job name
#SBATCH --output=logs/eval_%j.out      # Save stdout to file, %j = job ID
#SBATCH --error=logs/eval_%j.err       # Save stderr to file
#SBATCH --time=04:00:00                 # Time limit hrs:min:sec
#SBATCH --partition=gpu                 # Partition (adjust to your cluster)
#SBATCH --gres=gpu:2                    # Request 1 GPU
#SBATCH --cpus-per-task=4               # CPU cores per task
#SBATCH --mem=32G                       # Memory per node

# Activate conda
source ~/miniconda/etc/profile.d/conda.sh
conda activate thomas

save_dir_name="./model_eval"
mkdir -p $save_dir_name

python evaluation_3d.py \
  --eval_downsamp_xyz 2 \
  --ckpt ./model_dir/checkpoint_latest.pth.tar_pdenet_best.pth.tar \
  --data_folder_training '' \
  --data_folder_evaluation '' \
  --train_data ./data/MHD_64_train.npz \
  --eval_data ./data/MHD_64_val.npz  \
  --save_path $save_dir_name \
  --lres_interp linear \
  --lres_filter none \
  --eval_xres 64 \
  --eval_yres 64 \
  --eval_zres 64 \
  --frame_rate 1 \
  --nbr_val_samples 20 \
  --eval_pseudo_batch_size 1000 \
  --save_video 

