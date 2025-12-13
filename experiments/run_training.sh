#!/bin/bash
#SBATCH --job-name=unet3d_train      # Job name
#SBATCH --output=logs/train_%j.out      # Save stdout to file, %j = job ID
#SBATCH --error=logs/train_%j.err       # Save stderr to file
#SBATCH --time=06:00:00                 # Time limit hrs:min:sec
#SBATCH --partition=gpu                 # Partition (adjust to your cluster)
#SBATCH --gres=gpu:2                    # Request 1 GPU
#SBATCH --cpus-per-task=4               # CPU cores per task
#SBATCH --mem=32G                       # Memory per node

# Activate conda
source ~/miniconda/etc/profile.d/conda.sh
conda activate thomas

# Make sure logs dir exists
log_dir_name="./model_dir"
mkdir -p $log_dir_name


# Run training
python train_3d.py \
  --batch_size_per_gpu 8 \
  --epochs 30 \
  --pseudo_epoch_size 500 \
  --lr 1e-2 \
  --seed 1 \
  --data_folder_training '' \
  --data_folder_evaluation '' \
  --train_data ./data/MHD_64_train.npz \
  --eval_data ./data/MHD_64_val.npz  \
  --log_interval 20 \
  --log_dir $log_dir_name \
  --optim adam \
  --ny 32 --nx 32 --nz 32 \
  --downsamp_xyz 2 \
  --n_samp_pts_per_crop 512 \
  --lat_dims 5 \
  --unet_nf 5 \
  --unet_mf 64 \
  --imnet_nf 5 \
  --reg_loss_type l1 \
  --normalize_channels \
  --alpha_pde 0.01 \
  --pseudo_batch_size 128 \
  --clip_grad 1. \
  --lres_filter none \
  --lres_interp linear \
  --nonlin leakyrelu

