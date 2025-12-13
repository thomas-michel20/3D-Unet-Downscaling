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

save_dir_name="./model_ABL_square_res_32_eval"
mkdir -p $save_dir_name
#eval_results_16_alpha=0_0
#3d_isotropic_validation_16_0_2.npz
#'training_3d_16_alpha=0.0'

python evaluation_3d.py \
  --eval_downsamp_xyz 2 \
  --ckpt ./model_ABL_square_res_32/checkpoint_latest.pth.tar_pdenet_best.pth.tar \
  --data_folder_training ./data/ABL_Data_Training \
  --data_folder_evaluation ./data/ABL_Data_Evaluation \
  --train_data none \
  --eval_data none  \
  --save_path $save_dir_name \
  --lres_interp linear \
  --lres_filter none \
  --eval_xres 32 \
  --eval_yres 32 \
  --eval_zres 32 \
  --frame_rate 1 \
  --nbr_val_samples 20 \
  --eval_pseudo_batch_size 1000 \
  --save_video 

