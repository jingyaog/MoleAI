#!/bin/bash
#SBATCH --job-name=llava_attack
#SBATCH --output=llava_attack_%j.out
#SBATCH --error=llava_attack_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# #SBATCH --constraint=a6000|a5000

# # Load required modules
# module load anaconda

# # Initialize conda for bash shell
# eval "$(conda shell.bash hook)"

# # Activate conda environment
# conda activate llava_attack

# Navigate to working directory
cd /users/jgong42/csci1470/MoleAI

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Starting at: $(date)"
echo "Working directory: $(pwd)"
echo "GPU assigned: $CUDA_VISIBLE_DEVICES"

# Run the visual attack on the first 1667 images
python llava_visual_attack.py \
    --model-path ckpts/llava_llama_2_13b_chat_freeze \
    --n_iters 1000 \
    --eps 16 \
    --alpha 1 \
    --save_dir bear_adv \
    --image_dir bear

echo "Finished at: $(date)"
