#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --constraint=a6000|a5000  # Prefer high VRAM cards
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G           # LLaVA needs significant RAM to load (matching teammate's config)
#SBATCH --time=24:00:00     # 24 hours (Safety margin for ~1600 images)
#SBATCH --job-name="LLaVA-Attack"
#SBATCH --output=logs/attack-%x.%j.out
#SBATCH --error=logs/attack-%x.%j.err

# --- CONFIGURATION FOR TEAM MEMBERS ---
# Person A: 0 to 1667
# Person B: 1667 to 3334
# Person C: 3334 to -1 (-1 means end of list)
# Accept command-line arguments: sbatch slurm.sh [START_IDX] [END_IDX]
# If not provided, uses defaults below
START_IDX=${1:-0}
END_IDX=${2:-1667}
# --------------------------------------

echo "=========================================="
echo "Job started on $(hostname) at $(date)"
echo "Processing images from index $START_IDX to $END_IDX"
echo "=========================================="

# 1. Load Environment
# (Using the module and conda env we set up previously)
module load anaconda
source ~/.bashrc
conda activate jailbreak

# 1.5. Install missing dependencies if needed
echo "Checking for missing dependencies..."
python -c "import einops" 2>/dev/null || pip install -q einops

# 2. Critical Exports for LLaVA/Bitsandbytes
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# Redirect cache to scratch to avoid quota issues
export TORCH_HOME="/users/cvutha/scratch/cache/torch"
export HF_HOME="/users/cvutha/scratch/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/users/cvutha/scratch/cache/huggingface"
# GPU memory management - help with fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 3. Navigate to Project Directory
cd /oscar/home/cvutha/Desktop/MoleAI

# Ensure output directory exists
mkdir -p ./val2017_adv

# 4. Run the Attack
# use --n_iters 50 for speed (detection dataset generation)
# use --image_dir to point to your COCO val2017 folder
# use --load_8bit to reduce GPU memory usage (recommended for 24GB GPUs)
python -u llava_visual_attack.py \
  --image_dir /users/cvutha/Desktop/MoleAI/val2017 \
  --save_dir ./val2017_adv \
  --model-path /users/cvutha/scratch/MoleAI_ckpts/llava-llama-2-13b-chat-lightning-preview \
  --n_iters 50 \
  --alpha 2 \
  --eps 16 \
  --load_4bit \
  --start_index $START_IDX \
  --end_index $END_IDX

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="