#!/bin/bash
# Quick script to install einops in the jailbreak conda environment
# Run this ONCE before submitting your SLURM jobs

module load anaconda
source ~/.bashrc
conda activate jailbreak

echo "Installing einops in jailbreak environment..."
pip install einops

echo "Verifying installation..."
python -c "import einops; print(f'✓ einops {einops.__version__} installed successfully!')"

echo "Done! You can now submit your SLURM jobs."

