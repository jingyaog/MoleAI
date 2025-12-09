# Setup Instructions for Teammates

## Required Files to Clone/Setup

### 1. Clone the Visual-Adversarial-Examples Repository

```bash
cd /users/cvutha/Desktop/MoleAI
git clone https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models.git
```

### 2. Apply Patches to the Cloned Repository

After cloning, you need to apply two patches to fix model loading and memory issues:

#### Patch 1: `Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/llava_llama_2/utils.py`

Around line 144, update the `load_pretrained_model` call to:
```python
# Get quantization flags if they exist
load_8bit = getattr(args, 'load_8bit', False)
load_4bit = getattr(args, 'load_4bit', False)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path, args.model_base, model_name, 
    load_8bit=load_8bit, load_4bit=load_4bit
)
```

#### Patch 2: `Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/llava_llama_2/model/builder.py`

1. **Fix LLaVA model detection** (around line 35-50):
   - Replace the `if 'llava_llama_2' in model_name.lower():` check with code that also checks for `'llava-llama-2'` or `'llava'` in the model name, or checks the config.json file for `model_type: "llava"`.

2. **Fix vision tower memory handling** (around line 145-172):
   - Add try-except for CUDA OOM when moving vision tower to GPU
   - Add GPU cache clearing before attempting to move vision tower

See the modified files in the repository for exact changes.

### 3. Install Dependencies

```bash
conda activate jailbreak
pip install einops
```

Or run:
```bash
bash install_einops.sh
```

### 4. Ensure Model Checkpoints Are Available

The model should be located at:
`/users/cvutha/scratch/MoleAI_ckpts/llava-llama-2-13b-chat-lightning-preview`

## Running the Attack

```bash
sbatch slurm.sh [START_IDX] [END_IDX]
```

Example:
```bash
sbatch slurm.sh 0 1667      # Person A
sbatch slurm.sh 1667 3334   # Person B  
sbatch slurm.sh 3334 -1     # Person C (-1 means to end)
```

## Current Configuration

- **Quantization**: 4-bit (reduces GPU memory by ~75%)
- **RAM**: 64GB
- **GPU**: A6000 (24GB VRAM)
- **Batch size**: 1 (reduced for memory efficiency)
- **GPU Memory Management**: Enabled with fragmentation handling

