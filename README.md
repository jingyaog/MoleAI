# MoleAI: Adversarial Attacks on Vision-Language Models

## Overview

MoleAI is a research project investigating adversarial attacks on Vision-Language Models (VLMs) and developing detection mechanisms for such attacks. The project implements adversarial image generation techniques targeting models like LLaVA, MiniGPT-4, and InstructBLIP, followed by training classifiers to detect these adversarial perturbations.

## Research Goals

1. **Adversarial Attack Generation**: Create adversarial images that cause VLMs to generate harmful or unintended outputs
2. **Detection Mechanisms**: Train neural network detectors to identify adversarial images
3. **Robustness Analysis**: Evaluate VLM vulnerability and detector effectiveness across different architectures

## Project Structure

```
MoleAI/
├── blip_utils/              # InstructBLIP attack utilities
│   └── visual_attacker.py   # TensorFlow-based BLIP attacker
├── llava_utils/             # LLaVA attack utilities
│   └── attacker.py          # PyTorch/TF hybrid LLaVA attacker
├── minigpt_utils/           # MiniGPT-4 attack utilities
│   ├── tf_generator.py      # Text generation wrapper
│   ├── tf_prompt_wrapper.py # Prompt formatting utilities
│   └── tf_visual_attacker.py # MiniGPT adversarial attack
├── scripts/
│   ├── attacks/             # Attack implementation scripts
│   │   ├── instructblip_inference.py
│   │   ├── instructblip_visual_attack.py
│   │   ├── llava_tf_visual_attack.py
│   │   └── minigpt_tf_visual_attack.py
│   ├── models/              # Model testing and detector training
│   │   ├── test_adversarial.py
│   │   ├── test_jailbreak.py
│   │   ├── train_detector.py      # CLIP-based detector
│   │   └── train_detector_cnn.py  # CNN-based detector
│   ├── plot/                # Visualization scripts
│   │   ├── eval_timeline.py
│   │   └── plot_training.py
│   └── preprocess/          # Data preprocessing
│       ├── preprocess.py
│       └── resize_clean_images.py
├── harmful_corpus/          # Target harmful text corpus
├── adversarial_images/      # Generated adversarial examples
├── val2017/                 # COCO validation images (clean)
├── val2017_adv/            # COCO adversarial images
├── slurm.sh                # SLURM batch job script
└── README.md
```

## Key Components

### 1. Adversarial Attack Methods

The project implements **Projected Gradient Descent (PGD)** attacks optimized for vision-language models:

- **Unconstrained Attacks**: Generate arbitrary adversarial images from random noise
- **Constrained Attacks**: Add bounded perturbations (L∞ norm) to existing images
- **Target-Driven Optimization**: Force models to generate specific harmful outputs

**Supported Models**:
- **LLaVA** (Large Language and Vision Assistant)
- **MiniGPT-4** (Multimodal conversational AI)
- **InstructBLIP** (Instruction-tuned BLIP-2)

### 2. Attack Pipeline

```python
# Core attack algorithm (simplified)
for iteration in range(num_iterations):
    # Sample harmful target outputs
    targets = random.sample(harmful_corpus, batch_size)
    
    # Compute adversarial image
    perturbed_image = base_image + perturbation
    
    # Forward pass: force model to generate targets
    loss = model(perturbed_image, targets)
    
    # Backward pass: update perturbation
    gradient = compute_gradient(loss, perturbation)
    perturbation -= alpha * sign(gradient)
    
    # Project to valid range
    perturbation = clip(perturbation, -epsilon, epsilon)
```

### 3. Detection System

**Architecture**: Transfer learning approach using frozen vision encoders

**Supported Backbones**:
- ResNet-18/34/50 (CNN-based)
- CLIP RN50 (vision-language)
- MobileNet-v2 (lightweight)
- EfficientNet-B0 (efficient)

**Training Strategy**:
1. Freeze pretrained encoder (e.g., ResNet-18 ImageNet weights)
2. Train lightweight classification head (256-dim hidden layer)
3. Binary classification: Clean vs. Adversarial

**Performance** (ResNet-18 on validation set):
- Train Accuracy: **92.9%** (10 epochs)
- Test Accuracy: **91.6%** (best: epoch 7)

### 4. Experimental Setup

**Computing Resources**:
- GPU: NVIDIA A6000 (24GB VRAM)
- Memory: 64GB RAM
- Quantization: 4-bit model loading for memory efficiency
- Batch Size: 1-8 (memory-dependent)

**Attack Parameters**:
- Iterations: 50-5000 (detection dataset vs. research quality)
- Step size (α): 1-2/255
- Perturbation bound (ε): 16-128/255
- Target corpus: ~1000 harmful instructions

## Installation

### 1. Environment Setup

```bash
# Create conda environment
conda create -n jailbreak python=3.10
conda activate jailbreak

# Install PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install TensorFlow (GPU support)
pip install tensorflow[and-cuda]

# Install core dependencies
pip install transformers accelerate bitsandbytes
pip install einops pillow tqdm matplotlib seaborn pandas
pip install open_clip_torch  # For CLIP-based detector
```

### 2. Clone External Repository

```bash
git clone https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models.git
```

### 3. Apply Required Patches

**Patch 1**: Fix quantization support in `llava_llama_2/utils.py`:

```python
# Around line 144
load_8bit = getattr(args, 'load_8bit', False)
load_4bit = getattr(args, 'load_4bit', False)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path, args.model_base, model_name, 
    load_8bit=load_8bit, load_4bit=load_4bit
)
```

**Patch 2**: Fix memory handling in `llava_llama_2/model/builder.py`:
- Add flexible model name detection (llava/llava-llama-2)
- Add CUDA OOM exception handling for vision tower loading
- Clear GPU cache before loading vision components

### 4. Download Model Checkpoints

```bash
# LLaVA-LLaMA-2-13B
# Download from HuggingFace or official sources
# Expected location: ckpts/llava-llama-2-13b-chat-lightning-preview
```

## Usage

### Running Adversarial Attacks

#### Single Image Attack

```bash
python scripts/attacks/llava_tf_visual_attack.py \
  --model-path ckpts/llava_llama_2_13b_chat_freeze \
  --n_iters 5000 \
  --eps 32 \
  --alpha 1 \
  --constrained \
  --save_dir results_pt
```

#### Batch Processing (SLURM)

```bash
# Distribute workload across team members
sbatch slurm.sh 0 1667      # Person A: images 0-1667
sbatch slurm.sh 1667 3334   # Person B: images 1667-3334
sbatch slurm.sh 3334 -1     # Person C: images 3334-end
```

### Training Adversarial Detector

#### Prepare Dataset

```bash
# Resize clean images to match adversarial preprocessing
python scripts/preprocess/resize_clean_images.py \
  --input_dir val2017 \
  --output_dir val2017_resized \
  --size 224 \
  --method clip
```

#### Train Detector

```bash
python scripts/models/train_detector_cnn.py \
  --clean_dir val2017_resized \
  --adv_dir val2017_adv \
  --backbone resnet18 \
  --epochs 10 \
  --batch_size 32
```

#### Evaluate Detector

```bash
# Test on single image
python scripts/models/test_detector.py \
  --checkpoint jailbreak_detector_resnet18.pth \
  --image adversarial_images/bad_prompt.bmp

# Test on directory
python scripts/models/test_detector.py \
  --checkpoint jailbreak_detector_resnet18.pth \
  --directory val2017_adv/ \
  --threshold 0.5
```

### Testing Jailbreak Success

```bash
# Test model response to adversarial image
python scripts/models/test_jailbreak.py \
  --model-path ckpts/llava_llama_2_13b_chat_freeze \
  --image-file results_pt/bad_prompt.bmp \
  --use-corpus  # Sample random harmful prompt
```

### Visualization

```bash
# Plot detector training metrics
python scripts/plot/plot_training.py \
  --metrics_path training_metrics_resnet18.json \
  --output_dir plots/

# Analyze attack progression
python scripts/plot/eval_timeline.py \
  --model_path jailbreak_detector_resnet18.pth \
  --image_dir experiment_results/ \
  --output_csv attack_timeline.csv
```

## Implementation Details

### Attack Algorithm

The adversarial image generator implements a constrained PGD attack:

1. **Initialization**: Random perturbation δ sampled from [-ε, ε]
2. **Optimization Loop**:
   - Compute adversarial image: x_adv = x + δ
   - Sample target harmful outputs
   - Forward pass through VLM
   - Compute cross-entropy loss on target tokens
   - Backpropagate to compute ∇δ L
   - Update: δ ← δ - α · sign(∇δ L)
   - Project: δ ← clip(δ, -ε, ε)
3. **Image Constraints**: Ensure x_adv ∈ [0, 1]

### Detector Architecture

```
Input Image (224×224×3)
    ↓
Pretrained Encoder (frozen)
    ↓ (features: 512-2048 dim)
Classification Head:
    Linear(encoder_dim → 256)
    ReLU
    Dropout(0.1)
    Linear(256 → 1)
    ↓
Adversarial Probability
```

**Loss Function**: Binary Cross-Entropy with Logits

**Training**: Adam optimizer (lr=1e-3), 5-10 epochs

## Experimental Results

### Attack Effectiveness

- **Success Rate**: >90% jailbreak success on harmful prompts
- **Convergence**: Effective attacks generated within 500-2000 iterations
- **Transferability**: Moderate cross-model transfer (40-60%)

### Detection Performance

| Backbone | Train Acc | Test Acc | Parameters |
|----------|-----------|----------|------------|
| ResNet-18 | 92.9% | 91.6% | ~12M (trainable: ~200K) |
| ResNet-50 | 94.1% | 89.3% | ~26M (trainable: ~500K) |
| CLIP RN50 | 91.2% | 88.7% | ~38M (trainable: ~260K) |

**Key Findings**:
- Lightweight heads sufficient for detection
- ResNet-18 provides best accuracy/efficiency trade-off
- Minimal overfitting with proper regularization

## Security Considerations

**Responsible Research**: This project is for academic research on AI safety and robustness.

- All harmful corpora are curated for research purposes only
- Adversarial images should not be deployed maliciously
- Results inform defensive mechanisms and model hardening
- Follow institutional ethics guidelines for AI red-teaming

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{moleai2024,
  title={MoleAI: Adversarial Attacks and Detection for Vision-Language Models},
  author={[Your Names]},
  year={2024},
  note={Research project on VLM robustness}
}
```

## Acknowledgments

- Based on methodology from [Visual Adversarial Examples Jailbreak Large Language Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models)
- LLaVA model: [Liu et al. 2023]
- Detection architectures inspired by adversarial machine learning literature
- Computing resources provided by [Brown University]

## License

This project is released for academic research purposes. See individual model licenses for pretrained weights.


---

**Last Updated**: December 2024
