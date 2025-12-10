"""
TensorFlow implementation of LLaVA visual adversarial attack.

This script replicates the PyTorch LLaVA attack using TensorFlow/Keras,
leveraging HuggingFace transformers and a PyTorch-TensorFlow bridge.
"""

import argparse
import tensorflow as tf
import torch
import os
import sys
import csv
import numpy as np
from PIL import Image

# Add PyTorch implementation to path for model loading
sys.path.append('/users/jgong42/csci1470/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models')

from llava_tf.model_loader import get_model
from llava_tf_utils.tf_attacker import TFAttacker
from llava_llama_2_utils import prompt_wrapper


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLaVA Visual Attack - TensorFlow Implementation")
    parser.add_argument("--model-path", type=str, 
                       default="ckpts/llava_llama_2_13b_chat_freeze",
                       help="Path to LLaVA model checkpoint")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, 
                       help="GPU ID to use")
    parser.add_argument("--n_iters", type=int, default=5000, 
                       help="Number of attack iterations")
    parser.add_argument('--eps', type=int, default=32, 
                       help="Epsilon for attack budget (0-255)")
    parser.add_argument('--alpha', type=int, default=1, 
                       help="Step size for attack (0-255)")
    parser.add_argument("--constrained", default=False, action='store_true',
                       help="Use constrained attack (L-inf bounded)")
    parser.add_argument("--save_dir", type=str, default='results_tf',
                       help="Directory to save results")
    
    args = parser.parse_args()
    return args


def load_image(image_path):
    """Load and return PIL image."""
    image = Image.open(image_path).convert('RGB')
    return image


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    print('=' * 60)
    print('LLaVA Visual Attack - TensorFlow Implementation')
    print('=' * 60)
    
    # Check TensorFlow GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f'TensorFlow GPUs available: {len(gpus)}')
    if gpus:
        print(f'Using GPU: {gpus[0]}')
        # Set memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f'Memory growth setting error: {e}')
    
    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f'Created save directory: {args.save_dir}')
    
    # Initialize model
    print('\n>>> Initializing Models')
    print(f'Model path: {args.model_path}')
    
    tokenizer, model, image_processor, model_name = get_model(args.model_path, args.gpu_id)
    print('[Model Initialization Finished]\n')
    
    # Load harmful corpus
    print('>>> Loading harmful corpus')
    file = open("harmful_corpus/derogatory_corpus.csv", "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    
    targets = []
    num = len(data)
    for i in range(num):
        targets.append(data[i][0])
    
    print(f'Loaded {len(targets)} harmful targets')
    print(f'Sample targets: {targets[:3]}')
    
    # Initialize attacker
    print('\n>>> Initializing Attacker')
    my_attacker = TFAttacker(
        args, model, tokenizer, targets, 
        device=f'cuda:{args.gpu_id}',
        image_processor=image_processor
    )
    
    # Load template image
    template_img = 'adversarial_images/clean.jpeg'
    image = load_image(template_img)
    image = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    image_np = image.numpy()  # Convert to numpy for TensorFlow
    
    print(f'Image shape: {image_np.shape}')
    
    # Prepare text prompt
    text_prompt_template = prompt_wrapper.prepare_text_prompt('')
    print(f'Text prompt template: {text_prompt_template[:100]}...')
    
    # Run attack
    print('\n>>> Starting Attack')
    if not args.constrained:
        print('[Mode: Unconstrained]')
        print('Note: Unconstrained attack not yet implemented in TF version')
        print('Please use --constrained flag')
        return
    else:
        print('[Mode: Constrained L-infinity]')
        print(f'Epsilon: {args.eps}/255 = {args.eps/255:.4f}')
        print(f'Alpha: {args.alpha}/255 = {args.alpha/255:.4f}')
        print(f'Iterations: {args.n_iters}')
        
        adv_img_prompt = my_attacker.attack_constrained(
            text_prompt_template,
            img=image_np,
            batch_size=2,
            num_iter=args.n_iters,
            alpha=args.alpha / 255,
            epsilon=args.eps / 255
        )
    
    # Save final adversarial image
    adv_img = np.clip(adv_img_prompt.squeeze(0).transpose(1, 2, 0), 0, 1)
    adv_img_pil = Image.fromarray((adv_img * 255).astype(np.uint8))
    adv_img_pil.save(f'{args.save_dir}/bad_prompt.bmp')
    
    print(f'\n[Attack Complete]')
    print(f'Adversarial image saved to: {args.save_dir}/bad_prompt.bmp')
    print(f'Loss curve saved to: {args.save_dir}/loss_curve.png')


if __name__ == '__main__':
    main()
