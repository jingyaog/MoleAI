import argparse
import torch
import os
import sys
import csv
import numpy as np
from PIL import Image
from torchvision.utils import save_image

# Add the other repo to path to import model utils
sys.path.append('/users/jgong42/csci1470/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models')

from llava_llama_2.utils import get_model
from llava_llama_2_utils import prompt_wrapper
from llava_utils.attacker import Attacker

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA Visual Attack")
    parser.add_argument("--model-path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--n_iters", type=int, default=2000)
    parser.add_argument('--eps', type=int, default=16)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument("--save_dir", type=str, default='results_pt',
                        help='Directory where adversarial images will be saved')
    parser.add_argument("--image_dir", type=str, default='adversarial_images',
                        help='Directory containing images to attack')
    
    parser.add_argument("--start_index", type=int, default=0, help="Index to start at")
    parser.add_argument("--end_index", type=int, default=-1, help="Index to end at (-1 for all)")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = f'cuda:{args.gpu_id}'
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    print('>>> Initializing Models')
    
    class ModelArgs:
        def __init__(self, model_path, model_base, gpu_id):
            self.model_path = model_path
            self.model_base = model_base
            self.gpu_id = gpu_id
            self.low_resource = False 
            
    model_args = ModelArgs(args.model_path, args.model_base, args.gpu_id)
    
    tokenizer, model, image_processor, model_name = get_model(model_args)
    model.eval()
    
    print('[Model Initialization Finished]')
    
    print('>>> Loading harmful corpus')
    with open("harmful_corpus/derogatory_corpus.csv", "r") as f:
        data = list(csv.reader(f, delimiter=","))
    
    targets = [row[0] for row in data]
    print(f'Loaded {len(targets)} targets')
    
    print('>>> Initializing Attacker')
    attacker = Attacker(args, model, tokenizer, targets, device=device)

    # Get all image files from the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    # IMPORTANT: Sorted ensures everyone gets the same order
    image_files = sorted([f for f in os.listdir(args.image_dir)
                   if os.path.splitext(f)[1].lower() in image_extensions])

    if not image_files:
        print(f'No images found in {args.image_dir}')
        return

    # --- SLICING ---
    total_images = len(image_files)
    end_idx = args.end_index if args.end_index != -1 else total_images
    # Ensure end_idx doesn't exceed bounds
    end_idx = min(end_idx, total_images)
    
    # Slice the list for this specific user
    my_batch = image_files[args.start_index : end_idx]
    
    print(f"Total Dataset Size: {total_images}")
    print(f"Processing range: {args.start_index} to {end_idx}")
    print(f"Images in this job: {len(my_batch)}")
    # ---------------------

    # Prepare prompt
    text_prompt_template = prompt_wrapper.prepare_text_prompt('')

    # Attack each image in the assigned slice
    for idx, img_file in enumerate(my_batch):
        # Calculate true index for logging
        true_idx = args.start_index + idx
        img_path = os.path.join(args.image_dir, img_file)
        print(f'\n>>> [{true_idx+1}/{total_images}] Processing {img_file}')

        # Check if output already exists (Resume capability)
        base_name = os.path.splitext(img_file)[0]
        save_path = f'{args.save_dir}/adv_{base_name}.bmp'
        
        if os.path.exists(save_path):
             print(f"Skipping {img_file}, already exists.")
             continue

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue

        # Preprocess image
        img_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)

        # Attack
        adv_img = attacker.attack_constrained(
            text_prompt_template,
            img=img_tensor,
            batch_size=2,
            num_iter=args.n_iters,
            alpha=args.alpha / 255,
            epsilon=args.eps / 255
        )

        # Save
        save_image(adv_img, save_path)
        print(f'Saved adversarial image to {save_path}')

    print(f'\n>>> Attack finished.')

if __name__ == '__main__':
    main()