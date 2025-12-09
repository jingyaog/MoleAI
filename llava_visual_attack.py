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
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = f'cuda:{args.gpu_id}'
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    print('>>> Initializing Models')
    
    # Create a dummy args object for get_model
    class ModelArgs:
        def __init__(self, model_path, model_base, gpu_id):
            self.model_path = model_path
            self.model_base = model_base
            self.gpu_id = gpu_id
            self.low_resource = False # Assuming this is needed
            
    model_args = ModelArgs(args.model_path, args.model_base, args.gpu_id)
    
    # get_model returns: tokenizer, model, image_processor, model_name
    tokenizer, model, image_processor, model_name = get_model(model_args)
    # print(f"DEBUG: Image Processor Mean: {image_processor.image_mean}")
    # print(f"DEBUG: Image Processor Std: {image_processor.image_std}")
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
    image_files = [f for f in os.listdir(args.image_dir)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print(f'No images found in {args.image_dir}')
        return

    print(f'Found {len(image_files)} images to attack')

    # Prepare prompt
    text_prompt_template = prompt_wrapper.prepare_text_prompt('')

    # Attack each image
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(args.image_dir, img_file)
        print(f'\n>>> [{idx+1}/{len(image_files)}] Processing {img_file}')

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Preprocess image
        img_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].to(device)

        print(f'Image tensor shape: {img_tensor.shape}')

        print('>>> Starting Attack')

        adv_img = attacker.attack_constrained(
            text_prompt_template,
            img=img_tensor,
            batch_size=2,
            num_iter=args.n_iters,
            alpha=args.alpha / 255,
            epsilon=args.eps / 255
        )

        # Save adversarial image with original filename
        base_name = os.path.splitext(img_file)[0]
        save_path = f'{args.save_dir}/adv_{base_name}.bmp'
        save_image(adv_img, save_path)
        print(f'Saved adversarial image to {save_path}')

    print(f'\n>>> Attack finished. Processed {len(image_files)} images.')

if __name__ == '__main__':
    main()
