#!/usr/bin/env python3
"""
Resize clean images to 224x224 to match adversarial image preprocessing.
This ensures fair comparison during detector training.
"""

import argparse
import os
from PIL import Image
from tqdm import tqdm
import open_clip

def resize_with_clip_preprocessing(image, target_size=224):
    """
    Apply the same resize/crop strategy as CLIP preprocessing.
    This matches what happens during adversarial image generation.

    Args:
        image: PIL Image
        target_size: Target size (default 224 for CLIP)

    Returns:
        Resized PIL Image
    """
    # Get CLIP's preprocessing pipeline
    _, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')

    # Apply preprocessing and convert back to PIL
    import torch
    tensor = preprocess(image)

    # Convert tensor back to PIL Image
    # Denormalize: CLIP uses mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # Denormalize
    tensor = tensor * std + mean

    # Clamp to [0, 1] and convert to [0, 255]
    tensor = torch.clamp(tensor, 0, 1)
    tensor = (tensor * 255).byte()

    # Convert to PIL
    import torchvision.transforms as transforms
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)


def resize_simple(image, target_size=224):
    """
    Simple resize without normalization (alternative method).
    Uses bicubic interpolation to match torchvision default.
    """
    return image.resize((target_size, target_size), Image.BICUBIC)


def main():
    parser = argparse.ArgumentParser(description='Resize clean images to match adversarial preprocessing')
    parser.add_argument('--input_dir', type=str, default='val2017',
                        help='Directory containing original images')
    parser.add_argument('--output_dir', type=str, default='val2017_resized',
                        help='Directory to save resized images')
    parser.add_argument('--size', type=int, default=224,
                        help='Target size (default: 224 for CLIP)')
    parser.add_argument('--format', type=str, default='bmp', choices=['bmp', 'png', 'jpg'],
                        help='Output format (default: bmp to match adversarial images)')
    parser.add_argument('--method', type=str, default='clip', choices=['clip', 'simple'],
                        help='Resize method: "clip" uses CLIP preprocessing, "simple" uses basic resize')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = sorted([f for f in os.listdir(args.input_dir)
                         if os.path.splitext(f)[1].lower() in image_extensions])

    if not image_files:
        print(f'No images found in {args.input_dir}')
        return

    print(f'Found {len(image_files)} images in {args.input_dir}')
    print(f'Resizing to {args.size}x{args.size} using {args.method} method')
    print(f'Saving to {args.output_dir} as .{args.format}')

    # Process each image
    for img_file in tqdm(image_files, desc='Resizing images'):
        input_path = os.path.join(args.input_dir, img_file)

        # Change extension to match output format
        base_name = os.path.splitext(img_file)[0]
        output_file = f'{base_name}.{args.format}'
        output_path = os.path.join(args.output_dir, output_file)

        # Skip if already exists
        if os.path.exists(output_path):
            continue

        try:
            # Load image
            image = Image.open(input_path).convert('RGB')

            # Resize using selected method
            if args.method == 'clip':
                resized = resize_with_clip_preprocessing(image, args.size)
            else:
                resized = resize_simple(image, args.size)

            # Save
            resized.save(output_path)

        except Exception as e:
            print(f'\nError processing {img_file}: {e}')
            continue

    print(f'\nDone! Resized images saved to {args.output_dir}')

    # Print statistics
    output_files = [f for f in os.listdir(args.output_dir)
                   if os.path.splitext(f)[1].lower() == f'.{args.format}']
    print(f'Total output images: {len(output_files)}')


if __name__ == '__main__':
    main()
