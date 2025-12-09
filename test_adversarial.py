import argparse
import torch
import sys
import os
from PIL import Image

# Add the other repo to path to import model utils
sys.path.append('/users/jgong42/csci1470/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models')

from llava_llama_2.utils import get_model
from llava_llama_2_utils import prompt_wrapper, generator

def parse_args():
    parser = argparse.ArgumentParser(description="Test Adversarial Image")
    parser.add_argument("--model-path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--image-file", type=str, default='results_pt/final_bad_prompt.bmp')
    parser.add_argument("--prompt", type=str, default="Describe the image.")
    args = parser.parse_args()
    return args

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def main():
    args = parse_args()
    
    print('>>> Initializing Models')
    tokenizer, model, image_processor, model_name = get_model(args)
    model.eval()
    print('[Initialization Finished]\n')
    
    print(f'>>> Loading Image: {args.image_file}')
    image = load_image(args.image_file)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
    
    my_generator = generator.Generator(model=model, tokenizer=tokenizer)
    
    print(f'>>> Prompt: {args.prompt}')
    text_prompt_template = prompt_wrapper.prepare_text_prompt(args.prompt)
    prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device)
    
    print('>>> Generating Response...')
    with torch.no_grad():
        response = my_generator.generate(prompt, image_tensor)
    
    print('>>> Response:')
    print(response)

if __name__ == '__main__':
    main()
