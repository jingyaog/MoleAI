import torch
from tqdm import tqdm
import random
from torchvision.utils import save_image
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# We assume sys.path is set up correctly in the main script to import these
from llava_llama_2_utils import prompt_wrapper

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

class Attacker:
    def __init__(self, args, model, tokenizer, targets, device='cuda:0', is_rtp=False):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.is_rtp = is_rtp
        self.targets = targets
        self.num_targets = len(targets)
        self.loss_buffer = []
        
        # Freeze model
        self.model.eval()
        self.model.requires_grad_(False)

    def attack_constrained(self, text_prompt, img, batch_size=8, num_iter=2000, alpha=1/255, epsilon=128/255):
        print('>>> batch_size:', batch_size)
        
        # img is expected to be [1, 3, H, W] and normalized? 
        # In original code, it seems `img` passed to this function is NOT normalized if it calls `denormalize(img)` immediately?
        # Wait, `x = denormalize(img)`. If `img` was not normalized, `denormalize` would make it huge.
        # So `img` MUST be normalized.
        
        adv_noise = torch.rand_like(img).to(self.device) * 2 * epsilon - epsilon
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()
        
        # Correct Prompt initialization
        prompt = prompt_wrapper.Prompt(self.model, self.tokenizer, text_prompts=text_prompt, device=self.device)
        
        for t in tqdm(range(num_iter + 1)):
            batch_targets = random.sample(self.targets, batch_size)
            
            x_adv = x + adv_noise
            x_adv = normalize(x_adv)
            
            # Pass x_adv to attack_loss
            target_loss = self.attack_loss(prompt, x_adv, batch_targets)
            target_loss.backward()
            
            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()
            
            # Clear GPU cache periodically to prevent memory accumulation
            if t % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.loss_buffer.append(target_loss.item())
            
            # Debug: print gradient stats
            if t % 1 == 0:
                grad_norm = adv_noise.grad.norm().item()
                print(f"Iter {t}, Loss: {target_loss.item()}, Grad Norm: {grad_norm}")
                if torch.isnan(adv_noise.grad).any():
                    print("NAN IN GRADIENT!")
                self.plot_loss()
                
            if t % 100 == 0:
                print(f'######### Output - Iter = {t} ##########')
                x_adv = x + adv_noise
                x_adv = normalize(x_adv)
                
                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)
                save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, t))
                
        return denormalize(x_adv).detach().cpu()

    def plot_loss(self):
        sns.set_theme()
        num_iters = len(self.loss_buffer)
        x_ticks = list(range(0, num_iters))
        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig('%s/loss_curve.png' % (self.args.save_dir))
        plt.clf()

    def attack_loss(self, prompts, images, targets):
        # Copied from llava_llama_2_utils/visual_attacker.py
        
        context_length = prompts.context_length
        context_input_ids = prompts.input_ids
        batch_size = len(targets)
        
        if len(context_input_ids) == 1:
            context_length = context_length * batch_size
            context_input_ids = context_input_ids * batch_size
            
        # Repeat images for batch
        images = images.repeat(batch_size, 1, 1, 1)
        
        assert len(context_input_ids) == len(targets), f"Unmatched batch size {len(context_input_ids)} != {len(targets)}"
        
        # Tokenize targets
        # Note: self.tokenizer(targets) returns a BatchEncoding. 
        # We need to handle the BOS token.
        input_ids_list = self.tokenizer(targets).input_ids
        # Debug: print first target and its tokens
        if len(self.loss_buffer) == 0: # Only print once
            print(f"DEBUG: Target 0: {targets[0]}")
            print(f"DEBUG: Tokens 0: {input_ids_list[0]}")
            print(f"DEBUG: Sliced 0: {input_ids_list[0][1:]}")
        
        to_regress_tokens = [torch.as_tensor([item[1:]]).to(self.device) for item in input_ids_list]
        
        seq_tokens_length = []
        labels = []
        input_ids = []
        
        for i, item in enumerate(to_regress_tokens):
            L = item.shape[1] + context_length[i]
            seq_tokens_length.append(L)
            
            context_mask = torch.full([1, context_length[i]], -100, dtype=to_regress_tokens[0].dtype, device=self.device)
            labels.append(torch.cat([context_mask, item], dim=1))
            input_ids.append(torch.cat([context_input_ids[i], item], dim=1))
            
        pad = torch.full([1, 1], 0, dtype=to_regress_tokens[0].dtype, device=self.device)
        
        max_length = max(seq_tokens_length)
        attention_mask = []
        
        for i in range(batch_size):
            num_to_pad = max_length - seq_tokens_length[i]
            
            padding_mask = torch.full([1, num_to_pad], -100, dtype=torch.long, device=self.device)
            labels[i] = torch.cat([labels[i], padding_mask], dim=1)
            
            input_ids[i] = torch.cat([input_ids[i], pad.repeat(1, num_to_pad)], dim=1)
            attention_mask.append(torch.LongTensor([[1] * seq_tokens_length[i] + [0] * num_to_pad]).to(self.device))
            
        labels = torch.cat(labels, dim=0).to(self.device)
        input_ids = torch.cat(input_ids, dim=0).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
            images=images.half(),
        )
        return outputs.loss
