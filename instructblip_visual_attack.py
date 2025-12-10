import argparse
import os
import random
import numpy as np
import tensorflow as tf
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image

from lavis.models import load_model_and_preprocess
from blip_utils import visual_attacker


def parse_args():

    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")

    args = parser.parse_args()
    return args

def configure_tf_gpu(gpu_id):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        except RuntimeError:
            pass

def load_image_as_torch(path, vis_processor,device):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3)
    imp_np = img.numpy()
    pil_img = Image.fromarray(imp_np)
    torch_img= vis_processor["eval"](pil_img).unsqueeze(0).to(device)
    return torch_img

print(">>> Initializing Models")

args = parse_args()
configure_tf_gpu(args.gpu_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processor, _ = load_model_and_preprocess(
    name="blip2_vicuna_instruct",
    model_type="vicuna13b",
    is_eval=True,
    device=device,
)
model.eval()

print("[Initialization Finished]\n")

os.makedirs(args.save_dir, exist_ok=True)

targets = []
with open("harmful_corpus/derogatory_corpus.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        targets.append(row[0])

my_attacker = visual_attacker.Attacker(args,model,targets,device=model.device,is_rtp=False)

template_path = "adversarial_images/clean.jpeg"
img = load_image_as_torch(template_path, vis_processor, device)

if not args.constrained:
    adv_img = my_attacker.attack_unconstrained(
        img=img,
        batch_size=8,
        num_iter=args.n_iters,
        alpha=args.alpha/255.0
    )
else:
    adv_img = my_attacker.attack_constrained(
        img=img,
        batch_size=8,
        num_iter=args.n_iters,
        alpha=args.alpha/255.0,
        epsilon=args.eps/255.0
    )
save_path = f"{args.save_dir}/bad_prompt.bmp"
save_image(adv_img, save_path)

print(f"[Done] Saved adversarial image to {save_path}")