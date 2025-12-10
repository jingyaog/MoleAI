import argparse
import os
import glob
import torch
import re
import pandas as pd
from PIL import Image
from torchvision import transforms
# Import the architecture from your training script
from train_detector_cnn import JailbreakDetectorCNN

def get_iteration(filename):
    """Extracts the iteration number from 'bad_prompt_temp_1200.bmp'"""
    match = re.search(r'temp_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="jailbreak_detector_resnet18.pth", help="Path to saved model")
    parser.add_argument("--image_dir", default="experiment_results", help="Folder containing attack history")
    parser.add_argument("--output_csv", default="attack_timeline.csv", help="Where to save results")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "mobilenet_v2", "efficientnet_b0"],
                        help="CNN backbone architecture (must match the trained model)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Device: {device}")

    # 1. Load Model
    print(">>> Loading Detector...")
    # Use the same backbone you used in training!
    model = JailbreakDetectorCNN(backbone=args.backbone).to(device)

    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print(">>> Weights loaded successfully.")
    else:
        print(f"ERROR: Model file {args.model_path} not found! Train it first.")
        return

    # 2. Load Preprocessing Transforms
    # ImageNet normalization (standard for pretrained models)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Find Images
    # We look for the temp files generated during the attack
    image_paths = glob.glob(os.path.join(args.image_dir, "bad_prompt_temp_*.bmp"))
    
    # Also add the final one if it has a different name
    if os.path.exists(os.path.join(args.image_dir, "bad_prompt.bmp")):
        image_paths.append(os.path.join(args.image_dir, "bad_prompt.bmp"))
        
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return

    # Sort by iteration
    image_paths.sort(key=lambda x: get_iteration(os.path.basename(x)))

    print(f">>> Found {len(image_paths)} images. Starting evaluation...")

    results = []

    # 4. Evaluation Loop
    with torch.no_grad():
        for path in image_paths:
            try:
                # Load Image
                image = Image.open(path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)

                # Inference
                logits = model(image_tensor)
                prob = torch.sigmoid(logits).item() # Probability of being Adversarial
                
                # Metadata
                filename = os.path.basename(path)
                iteration = get_iteration(filename)
                
                # Special case for the final image if it doesn't have a number
                if "bad_prompt.bmp" in filename:
                    iteration = 5000 # Or whatever your max was
                
                print(f"Iter: {iteration:<5} | Prob: {prob:.4f} | {filename}")
                
                results.append({
                    "iteration": iteration,
                    "probability": prob,
                    "filename": filename
                })
            except Exception as e:
                print(f"Error processing {path}: {e}")

    # 5. Save Data
    df = pd.DataFrame(results)
    df = df.sort_values("iteration")
    df.to_csv(args.output_csv, index=False)
    print(f"\n>>> Results saved to {args.output_csv}")
    print(">>> You can now plot this data to show how quickly the detector spots the attack.")

if __name__ == "__main__":
    main()
