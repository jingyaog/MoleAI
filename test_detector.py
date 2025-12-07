import argparse
import os
import glob
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
import numpy as np

# --- THE ARCHITECTURE ---
class JailbreakDetector(nn.Module):
    def __init__(self, base_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        # Load the Pre-trained Vision Transformer
        self.encoder = CLIPVisionModel.from_pretrained(base_model_name)
        
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # The Classification Head 
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1) 
        )

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        features = outputs.pooler_output
        logits = self.classifier(features)
        return logits


def load_model(checkpoint_path, device="cuda"):
    """Load the trained detector model."""
    print(f">>> Loading model from {checkpoint_path}")
    
    model = JailbreakDetector().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print(">>> Model loaded successfully")
    return model


def predict_image(model, processor, image_path, device="cuda", threshold=0.5):
    """Predict if a single image is adversarial."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        
        # Get prediction
        with torch.no_grad():
            logits = model(pixel_values)
            prob = torch.sigmoid(logits).item()  # Convert logits to probability
        
        # Classify (0 = clean, 1 = adversarial)
        is_adversarial = prob > threshold
        
        return {
            "image_path": image_path,
            "probability": prob,
            "is_adversarial": is_adversarial,
            "prediction": "ADVERSARIAL" if is_adversarial else "CLEAN"
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def test_directory(model, processor, directory, device="cuda", threshold=0.5):
    """Test all images in a directory."""
    # Find all image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
        image_files.extend(glob.glob(os.path.join(directory, ext.upper())))
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"No images found in {directory}")
        return []
    
    print(f"\n>>> Testing {len(image_files)} images from {directory}")
    print("-" * 80)
    
    results = []
    for image_path in image_files:
        result = predict_image(model, processor, image_path, device, threshold)
        if result:
            results.append(result)
            print(f"{result['prediction']:12s} | Prob: {result['probability']:.4f} | {os.path.basename(result['image_path'])}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test the jailbreak detector")
    parser.add_argument("--checkpoint", type=str, default="jailbreak_detector.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image to test")
    parser.add_argument("--directory", type=str, default=None,
                        help="Path to directory containing images to test")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load model
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")
    
    model = load_model(args.checkpoint, args.device)
    
    # Load processor
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Test single image or directory
    if args.image:
        print(f"\n>>> Testing single image: {args.image}")
        result = predict_image(model, processor, args.image, args.device, args.threshold)
        if result:
            print("-" * 80)
            print(f"Image: {result['image_path']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Adversarial Probability: {result['probability']:.4f}")
            print(f"Threshold: {args.threshold}")
    
    elif args.directory:
        results = test_directory(model, processor, args.directory, args.device, args.threshold)
        
        # Summary statistics
        if results:
            adversarial_count = sum(1 for r in results if r['is_adversarial'])
            clean_count = len(results) - adversarial_count
            avg_prob = np.mean([r['probability'] for r in results])
            
            print("-" * 80)
            print(f"\nSummary:")
            print(f"  Total images: {len(results)}")
            print(f"  Predicted Adversarial: {adversarial_count} ({100*adversarial_count/len(results):.1f}%)")
            print(f"  Predicted Clean: {clean_count} ({100*clean_count/len(results):.1f}%)")
            print(f"  Average Probability: {avg_prob:.4f}")
    else:
        print("Please provide either --image or --directory to test")
        print("\nExample usage:")
        print("  python test_detector.py --image adversarial_images/clean.jpeg")
        print("  python test_detector.py --directory micro_data/adv/")
        print("  python test_detector.py --directory micro_data/clean/ --threshold 0.5")


if __name__ == "__main__":
    main()

