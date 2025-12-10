import argparse
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import open_clip
from tqdm import tqdm

# --- THE ARCHITECTURE ---
class JailbreakDetector(nn.Module):
    def __init__(self, model_name="RN50", pretrained="openai"):
        super().__init__()
        print(f">>> Loading Backbone: {model_name} (pretrained: {pretrained})")
        # Load the Pre-trained CLIP ResNet model
        self.encoder, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

        # Freeze the encoder (We only train the head)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # The Classification Head
        # We use a simple MLP. Input is 1024 (CLIP RN50 embedding size).
        # We do NOT use Sigmoid here. We output raw logits for stability.
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, images):
        # 1. Pass image through frozen CLIP encoder
        with torch.no_grad():
            features = self.encoder.encode_image(images)

        # 2. Normalize features (CLIP outputs are typically normalized)
        features = features.float()

        # 3. Pass through classifier
        logits = self.classifier(features)
        return logits

# --- DATASET ---
class PairedDataset(Dataset):
    def __init__(self, clean_dir, adv_dir, preprocess):
        # Check if directories exist
        if not os.path.exists(clean_dir):
            raise ValueError(f"Clean directory does not exist: {clean_dir}")
        if not os.path.exists(adv_dir):
            raise ValueError(f"Adversarial directory does not exist: {adv_dir}")

        # Get list of files (support common image extensions)
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        self.clean_files = []
        for ext in image_extensions:
            self.clean_files.extend(glob.glob(os.path.join(clean_dir, ext)))
            self.clean_files.extend(glob.glob(os.path.join(clean_dir, ext.upper())))
        self.clean_files = sorted(self.clean_files)

        self.adv_files = []
        for ext in image_extensions:
            self.adv_files.extend(glob.glob(os.path.join(adv_dir, ext)))
            self.adv_files.extend(glob.glob(os.path.join(adv_dir, ext.upper())))
        self.adv_files = sorted(self.adv_files)

        self.preprocess = preprocess

        # Create dataset list
        self.data = []
        # Label 0.0 = Clean
        for f in self.clean_files:
            self.data.append((f, 0.0))
        # Label 1.0 = Adversarial
        for f in self.adv_files:
            self.data.append((f, 1.0))

        print(f"Dataset Loaded: {len(self.clean_files)} Clean, {len(self.adv_files)} Adversarial")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        try:
            image = Image.open(path).convert("RGB")
            # Preprocess (Resize/Normalize) using OpenCLIP preprocessing
            image_tensor = self.preprocess(image)
            return {
                "pixel_values": image_tensor,
                "labels": torch.tensor(label, dtype=torch.float)
            }
        except Exception as e:
            # Handle corrupted images gracefully
            print(f"Error loading {path}: {e}")
            return self.__getitem__((idx + 1) % len(self.data))

# --- TRAINING LOOP ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--adv_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Device: {device}")

    # Load Preprocessing transforms for RN50
    # This downloads and sets up the preprocessing pipeline
    _, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')

    # Load Data
    full_dataset = PairedDataset(args.clean_dir, args.adv_dir, preprocess)
    
    # Check if dataset is empty
    if len(full_dataset) == 0:
        raise ValueError(
            f"Dataset is empty! Check that directories exist and contain files:\n"
            f"  Clean dir: {args.clean_dir}\n"
            f"  Adversarial dir: {args.adv_dir}\n"
            f"  Found {len(full_dataset.clean_files)} clean files and {len(full_dataset.adv_files)} adversarial files"
        )
    
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize Model
    model = JailbreakDetector().to(device)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    # BCEWithLogitsLoss includes the Sigmoid + BCELoss in one stable function
    criterion = nn.BCEWithLogitsLoss()

    print(">>> Starting Training...")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            imgs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy Calculation (Sigmoid > 0.5 is equivalent to Logits > 0)
            preds = (logits > 0).float() 
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({"Loss": loss.item()})

        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {100*correct/total:.2f}%")

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device).unsqueeze(1)
                
                logits = model(imgs)
                preds = (logits > 0).float()
                
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        
        print(f"TEST ACCURACY: {100*test_correct/test_total:.2f}%")

    # Save
    torch.save(model.state_dict(), "jailbreak_detector.pth")
    print(">>> Detector Saved.")

if __name__ == "__main__":
    main()
