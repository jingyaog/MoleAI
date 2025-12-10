import argparse
import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

# --- THE ARCHITECTURE ---
class JailbreakDetectorCNN(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        print(f">>> Loading Backbone: {backbone} (pretrained: {pretrained})")

        # Load the Pre-trained CNN backbone
        weights = "DEFAULT" if pretrained else None
        if backbone == "resnet18":
            self.encoder = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone == "resnet50":
            self.encoder = models.resnet50(weights=weights)
            feature_dim = 2048
        elif backbone == "resnet34":
            self.encoder = models.resnet34(weights=weights)
            feature_dim = 512
        elif backbone == "mobilenet_v2":
            self.encoder = models.mobilenet_v2(weights=weights)
            feature_dim = 1280
        elif backbone == "efficientnet_b0":
            self.encoder = models.efficientnet_b0(weights=weights)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the final classification layer
        if backbone.startswith("resnet"):
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        elif backbone == "mobilenet_v2":
            self.encoder.classifier = nn.Identity()
        elif backbone.startswith("efficientnet"):
            self.encoder.classifier = nn.Identity()

        # Freeze the encoder (We only train the head)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # The Classification Head
        # We use a simple MLP. Input depends on the backbone.
        # We do NOT use Sigmoid here. We output raw logits for stability.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, images):
        # 1. Pass image through frozen CNN encoder
        with torch.no_grad():
            features = self.encoder(images)

        # 2. Ensure features are flattened
        if len(features.shape) == 4:
            features = features.squeeze(-1).squeeze(-1)

        # 3. Pass through classifier
        logits = self.classifier(features)
        return logits

# --- DATASET ---
class PairedDataset(Dataset):
    def __init__(self, clean_dir, adv_dir, transform):
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

        self.transform = transform

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
            # Apply transforms (Resize/Normalize) using torchvision
            image_tensor = self.transform(image)
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
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "mobilenet_v2", "efficientnet_b0"],
                        help="CNN backbone architecture")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Device: {device}")

    # ImageNet normalization (standard for pretrained models)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Data
    full_dataset = PairedDataset(args.clean_dir, args.adv_dir, transform)

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
    model = JailbreakDetectorCNN(backbone=args.backbone).to(device)

    # Optimizer & Loss
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    # BCEWithLogitsLoss includes the Sigmoid + BCELoss in one stable function
    criterion = nn.BCEWithLogitsLoss()

    print(">>> Starting Training...")

    # Track metrics for plotting
    metrics_history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": []
    }

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

        avg_train_loss = train_loss/len(train_loader)
        train_accuracy = 100*correct/total
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")

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

        test_accuracy = 100*test_correct/test_total
        print(f"TEST ACCURACY: {test_accuracy:.2f}%")

        # Save metrics
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["train_acc"].append(train_accuracy)
        metrics_history["test_acc"].append(test_accuracy)

    # Save model
    save_path = f"jailbreak_detector_{args.backbone}.pth"
    torch.save(model.state_dict(), save_path)
    print(f">>> Detector Saved to {save_path}")

    # Save metrics
    metrics_path = f"training_metrics_{args.backbone}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    print(f">>> Training metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
