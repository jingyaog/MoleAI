import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
import numpy as np
import random


transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),transforms.Normalize(mean = [0.485,0.456,0.406],std= [0.229,0.224,0.255])])
dataset = load_dataset("coco", split = "validation")
dataset = dataset.select(range(5000))

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image)


image_path = dataset[0]['image']
image_tensor = preprocess_image(image_path)
print(image_tensor.shape)

class CocoDataset(torch.utils.data.Dataset):
    def _init_(self,dataset,transform = None):
        self.dataset = dataset
        self.transform = transform

    def _len_(self):
        return len(self.dataset)

    def _getitem_(self,idx):
        image_path = self.dataset[idx]['image']
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
    
coco_dataset = CocoDataset(dataset,transform = transform)
batch_size = 16
dataloader = DataLoader(coco_dataset,batch_size = batch_size, shuffle = True)

model_name = "google/vit-base-patch16-224-in21k"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
loss_fn  = nn.CrossEntropyLoss()
num_epochs = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, images in enumerate(dataloader):
        images = images.to(device)
        labels = torch.tensor([random.randint(0,80)]* batch_size).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs.logits

        loss = loss_fn(logits,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}]/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss/ len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")


os.makedirs("models",exist_ok = True)
model.save_pretrained("model/vit_finetuned")

model.eval()
sample_image = coco_dataset[0]
sample_image = sample_image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(sample_image)
    logits = output.logits
    predicted_class = torch.argmax(logits, dim =1).item()
    print(f"Predicated class: {predicted_class}")

        
