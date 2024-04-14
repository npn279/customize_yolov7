import os
import json

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.yolo import Model
import argparse
import argparse

parser = argparse.ArgumentParser(description='Custom Training')
parser.add_argument('--config', type=str, default='cfg/training/yolov7.yaml', help='Path to the config file')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
parser.add_argument('--weight', type=str, default='yolov7.pt', help='Path to the weight file')
args = parser.parse_args()

config = args.config
device = args.device
weight = args.weight

# Backbone
backbone = Model(cfg=config, only_backbone=True).to(device)
state_dict = torch.load(weight, map_location=device)['model'].float().state_dict()
backbone.load_state_dict(state_dict, strict=False)

# Head Task A
class HeadA(nn.Module):
    def __init__(self, backbone):
        super(HeadA, self).__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
    
modelA = HeadA(backbone).to(device)
# freeze backbone in modelA
for param in modelA.backbone.parameters():
    param.requires_grad = False

# Head Task B
class HeadB(nn.Module):
    def __init__(self, backbone):
        super(HeadB, self).__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(1024, 70)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
    
modelB = HeadB(backbone).to(device)
# freeze backbone in modelA
for param in modelB.backbone.parameters():
    param.requires_grad = False

# Data
class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, categories, transform):
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        self.img_dir = img_dir
        self.categories = categories
        self.transform = transform
        self.imageid2classid = {x['image_id']: x['category_id'] - 1 for x in annotations['annotations'] if x['category_id'] in categories}

    def __len__(self):
        return len(self.imageid2classid)

    def __getitem__(self, idx):
        file = list(self.imageid2classid.keys())[idx]
        label = self.imageid2classid[file]
        img_path = os.path.join(self.img_dir, str(file).zfill(12) + '.jpg')
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label
        
taskA_categories = list(range(1, 11))
taskB_categories = list(range(11, 81))

# DataLoader
transform = transforms.Compose([
        transforms.Resize((256, 256)),           # Resize all images to 256x256
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB if needed
        transforms.ToTensor(),                   # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean
                                std=[0.229, 0.224, 0.225])   # and standard deviation
    ])

datasetA = CustomDataset('instances_val2017.json', 'val2017', taskA_categories, transform)
dataloaderA = DataLoader(datasetA, batch_size=32, shuffle=True)

datasetB = CustomDataset('instances_val2017.json', 'val2017', taskB_categories, transform)
dataloaderB = DataLoader(datasetB, batch_size=32, shuffle=True)

# Training Task A


import torch.optim as optim

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelA.head.parameters(), lr=0.001)  # Only optimize the head parameters

# Training loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    print('Training complete')

# Now, call the training function
train_model(modelA, dataloaderA, criterion, optimizer)
