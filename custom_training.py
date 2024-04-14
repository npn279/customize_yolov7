import os
import sys
sys.path.append('./')
import json

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.yolo import Model
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
import torch.nn as nn
import torch.nn.functional as F


class HeadA(nn.Module):
    def __init__(self, backbone):
        super(HeadA, self).__init__()
        self.backbone = backbone
        
        # New Convolutional Layer
        self.conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        # Adaptive pooling to flatten the output of the convolution to a fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout and Batch Normalization
        self.dropout = nn.Dropout(0.1)
        self.batch_norm = nn.BatchNorm1d(512)  # Adjust to match the output of the conv layer
        
        # Linear layer
        self.head = nn.Linear(512, 70)  # Adjust the input features to match the output of the conv layer

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        
        # Flatten the output for processing in dense layers
        x = torch.flatten(x, 1)
        
        # Regularization and normalization
        x = self.dropout(x)
        x = self.batch_norm(x)
        
        # Final linear layer
        x = self.head(x)

        return x
    
class HeadB(nn.Module):
    def __init__(self, backbone):
        super(HeadB, self).__init__()
        self.backbone = backbone
        
        # New Convolutional Layer
        self.conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        # Adaptive pooling to flatten the output of the convolution to a fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dropout and Batch Normalization
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm2d(512)  # Adjust to match the output of the conv layer
        
        # Linear layer
        self.head = nn.Linear(512, 70)  # Adjust the input features to match the output of the conv layer

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        
        # Flatten the output for processing in dense layers
        x = torch.flatten(x, 1)
        
        # Regularization and normalization
        x = self.dropout(x)
        x = self.batch_norm(x)
        
        # Final linear layer
        x = self.head(x)

        return x

    
# # Head Task B
# class HeadB(nn.Module):
#     def __init__(self, backbone):
#         super(HeadB, self).__init__()
#         self.backbone = backbone
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.head = nn.Linear(1024, 70)

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.pool(x)
#         x = torch.flatten(x, 1)
#         x = self.head(x)
#         return x
    
# Data
class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, categories, transform):
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        self.img_dir = img_dir
        self.categories = categories
        self.transform = transform
        self.img2idx = {x['image_id']: x['category_id'] for x in annotations['annotations'] if x['category_id'] in categories}
        self.idx2cls = {cid: i for i, cid in enumerate(sorted(list(set(self.img2idx.values()))))}       
        self.img2cls = {img: self.idx2cls[cls] for img, cls in self.img2idx.items()}

    def __len__(self):
        return len(self.img2cls)

    def __getitem__(self, idx):
        file = list(self.img2cls.keys())[idx]
        label = self.img2cls[file]
        img_path = os.path.join(self.img_dir, str(file).zfill(12) + '.jpg')
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label

def inference(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total}%')

def train_model(model, train_loader, val_loader, num_epochs=1):
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=0.001)
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
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

            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                with torch.no_grad():
                    inference(model, val_loader)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    print('Training complete')

# DataLoader
taskA_categories = list(range(1, 11))
taskB_categories = list(range(11, 91))
transform = transforms.Compose([
        transforms.Resize((256, 256)),           # Resize all images to 256x256
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB if needed
        transforms.ToTensor(),                   # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean
                                std=[0.229, 0.224, 0.225])   # and standard deviation
    ])

# datasetA = CustomDataset('instances_val2017.json', 'val2017', taskA_categories, transform)
# train_size = int(0.9 * len(datasetA))
# val_size = len(datasetA) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(datasetA, [train_size, val_size])
# train_loader_A = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader_A = DataLoader(val_dataset, batch_size=32, shuffle=False)

# modelA = HeadA(backbone).to(device)
# for param in modelA.backbone.parameters():
#     param.requires_grad = False

# print('Training Task A')
# train_model(modelA, train_loader_A, val_loader_A, num_epochs=1)
# torch.save(modelA.state_dict(), 'taskA.pth')


datasetB = CustomDataset('instances_val2017.json', 'val2017', taskB_categories, transform)
train_size_B = int(0.9 * len(datasetB))
val_size_B = len(datasetB) - train_size_B
train_dataset_B, val_dataset_B = torch.utils.data.random_split(datasetB, [train_size_B, val_size_B])
train_loader_B = DataLoader(train_dataset_B, batch_size=32, shuffle=True)
val_loader_B = DataLoader(val_dataset_B, batch_size=32, shuffle=False)

modelB = HeadB(backbone).to(device)
for param in modelB.backbone.parameters():
    param.requires_grad = False

print('Training Task B')
train_model(modelB, train_loader_B, val_loader_B, num_epochs=5)
torch.save(modelB.state_dict(), 'taskB.pth')


