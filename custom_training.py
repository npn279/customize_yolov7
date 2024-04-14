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
train_size = int(0.8 * len(datasetA))
val_size = len(datasetA) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(datasetA, [train_size, val_size])
train_loader_A = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader_A = DataLoader(val_dataset, batch_size=32, shuffle=False)

datasetB = CustomDataset('instances_val2017.json', 'val2017', taskB_categories, transform)
train_size = int(0.8 * len(datasetB))
val_size = len(datasetB) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(datasetB, [train_size, val_size])
train_loader_B = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader_B = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Training loop
def train_model(model, dataloader, num_epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(modelA.head.parameters(), lr=0.001)
    model.train()  # Set model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(dataloader):
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
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
    
    print('Training complete')

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

print('Training Task A')
train_model(modelA, train_loader_A)
torch.save(modelA.state_dict(), 'taskA.pth')
print('Inference Task A')
inference(modelA, val_loader_A)

print('Training Task B')
train_model(modelB, train_loader_B)
torch.save(modelB.state_dict(), 'taskB.pth')
print('Inference Task B')
inference(modelB, val_loader_B)


