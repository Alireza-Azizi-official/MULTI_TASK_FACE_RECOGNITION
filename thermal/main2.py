from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import cv2 
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from layers import L1Dist

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class L1Dist(nn.Module):
    def __init__(self):
        super(L1Dist, self).__init__()
    
    def forward(self, x1, x2):
        return torch.abs(x1 - x2)
    
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size = 10)
        self.conv2 = nn.Conv2d(128, 256, kernel_size = 7)
        self.conv3 = nn.Conv2d(256, 512, kernel_size = 4)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size = 4)
        
        self.fc1  = nn.Linear(1024 * 2 * 2, 8192)
        self.dropout = nn.Dropout(0.4)
        
        
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(512)
        self.batch_norm4 = nn.BatchNorm2d(1024)
        
        self.fc2 = nn.Linear(8192, 1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward_one(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return x 
    
    def format (self, anchor, positive, negative):
        anchor_output = self.forward_one(anchor)
        positive_output = self.forward_one(positive)
        negative_output = self.forward_one(negative)
        return anchor_output, positive_output, negative_output
    
    def predict(self, x1, x2):
        l1_dist = L1Dist()
        distance = l1_dist(x1, x2)
        x = self.fc2(distance)
        return self.sigmoid(x)
    
    
def train_step(anchor, positive, negative, model, criterion, optimizer):
    optimizer.zero_grad()
    
    anchor_output = model.forward_one(anchor)
    positive_output = model.forward_one(positive)
    negative_output = model.forward_one(negative)
    
    positive_distance = L1Dist()(anchor_output, positive_output)
    negative_distnace = L1Dist()(anchor_output, negative_output)
    
    positive_labels = torch.ones(positive_distance.size(), device = device)
    negative_labels = torch.zeros(negative_distnace.size(), device = device)
    
    positive_loss = criterion(positive_distance, positive_labels)
    negative_loss = criterion(negative_distnace, negative_labels)
    
    loss = positive_loss + negative_loss
    loss.backward()
    optimizer.step()
    return loss.item()
    
    
def train(model, train_loader, num_epochs=50, lr=0.0001, patience=5):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)  

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for batch in train_loader:
            anchor, positive, negative = batch

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            loss = train_step(anchor, positive, negative, model, criterion, optimizer)
            running_loss += loss

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'siamese_model_thermal.pth') 
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break

    print("Training completed!")

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

class SiameseDataset(Dataset):
    def __init__(self, anchor_dir, positive_dir, negative_dir, transform = None):
        self.anchor_images = [os.path.join(anchor_dir, f) for f in os.listdir(anchor_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.positive_images = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        negative_images = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.negative_images =negative_images
        
        self.transform = transform
        
    def __len__(self):
        return min(len(self.anchor_images), len(self.positive_images), len(self.negative_images))
        
    def __getitem__(self, idx):
        anchor = cv2.imread(self.anchor_images[idx])
        positive = cv2.imread(self.positive_images[idx])
        negative = cv2.imread(self.negative_images[idx])

        anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
        positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)
        negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


anchor_dir = r'C:\Users\Administrator\Desktop\face_recognition2_finished\thermal\data\anchor'
positive_dir = r'C:\Users\Administrator\Desktop\face_recognition2_finished\thermal\data\positive'
negative_dir =  r'C:\Users\Administrator\Desktop\face_recognition2_finished\thermal\data\negative'

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

train_dataset = SiameseDataset(anchor_dir, positive_dir, negative_dir, transform = transform)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)

model = SiameseNetwork().to(device)
model.apply(weights_init)

train(model, train_loader, num_epochs = 50 )