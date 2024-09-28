import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Check if the model is using GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")  

positive_path = r'C:\Users\Administrator\Desktop\face_recognition2_finished\data\positive'
negative_path = r'C:\Users\Administrator\Desktop\face_recognition2_finished\data\negative'
anchor_path = r'C:\Users\Administrator\Desktop\face_recognition2_finished\data\anchor'

# Data Augmentation Function
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(100),
    transforms.ToTensor()
])

# Custom Dataset
class SiameseDataset(Dataset):
    def __init__(self, anchor_path, positive_path, negative_path, transform=None):
        self.anchor_images   = [os.path.join(anchor_path, img) for img in os.listdir(anchor_path)]
        self.positive_images = [os.path.join(positive_path, img) for img in os.listdir(positive_path)]
        self.negative_images = [os.path.join(negative_path, img) for img in os.listdir(negative_path)]
        self.transform = transform
        
        # Ensure all lists have the same length
        self.min_length = min(len(self.anchor_images), len(self.positive_images), len(self.negative_images))

    def __len__(self):
        return self.min_length

    def __getitem__(self, idx):
        anchor_image   = Image.open(self.anchor_images[idx % len(self.anchor_images)]).convert('RGB')
        positive_image = Image.open(self.positive_images[idx % len(self.positive_images)]).convert('RGB')
        negative_image = Image.open(self.negative_images[idx % len(self.negative_images)]).convert('RGB')

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to draw a red box around detected faces
def detect_and_draw_face(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)  

    return frame, faces

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=10)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4)
        self.fc1 = nn.Linear(512 * 5 * 5, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.dropout = nn.Dropout(0.4)  # Dropout for Regularization
        self.batch_norm = nn.BatchNorm2d(512)  # Batch Normalization for stability
        
    def forward_one(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv4(x))
        x = self.batch_norm(x)  # Batch Normalization
        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # Apply Dropout
        x = nn.Sigmoid()(self.fc1(x))
        return x

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_one(anchor)
        positive_output = self.forward_one(positive)
        negative_output = self.forward_one(negative)
        return anchor_output, positive_output, negative_output

# Function to verify by computing L1 distance between two outputs
def verify(anchor_output, validation_output):
    l1_distance = torch.abs(anchor_output - validation_output)
    return l1_distance

# Real-time verification using OpenCV
def verify_real_time(model, threshold=0.5):
    input_img_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
    input_img = Image.open(input_img_path).convert('RGB')
    input_img = transform(input_img).unsqueeze(0).to(device)
    
    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        validation_img_path = os.path.join('application_data', 'verification_images', image)
        validation_img = Image.open(validation_img_path).convert('RGB')
        validation_img = transform(validation_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            input_output = model.forward_one(input_img)
            validation_output = model.forward_one(validation_img)
            
            # Call the newly added verify function here
            l1_distance = verify(input_output, validation_output)
            distance = torch.norm(l1_distance).item()
            results.append(distance < threshold)

    verified = np.mean(results) > threshold
    return results, verified

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_output, positive_output, negative_output):
        pos_dist = torch.norm(anchor_output - positive_output, dim=1)
        neg_dist = torch.norm(anchor_output - negative_output, dim=1)
        loss = torch.mean(torch.clamp(self.margin + pos_dist - neg_dist, min=0))
        return loss

# Dataset and DataLoader
siamese_dataset = SiameseDataset(anchor_path, positive_path, negative_path, transform=transform)
train_loader = DataLoader(siamese_dataset, batch_size=64, shuffle=True)  # Batch size set to 64

# Model Initialization
model = SiameseNetwork().to(device)

# Save and load the model
torch.save(model.state_dict(), 'siamese_model.pth')
model.load_state_dict(torch.load(r'C:\Users\Administrator\Desktop\face_recognition2_finished\siamese_model.pth'))
model.eval()

# Model, Loss, and Optimizer
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Weight decay added

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Training loop with early stopping
def train_model(model, train_loader, criterion, optimizer, epochs = 50):
    model.train()
    early_stopping = EarlyStopping(patience = 7)
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            print(f"Data is on device: {anchor.device}")  # Check if data is on GPU or CPU
            
            # Forward pass
            anchor_output, positive_output, negative_output = model(anchor, positive, negative)
            loss = criterion(anchor_output, positive_output, negative_output)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Check for early stopping based on validation loss
        early_stopping(avg_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
    print("Training complete!")

# Train the model with 50 epochs and early stopping
train_model(model, train_loader, criterion, optimizer, epochs = 50)

# Verification and face detection loop with red box
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        # Detect and draw face(s)
        frame_with_box, faces = detect_and_draw_face(frame)
        
        # Display the frame with a red box around detected faces
        cv2.imshow('Verification', frame_with_box)

        # Check if no faces were detected
        if len(faces) == 0:
            print("No face detected")
            continue
        
        # Save the image and run verification if 'v' is pressed
        if cv2.waitKey(10) & 0xFF == ord('v') and len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_region = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join('application_data', 'input_image', 'input_image.jpg'), face_region)
            
            # Run verification
            results, verified = verify_real_time(model, 0.9)
            print(f"Verification result: {verified}")
        
        # Quit the loop if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
