# SSL issue

import os
import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)

os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"

import ssl
ssl._create_default_https_context = ssl._create_unverified_context




import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def split_dataset(dataset, val_split=0.2, omit_class=None):
    """
    Split dataset into train and validation sets, optionally omitting a class
    """
    # Filter out the omit_class if specified
    if omit_class is not None:
        indices = [i for i, (_, label) in enumerate(dataset) if label != omit_class]
        dataset = Subset(dataset, indices)
    
    # Split indices
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))), 
        test_size=val_split, 
        stratify=[dataset[i][1] for i in range(len(dataset))]
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset

def train_model(train_loader, val_loader, num_classes, epochs=10, learning_rate=0.001):
    """
    Train model and track performance
    """
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, loss, and optimizer
    net = Net(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
    # Tracking metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Record metrics
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(100 * train_correct / train_total)
        val_accuracies.append(100 * val_correct / val_total)
        
        print(f'Epoch {epoch+1}: '
              f'Train Loss: {train_losses[-1]:.4f}, '
              f'Train Acc: {train_accuracies[-1]:.2f}%, '
              f'Val Loss: {val_losses[-1]:.4f}, '
              f'Val Acc: {val_accuracies[-1]:.2f}%')
    
    # Per-class accuracy
    class_correct = list(0. for _ in range(num_classes))
    class_total = list(0. for _ in range(num_classes))
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    class_accuracies = [100 * class_correct[i] / class_total[i] for i in range(num_classes)]
    
    # Plotting
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    
    # Per-class accuracy plot
    plt.subplot(2, 2, 3)
    plt.bar(range(num_classes), class_accuracies)
    plt.title('Per-Class Validation Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(num_classes), range(num_classes))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'model': net,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'class_accuracies': class_accuracies
    }

def main():
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load full dataset
    full_dataset = torchvision.datasets.CIFAR10(
        root='/mnt/data/datasets', 
        # root='./data', 
        train=True, download=True, 
        transform=transform)
    
    # Choose class to omit (e.g., 'cat' which is index 3)
    omit_class = 3  # cat
    
    # Split dataset
    train_dataset, val_dataset = split_dataset(full_dataset, val_split=0.2, omit_class=omit_class)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Train and evaluate
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    results = train_model(train_loader, val_loader, num_classes=len(classes)-1)

if __name__ == '__main__':
    main()
