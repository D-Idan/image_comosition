# SSL issue
import json
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


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, Dataset
from torchvision.datasets import ImageFolder

class Net(nn.Module):
    def __init__(self, num_classes=9):
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

def train_model(train_loader, val_loader, num_classes, model_name='default', epochs=10, learning_rate=0.001):
    """
    Train model and track performance with results saving
    """
    # Create results directory if it doesn't exist
    results_dir = './model_results'
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        print(f'Epoch {epoch + 1}: '
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
    plt.title(f'{model_name} - Loss over Epochs')
    plt.legend()

    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title(f'{model_name} - Accuracy over Epochs')
    plt.legend()

    # Per-class accuracy plot
    plt.subplot(2, 2, 3)
    plt.bar(range(num_classes), class_accuracies)
    plt.title(f'{model_name} - Per-Class Validation Accuracy')
    plt.xlabel('Remapped Class Index')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(num_classes), range(num_classes))

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_performance.png'))
    plt.close()

    # Save results to JSON
    results = {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'class_accuracies': class_accuracies,
        'final_train_accuracy': train_accuracies[-1],
        'final_val_accuracy': val_accuracies[-1]
    }

    # Save results to a JSON file
    with open(os.path.join(results_dir, f'{model_name}_results.json'), 'w') as f:
        json.dump(results, f)

    # Save model
    torch.save(net.state_dict(), os.path.join(results_dir, f'{model_name}_model.pth'))

    return results

def main():
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset from directories
    root_dir = './prepared_data'
    train_dataset = ImageFolder(root=os.path.join(root_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(root_dir, 'val'), transform=transform)
    test_dataset = ImageFolder(root=os.path.join(root_dir, 'test'), transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Train and evaluate
    classes = train_dataset.classes
    print("Training on classes:", classes)

    # Train multiple models with different hyperparameters
    models_to_train = [
        {'name': 'model_lr_0.001', 'learning_rate': 0.001},
        {'name': 'model_lr_0.0001', 'learning_rate': 0.0001},
        {'name': 'model_epochs_5', 'learning_rate': 0.001, 'epochs': 5},
        {'name': 'model_epochs_15', 'learning_rate': 0.001, 'epochs': 15}
    ]

    all_results = {}
    for model_config in models_to_train:
        print(f"\nTraining {model_config['name']}:")
        model_results = train_model(
            train_loader,
            val_loader,
            num_classes=len(classes),
            model_name=model_config['name'],
            learning_rate=model_config.get('learning_rate', 0.001),
            epochs=model_config.get('epochs', 10)
        )
        all_results[model_config['name']] = model_results

    # Save comprehensive results
    with open('./model_results/comprehensive_results.json', 'w') as f:
        json.dump(all_results, f)

if __name__ == '__main__':
    main()