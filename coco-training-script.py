# SSL issue
import json
import os
import requests
from huggingface_hub import configure_http_backend

from prepared_data_coco import path_train_original, path_train_one_percent, path_train_double_one_percent, path_val


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
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, Dataset
from torchvision.datasets import ImageFolder

def Net(num_classes=9):
    # Load the ResNet-18 model pre-trained on ImageNet
    model = models.resnet18(pretrained=True)
    # Modify the final fully connected layer for 80 classes (COCO categories)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(train_loader, val_loader, num_classes, model_name='default', epochs=2, learning_rate=0.001):
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

def paths_to_dataloader(path, transform):
    dataset = ImageFolder(root=path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    # Calculate class distribution
    class_counts = [0] * len(dataset.classes)
    for _, label in dataset:
        class_counts[label] += 1

    return dataloader, class_counts

def main():
    # Transformations
    # transform = transforms.Compose([
    #
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 for consistency
        transforms.RandomCrop((224, 224)),  # Random crop to 224x224
        transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize(  # Normalize with ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Train multiple models with different hyperparameters
    models_to_train = [
        {'name': 'model_original', 'learning_rate': 0.001, 'epochs': 2, 'path_train': path_train_original},
        {'name': 'model_5_pct', 'learning_rate': 0.001, 'epochs': 2, 'path_train': path_train_one_percent},
        {'name': 'model_10_pct', 'learning_rate': 0.001, 'epochs': 2, 'path_train': path_train_double_one_percent},
    ]

    val_loader, _ = paths_to_dataloader(path_val, transform)

    comprehensive_results = {}
    class_count_dict = {}
    all_results = {}
    for model_config in models_to_train:
        print(f"\nTraining {model_config['name']}:")
        train_loader, class_count = paths_to_dataloader(model_config['path_train'], transform)
        model_results = train_model(
            train_loader,
            val_loader,
            num_classes=len(class_count),
            model_name=model_config['name'],
            learning_rate=model_config.get('learning_rate', 0.001),
            epochs=model_config.get('epochs', 10)
        )
        model_name = model_config['name']
        class_count_dict[model_name] = class_count
        all_results[model_name] = model_results

    # Save comprehensive results
    comprehensive_results['models'] = all_results
    comprehensive_results['class_counts'] = class_count_dict
    with open('./model_results/comprehensive_results.json', 'w') as f:
        json.dump(comprehensive_results, f)


if __name__ == '__main__':
    main()