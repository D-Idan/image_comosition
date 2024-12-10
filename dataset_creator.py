# SSL issue

import os
import requests
import torchvision
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)

os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# --

import os
import shutil
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, Dataset


def prepare_data_directory(dataset, root_dir, classes, val_split=0.2, test_split=0.1):
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # Create directories for train, validation, and test
    splits = ['train', 'val', 'test']
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(root_dir, split, cls), exist_ok=True)

    # Split dataset
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(indices, test_size=(val_split + test_split), stratify=[dataset.targets[i] for i in indices])
    val_indices, test_indices = train_test_split(temp_indices, test_size=(test_split / (val_split + test_split)), stratify=[dataset.targets[i] for i in temp_indices])

    split_indices = {'train': train_indices, 'val': val_indices, 'test': test_indices}

    for split, split_idx in split_indices.items():
        for idx in split_idx:
            img, label = dataset[idx]
            class_name = classes[label]
            save_path = os.path.join(root_dir, split, class_name, f"{idx}.png")
            torchvision.transforms.ToPILImage()(img).save(save_path)

# Example usage
original_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
omit_class = 3  # Omit 'cat'
omit_class = -1  # Not omitting any class
classes = [cls for i, cls in enumerate(original_classes) if i != omit_class]
root_dir = './prepared_data_cifar'

# Transform and Dataset
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

# Prepare data directory
prepare_data_directory(full_dataset, root_dir, classes)
