# SSL issue
import json
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
import argparse
import requests
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset, DataLoader
from sklearn.model_selection import train_test_split
from huggingface_hub import configure_http_backend

# SSL Workaround
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)

os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def prepare_data_directory(dataset, root_dir, classes, val_split=0.2, test_split=0.1):
    """
    Save dataset into train/val/test directories with class-wise subdirectories.
    """
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
    targets = [dataset.targets[i] for i in indices]  # CIFAR-10-style targets
    train_indices, temp_indices = train_test_split(indices, test_size=(val_split + test_split), stratify=targets)
    val_indices, test_indices = train_test_split(temp_indices, test_size=(test_split / (val_split + test_split)), stratify=[targets[i] for i in temp_indices])

    split_indices = {'train': train_indices, 'val': val_indices, 'test': test_indices}

    for split, split_idx in split_indices.items():
        for idx in split_idx:
            img, label = dataset[idx]
            class_name = classes[label]
            save_path = os.path.join(root_dir, split, class_name, f"{idx}.png")
            torchvision.transforms.ToPILImage()(img).save(save_path)

def prepare_coco_data_directory(dataset, root_dir, classes, num_images=100, val_split=0.2, test_split=0.1):
    """
    Save COCO dataset into train/val/test directories with class-wise subdirectories.
    """
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # Create directories for train, validation, and test
    splits = ['train', 'val', 'test']
    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(root_dir, split, str(cls)), exist_ok=True)

    # Collect and limit dataset size
    indices = list(range(len(dataset)))
    indices = indices[:num_images]  # Limit to num_images if specified
    train_indices, temp_indices = train_test_split(indices, test_size=(val_split + test_split))
    val_indices, test_indices = train_test_split(temp_indices, test_size=(test_split / (val_split + test_split)))

    split_indices = {'train': train_indices, 'val': val_indices, 'test': test_indices}

    for split, split_idx in split_indices.items():
        for idx in split_idx:
            img, targets = dataset[idx]
            for target in targets:  # Handle multiple labels per image
                label = target['category_id']
                class_name = classes[label]
                save_path = os.path.join(root_dir, split, class_name, f"{idx}.png")
                torchvision.transforms.ToPILImage()(img).save(save_path)


def load_dataset(dataset_flag, root_dir, transform, num_images=100):
    """
    Load CIFAR-10 or mini COCO based on dataset_flag.
    """
    if dataset_flag == 'cifar':
        original_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        omit_class = -1  # Adjust as needed
        classes = [cls for i, cls in enumerate(original_classes) if i != omit_class]

        # Load full dataset
        full_dataset = torchvision.datasets.CIFAR10(
            root='/mnt/data/datasets',
            train=True, download=True,
            transform=transform)

        prepare_data_directory(full_dataset, root_dir, classes)
    elif dataset_flag == 'coco':
        # Map class indices to category names if available in the annotations
        with open('/mnt/data/datasets/mini_coco/annotations/instances_train2017.json') as f:
            coco_annotations = json.load(f)
            categories = coco_annotations['categories']
            classes = {cat['id']: cat['name'] for cat in categories}

        full_dataset = torchvision.datasets.CocoDetection(
            root='/mnt/data/datasets/mini_coco/images',
            annFile='/mnt/data/datasets/mini_coco/annotations/instances_train2017.json',
            transform=transform
        )
        prepare_coco_data_directory(full_dataset, root_dir, classes, num_images=num_images)
    else:
        raise ValueError("Unsupported dataset flag. Use 'cifar' or 'coco'.")
    return classes

def main():
    parser = argparse.ArgumentParser(description='Dataset Preparation')
    parser.add_argument('--dataset', type=str, choices=['cifar', 'coco'], default='coco', help='Dataset to use: cifar or coco')
    parser.add_argument('--num-images', type=int, default=100, help='Number of images to use for COCO dataset')
    args = parser.parse_args()

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Prepare dataset
    root_dir = './prepared_data_coco' if args.dataset == 'coco' else './prepared_data_cifar'
    classes = load_dataset(args.dataset, root_dir, transform)

    # Load dataset from directories
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(root_dir, 'train'), transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(root_dir, 'val'), transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(root_dir, 'test'), transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print("Classes:", train_dataset.classes)
    # Continue with training...

if __name__ == '__main__':
    main()
