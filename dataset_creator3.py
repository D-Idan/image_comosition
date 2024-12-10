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

############################################ Data Preparation ############################################ COCO Dataset
# wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# Unzip the annotations
# unzip annotations_trainval2017.zip
############################################################################################################

import os
import shutil
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

def prepare_data_directory_coco(data_dir, data_type, ann_file, root_dir, classes, val_split=0.2, test_split=0.1):
    # Initialize COCO API
    coco = COCO(ann_file)

    # Get category IDs for the specified classes
    cat_ids = coco.getCatIds(catNms=classes)
    img_ids = []
    for cat_id in cat_ids:
        img_ids.extend(coco.getImgIds(catIds=cat_id))
    print(f"Total images found for classes {classes}: {len(img_ids)}")

    # Shuffle and split image IDs into train, val, and test
    np.random.shuffle(img_ids)
    total_samples = len(img_ids)
    test_count = int(test_split * total_samples)
    val_count = int(val_split * total_samples)
    train_count = total_samples - val_count - test_count

    train_ids = img_ids[:train_count]
    val_ids = img_ids[train_count:train_count + val_count]
    test_ids = img_ids[train_count + val_count:]

    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}

    # Prepare directory structure
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir, exist_ok=True)

    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(root_dir, split, cls), exist_ok=True)

    # Process images and save them into respective folders
    for split, ids in tqdm(splits.items(), desc=f'Processing dataset'):
        for img_id in tqdm(ids, desc=f'Processing {split} images'):
            img_info = coco.loadImgs(img_id)[0]
            img_url = img_info['coco_url']
            img_filename = img_info['file_name']

            # Load image
            try:
                image = io.imread(img_url)
                image = Image.fromarray(image)
            except Exception as e:
                print(f"Error loading image {img_url}: {e}")
                continue

            # Load annotations for the image
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                class_id = ann['category_id']
                class_name = coco.loadCats([class_id])[0]['name']

                if class_name in classes:
                    # Save image in the respective class folder
                    class_dir = os.path.join(root_dir, split, class_name)
                    save_path = os.path.join(class_dir, img_filename)
                    image.save(save_path)
                    break  # Save only once per image for simplicity

    print(f"Dataset prepared at {root_dir}")

# Example usage
data_dir = './coco_annotations'
# data_type = 'train2017'
data_type = 'val2017'
ann_file = f'{data_dir}/annotations/instances_{data_type}.json'

original_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

original_classes = ["car", "truck", "horse", "zebra", "cow"]
omit_class = "dog"  # Omit 'dog'
omit_class = -1  # Omit no class
classes = [cls for cls in original_classes if cls != omit_class]

root_dir = './prepared_data_coco'

# Prepare data directory
prepare_data_directory_coco(data_dir, data_type, ann_file, root_dir, classes)


