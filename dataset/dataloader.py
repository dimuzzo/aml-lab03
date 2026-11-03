import os
import shutil
import requests
from io import BytesIO
from zipfile import ZipFile
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

DATASET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
DATASET_DIR = 'data/tiny-imagenet-200'


def download_and_extract(dataset_url=DATASET_URL, extract_to='data'):
    """Download and prepare the Tiny-ImageNet dataset if not already present."""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(extract_to, exist_ok=True)
        print("Downloading Tiny-ImageNet dataset...")
        response = requests.get(dataset_url)
        if response.status_code == 200:
            with ZipFile(BytesIO(response.content)) as zip_file:
                zip_file.extractall(extract_to)
            print("Download and extraction complete.")
        else:
            raise RuntimeError(f"Failed to download dataset. Status code: {response.status_code}")

    # Reorganize validation directory structure for PyTorch ImageFolder
    val_dir = os.path.join(DATASET_DIR, 'val')
    images_dir = os.path.join(val_dir, 'images')
    annotations_file = os.path.join(val_dir, 'val_annotations.txt')

    if os.path.exists(images_dir) and os.path.exists(annotations_file):
        print("Restructuring validation directory...")
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                fn, cls = parts[0], parts[1]
                cls_dir = os.path.join(val_dir, cls)
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(images_dir, fn)
                dst = os.path.join(cls_dir, fn)
                if os.path.exists(src):
                    shutil.copyfile(src, dst)
        shutil.rmtree(images_dir)
        print("Validation directory restructured successfully.")


def get_transforms():
    """Returns preprocessing transforms for inputs."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_dataloaders(batch_size=32, num_workers=2):
    """Initializes dataset loaders for train and validation splits."""
    download_and_extract()
    transform = get_transforms()

    train_dataset = ImageFolder(root=os.path.join(DATASET_DIR, 'train'), transform=transform)
    val_dataset = ImageFolder(root=os.path.join(DATASET_DIR, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def denormalize(image):
    """Reverts normalization transforms for image plotting."""
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.clip(image * std + mean, 0, 1)


if __name__ == '__main__':
    # Visual sanity check
    train_loader, _ = get_dataloaders(batch_size=10, num_workers=0)
    images, labels = next(iter(train_loader))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for idx, ax in enumerate(axes.flatten()):
        ax.imshow(denormalize(images[idx]))
        ax.set_title(f"Label: {labels[idx].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()