import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir, batch_size=32, shuffle=True):
    """
    Creates PyTorch DataLoader using ImageFolder structure

    Args:
        data_dir (str): Path to image dataset folder
        batch_size (int): Batch size
        shuffle (bool): Shuffle data or not

    Returns:
        dataloader, class_names
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ Dataset path not found: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,          # 🔑 Windows safe
        pin_memory=torch.cuda.is_available()
    )

    return dataloader, dataset.classes
