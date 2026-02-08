# utils/dataloader.py

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

class CIFAR10Pairs(Dataset):
    def __init__(self, root='./data', train=True):
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset) // 2

    def __getitem__(self, idx):
        # Pair images: i and i+1
        cover, _ = self.dataset[2 * idx]
        secret, _ = self.dataset[2 * idx + 1]
        return cover, secret

def get_dataloaders(batch_size=16):
    train_data = CIFAR10Pairs(train=True)
    test_data = CIFAR10Pairs(train=False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
