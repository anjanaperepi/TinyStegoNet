# scripts/save_pairs.py
import sys
import os

# Fix path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import torch
from utils.dataloader import CIFAR10Pairs


def save_dataset_pairs(train=True):
    dataset = CIFAR10Pairs(train=train)

    covers = []
    secrets = []

    for i in range(len(dataset)):
        cover, secret = dataset[i]
        covers.append(cover)
        secrets.append(secret)

    covers_tensor = torch.stack(covers)
    secrets_tensor = torch.stack(secrets)

    tag = 'train' if train else 'test'
    os.makedirs('data/pairs', exist_ok=True)
    torch.save(covers_tensor, f'data/pairs/{tag}_covers.pt')
    torch.save(secrets_tensor, f'data/pairs/{tag}_secrets.pt')

    print(f"Saved {len(dataset)} {tag} pairs.")

if __name__ == "__main__":
    save_dataset_pairs(train=True)
    save_dataset_pairs(train=False)
