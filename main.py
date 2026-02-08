# main.py

from utils.dataloader import get_dataloaders
import matplotlib.pyplot as plt
import torch

train_loader, _ = get_dataloaders()

# Visualize one cover-secret pair
for cover, secret in train_loader:
    print("Cover shape:", cover.shape)
    print("Secret shape:", secret.shape)
    plt.figure(figsize=(4, 2))
    plt.subplot(1, 2, 1)
    plt.imshow(cover[0].permute(1, 2, 0))
    plt.title("Cover")
    plt.subplot(1, 2, 2)
    plt.imshow(secret[0].permute(1, 2, 0))
    plt.title("Secret")
    plt.show()
    break
import torch
from utils.dataloader import CIFAR10Pairs

dataset = CIFAR10Pairs(train=True)

covers = []
secrets = []

for i in range(len(dataset)):
    cover, secret = dataset[i]
    covers.append(cover)
    secrets.append(secret)

covers_tensor = torch.stack(covers)
secrets_tensor = torch.stack(secrets)

# Save to disk
torch.save(covers_tensor, 'data/pairs/train_covers.pt')
torch.save(secrets_tensor, 'data/pairs/train_secrets.pt')
