# training/train.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from models.tinystegonet import TinyStegoNet
import matplotlib.pyplot as plt

# Load dataset
train_covers = torch.load("data/pairs/train_covers.pt")
train_secrets = torch.load("data/pairs/train_secrets.pt")

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare model
model = TinyStegoNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Data loader
train_dataset = torch.utils.data.TensorDataset(train_covers, train_secrets)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
loss_history = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for cover, secret in train_loader:
        cover, secret = cover.to(DEVICE), secret.to(DEVICE)

        optimizer.zero_grad()
        stego, recovered = model(cover, secret)

        loss_cover = loss_fn(stego, cover)
        loss_secret = loss_fn(recovered, secret)
        loss = loss_cover + loss_secret
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

# Save model
os.makedirs("models/saved", exist_ok=True)
torch.save(model.state_dict(), "models/saved/tinystegonet.pth")

# Save loss plot
os.makedirs("results", exist_ok=True)
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("results/loss_plot.png")
plt.show()
