# models/tinystegonet.py
import sys
import os

# Fix path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn

# Simple convolutional block
def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )

# Encoder: combines cover + secret into a stego image
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.input_channels = 6  # cover(3) + secret(3)

        self.encoder = nn.Sequential(
            conv_block(6, 32),
            conv_block(32, 64),
            conv_block(64, 64),
            nn.Conv2d(64, 3, kernel_size=1)  # Output: stego image (3 channels)
        )

    def forward(self, cover, secret):
        x = torch.cat([cover, secret], dim=1)  # Concatenate along channel dimension
        stego = self.encoder(x)
        return stego

# Decoder: recovers secret from stego image
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 64),
            nn.Conv2d(64, 3, kernel_size=1)  # Reconstructed secret image
        )

    def forward(self, stego):
        recovered = self.decoder(stego)
        return recovered

# Combined model
class TinyStegoNet(nn.Module):
    def __init__(self):
        super(TinyStegoNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, cover, secret):
        stego = self.encoder(cover, secret)
        recovered = self.decoder(stego)
        return stego, recovered
