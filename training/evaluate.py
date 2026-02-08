# training/evaluate.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.tinystegonet import TinyStegoNet
from skimage.metrics import structural_similarity as ssim

# --- Load model ---
model = TinyStegoNet()
model.load_state_dict(torch.load("models/saved/tinystegonet.pth", map_location='cpu'))
model.eval()

# --- Load test data ---
covers = torch.load("data/pairs/test_covers.pt")
secrets = torch.load("data/pairs/test_secrets.pt")

# Evaluate on first 10 samples (you can change this)
n = 10
covers = covers[:n]
secrets = secrets[:n]

psnr_list = []
ssim_list = []
ber_list = []

for i in range(n):
    with torch.no_grad():
        stego, recovered = model(covers[i].unsqueeze(0), secrets[i].unsqueeze(0))

    stego_np = stego.squeeze().permute(1, 2, 0).numpy()
    recovered_np = recovered.squeeze().permute(1, 2, 0).numpy()
    secret_np = secrets[i].permute(1, 2, 0).numpy()

    # Clip and scale
    recovered_np = np.clip(recovered_np, 0, 1)
    secret_np = np.clip(secret_np, 0, 1)

    # PSNR
    mse = np.mean((recovered_np - secret_np) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    psnr_list.append(psnr)

    # SSIM
    ssim_val = ssim(secret_np, recovered_np, channel_axis=-1, data_range=1.0)

    ssim_list.append(ssim_val)

    # BER
    secret_bin = (secret_np * 255).astype(np.uint8)
    recovered_bin = (recovered_np * 255).astype(np.uint8)
    ber = np.mean(secret_bin != recovered_bin)
    ber_list.append(ber)

    # Visualize (first sample only)
    if i == 0:
        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        axs[0].imshow(covers[i].permute(1, 2, 0))
        axs[0].set_title("Cover")
        axs[1].imshow(secrets[i].permute(1, 2, 0))
        axs[1].set_title("Secret")
        axs[2].imshow(recovered.squeeze().permute(1, 2, 0).numpy())
        axs[2].set_title("Recovered Secret")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/sample_output.png")
        plt.show()

# Final scores
print(f"\nAverage PSNR: {np.mean(psnr_list):.2f} dB")
print(f"Average SSIM: {np.mean(ssim_list):.4f}")
print(f"Average BER : {np.mean(ber_list):.4f}")

# Plot metrics
plt.figure()
plt.plot(psnr_list, label='PSNR')
plt.plot(ssim_list, label='SSIM')
plt.plot(ber_list, label='BER')
plt.xlabel("Sample")
plt.ylabel("Score")
plt.title("Evaluation Metrics")
plt.legend()
plt.grid()
plt.savefig("results/eval_metrics_plot.png")
plt.show()
