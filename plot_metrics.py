import pandas as pd
import matplotlib.pyplot as plt

# Read metrics log
df = pd.read_csv("metrics.csv", parse_dates=['Timestamp'])

# Plot 1: PSNR and SSIM over time
plt.figure(figsize=(10, 5))
plt.plot(df['Timestamp'], df['PSNR'], label='PSNR (dB)', marker='o', color='blue')
plt.plot(df['Timestamp'], df['SSIM'], label='SSIM', marker='s', color='green')
plt.xticks(rotation=45)
plt.title("PSNR and SSIM Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("psnr_ssim_trend.png")
plt.close()

# Plot 2: BER over time
plt.figure(figsize=(10, 5))
plt.bar(df['Timestamp'], df['BER'], color='tomato')
plt.xticks(rotation=45)
plt.title("Bit Error Rate (BER) Over Time")
plt.xlabel("Timestamp")
plt.ylabel("BER")
plt.tight_layout()
plt.savefig("ber_trend.png")
plt.close()

print("✅ Plots saved as 'psnr_ssim_trend.png' and 'ber_trend.png'")
