import os
import sys
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms
from PIL import Image

# Add root path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.tinystegonet import TinyStegoNet

app = Flask(__name__)

# Full path to static folder
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
app.config['UPLOAD_FOLDER'] = STATIC_DIR

# Load model
model = TinyStegoNet()
try:
    model.load_state_dict(torch.load("models/saved/tinystegonet.pth", map_location='cpu'))
    model.eval()
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print("[ERROR] Failed to load model:", e)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def tensor_to_image(tensor):
    tensor = tensor.squeeze().detach().permute(1, 2, 0).numpy()
    tensor = (tensor * 255).clip(0, 255).astype('uint8')
    return Image.fromarray(tensor)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            cover_file = request.files['cover']
            secret_file = request.files['secret']
            print("[INFO] Files received:", cover_file.filename, secret_file.filename)

            cover_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cover.png')
            secret_path = os.path.join(app.config['UPLOAD_FOLDER'], 'secret.png')
            cover_file.save(cover_path)
            secret_file.save(secret_path)

            cover_img = Image.open(cover_path).convert('RGB')
            cover_img = transform(cover_img)
            cover_img = cover_img.unsqueeze(0)

            secret_img = Image.open(secret_path).convert('RGB')
            secret_img = transform(secret_img)
            secret_img = secret_img.unsqueeze(0)


            with torch.no_grad():
                stego, recovered = model(cover_img, secret_img)

            stego_img = tensor_to_image(stego)
            recovered_img = tensor_to_image(recovered)

            stego_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'stego.png'))
            recovered_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'recovered.png'))

            print("[INFO] Images processed and saved.")
            return render_template('index.html',
                                   cover='static/cover.png',
                                   secret='static/secret.png',
                                   stego='static/stego.png',
                                   recovered='static/recovered.png')

        except Exception as e:
            print("[ERROR]", e)
            return f"<h3>Upload Failed:</h3><pre>{e}</pre>"

    # If GET request — clear previous images
    try:
        for f in ['cover.png', 'secret.png', 'stego.png', 'recovered.png']:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[INFO] Removed {file_path}")
    except Exception as cleanup_error:
        print(f"[WARN] Cleanup error: {cleanup_error}")

    return render_template('index.html')

@app.route('/test-save')
def test_save():
    from PIL import Image
    img = Image.new('RGB', (64, 64), color='green')
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'test.png')
    img.save(path)
    return f"<h3>Image saved to: {path}</h3><img src='/static/test.png'>"

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from PIL import Image

@app.route('/metrics')
def metrics():
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        from PIL import Image
        import numpy as np

        def load_and_resize(path, size=(64, 64)):
            img = Image.open(path).convert('RGB').resize(size)
            return np.array(img)

        base_path = app.config['UPLOAD_FOLDER']
        cover = load_and_resize(os.path.join(base_path, 'cover.png'))
        stego = load_and_resize(os.path.join(base_path, 'stego.png'))
        secret = load_and_resize(os.path.join(base_path, 'secret.png'))
        recovered = load_and_resize(os.path.join(base_path, 'recovered.png'))

        psnr = peak_signal_noise_ratio(cover, stego, data_range=255)
        ssim = structural_similarity(cover, stego, channel_axis=-1, data_range=255)
        ber = np.sum(secret != recovered) / secret.size

        import csv
        from datetime import datetime

        # CSV logging block
        csv_file = os.path.join(os.path.dirname(__file__), 'metrics.csv')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if os.path.getsize(csv_file) == 0:
                writer.writerow(['Timestamp', 'PSNR', 'SSIM', 'BER'])
            writer.writerow([timestamp, psnr, ssim, ber])


        return render_template('metrics.html', psnr=psnr, ssim=ssim, ber=ber)

    except Exception as e:
        return f"<h3>Error loading or processing images:</h3><pre>{e}</pre>"




if __name__ == '__main__':
    app.run(debug=True)
