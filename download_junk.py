import ssl
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10
import os

# 1. Setup
ssl._create_default_https_context = ssl._create_unverified_context
save_dir = "gatekeeper_dataset/non_crop"
os.makedirs(save_dir, exist_ok=True)

# 2. Download Data
print("Downloading CIFAR-10 data...")
(x_train, _), (_, _) = cifar10.load_data()

# 3. Save 450 Random Images (Increased count)
count = 450
print(f"Generating {count} junk images to fill the gap...")

indices = np.random.choice(len(x_train), count, replace=False)

for i, idx in enumerate(indices):
    img = Image.fromarray(x_train[idx])
    # Upscale to 224x224 so they match your other images
    img = img.resize((224, 224), Image.Resampling.NEAREST) 
    img.save(f"{save_dir}/junk_cifar_{i}.jpg")

print(f"✅ Success! Added {count} images to {save_dir}")