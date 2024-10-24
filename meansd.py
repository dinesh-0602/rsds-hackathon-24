import os
import numpy as np
from PIL import Image

# Paths to your dataset directories
data_dir = "datasets/fire_scars_train_val/train"

# Initialize lists for each band
sum_means = None
sum_stds = None
n_images = 0  # Counter for number of images

# List the bands you're working with (assuming 6 bands)
bands = ['BLUE', 'GREEN', 'RED', 'NIR_NARROW', 'SWIR_1', 'SWIR_2']

# Iterate over the images in your dataset
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith("_merged.tif"):  # Change the pattern based on your image filenames
            img_path = os.path.join(root, file)
            
            # Load image (assuming TIFF format, change if needed)
            img = Image.open(img_path)
            img = np.array(img)

            # If the image has 6 bands (channels)
            if img.shape[-1] == len(bands):
                n_images += 1
                img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

                # Compute mean and std per band and accumulate
                means = np.mean(img, axis=(0, 1))
                stds = np.std(img, axis=(0, 1))
                
                if sum_means is None:
                    sum_means = means
                    sum_stds = stds
                else:
                    sum_means += means
                    sum_stds += stds

# Finalize means and stds by dividing by the number of images
if n_images > 0:
    means = sum_means / n_images
    stds = sum_stds / n_images
else:
    means = stds = np.zeros(len(bands))

# Print results
print(f"Means per band: {means}")
print(f"Stds per band: {stds}")
