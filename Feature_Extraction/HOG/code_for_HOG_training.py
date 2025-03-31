from google.colab import drive
drive.mount('/content/drive')
import zipfile
import os
zip_path = "/content/drive/My Drive/Fruits_Classification_Dataset.zip"  # Path to ZIP file in Drive
extract_path = "/content/Fruits_Classification_Dataset"  # Destination path in Colab
# Create the extraction folder if it doesn't exist
os.makedirs(extract_path, exist_ok=True)
# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print(f"Dataset extracted to: {extract_path}")
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from tqdm import tqdm  # Progress bar
from google.colab import drive  # Import Google Drive module

# ✅ Mount Google Drive
drive.mount('/content/drive')

# ✅ Define save path inside Google Drive
save_path = "/content/drive/MyDrive/fruits_hog_features.csv"

# Path to Training folder
dataset_path = "Fruits_Classification_Dataset/fruits-360_dataset_100x100/fruits-360/Training"

# HOG Parameters
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
orientations = 9  # Number of gradient orientations

# List to store data
data = []

# Loop through each class folder
for fruit_class in tqdm(os.listdir(dataset_path), desc="Processing Classes"):
    class_path = os.path.join(dataset_path, fruit_class)

    if os.path.isdir(class_path):  # Ensure it's a directory
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # Read the image
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                continue  # Skip corrupted images

            # Resize to 32x32
            img_resized = cv2.resize(img, (32, 32))

            # Convert to grayscale for HOG extraction
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            # Extract HOG features
            hog_features = hog(
                gray, 
                orientations=orientations, 
                pixels_per_cell=pixels_per_cell, 
                cells_per_block=cells_per_block, 
                block_norm='L2-Hys',
                visualize=False
            )

            # Append features along with class label
            data.append([fruit_class] + list(hog_features))

# Convert data to a DataFrame
columns = ["Class"] + [f"Feature_{i}" for i in range(len(hog_features))]
df = pd.DataFrame(data, columns=columns)

# ✅ Save CSV directly in Google Drive
df.to_csv(save_path, index=False)

print(f"✅ CSV file saved in Google Drive: {save_path}")

