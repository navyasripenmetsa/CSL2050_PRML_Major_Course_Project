from google.colab import drive
drive.mount('/content/drive')
dataset_path = "/content/drive/My Drive/Fruits_Classification_Dataset/fruits-360_dataset_100x100/fruits-360/Test"
import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import hog
from skimage import color
from skimage.transform import resize

# Set dataset and output paths
dataset_path = "/content/drive/My Drive/Fruits_Classification_Dataset/fruits-360_dataset_100x100/fruits-360/Test"
output_csv_path = "/content/HOG_features.csv"  # CSV file saved in Colab first

# Function to extract HOG features
def extract_hog_features(image):
    image = resize(image, (32, 32))  # Resize to 32x32
    gray_image = color.rgb2gray(image)  # Convert to grayscale
    features = hog(gray_image, pixels_per_cell=(4, 4), cells_per_block=(2, 2), feature_vector=True)
    return features

# Process all images in subdirectories
data = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):  # Check for image formats
            img_path = os.path.join(root, file)
            image = cv2.imread(img_path)
            
            if image is not None:
                features = extract_hog_features(image)
                label = os.path.basename(root)  # Folder name as class label
                data.append([file, label] + list(features))

# Convert to DataFrame and save as CSV
columns = ["Image_Name", "Label"] + [f"Feature_{i}" for i in range(len(features))]
df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv_path, index=False)

print(f"Extracted HOG features saved at: {output_csv_path}")
from google.colab import files
files.download(output_csv_path)
