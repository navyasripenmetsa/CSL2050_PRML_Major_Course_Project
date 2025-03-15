import cv2
import numpy as np
import os
from skimage.feature import hog
import pandas as pd

# Path to your training dataset
train_folder = r"Fruits_Classification_Dataset/fruits-360_dataset_original-size/fruits-360-original-size/Training"

# Get class names (folder names)
class_labels = os.listdir(train_folder)  # Assuming they're already in the desired order

data = []
labels = []

for class_name in class_labels:
    class_path = os.path.join(train_folder, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        
        # Read image and convert to grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))  # Resize for consistency
        
        # Compute HOG features
        hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9)

        
        data.append(hog_features)
        labels.append(class_name)  # Keep class names

# Convert to DataFrame
df = pd.DataFrame(data)
df['Class'] = labels  # Add class column

# Save as CSV with the new name
csv_path = r"C:/Users/navya/Downloads/HOG_training.csv"
df.to_csv(csv_path, index=False)

print(f"HOG features extracted and saved at: {csv_path}")
