import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the dataset path dynamically (relative to script directory)
dataset_path = os.path.join(script_dir, "Fruits_Classification_Dataset", "fruits-360_dataset_100x100", "fruits-360", "Training")

# Check if the dataset path exists
if not os.path.exists(dataset_path):
    print(f"Error: Dataset path '{dataset_path}' does not exist!")
    exit()

# Number of bins per channel
bins = 8

# Output CSV file
output_csv = os.path.join(script_dir, "colour_Histogram_Training.csv")

# List to store extracted features
data = []

# Process each fruit folder
for fruit_class in tqdm(os.listdir(dataset_path), desc="Processing Fruits"):
    fruit_folder = os.path.join(dataset_path, fruit_class)

    # Skip non-folder files
    if not os.path.isdir(fruit_folder):
        continue

    # Process each image in the folder
    for image_name in os.listdir(fruit_folder):
        image_path = os.path.join(fruit_folder, image_name)

        # Read image using OpenCV
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not read {image_path}")
            continue

        # Convert image to RGB (OpenCV loads images in BGR by default)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Compute color histogram (8 bins per channel)
        hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256]).flatten()
        hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256]).flatten()
        hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256]).flatten()

        # Normalize histograms
        hist_r /= hist_r.sum() if hist_r.sum() > 0 else 1
        hist_g /= hist_g.sum() if hist_g.sum() > 0 else 1
        hist_b /= hist_b.sum() if hist_b.sum() > 0 else 1

        # Concatenate features (R, G, B histograms)
        features = np.concatenate([hist_r, hist_g, hist_b])

        # Append to dataset
        data.append([image_name, fruit_class] + features.tolist())

# Convert to DataFrame
columns = ["filename", "class"] + [f"bin_{i}" for i in range(3 * bins)]
df = pd.DataFrame(data, columns=columns)

# Save as CSV
df.to_csv(output_csv, index=False)

print(f"Feature extraction complete! Saved to {output_csv}")
