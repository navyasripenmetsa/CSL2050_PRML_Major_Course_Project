import os  
import numpy as np  
import pandas as pd  
from skimage.feature import hog  
from skimage.io import imread  
from skimage.color import rgb2gray  
from skimage.transform import resize  
import shutil  
from google.colab import drive  

# Mount Google Drive  
drive.mount('/content/drive', force_remount=True)  

# Set the **already extracted dataset** path (Update this if needed)
dataset_path = "/content/drive/My Drive/Fruits_Classification_Dataset/fruits-360_dataset_100x100/fruits-360/Test"  

# Set the path to save extracted HOG features as CSV files  
output_csv_path = "/content/drive/My Drive/HOG_features_CSV_files_Test"  

# Create the output directory if it does not exist  
os.makedirs(output_csv_path, exist_ok=True)  

# Get all class folders inside the dataset  
class_folders = sorted([folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))])  

# HOG feature extraction parameters  
hog_params = {  
    "orientations": 9,  
    "pixels_per_cell": (8, 8),  
    "cells_per_block": (2, 2),  
    "block_norm": "L2-Hys"  
}  

# Process images folder by folder  
for class_name in class_folders:  
    class_folder_path = os.path.join(dataset_path, class_name)  
    hog_features_list = []  

    # Process each image in the folder  
    for img_name in os.listdir(class_folder_path):  
        img_path = os.path.join(class_folder_path, img_name)  

        try:  
            img = imread(img_path)  

            if img is None:  
                print(f"⚠️ Skipping unreadable file: {img_path}")  
                continue  

            # Convert image to grayscale  
            img_gray = rgb2gray(img)  

            # Resize image to 96x96  
            img_resized = resize(img_gray, (96, 96))  

            # Extract HOG features  
            hog_features = hog(img_resized, **hog_params)  

            # Append features along with the class name  
            hog_features_list.append(np.append(hog_features, class_name))  

        except Exception as e:  
            print(f"⚠️ Error processing {img_name}: {e}")  
            continue  

    # Save extracted features as CSV if any valid images were processed  
    if hog_features_list:  
        df = pd.DataFrame(hog_features_list)  
        df.to_csv(os.path.join(output_csv_path, f"{class_name}.csv"), index=False, header=False)  
        print(f" Saved: {class_name}.csv")  
    else:  
        print(f"No valid images found for class: {class_name}")  

# Zip all CSV files for easy download  
zip_filename = "/content/drive/My Drive/HOG_features_CSV_files_Test.zip"  
shutil.make_archive(zip_filename.replace(".zip", ""), 'zip', output_csv_path)  

print("\n All HOG feature CSV files saved and zipped successfully!")  
print(f" Zipped file created: {zip_filename}")  
