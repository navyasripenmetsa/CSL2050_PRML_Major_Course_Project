import os  
import numpy as np  
import pandas as pd  
from skimage.feature import hog  
from skimage.io import imread  
from skimage.color import rgb2gray  
from skimage.transform import resize  
import shutil  

# Set the correct dataset path where images are stored
dataset_path = "Fruits_Classification_Dataset/fruits-360_dataset_100x100/fruits-360/Training"  # Update this if needed  

# Set the path where extracted HOG features should be saved as CSV files  
output_csv_path = "/content/drive/My Drive/HOG_features_CSV_files"  

# Create the output directory if it does not exist  
os.makedirs(output_csv_path, exist_ok=True)  

# Get the list of all folders (each folder represents a fruit class) and sort them alphabetically  
class_folders = sorted([folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))])  

# Set the parameters for HOG feature extraction  
hog_params = {  
    "orientations": 9,  # Number of gradient orientations  
    "pixels_per_cell": (8, 8),  # Size of each cell in pixels  
    "cells_per_block": (2, 2),  # Number of cells in each block  
    "block_norm": "L2-Hys"  # Normalization method for blocks  
}  

# Loop through each class folder to process images  
for class_name in class_folders:  
    class_folder_path = os.path.join(dataset_path, class_name)  # Get full path of the class folder  
    hog_features_list = []  # Create an empty list to store extracted features  

    # Loop through each image in the class folder  
    for img_name in os.listdir(class_folder_path):  
        img_path = os.path.join(class_folder_path, img_name)  # Get full path of the image  
        img = imread(img_path)  # Read the image file  

        # Convert the image to grayscale because HOG works on single-channel images  
        img_gray = rgb2gray(img)  

        # Resize the grayscale image to 96x96 to maintain consistency  
        img_resized = resize(img_gray, (96, 96))  

        # Extract HOG features from the resized grayscale image  
        hog_features = hog(img_resized, **hog_params)  

        # Append extracted features along with the class name  
        hog_features_list.append(np.append(hog_features, class_name))  

    # Convert the extracted features list into a DataFrame  
    df = pd.DataFrame(hog_features_list)  

    # Save the DataFrame as a CSV file (one file per class)  
    df.to_csv(os.path.join(output_csv_path, f"{class_name}.csv"), index=False, header=False)  

    # Print a message to confirm that the file has been saved  
    print(f" Saved: {class_name}.csv")  

# Print a final message after all files are saved  
print("\nAll HOG feature CSV files saved successfully!")  

# Create a zip file containing all CSV files for easier download  
zip_filename = "/content/drive/My Drive/HOG_features_CSV_files.zip"  
shutil.make_archive(zip_filename.replace(".zip", ""), 'zip', output_csv_path)  

# Print a message to confirm that the zip file has been created  
print(f"\n Zipped file created: {zip_filename}")  
