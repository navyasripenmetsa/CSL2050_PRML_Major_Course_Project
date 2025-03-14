import os
import pandas as pd
import re  

meta_folder = "Fruits_Classification_Dataset/fruits-360_dataset_original-size/fruits-360-original-size/Meta"

data = {}
attributes = set()

for fruit in os.listdir(meta_folder):
    fruit_folder = os.path.join(meta_folder, fruit)
    info_file = os.path.join(fruit_folder, "info.txt")

    if os.path.exists(info_file):
        fruit_data = {"Fruit": fruit}  # Store data for this fruit

        with open(info_file, "r") as file:
            for line in file:
                line = line.strip()
                
                if "=" in line:
                    key, value = line.split("=")
                    
                    match = re.match(r"(.+?)\[(\d+)\]", key.strip())
                    if match:
                        attribute_name, _ = match.groups()
                        value = int(value.strip())

                        fruit_data[attribute_name] = value  # Store attribute value
                        attributes.add(attribute_name)  # Track all attributes

        data[fruit] = fruit_data  # Store the structured data

# Ensure all fruits have the same attributes (fill missing values with None)
attributes = sorted(attributes)  # Sort attributes for consistency
columns = ["Fruit"] + attributes

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(data, orient="index", columns=columns)

# Save to CSV file
csv_path = "formatted_metadata.csv"
df.to_csv(csv_path, index=False)

print(f"Formatted metadata saved to {csv_path}")

