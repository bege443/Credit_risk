
import os
import pandas as pd

dataset_folder = "dataset"

print("CHECKING DATASET FOLDER")
print("=======================")

if not os.path.exists(dataset_folder):
    print("Dataset folder not found!")
else:
    csv_files = [
        file for file in os.listdir(dataset_folder)
        if file.lower().endswith(".csv")
    ]

    print("CSV files found:", csv_files)

    for file in csv_files:
        path = os.path.join(dataset_folder, file)
        df = pd.read_csv(path)

        print("\nFile:", file)
        print("Dataset shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print("Missing values:", df.isnull().sum().sum())
        print("\nFirst five records:")
        print(df.head())
