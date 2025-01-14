# Import necessary libraries
import pandas as pd  # For data handling
import seaborn as sns  # For loading the dataset
import matplotlib.pyplot as plt  # For creating plots

# Print a message to confirm setup is working
print("Libraries imported successfully!")

# Load the Palmer Penguins dataset using seaborn
penguins = sns.load_dataset('penguins')

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(penguins.head())

# Display the number of rows and columns
print(f"The dataset contains {penguins.shape[0]} rows and {penguins.shape[1]} columns.")

# Display the names of the columns
print("Column names:", penguins.columns.tolist())

# Check for missing values in each column
print("Missing values in each column:")
print(penguins.isnull().sum())

# Task 2.1: Drop rows with missing values
penguins_cleaned = penguins.dropna()
print(f"Rows before cleaning: {len(penguins)}")
print(f"Rows after cleaning: {len(penguins_cleaned)}")

# Task 2.2: Check and drop duplicate rows
duplicates = penguins_cleaned.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
penguins_cleaned = penguins_cleaned.drop_duplicates()

# Task 2.3: Reset the index
penguins_cleaned.reset_index(drop=True, inplace=True)
print("First 5 rows after cleaning:")
print(penguins_cleaned.head())

# Task 2.4: Verify the cleaned dataset
print("Missing values after cleaning:")
print(penguins_cleaned.isnull().sum())
print(f"Number of duplicate rows after cleaning: {penguins_cleaned.duplicated().sum()}")