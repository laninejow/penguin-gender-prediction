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
