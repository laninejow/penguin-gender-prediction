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

# Task 3.1: Check basic statistics
print("Summary statistics for numerical columns:")
print(penguins_cleaned.describe())

# Task 3.2: Plot distributions of numerical features
penguins_cleaned.hist(figsize=(10, 8), bins=20)
plt.suptitle("Distribution of Numerical Features")
plt.show()

# Task 3.3: Visualize feature differences by gender
sns.boxplot(x='sex', y='bill_length_mm', data=penguins_cleaned)
plt.title('Bill Length by Gender')
plt.show()

sns.boxplot(x='sex', y='body_mass_g', data=penguins_cleaned)
plt.title('Body Mass by Gender')
plt.show()

# Task 3.4: Pair plot to show relationships between features
sns.pairplot(penguins_cleaned, hue='sex', diag_kind='kde')
plt.suptitle("Pair Plot of Numerical Features by Gender", y=1.02)
plt.show()

# Task 3.5: Correlation heatmap
corr_matrix = penguins_cleaned.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()
