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

# 2

# Drop rows with missing values
penguins_cleaned = penguins.dropna()
print(f"Rows before cleaning: {len(penguins)}")
print(f"Rows after cleaning: {len(penguins_cleaned)}")

# Check and drop duplicate rows
duplicates = penguins_cleaned.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
penguins_cleaned = penguins_cleaned.drop_duplicates()

# Reset the index
penguins_cleaned.reset_index(drop=True, inplace=True)
print("First 5 rows after cleaning:")
print(penguins_cleaned.head())

# Verify the cleaned dataset
print("Missing values after cleaning:")
print(penguins_cleaned.isnull().sum())
print(f"Number of duplicate rows after cleaning: {penguins_cleaned.duplicated().sum()}")

# 3

# Check basic statistics
print("Summary statistics for numerical columns:")
print(penguins_cleaned.describe())

# Plot distributions of numerical features
penguins_cleaned.hist(figsize=(10, 8), bins=20)
plt.suptitle("Distribution of Numerical Features")
plt.show()

# Visualize feature differences by gender
sns.boxplot(x='sex', y='bill_length_mm', data=penguins_cleaned)
plt.title('Bill Length by Gender')
plt.show()

sns.boxplot(x='sex', y='body_mass_g', data=penguins_cleaned)
plt.title('Body Mass by Gender')
plt.show()

# Pair plot to show relationships between features
sns.pairplot(penguins_cleaned, hue='sex', diag_kind='kde')
plt.suptitle("Pair Plot of Numerical Features by Gender", y=1.02)
plt.show()

# Correlation heatmap
corr_matrix = penguins_cleaned.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# 4

# List all columns
print("All columns:", penguins_cleaned.columns.tolist())

# Identify numerical columns (excluding categorical columns like 'species' and 'island')
numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("Selected numerical features:", numerical_features)

# Identify numerical features
numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("Selected numerical features:", numerical_features)

# Check correlations among numerical features
corr_matrix = penguins_cleaned[numerical_features].corr()
print("Correlation matrix:")
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap of Selected Features")
plt.show()

# Choose final features
final_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
print("Final selected features:", final_features)

# Prepare the data for modeling
X = penguins_cleaned[final_features]
y = penguins_cleaned['sex']
print("Features (X):")
print(X.head())
print("Target (y):")
print(y.head())

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)} rows")
print(f"Testing set size: {len(X_test)} rows")

# 5

# Choose and initialize the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)
print("Model training complete!")

# Make predictions
y_pred = model.predict(X_test)
print("Predictions on the test set:", y_pred[:10])

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Save the model
import joblib
joblib.dump(model, 'penguin_gender_model.pkl')
print("Model saved as 'penguin_gender_model.pkl'")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")
print("Classification Report for Decision Tree:")
print(classification_report(y_test, y_pred_dt))

# Random Forest
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

# KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn:.2f}")
print("Classification Report for KNN:")
print(classification_report(y_test, y_pred_knn))

# Compare model performance
model_performance = {
    "Logistic Regression": accuracy,
    "Decision Tree": accuracy_dt,
    "Random Forest": accuracy_rf,
    "KNN": accuracy_knn
}

print("Model Performance Comparison:")
for model, acc in model_performance.items():
    print(f"{model}: {acc:.2f}")
