# Import necessary libraries
import pandas as pd  # Used for data manipulation and analysis, including reading CSV files and handling structured data.
import numpy as np  # Provides support for numerical computations and handling large multidimensional arrays.
import matplotlib.pyplot as plt  # Used for data visualization, creating graphs and charts to explore data distributions.
import seaborn as sns  # Built on top of Matplotlib, provides enhanced visualization capabilities for statistical data.
from sklearn.model_selection import train_test_split  # Splits the dataset into training and testing sets for model evaluation.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # Encodes categorical variables into numerical format for ML models.
from sklearn.ensemble import RandomForestRegressor  # Implements the Random Forest algorithm for car price prediction.
from sklearn.metrics import mean_squared_error, r2_score  # Evaluates model performance using MSE and R² metrics.

# Load Dataset
file_path = r"D:\Ifayani\DataSet Car.csv"
df = pd.read_csv(file_path)

# Drop irrelevant column (Unnamed index)
df.drop(columns=["Unnamed: 0"], inplace=True)

# Step 1: Exploratory Data Analysis (EDA)


#  Display basic info about dataset
df.info()
print("\nSummary statistics:\n", df.describe())

# Check missing values
missing_values = df.isnull().sum()
print("\nMissing values in dataset:\n", missing_values[missing_values > 0])

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
print("\nCategorical Columns:", categorical_columns)

# Display sample values before encoding
print("\nSample Values from Categorical Columns:")
print(df[categorical_columns].head())

# Convert categorical columns to numerical before correlation analysis
df_encoded = df.copy()

# Apply Label Encoding to categorical features for correlation heatmap
for col in categorical_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Correlation Heatmap (After Encoding)
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Price Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df["Price"], bins=50, kde=True, color="blue")
plt.title("Car Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Boxplot of Price
plt.figure(figsize=(8, 6))
sns.boxplot(y=df["Price"])
plt.title("Box Plot of Car Prices")
plt.show()

# Step 2: Preprocessing - Encoding Categorical Features

# Apply Label Encoding for all categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Force conversion to string before encoding
    label_encoders[col] = le

# Verify all categorical columns are now numeric
print("\nData Types After Encoding:")
print(df.dtypes)

# Display updated dataset sample
print("\nUpdated Dataset Sample:")
print(df.head())

# Train-Test Split
X = df.drop(columns=["Price"])  # Features
y = df["Price"]  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (R²): {r2:.4f}")

# Visualizations

# Feature Importance Plot
feature_importance = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(np.array(X_train.columns)[sorted_idx][:10], feature_importance[sorted_idx][:10], color="blue")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Important Features in Car Price Prediction")
plt.gca().invert_yaxis()
plt.show()

# Actual vs Predicted Prices (Scatter Plot)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5, label="Predicted Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal Prediction")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices (Random Forest)")
plt.legend()
plt.show()

# Price Distribution Before & After Prediction
plt.figure(figsize=(8, 6))
plt.hist(y_test, bins=30, alpha=0.7, color="blue", label="Actual Prices")
plt.hist(y_pred, bins=30, alpha=0.7, color="red", label="Predicted Prices")
plt.xlabel("Car Price")
plt.ylabel("Frequency")
plt.title("Car Price Distribution: Actual vs Predicted")
plt.legend()
plt.show()

# Error Distribution Plot
errors = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(errors, bins=30, kde=True, color="purple")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution (Random Forest)")
plt.show()