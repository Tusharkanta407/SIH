import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# File paths
file_2011 = "C:\\Users\\hp\\OneDrive\\Documents\\Desktop\\new project\\data\\State_wise_Ground_Water_Resources_Data_of_India_(In_bcm)_of_2011.csv"
file_2017 = "C:\\Users\\hp\\OneDrive\\Documents\\Desktop\\new project\\data\\State_wise_Ground_Water_Resources_Data_of_India_(In_bcm)_of_2017.csv"

# Read the data
data_2011 = pd.read_csv(file_2011, on_bad_lines='skip')
data_2017 = pd.read_csv(file_2017, on_bad_lines='skip')

# Print first few rows to understand data
print("2011 Data Sample:")
print(data_2011.head())
print("\n2017 Data Sample:")
print(data_2017.head())

# Print columns and types
print("2011 Data Columns and Types:")
print(data_2011.dtypes)
print("\n2017 Data Columns and Types:")
print(data_2017.dtypes)

# Clean column names
data_2011.columns = [col.strip() for col in data_2011.columns]
data_2017.columns = [col.strip() for col in data_2017.columns]

# Rename columns to a consistent name
data_2011.rename(columns={'State wise Ground Water Resources Data of India (In bcm) of 2017': 'Stage of Goundwater Development (%)'}, inplace=True)
data_2017.rename(columns={'State wise Ground Water Resources Data of India (In bcm) of 2017': 'Stage of Goundwater Development (%)'}, inplace=True)

# Combine datasets
data_combined = pd.concat([data_2011, data_2017], axis=0)

# Print combined data info
print("Combined Data Columns and Types:")
print(data_combined.dtypes)
print("\nCombined Data Sample:")
print(data_combined.head())

# Check if target column exists
target_column = 'Stage of Goundwater Development (%)'
if target_column not in data_combined.columns:
    raise ValueError(f"Target column '{target_column}' not found in the data")

# Convert columns to numeric, coerce errors to NaN
data_combined = data_combined.apply(pd.to_numeric, errors='coerce')

# Print data types after conversion
print("Data Types After Conversion:")
print(data_combined.dtypes)

# Check number of rows before and after dropping NaN
print(f"Number of rows before dropping NaN: {data_combined.shape[0]}")
data_combined.dropna(inplace=True)
print(f"Number of rows after dropping NaN: {data_combined.shape[0]}")

# Check if we have enough data
if data_combined.empty:
    raise ValueError("Data is empty after preprocessing")

# Prepare features and target
X = data_combined.drop(columns=[target_column], errors='ignore')  # Features
y = data_combined[target_column]  # Target

# Ensure there's data to split
if X.empty or y.empty:
    raise ValueError("Feature or target data is empty after preprocessing")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict for 2019 data
file_2019 = 'path/to/your/2019_data.csv'
data_2019 = pd.read_csv(file_2019, on_bad_lines='skip')

# Clean and preprocess 2019 data
data_2019.columns = [col.strip() for col in data_2019.columns]
data_2019.rename(columns={'State wise Ground Water Resources Data of India (In bcm) of 2017': 'Stage of Goundwater Development (%)'}, inplace=True)
data_2019 = data_2019.apply(pd.to_numeric, errors='coerce')
data_2019.fillna(data_2019.mean(), inplace=True)

# Prepare 2019 data for prediction
X_2019 = data_2019.drop(columns=[target_column], errors='ignore')
y_2019_pred = model.predict(X_2019)

print("Predicted 2019 Groundwater Levels:")
print(y_2019_pred)
