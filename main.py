import pandas as pd
import numpy as np

# Import functions from Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Load the dataset
DATA_PATH = "house_price_dataset.csv"      
df = pd.read_csv(DATA_PATH)
print(df)
df.head()
cols = ['City', 'Locality_Tier', 'Super_Area_sqft', 'Carpet_Area_sqft','Furnishing','Crime_Rate_Index','Market_Price_INR']
df=df[cols]
# -------------------- Data Cleaning --------------------

# Check for missing values
print(df.isnull().sum())

# Convert categorical columns into numerical values
df = pd.get_dummies(
    df,
    columns=['City', 'Locality_Tier', 'Furnishing'],
    drop_first=True,
    dtype=int
)
# Show all columns
pd.set_option('display.max_columns', None)

# Display the first 20 rows
print("First 20 rows:")
print(df.head(20))

# Display the last 20 rows
print("\nLast 20 rows:")
print(df.tail(20))

df.head()
X=df.drop('Market_Price_INR', axis=1)
y=df['Market_Price_INR']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, preds))
print("R2 Score:", r2_score(y_test, preds))