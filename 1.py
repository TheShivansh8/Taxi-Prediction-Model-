# ============================================
# New York Taxi Fare Prediction
# Linear Regression
# ============================================

# Import Libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================================
# Step 1: Load Dataset
# ============================================

df = pd.read_csv(r"C:\Users\shiva_dwq2zco\Downloads\new_york_taxi.csv")

# ============================================
# Step 2: Display Dataset
# ============================================

print("First 5 Rows")
print(df.head())

print("\nColumns in Dataset")
print(df.columns)

# ============================================
# Step 3: Remove Missing Values
# ============================================

df = df.dropna()

# ============================================
# Step 4: Select Features and Target
# ============================================

X = df[['trip_distance',
        'passenger_count',
        'RatecodeID',
        'PULocationID',
        'DOLocationID']]

y = df['fare_amount']

# ============================================
# Step 5: Split Dataset
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ============================================
# Step 6: Train Model
# ============================================

model = LinearRegression()

model.fit(X_train, y_train)

# ============================================
# Step 7: Prediction
# ============================================

y_pred = model.predict(X_test)

# ============================================
# Step 8: Evaluation
# ============================================

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== Model Performance ==========")
print("Mean Squared Error :", mse)
print("R² Score :", r2)

# ============================================
# Step 9: Coefficients
# ============================================

print("\nIntercept")
print(model.intercept_)

print("\nFeature Coefficients")

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature} : {coefficient}")

# ============================================
# Step 10: Actual vs Predicted
# ============================================

results = pd.DataFrame({
    "Actual Fare": y_test.values,
    "Predicted Fare": y_pred
})

print("\nActual vs Predicted")
print(results.head(10))
