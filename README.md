# New York Taxi Fare Prediction Using Linear Regression

## Project Overview

This project implements a **Linear Regression** model to predict taxi fares using the **New York Taxi dataset**. The objective is to understand the complete machine learning regression workflow, including data preprocessing, feature selection, model training, prediction, and performance evaluation.

---

## Dataset

The dataset contains taxi trip information, including:

- VendorID
- Passenger Count
- Trip Distance
- Rate Code ID
- Pickup Location ID
- Drop-off Location ID
- Fare Amount (Target Variable)

The target variable is:

- **fare_amount**

---

## Features Used

The following features were selected for model training:

- trip_distance
- passenger_count
- RatecodeID
- PULocationID
- DOLocationID

Target:

- fare_amount

---

## Technologies Used

- Python 3.x
- Pandas
- Scikit-learn

---

## Libraries

Install the required libraries before running the project.

```bash
pip install pandas scikit-learn
```

---

## Project Workflow

1. Load the dataset.
2. Remove missing values.
3. Select relevant features.
4. Split the dataset into training and testing sets.
5. Train a Linear Regression model.
6. Predict taxi fares.
7. Evaluate the model using:
   - Mean Squared Error (MSE)
   - R² Score

---

## File Structure

```
TaxiFarePrediction/
│
├── new_york_taxi.csv
├── linear_regression.py
├── README.md
```

---

## How to Run

1. Place the dataset (`new_york_taxi.csv`) in the project folder.
2. Open a terminal in the project directory.
3. Run the program:

```bash
python linear_regression.py
```

---

## Expected Output

The program displays:

- First five rows of the dataset
- Dataset columns
- Mean Squared Error (MSE)
- R² Score
- Model coefficients
- Actual vs Predicted fare values

---

## Machine Learning Algorithm

**Algorithm:** Linear Regression

Linear Regression is a supervised machine learning algorithm used for predicting continuous numerical values. It models the relationship between independent variables and the target variable by fitting a linear equation to the observed data.

---

## Evaluation Metrics

### Mean Squared Error (MSE)

Measures the average squared difference between the actual and predicted fare values.

### R² Score

Represents how well the regression model explains the variance in the target variable. A value closer to 1 indicates better model performance.

---

## Future Enhancements

- Feature engineering
- Outlier detection
- Hyperparameter optimization
- Comparison with Decision Tree and Random Forest Regression
- Cross-validation
- Data visualization

---

## Author

**Mr. Prince**

Machine Learning Mini Project
