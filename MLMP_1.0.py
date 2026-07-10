import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, confusion_matrix
)

# DATASET 1: HOUSE PRICE PREDICTION (REGRESSION TASK)

# Load the Dataset via File Dialog 

# Hide the main Tkinter root window so only the file picker pops up

root = tk.Tk()
root.withdraw()

# Open the explorer window to choose the house prices CSV file

selected_path = filedialog.askopenfilename(
    title="Select train_house.CSV File",
    filetypes=[("CSV Files", "*.csv")]
)

if selected_path:
    house_prices = pd.read_csv(selected_path)

# Feature and Target Separation 

# Split the dataset into independent variables (X) and target variable (y)

X_house = house_prices.drop("SalePrice", axis=1)
y_house = house_prices["SalePrice"]

# Identify numerical and categorical columns for distinct preprocessing

num_features = X_house.select_dtypes(include=["int64", "float64"]).columns
cat_features = X_house.select_dtypes(include=["object"]).columns

# Data Preprocessing 

# Handle missing values in numerical columns by replacing NaNs with the median

X_num = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(X_house[num_features]),
    columns=num_features
)

# Convert categorical columns into dummy/indicator variables (One-Hot Encoding)

X_cat = pd.get_dummies(X_house[cat_features])

# Recombine processed numerical and categorical features

X_house_processed = pd.concat([X_num, X_cat], axis=1)

# Feature Scaling: Standardize features to have a mean of 0 and variance of 1

scaler_house = StandardScaler()
X_house_processed = scaler_house.fit_transform(X_house_processed)

# Train-Test Split 

# Split the data into 80% training and 20% testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X_house_processed, y_house, test_size=0.2, random_state=42
)

# Initialize a list to accumulate runtime and baseline performance metrics

regression_results = []

# Model 1: Linear Regression 

lr = LinearRegression()

# Time the training process

start = time.perf_counter()
lr.fit(X_train, y_train)
train_time = time.perf_counter() - start

# Time the prediction process

start = time.perf_counter()
y_pred_lr = lr.predict(X_test)
pred_time = time.perf_counter() - start

regression_results.append([
    "Linear Regression", r2_score(y_test, y_pred_lr), train_time, pred_time
])

# Model 2: Polynomial Regression 

# Generate polynomial features (interaction and squared terms) up to degree 2

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_reg = LinearRegression()

# Time both the feature transformation and model fitting

start = time.perf_counter()
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
poly_reg.fit(X_train_poly, y_train)
train_time = time.perf_counter() - start

# Time the prediction process using the polynomial features

start = time.perf_counter()
y_pred_poly = poly_reg.predict(X_test_poly)
pred_time = time.perf_counter() - start

regression_results.append([
    "Polynomial Regression", r2_score(y_test, y_pred_poly), train_time, pred_time
])

# Model 3: K-Nearest Neighbors (KNN) Regressor 

knn_reg = KNeighborsRegressor(n_neighbors=5)

# Time the training process

start = time.perf_counter()
knn_reg.fit(X_train, y_train)
train_time = time.perf_counter() - start

# Time the prediction process

start = time.perf_counter()
y_pred_knn = knn_reg.predict(X_test)
pred_time = time.perf_counter() - start

regression_results.append([
    "KNN Regressor", r2_score(y_test, y_pred_knn), train_time, pred_time
])

# Model 4: Decision Tree Regressor 

dt_reg = DecisionTreeRegressor(random_state=42)

# Time the training process

start = time.perf_counter()
dt_reg.fit(X_train, y_train)
train_time = time.perf_counter() - start

# Time the prediction process

start = time.perf_counter()
y_pred_dt = dt_reg.predict(X_test)
pred_time = time.perf_counter() - start

regression_results.append([
    "Decision Tree Regressor", r2_score(y_test, y_pred_dt), train_time, pred_time
])

# Print Regression Results Summary 

print("\n=== Regression Model Efficiency & Baseline Performance ===")
results_reg_time = pd.DataFrame(
    regression_results,
    columns=["Model", "R² Score", "Training Time (s)", "Prediction Time (s)"]
)
print(results_reg_time)

# Helper function to compute a full range of regression evaluation metrics

def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

# Compile and print all detailed error metrics into a side-by-side DataFrame

results_reg = pd.DataFrame({
    "Linear Regression": regression_metrics(y_test, y_pred_lr),
    "Polynomial Regression": regression_metrics(y_test, y_pred_poly),
    "KNN Regressor": regression_metrics(y_test, y_pred_knn),
    "Decision Tree": regression_metrics(y_test, y_pred_dt)
})
print("\n=== Detailed Regression Metrics ===")
print(results_reg)

# Data Visualization (Regression) 

# Scatter plot comparing actual vs. predicted values for Polynomial Regression

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_poly, alpha=0.6)

# Plot a reference line representing a perfect prediction (y = x)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("Actual vs Predicted House Prices (Polynomial Regression)")
plt.tight_layout()
plt.show()


# DATASET 2: TITANIC SURVIVAL PREDICTION (CLASSIFICATION TASK)


# Load the Dataset via File Dialog 
root = tk.Tk()
root.withdraw()

selected_path = filedialog.askopenfilename(
    title="Select train_titanic.CSV File",
    filetypes=[("CSV Files", "*.csv")]
)

if selected_path:
    titanic = pd.read_csv(selected_path)

# Feature and Target Separation & Cleaning 

# Remove uninformative/high-cardinality metadata columns

titanic = titanic.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Split features and target variable

X_titanic = titanic.drop("Survived", axis=1)
y_titanic = titanic["Survived"]

# Convert categorical variables into binary dummy variables (drop_first avoids collinearity)

X_titanic = pd.get_dummies(X_titanic, drop_first=True)

# Fill up any missing value using column median

X_titanic = SimpleImputer(strategy="median").fit_transform(X_titanic)

# Train-Test Split & Scaling 

# Split classification data into 80% training and 20% testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X_titanic, y_titanic, test_size=0.2, random_state=42
)

# Normalize the dataset so that distance based algorithms (SVM, KNN) work properly

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an empty list to hold time and accuracy results

classification_results = []

# Model 1: Logistic Regression 

log_reg = LogisticRegression(max_iter=1000)

# Time the training process

start = time.perf_counter()
log_reg.fit(X_train, y_train)
train_time = time.perf_counter() - start

# Time the prediction process

start = time.perf_counter()
y_pred_log = log_reg.predict(X_test)
pred_time = time.perf_counter() - start

classification_results.append([
    "Logistic Regression", accuracy_score(y_test, y_pred_log), train_time, pred_time
])

# Model 2: Support Vector Machine (SVM) Classifier 

svm = SVC()

# Time the training process

start = time.perf_counter()
svm.fit(X_train, y_train)
train_time = time.perf_counter() - start

# Time the prediction process

start = time.perf_counter()
y_pred_svm = svm.predict(X_test)
pred_time = time.perf_counter() - start

classification_results.append([
    "SVM", accuracy_score(y_test, y_pred_svm), train_time, pred_time
])

# Model 3: K-Nearest Neighbors (KNN) Classifier 

knn_clf = KNeighborsClassifier(n_neighbors=5)

# Time the training process

start = time.perf_counter()
knn_clf.fit(X_train, y_train)
train_time = time.perf_counter() - start

# Time the prediction process

start = time.perf_counter()
y_pred_knn = knn_clf.predict(X_test)
pred_time = time.perf_counter() - start

classification_results.append([
    "KNN", accuracy_score(y_test, y_pred_knn), train_time, pred_time
])

# Model 4: Decision Tree Classifier 

dt_clf = DecisionTreeClassifier(random_state=42)

# Time the training process

start = time.perf_counter()
dt_clf.fit(X_train, y_train)
train_time = time.perf_counter() - start

# Time the prediction process

start = time.perf_counter()
y_pred_dt = dt_clf.predict(X_test)
pred_time = time.perf_counter() - start

classification_results.append([
    "Decision Tree", accuracy_score(y_test, y_pred_dt), train_time, pred_time
])

# Print Classification Results Summary 

print("\n=== Classification Model Efficiency & Baseline Performance ===")
results_clf_time = pd.DataFrame(
    classification_results,
    columns=["Model", "Accuracy", "Training Time (s)", "Prediction Time (s)"]
)
print(results_clf_time)

# Function used to calculate various classification evaluation metrics

def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }

# Generate and display a DataFrame containing all classification metrics side-by-side

results_clf = pd.DataFrame({
    "Logistic Regression": classification_metrics(y_test, y_pred_log),
    "SVM": classification_metrics(y_test, y_pred_svm),
    "KNN": classification_metrics(y_test, y_pred_knn),
    "Decision Tree": classification_metrics(y_test, y_pred_dt)
})
print("\n=== Detailed Classification Metrics ===")
print(results_clf)

# Data Visualization (Classification) 

# Generate and plot a heatmap representing the Confusion Matrix for Logistic Regression

plt.figure()
cm = confusion_matrix(y_test, y_pred_log)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Logistic Regression")
plt.show()