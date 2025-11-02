#!/usr/bin/env python
# coding: utf-8

# ===============================================================
#   CO₂ Contact Angle Prediction Challenge
#   Author: Ahaji Victor Kelechi
#   Description: Gradient Boosting model for contact angle prediction
# ===============================================================

# --- Import Libraries ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.linalg import pinv
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# File Paths
train_path = r"C:\Users\Ahaji Kelechi\Desktop\ML Projects\Contact_angle\Train.csv"
test_path = r"C:\Users\Ahaji Kelechi\Desktop\ML Projects\Contact_angle\Test.csv"

# Check for Dataset Existence
if not os.path.exists(train_path):
    raise FileNotFoundError(f"❌ Train file not found at {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"❌ Test file not found at {test_path}")

print("✅ Dataset files found successfully.\n")

# Read in Datasets
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Sort training data for consistency
train = train.sort_values(by=['mineral', 'contact_type', 'theta0']).reset_index(drop=True)

# Quick Checks
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nMissing values in Train:\n", train.isnull().sum())
print("\nMissing values in Test:\n", test.isnull().sum())

# Plot KDE Distribution Function
def plot_overall_kde(df, variables=None, log_transform=False):
    """
    Plots KDE (Kernel Density Estimate) for selected numeric variables.
    """
    if variables is None:
        variables = df.select_dtypes(include="number").columns.tolist()

    fig, axes = plt.subplots(1, len(variables), figsize=(16, 4))

    if len(variables) == 1:
        axes = [axes]

    for j, var in enumerate(variables):
        ax = axes[j]
        values = df[var].dropna()

        if log_transform:
            values = np.log1p(values.clip(lower=0))

        sns.kdeplot(values, ax=ax, fill=True, color='steelblue', alpha=0.6, linewidth=2)
        ax.set_title(f"{var}" + (" (log)" if log_transform else ""), fontsize=11)
        ax.set_xlabel(var)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# Plot Distributions
plot_overall_kde(train, ["pressure", "temperature", "salinity", "contact_angle"], log_transform=True)

plt.figure(figsize=(10, 6))
sns.histplot(train["contact_angle"], kde=True, color='steelblue', bins=30)
plt.title("Distribution of Target: Contact Angle")
plt.xlabel("Contact Angle (°)")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()

sns.scatterplot(data=train, x='temperature', y='contact_angle', hue='mineral', alpha=0.8)
plt.title('Temperature vs Contact Angle by Mineral Type')
plt.show()

# Feature & Target Setup
target_col = "contact_angle"
feature_cols = ["pressure", "temperature", "salinity", "mineral", "contact_type", "theta0"]

cat_features = ["mineral", "contact_type"]
num_features_to_transform = ["pressure", "temperature", "salinity"]
num_features_pass = ["theta0"]

X = train[feature_cols]
y = train[target_col]

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
cat_transformer = OneHotEncoder(handle_unknown="ignore")
standard_scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, cat_features),
        ("num", standard_scaler, num_features_to_transform)
    ],
    remainder="passthrough"  # keeps theta0 as is
)

# Model
gbr_model = GradientBoostingRegressor(
    n_estimators=790,
    learning_rate=0.008,
    max_depth=6,
    subsample=0.85,
    random_state=42
)

# Pipeline
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", gbr_model)
])

# Fit Model
print("🚀 Training model...")
pipe.fit(X_train, y_train)
print("Model training completed.\n")

# --- Evaluate Model ---
y_pred = pipe.predict(X_val)

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

print(f"Model Performance Metrics:")
print(f"MSE:  {mse:.4f}")
print(f"MAE:  {mae:.4f}°")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%\n")

# Additional Metrics
def metrics(y_exp, y_pred):
    y_exp = np.array(y_exp, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    N = len(y_exp)
    AARE = (np.sum(np.abs((y_pred - y_exp) / y_exp)) / N) * 100
    RMSE = np.sqrt(np.sum((y_pred - y_exp) ** 2) / N)
    R = np.corrcoef(y_exp, y_pred)[0, 1]
    return AARE, RMSE, R

AARE, RMSE, R = metrics(y_val, y_pred)
print(f"AARE: {AARE:.4f}% | RMSE: {RMSE:.4f} | R: {R:.4f}\n")

# Plot Experimental vs Predicted
plt.scatter(y_val, y_pred, alpha=0.6, s=5)
plt.plot([0, 140], [0, 140], 'b--', lw=1)
plt.grid(True, linestyle='--', linewidth=0.2)
plt.xlabel('Experimental Contact Angle (°)')
plt.ylabel('Predicted Contact Angle (°)')
plt.text(10, 100, f'R = {R:.4f}\nRMSE = {RMSE:.2f}\nAARE = {AARE:.2f}%',
         fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
plt.xlim(0, 140)
plt.ylim(0, 140)
plt.show()

# Save Model
model_file = "contact_angle_pred_model.bin"
with open(model_file, "wb") as f_out:
    pickle.dump(pipe, f_out)

print(f" Model saved as: {model_file}\n")

with open(model_file, "rb") as f_in:
    model_ = pickle.load(f_in)

# Predict on unseen data
def predict_contact_angle(data):
    y_pred = model_.predict(data)
    # convert to numpy array of floats
    y_pred = np.array(y_pred, dtype=float)

    # Determine number of samples
    if isinstance(data, pd.DataFrame):
        ids = data.index
    elif isinstance(data, (np.ndarray,list)):
        ids = range(len(data))
    elif isinstance(data, dict):
        ids = range(len(next(iter(data.values()))))
    else:
        raise ValueError("Unsupported data type for prediction.")
    
    return pd.DataFrame({"id": ids, "contact_angle": y_pred})
