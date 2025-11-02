#!/usr/bin/env python
# coding: utf-8

# # **CO₂ Contact Angle Predition Challenge**
# 

# In[ ]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from numpy.linalg import pinv
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# In[ ]:


# Mount drive for data
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# read in dataset
train = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Contact_Angle Project/Train.csv")
train = train.sort_values(by=['mineral','contact_type', 'theta0']).reset_index(drop=True)

test = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Contact_Angle Project/Test.csv")

train.head()


# In[ ]:


# understanding if there are null values
train.isnull().sum()


# In[ ]:


# checking for null values
test.isna().sum()


# In[ ]:


# train data info
train.info()


# In[ ]:


# summary of train data
train.describe()


# In[ ]:


def plot_overall_kde(df, variables=None, log_transform=False):
    """
    Plots KDE (Kernel Density Estimate) for selected variables in the entire dataset
    (no grouping by mineral type).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing numeric variables.
    variables : list of str, optional
        List of numeric variables to plot. If None, uses all numeric columns.
    log_transform : bool, default=False
        If True, applies strict log-transform (log(x + 1e-6)) before plotting.
    """

    if variables is None:
        variables = df.select_dtypes(include="number").columns.tolist()

    fig, axes = plt.subplots(1, len(variables), figsize=(16, 4))

    # Handle the case of single variable
    if len(variables) == 1:
        axes = [axes]

    for j, var in enumerate(variables):
        ax = axes[j]
        values = df[var].dropna()

        if log_transform:values = np.log1p(values.clip(lower=0))

        sns.kdeplot(values, ax=ax, fill=True, color='steelblue', alpha=0.6, linewidth=2)
        ax.set_title(f"{var}" + (" (log)" if log_transform else ""), fontsize=11)
        ax.set_xlabel(var)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# In[ ]:


plot_overall_kde(
    train,
    variables=["pressure", "temperature", "salinity", "contact_angle"],
    log_transform=True
)


# In[ ]:


# Visualize the `Target` distribution
plt.figure(figsize=(10, 6))
sns.histplot(train["contact_angle"], kde=True, color='steelblue', bins=30)
plt.title("Distribution of Target", fontsize=14)
plt.xlabel("Target")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()



# In[ ]:





# In[ ]:


sns.scatterplot(
    data=train,
    x='temperature',
    y='contact_angle',
    hue='mineral',
    alpha=0.8
)

plt.title('Pressure (Quantile Transformed) vs Contact Angle (Quantile Transformed)')
plt.show()


# In[ ]:


train.columns


# ### **Model Building**

# In[ ]:


# --- Feature & Target Setup ---
target_col = "contact_angle"
feature_cols = ["pressure", "temperature", "salinity", "mineral", "contact_type", "theta0"]

cat_features = ["mineral", "contact_type"]
num_features_to_transform = ["pressure", "temperature", "salinity"]
num_features_pass = ["theta0"]

X = train[feature_cols]
y = train["contact_angle"]

# --- Split data ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Preprocessing ---
cat_transformer = OneHotEncoder(handle_unknown="ignore")
standard_scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_transformer, cat_features),
        ("num", standard_scaler, num_features_to_transform)
    ],
    remainder="passthrough"  # keeps theta0 as is
)

# --- Model ---
gbr_model = GradientBoostingRegressor(
    n_estimators= 790,
    learning_rate=0.008,
    max_depth= 6,
    subsample=0.85,
    random_state=42
)

# --- Pipeline ---
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", gbr_model)
])

# --- Fit model ---
pipe.fit(X_train, y_train)

# --- Predict in quantile space ---
y_pred = pipe.predict(X_val)

# --- Evaluate model ---
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"MSE: {mse:.4f}")
print(f"MAE (°): {mae:.4f}")
print(f"R²(°): {r2:.4f}")


# In[ ]:


# mcalculate mape
mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
print(f"Validation MAPE: {mape:.2f}%")


# In[ ]:


def metrics(y_exp, y_pred):
    y_exp = np.array(y_exp, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    N = len(y_exp)

    # AARE (%)
    AARE = (np.sum(np.abs((y_pred - y_exp) / y_exp)) / N) * 100

    # RMSE
    RMSE = np.sqrt(np.sum((y_pred - y_exp) ** 2) / N)

    # Correlation coefficient (R)
    numerator = np.sum((y_exp - np.mean(y_exp)) * (y_pred - np.mean(y_pred)))
    denominator = np.sqrt(np.sum((y_exp - np.mean(y_exp)) ** 2) * np.sum((y_pred - np.mean(y_pred)) ** 2))
    R = numerator / denominator

    return AARE, RMSE, R

AARE, RMSE, R = metrics(y_val, y_pred)
print(f"AARE (%) = {AARE:.4f}")
print(f"RMSE = {RMSE:.4f}")
print(f"R = {R:.4f}")


# In[ ]:


plt.scatter(y_val, y_pred, alpha=0.6, s=5)
plt.plot([0, 140], [0, 140], 'b--', lw=1)
plt.grid(True, which='both', linestyle='--', linewidth=0.2)
plt.tick_params(axis='both', which='both', direction='in',
                length=6, width=0.5, top=True, right=True)
plt.xlabel('Experimental Contact Angle (°)')
plt.ylabel('Predicted Contact Angle (°)')
# plt.title('Predicted vs Experimental Contact Angle')
plt.text(10, 100, f'R = {R:.4f}\nRMSE = {RMSE:.2f}\nAARE = {AARE:.2f}%', fontsize=10,
          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round', pad=0.5))
plt.xlim(0, 140)
plt.ylim(0, 140)
plt.show()


# In[ ]:


import pickle
# Save Model
model_file = f"contact_angle_pred_model.bin"

with open(model_file, "wb") as f_out:
    pickle.dump(pipe, f_out)

# Load Model 
with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)

# Predict on test data
test_pred = pipe.predict(test)
Prediction= pd.DataFrame({"id":test.index,
                           "contact_angle":test_pred})

print(f"Prediction\n {Prediction}")

