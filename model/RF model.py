import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap


np.random.seed(42)

# Load dataset
Plastic = pd.read_excel('train_random_all_1_6.xlsx')

X = Plastic.values[:, 0:6]
Y = Plastic.values[:, 6:12]

print(X)
print(f'X shape: {X.shape}')
print(f'First Y row:\n{Y[0]}')

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Manual random search for hyperparameters
best_score = -np.inf
best_params = None
results = []

for iteration in range(50):
    n_estimators = np.random.randint(1, 101)
    max_depth = np.random.randint(1, 51)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5
    )
    cv_score_mean = np.mean(cv_scores)

    results.append({
        'iteration': iteration + 1,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'cv_score': cv_score_mean
    })

    if cv_score_mean > best_score:
        best_score = cv_score_mean
        best_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }

print('Best parameters:', best_params)

# Train final model using selected hyperparameters
model = RandomForestRegressor(
    n_estimators=67,
    max_depth=27,
    random_state=42
)
model.fit(X_train, y_train)

# Target variable names
target_labels = ["PET", "PP_PE", "PVC", "PS", "PA", "PC"]

# --------------------------------------------------
# 1. Predict on training and test sets
# --------------------------------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# --------------------------------------------------
# 2. Compute residuals
# --------------------------------------------------
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# --------------------------------------------------
# 3. Export per-target prediction and residual data
# --------------------------------------------------
for i, target_name in enumerate(target_labels):

    train_actual = y_train[:, i]
    train_pred = y_train_pred[:, i]
    train_resid = train_residuals[:, i]

    test_actual = y_test[:, i]
    test_pred = y_test_pred[:, i]
    test_resid = test_residuals[:, i]

    train_df = pd.DataFrame({
        'train_actual': train_actual,
        'train_predicted': train_pred,
        'train_residual': train_resid
    })

    test_df = pd.DataFrame({
        'test_actual': test_actual,
        'test_predicted': test_pred,
        'test_residual': test_resid
    })

    max_length = max(len(train_df), len(test_df))
    combined_df = pd.DataFrame(index=range(max_length))

    for col in train_df.columns:
        combined_df[col] = np.nan
        combined_df.loc[:len(train_df) - 1, col] = train_df[col].values

    for col in test_df.columns:
        combined_df[col] = np.nan
        combined_df.loc[:len(test_df) - 1, col] = test_df[col].values

    print(f"{target_name}:")
    print(f"  Train samples: {len(train_actual)}")
    print(f"  Test samples: {len(test_actual)}")

print("All prediction and residual data prepared.")

# --------------------------------------------------
# 4. Enforce non-negativity and closure constraint
# --------------------------------------------------
def enforce_nonneg_and_closure(y_pred, total_sum=100.0, eps=1e-12):
    y_pred = np.asarray(y_pred)

    y_clipped = np.clip(y_pred, a_min=0.0, a_max=None)

    row_sums = np.sum(y_clipped, axis=1, keepdims=True)
    row_sums[row_sums < eps] = total_sum

    y_normalized = y_clipped / row_sums * total_sum

    return y_normalized


# Apply constraint correction (optional post-processing)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# --------------------------------------------------
# 5. Model performance evaluation
# --------------------------------------------------
train_r2 = r2_score(y_train, y_train_pred, multioutput='uniform_average')
test_r2 = r2_score(y_test, y_test_pred, multioutput='uniform_average')

train_rmse = mean_squared_error(
    y_train, y_train_pred, squared=False
)
test_rmse = mean_squared_error(
    y_test, y_test_pred, squared=False
)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("Train R2:", np.round(train_r2, 4))
print("Train RMSE:", np.round(train_rmse, 4))
print("Train MAE:", np.round(train_mae, 4))
print("Test R2:", np.round(test_r2, 4))
print("Test RMSE:", np.round(test_rmse, 4))
print("Test MAE:", np.round(test_mae, 4))

