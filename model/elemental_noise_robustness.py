import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ------------------------
# 1. Load dataset
# ------------------------
Plastic = pd.read_excel('train random all 1_6.xlsx')
X = Plastic.values[:, 0:6]  # C, H, O, N, H/C, O/C
Y = Plastic.values[:, 6:12]

# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ------------------------
# 2. Train Random Forest model
# ------------------------
model = RandomForestRegressor(n_estimators=67, max_depth=27, random_state=42)
model.fit(X_train, y_train)


# ------------------------
# 3. Function to add relative Gaussian noise to elemental features
# ------------------------
def add_elemental_noise(X, sigma=0.03, normalize=True, random_state=42):
    """
    Add relative Gaussian noise to elemental features.

    Parameters:
    X : ndarray, shape (n_samples, n_features)
    sigma : float
        Relative noise level (e.g., 0.03 for ±3%)
    normalize : bool
        Whether to renormalize elemental fractions to sum to average row sum
    """
    rng = np.random.default_rng(random_state)
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    X_noisy = X * (1 + noise)

    # Avoid negative values
    X_noisy[X_noisy < 0] = 0.0

    if normalize:
        row_sum = X_noisy.sum(axis=1, keepdims=True)
        X_noisy = X_noisy / row_sum * row_sum.mean()

    return X_noisy


# ------------------------
# 4. Evaluate model robustness to elemental noise
# ------------------------
noise_levels = [0.0, 0.01, 0.03, 0.05]
results = []

for sigma in noise_levels:
    if sigma == 0.0:
        X_test_noisy = X_test.copy()
    else:
        X_test_noisy = add_elemental_noise(
            X_test, sigma=sigma, normalize=True, random_state=42
        )

    y_pred = model.predict(X_test_noisy)

    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results.append([sigma, r2, rmse, mae])

# Convert results to DataFrame
results_df = pd.DataFrame(
    results, columns=['Noise level (σ)', 'R2', 'RMSE', 'MAE']
)

print(results_df)

# Export results to Excel
results_df.to_excel('elemental_noise_robustness_results.xlsx', index=False)
print("Robustness results exported to 'elemental_noise_robustness_results.xlsx'")

# ------------------------
# 5. Plot R² vs noise level
# ------------------------
plt.figure(figsize=(6, 4))
plt.plot(results_df['Noise level (σ)'] * 100, results_df['R2'], marker='o', color='tab:blue')
plt.xlabel('Elemental noise level (%)')
plt.ylabel('R²')
plt.title('Model robustness to elemental measurement noise')
plt.grid(True)
plt.tight_layout()
plt.show()
